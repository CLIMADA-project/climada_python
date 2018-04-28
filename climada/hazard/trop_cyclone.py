"""
Define TropCyclone class and IBTracs reader.
"""

__all__ = ['TropCyclone',
           'read_ibtracs',
           'gust_from_track',
           'interp_track']

import logging
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from pint import UnitRegistry
from pathos.multiprocessing import ProcessingPool as Pool

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.util.config import CONFIG
from climada.util.constants import GLB_CENTROIDS_MAT, ONE_LAT_KM
from climada.util.interpolation import dist_sqr_approx
from climada.util.files_handler import to_list, get_file_names
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
""" Hazard type acronym for Tropical Cyclone """

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 135, 1000]
""" Saffir-Simpson Hurricane Wind Scale """

INLAND_MAX_DIST_KM = 1000
""" Maximum inland distance of the centroids in km """

CENTR_NODE_MAX_DIST_KM = 300
""" Maximum distance between centroid and TC track node in km """

MIN_WIND_THRES_KN = 34
""" Minimum windspeed stored in knots (for storage management reasons)"""

ENS_SIZE = 9
"""Number of created tracks per original track """

class TropCyclone(Hazard):
    """Contains tropical cyclone events obtained from a csv IBTrACS file.

    Attributes:
        tracks (list(xarray.Dataset)): list of tropical cyclone tracks
    """

    def __init__(self, file_name='', description='', centroids=None):
        """Initialize values from given file, if given. Input file contains
        Hazard data.

        Parameters:
            file_name (str or list(str), optional): file name(s) or folder name
                containing the files to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids

        Raises:
            ValueError
        """
        Hazard.__init__(self, HAZ_TYPE, file_name, description, centroids)

    def set(self, files, descriptions='', centroids=None, model='H08'):
        """Set and check hazard, and centroids if not provided, from input
        csv IBTrACS file. Parallel through files.

        Parameters:
            files (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
            descriptions (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids
            model (str, optional): model to compute gust. Default Holland2008.

        Raises:
            ValueError
        """
        all_files = get_file_names(files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        centr_list = to_list(len(all_files), centroids, 'centroids')
        model_list = to_list(len(all_files), model, 'model')
        self.clear()
        tc_part = Pool().map(self._set_one, all_files, desc_list, centr_list,
                             model_list)
        for tc_haz in tc_part:
            self.append(tc_haz)

        # Compute events frequency
        delta_time = \
            np.max([track.time.dt.year for track in self.tracks]) - \
            np.min([track.time.dt.year for track in self.tracks]) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = self.frequency / delta_time / ens_size

    def append(self, hazard):
        """Check and append variables of input TropCyclone to current.
        Repeated events and centroids will be overwritten. Tracks are appended.

        Parameters:
            hazard (TropCyclone or Hazard): tropical cyclone data to append
                to current

        Raises:
            ValueError
        """
        start = False
        if self.event_id.size == 0:
            start = True
        super(TropCyclone, self).append(hazard)
        # the first append has already copied all the attributes
        if hasattr(hazard, 'tracks') and not start:
            self.tracks.extend(hazard.tracks)

    def plot_tracks(self):
        """Track over earth. Historical events are blue, probabilistic black.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if len(self.tracks) < self.event_id.size:
            LOGGER.warning('Number of tracks %s != number of events %s.',
                           len(self.tracks), self.event_id.size)

        _, axis = plot.make_map()
        axis = axis[0][0]
        min_lat, max_lat = 10000, -10000
        min_lon, max_lon = 10000, -10000
        for track in self.tracks:
            min_lat, max_lat = min(min_lat, np.min(track.lat.values)), \
                                max(max_lat, np.max(track.lat.values))
            min_lon, max_lon = min(min_lon, np.min(track.lon.values)), \
                                max(max_lon, np.max(track.lon.values))
        axis.set_extent(([min_lon, max_lon, min_lat, max_lat]))
        plot.add_shapes(axis)
        axis.set_title('TC tracks' + ''.join(self.tag.description))
        for track in self.tracks:
            if track.orig_event_flag:
                color = 'b'
            else:
                color = 'k'
            axis.plot(track.lon.values, track.lat.values, c=color)

    def clear(self):
        """Clear and reinitialize all data."""
        super(TropCyclone, self).clear()
        self.tracks = list() # [xr.Dataset()]

    def set_random_walk(self):
        """ For every track, compute ENS_SIZE probable tracks """
        prob_tracks = Pool().map(calc_random_walk, self.tracks)
        for tracks in prob_tracks:
            self.tracks.extend(tracks)

    @staticmethod
    def _set_one(file_name, description='', centroids=None, model='H08'):
        """ Set hazard from input file. If centroids are not provided, they are
        read from the same file.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            centroids (Centroids, optional): Centroids instance. Use global
                centroids if not provided.
            model (str, optional): model to compute gust. Default Holland2008.

        Raises:
            ValueError, KeyError

        Returns:
            TropicalCyclone
        """
        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, file_name, description)

        try:
            track = read_ibtracs(file_name)
        except pd.errors.ParserError:
            LOGGER.error('Provide a IBTraCS file in csv format containing'\
                         ' one TC track.')
            raise ValueError

        if centroids is None:
            centroids = Centroids(GLB_CENTROIDS_MAT, 'Global Nat centroids')

        LOGGER.info('Setting TC event from file: %s', file_name)
        new_haz.intensity = gust_from_track(track, centroids, model)
        new_haz.units = 'm/s'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        # frequency set when all tracks available
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.name]
        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        # store date of start
        new_haz.date = np.array([dt.datetime(
            track.time.dt.year[0], track.time.dt.month[0],
            track.time.dt.day[0]).toordinal()])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.tracks.append(track)
        return new_haz

def read_ibtracs(file_name):
    """Read IBTrACS track file.

        Parameters:
            file_name (str): file name containing one IBTrACS track to read

        Returns:
            xarray.Dataset
    """
    dfr = pd.read_csv(file_name)
    name = dfr['ibtracsID'].values[0]

    datetimes = list()
    for time in dfr['isotime'].values:
        year = np.fix(time/1e6)
        time = time - year*1e6
        month = np.fix(time/1e4)
        time = time - month*1e4
        day = np.fix(time/1e2)
        hour = time - day*1e2
        datetimes.append(dt.datetime(int(year), int(month), int(day), \
                                     int(hour)))

    lat = dfr['cgps_lat'].values
    lon = dfr['cgps_lon'].values
    cen_pres = dfr['pcen'].values
    max_sus_wind = dfr['vmax'].values
    max_sus_wind_unit = 'kn'
    cen_pres = _missing_pressure(cen_pres, max_sus_wind, lat, lon)

    track_ds = xr.Dataset()
    track_ds.coords['time'] = ('time', datetimes)
    track_ds.coords['lat'] = ('time', lat)
    track_ds.coords['lon'] = ('time', lon)
    track_ds['time_step'] = ('time', dfr['tint'].values)
    track_ds['radius_max_wind'] = ('time', dfr['rmax'].values)
    track_ds['max_sustained_wind'] = ('time', max_sus_wind)
    track_ds['central_pressure'] = ('time', cen_pres)
    track_ds['environmental_pressure'] = ('time', dfr['penv'].values)
    track_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
    track_ds.attrs['central_pressure_unit'] = 'mb'
    track_ds.attrs['name'] = name
    track_ds.attrs['orig_event_flag'] = bool(dfr['original_data'].values[0])
    track_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
    track_ds.attrs['basin'] = dfr['gen_basin'].values[0]
    track_ds.attrs['id_no'] = float(name.replace('N', '0').replace('S', '1'))
    track_ds.attrs['category'] = _set_category(max_sus_wind, max_sus_wind_unit)

    return track_ds

def gust_from_track(track, centroids=None, model='H08'):
    """ Compute wind gusts at centroids from track. Track is interpolated to
    configured time step.

    Parameters:
        track (xr.Dataset): track infomation
        centroids(Centroids, optional): centroids where gusts are computed. Use
            global centroids if not provided.
        model (str, optional): model to compute gust. Default Holland2008.

    Returns:
        sparse.csr_matrix
    """
    if centroids is None:
        centroids = Centroids(GLB_CENTROIDS_MAT, 'Global Nat centroids')

    # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
    coastal_centr = _coastal_centr_idx(centroids)

    # Interpolate each track to min_time_step values
    track_int = interp_track(track)

    # Compute wind gusts
    intensity = sparse.lil_matrix((1, centroids.id.size))
    intensity[0, coastal_centr] = _windfield_holland(track_int, \
             centroids.coord[coastal_centr, :], model)
    return sparse.csr_matrix(intensity)

def interp_track(track):
    """ Generate interpolated track values to time steps of min_time_step.

    Parameters:
        in_tracks (xr.Dataset): input track

    Returns:
        xr.Dataset
    """
    time_step = str(CONFIG['tc_time_step_h']) + 'H'
    track_int = track.resample(time=time_step).interpolate('linear')
    track_int['time_step'] = ('time', \
                             track_int.time.size * [CONFIG['tc_time_step_h']])
    track_int.coords['lat'] = track.lat.resample(time=time_step).\
                                interpolate('cubic')
    track_int.coords['lon'] = track.lon.resample(time=time_step).\
                                interpolate('cubic')
    track_int.attrs = track.attrs

    return track_int

def calc_random_walk(track, rand_unif_ini=None, rand_unif_ang=None):
    """ Generate random tracks from input track.

    Parameters:
        track (xr.Dataset): TC track
        rand_unif_ini (np.array, optional): array of uniform [0,1) random
            numbers of size 2
        rand_unif_ang (np.array, optional): array of uniform [0,1) random
            numbers of size ENS_SIZE x size track

    Returns:
        list(xr.Dataset)
    """
    # amplitude of max random starting point shift degree longitude
    ens_amp0 = 1.5
    # maximum angle of variation, =pi is like undirected, pi/4 means one quadrant
    max_angle = np.pi/10
    # amplitude of random walk wiggles in degree longitude for 'directed'
    ens_amp = 0.1

    n_dat = track.time.size
    if rand_unif_ini is None or rand_unif_ini.shape != (2, ENS_SIZE):
        rand_unif_ini = np.random.uniform(size=(2, ENS_SIZE))
    if rand_unif_ang is None or rand_unif_ang.size != ENS_SIZE*n_dat:
        rand_unif_ang = np.random.uniform(size=ENS_SIZE*n_dat)

    xy_ini = ens_amp0 * (rand_unif_ini - 0.5)
    tmp_ang = np.cumsum(2 * max_angle * rand_unif_ang - max_angle)
    coord_xy = np.empty((2, ENS_SIZE * n_dat))
    coord_xy[0] = np.cumsum(ens_amp * np.sin(tmp_ang))
    coord_xy[1] = np.cumsum(ens_amp * np.cos(tmp_ang))

    ens_track = list()
    for i_ens in range(ENS_SIZE):
        i_track = track.copy(True)

        d_xy = coord_xy[:, i_ens * n_dat: (i_ens + 1) * n_dat] - \
            np.expand_dims(coord_xy[:, i_ens * n_dat], axis=1)

        d_lat_lon = d_xy + np.expand_dims(xy_ini[:, i_ens], axis=1)

        i_track.lon.values = i_track.lon.values + d_lat_lon[0, :]
        i_track.lat.values = i_track.lat.values + d_lat_lon[1, :]
        i_track.attrs['orig_event_flag'] = False
        i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + str(i_ens+1)
        i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens+1)/100

        ens_track.append(i_track)

    return ens_track

def _missing_pressure(cen_pres, v_max, lat, lon):
    """Deal with missing central pressures."""
    if np.argwhere(cen_pres < 0).size > 0:
        cen_pres = 1024.388 + 0.047*lat - 0.029*lon - 0.818*v_max
    return cen_pres

def _set_category(max_sus_wind, max_sus_wind_unit):
    """Add storm category according to saffir-simpson hurricane scale
   -1 tropical depression
    0 tropical storm
    1 Hurrican category 1
    2 Hurrican category 2
    3 Hurrican category 3
    4 Hurrican category 4
    5 Hurrican category 5
    """
    ureg = UnitRegistry()
    if (max_sus_wind_unit == 'kn') or (max_sus_wind_unit == 'kt'):
        unit = ureg.knot
    elif max_sus_wind_unit == 'mph':
        unit = ureg.mile / ureg.hour
    elif max_sus_wind_unit == 'm/s':
        unit = ureg.meter / ureg.second
    elif max_sus_wind_unit == 'km/h':
        unit = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Wind not recorded in kn, conversion to kn needed.')
        raise ValueError
    max_wind_kn = (np.max(max_sus_wind) * unit).to(ureg.knot).magnitude

    return (np.argwhere(max_wind_kn < SAFFIR_SIM_CAT) - 1)[0][0]

def _windfield_holland(track, centroids, model='H08'):
    """ Compute windfields (in m/s) in centroids using Holland model 08.

    Parameters:
        track (xr.Dataset): track infomation
        centroids(2d np.array): each row is a centroid [lat, lon]
        model (str, optional): Holland model selection. Holland 2008 default.

    Returns:
        sparse.lil_matrix
    """
    ureg = UnitRegistry()

    # Minimum windspeed to m/s
    min_wind_threshold = (MIN_WIND_THRES_KN * ureg.knot).to(ureg.meter / \
                         ureg.second).magnitude

    # Make sure that CentralPressure never exceeds EnvironmentalPressure
    up_pr = np.argwhere(track.central_pressure.values >
                        track.environmental_pressure.values)
    track.central_pressure.values[up_pr] = \
        track.environmental_pressure.values[up_pr]

    # Extrapolate RadiusMaxWind from pressure if not given
    track['radius_max_wind'] = ('time', _extra_rad_max_wind(track, ureg))

    intensity = sparse.lil_matrix((1, centroids.shape[0]))
    centr_cos_lat = np.cos(centroids[:, 0] / 180 * np.pi)
    for i_node in range(1, track.time.size):
        # compute distance to all centroids
        r_arr = np.sqrt(dist_sqr_approx(centroids[:, 0], centroids[:, 1], \
            centr_cos_lat, track.lat.values[i_node], \
            track.lon.values[i_node])) * ONE_LAT_KM

        # Choose centroids that are close enough
        close_centr = np.argwhere(r_arr < CENTR_NODE_MAX_DIST_KM)
        r_arr = r_arr[close_centr]

        # m/s
        v_trans, v_trans_corr = _vtrans_holland(track, i_node, centroids,
                                                close_centr, r_arr, ureg)
        v_ang = _vang_holland(track, i_node, r_arr, v_trans, model)

        v_full = v_trans_corr + v_ang
        v_full[np.isnan(v_full)] = 0
        v_full[v_full < min_wind_threshold] = 0

        # keep maximum instantaneous wind
        intensity[0, close_centr] = np.maximum(
            intensity[0, close_centr].todense(), v_full)

    return intensity

def _coastal_centr_idx(centroids):
    """ Compute centroids indices which are inside INLAND_MAX_DIST_KM and
    with lat < 61 """
    if centroids.dist_coast.size == 0:
        centroids.calc_dist_to_coast()

    return np.logical_and(centroids.dist_coast < INLAND_MAX_DIST_KM,
                          centroids.lat < 61).nonzero()[0]

def _extra_rad_max_wind(track, ureg):
    """ Extrapolate RadiusMaxWind from pressure.

    Parameters:
        track (xr.Dataset): contains TC track information
        ureg (UnitRegistry): units handler

    Returns:
        np.array
    """
    # TODO: alwasy extrapolate???!!!
    # rmax thresholds in nm
    rmax_1 = 15
    rmax_2 = 25
    rmax_3 = 50
    # pressure in mb
    pres_1 = 950
    pres_2 = 980
    pres_3 = 1020
    track.radius_max_wind[track.central_pressure.values <= pres_1] = rmax_1
    to_change = np.logical_and(track.central_pressure.values > pres_1, \
                               track.central_pressure.values <= pres_2). \
                               nonzero()[0]
    track.radius_max_wind[to_change] = \
        (track.central_pressure[to_change] - pres_1) * \
        (rmax_2 - rmax_1)/(pres_2 - pres_1) + rmax_1
    to_change = np.argwhere(track.central_pressure.values > pres_2).squeeze()
    track.radius_max_wind[to_change] = \
        (track.central_pressure[to_change] - pres_2) * \
        (rmax_3 - rmax_2)/(pres_3 - pres_2) + rmax_2

    return (track.radius_max_wind.values * ureg.nautical_mile). \
        to(ureg.kilometer).magnitude

def _vtrans(track, i_node, ureg):
    """ Compute Hollands translation wind without correction  in m/s.

    Parameters:
        track (xr.Dataset): contains TC track information
        i_node (int): track node (point) to compute
        ureg (UnitRegistry): units handler

    Returns:
        float
    """
    dist = np.sqrt(dist_sqr_approx(track.lat[i_node - 1].values, \
        track.lon[i_node-1].values, \
        np.cos(track.lat[i_node - 1].values / 180 * np.pi), \
        track.lat[i_node].values, track.lon[i_node].values)) * ONE_LAT_KM
    dist = (dist * ureg.kilometer).to(ureg.nautical_mile).magnitude

    # nautical miles/hour, limit to 30 nmph
    v_trans = dist / track.time_step[i_node].values
    if v_trans > 30:
        v_trans = 30
    # to m/s
    return (v_trans * ureg.knot).to(ureg.meter / ureg.second).magnitude

def _vtrans_holland(track, i_node, centroids, close_centr, r_arr, ureg):
    """ Compute Hollands translation wind. Returns gust in m/s.

    Parameters:
        track (xr.Dataset): contains TC track information
        i_node (int): track node (point) to compute
        centroids(2d np.array): each row is a centroid [lat, lon]
        close_centr (np.array): indices of selected centroids, the coastal ones
        r_arr (np.array): distance between coastal centroids and track node
        ureg (UnitRegistry): units handler

    Returns:
        v_trans (float), v_trans corrected (np.array)
    """
    v_trans = _vtrans(track, i_node, ureg)

    if i_node == track.lon.size - 1:
        node_dx = (track.lon[i_node] - track.lon[i_node]).values
        node_dy = (track.lat[i_node] - track.lat[i_node]).values
    else:
        node_dx = (track.lon[i_node + 1] - track.lon[i_node]).values
        node_dy = (track.lat[i_node + 1] - track.lat[i_node]).values
    node_dlen = LA.norm([node_dx, node_dy])

    # we use the scalar product of the track forward vector and the vector
    # towards each centroid to figure the angle between and hence whether
    # the translational wind needs to be added (on the right side of the
    # track for Northern hemisphere) and to which extent (100% exactly 90
    # to the right of the track, zero in front of the track)

    # hence, rotate track forward vector 90 degrees clockwise, i.e.
    node_tmp = node_dx
    node_dx = node_dy
    node_dy = -node_tmp

    # the vector towards each centroid
    centroids_dlon = centroids[close_centr, 1] - track.lon[i_node].values
    centroids_dlat = centroids[close_centr, 0] - track.lat[i_node].values

    # scalar product, a*b=|a|*|b|*cos(phi), phi angle between vectors
    if node_dlen > 0:
        with np.errstate(invalid='print'):
            cos_phi = (centroids_dlon * node_dx + centroids_dlat * node_dy) / \
                LA.norm([centroids_dlon, centroids_dlat], axis=0) / node_dlen
    else:
        cos_phi = np.empty((close_centr.size, 1))
        cos_phi[:] = np.nan

    # southern hemisphere
    if track.lat[i_node] < 0:
        cos_phi = -cos_phi

    # calculate v_trans wind field array assuming that
    # - effect of v_trans decreases with distance from eye (r_arr_normed)
    # - v_trans is added 100% to the right of the track, 0% in front (cos_phi)
    with np.errstate(all='ignore'):
        r_arr_normed = track.radius_max_wind[i_node].values / r_arr
    r_arr_normed[r_arr_normed > 1] = 1

    return v_trans, v_trans * np.multiply(r_arr_normed, cos_phi)

def _bs_value(v_trans, penv, pcen, prepcen, lat, hol_xx, tint):
    """ Halland's 2008 b value computation.

    Parameters:
        v_trans (float): translational wind in m/s
        penv (float): environmental pressure
        pcen (float): central pressure
        prepcen (float): previous central pressure
        lat (float): latitude
        hol_xx (float): Holland's xx value
        tint (float): time step

    Returns:
        float
    """
    return -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv-pcen) + \
        0.03 * (pcen - prepcen) / tint - \
        0.014 * abs(lat) + 0.15 * v_trans**hol_xx + 1.0

def _stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord):
    """ Holland symmetric and static wind field (in m/s) according to
    Holland1980 or Holland2008m depending on hol_b parameter.

    Parameters:
        r_arr (np.array): distance between coastal centroids and track node
        r_max (float): radius_max_wind
        hol_b (float): Holland's b parameter
        penv (float): environmental pressure
        pcen (float): central pressure
        ycoord (float): latitude

    Returns:
        np.array
    """
    rho = 1.15
    f_val = 2 * 0.0000729 * np.sin(np.abs(ycoord) * np.pi / 180)
    # units are m/s
    v_arr = np.zeros(r_arr.shape)
    with np.errstate(all='ignore'):
        for i_vel in range(v_arr.size):
            v_arr[i_vel] = np.sqrt(
                100 * hol_b / rho * (r_max / r_arr[i_vel])**hol_b *
                (penv - pcen) * np.exp(-(r_max/r_arr[i_vel])**hol_b) +
                (1000 * 0.5 * r_arr[i_vel] * f_val)**2
                ) - 0.5 * 1000 * r_arr[i_vel] * f_val

    return v_arr

def _vang_holland(track, i_node, r_arr, v_trans, model='H08'):
    """ Compute Hollands angular wind filed.

    Parameters:
        track (xr.Dataset): contains TC track information
        i_node (int): track node (point) to compute
        v_trans (float): translational wind field
        model (str, optional): Holland model to use, default 2008.

    Returns:
        np.array
    """
    # data for windfield calculation
    penv = track.environmental_pressure[i_node]
    pcen = track.central_pressure[i_node]
    ycoord = track.lat[i_node]

    hol_xx = 0.6 * (1. - (penv - pcen) / 215)
    if model == 'H08':
        # adjust pressure at previous track point
        pre_pcen = track.central_pressure[i_node - 1]
        if pre_pcen < 850:
            pre_pcen = track.central_pressure[i_node]

        hol_b = _bs_value(v_trans, penv, pcen, pre_pcen, \
                           ycoord, hol_xx, track.time_step[i_node])
    else:
        # TODO H80: b=b_value(v_trans,vmax,penv,pcen,rho);
        raise NotImplementedError

    return _stat_holland(r_arr, track.radius_max_wind.values[i_node],
                         hol_b.values, penv.values, pcen.values, ycoord.values)
