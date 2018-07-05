"""
Define TropCyclone class and IBTracs reader.
"""

__all__ = ['TropCyclone',
           'TCTracks']

import logging
import datetime as dt
import itertools
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

class TCTracks(object):
    """Contains tropical cyclone tracks.

    Attributes:
        data (list(xarray.Dataset)): list of tropical cyclone tracks. Each
            track contains following attributes:
                - time (coords)
                - lat (coords)
                - lon (coords)
                - time_step
                - radius_max_wind
                - max_sustained_wind
                - central_pressure
                - environmental_pressure
                - max_sustained_wind_unit (attrs)
                - central_pressure_unit (attrs)
                - name (attrs)
                - orig_event_flag (attrs)
                - data_provider (attrs)
                - basin (attrs)
                - id_no (attrs)
                - category (attrs)
    """
    def __init__(self):
        """Empty constructor. Read csv IBTrACS files if provided. """
        self.data = list()

    def append(self, tracks):
        """Append tracks to current.

        Parameters:
            tracks (xarray.Dataset or list(xarray.Dataset)): tracks to append.
        """
        if not isinstance(tracks, list):
            tracks = [tracks]
        self.data.extend(tracks)

    def get_track(self, name_ev=None, date_ev=None):
        """Get all tracks with provided name that happened at provided date.

        Parameters:
            name_ev (str, optional): name of event
            date_ev (str, optional): date of the event in format

        Returns:
            xarray.Dataset
        """
        if not name_ev and not date_ev:
            return self.data[0]
        else:
            raise NotImplementedError

    def read_ibtracs_csv(self, file_names):
        """Clear and model tropical cyclone from input csv IBTrACS file.
        Parallel process.

        Parameters:
            file_names (str or list(str)): absolute file name(s) or
                folder name containing the files to read.
        """
        all_file = get_file_names(file_names)
        for file in all_file:
            self._read_one_csv(file)

    def retrieve_ibtracs(self, name_ev, date_ev):
        """ Download from IBTrACS repository a specific track.

        Parameters:
            name_ev (str, optional): name of event
            date_ev (str, optional): date of the event in format
        """
        raise NotImplementedError

    def interpolate_time(self, time_step_h=
                         CONFIG['trop_cyclone']['time_step_h']):
        """ Generate interpolated track values to time steps of min_time_step.

        Parameters:
            time_step_h (float): time step in hours to which to interpolate
        """
        new_list = list()
        for track in self.data:
            if track.time.size > 3:
                time_step = str(time_step_h) + 'H'
                track_int = track.resample(time=time_step). \
                            interpolate('linear')
                track_int['time_step'] = ('time', track_int.time.size *
                                          [time_step_h])
                track_int.coords['lat'] = track.lat.resample(time=time_step).\
                                          interpolate('cubic')
                track_int.coords['lon'] = track.lon.resample(time=time_step).\
                                          interpolate('cubic')
                track_int.attrs = track.attrs
            else:
                LOGGER.warning('Track interpolation not done. ' +
                               'Not enough elements for %s', track.name)
                track_int = track
            new_list.append(track_int)

        self.data = new_list

    def calc_random_walk(self, ens_size=9, rand_unif_ini=None,
                         rand_unif_ang=None, ens_amp0=1.5, max_angle=np.pi/10,
                         ens_amp=0.1):
        """ Generate random tracks for every track contained.

        Parameters:
            ens_size (int, optional): number of created tracks per original
                track. Default 9.
            rand_unif_ini (np.array, optional): array of uniform [0,1) random
                numbers of size 2
            rand_unif_ang (np.array, optional): array of uniform [0,1) random
                numbers of size ens_size x size track
            ens_amp0 (float, optional): amplitude of max random starting point
                shift degree longitude. Default: 1.5
            max_angle (float, optional): maximum angle of variation, =pi is
                like undirected, pi/4 means one quadrant. Default: pi/10
            ens_amp (float, optional): amplitude of random walk wiggles in
                degree longitude for 'directed'. Default: 0.1
        """
        ens_track = list()
        for track in self.data:
            n_dat = track.time.size
            if rand_unif_ini is None or rand_unif_ini.shape != (2, ens_size):
                rand_unif_ini = np.random.uniform(size=(2, ens_size))
            if rand_unif_ang is None or rand_unif_ang.size != ens_size*n_dat:
                rand_unif_ang = np.random.uniform(size=ens_size*n_dat)

            xy_ini = ens_amp0 * (rand_unif_ini - 0.5)
            tmp_ang = np.cumsum(2 * max_angle * rand_unif_ang - max_angle)
            coord_xy = np.empty((2, ens_size * n_dat))
            coord_xy[0] = np.cumsum(ens_amp * np.sin(tmp_ang))
            coord_xy[1] = np.cumsum(ens_amp * np.cos(tmp_ang))

            ens_track.append(track)
            for i_ens in range(ens_size):
                i_track = track.copy(True)

                d_xy = coord_xy[:, i_ens * n_dat: (i_ens + 1) * n_dat] - \
                    np.expand_dims(coord_xy[:, i_ens * n_dat], axis=1)

                d_lat_lon = d_xy + np.expand_dims(xy_ini[:, i_ens], axis=1)

                i_track.lon.values = i_track.lon.values + d_lat_lon[0, :]
                i_track.lat.values = i_track.lat.values + d_lat_lon[1, :]
                i_track.attrs['orig_event_flag'] = False
                i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + \
                                        str(i_ens+1)
                i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens+1)/100

                ens_track.append(i_track)

        self.data = ens_track

    def plot(self, title=None):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters:
            title (str, optional): plot title

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        deg_border = 0.5
        fig, axis = plot.make_map()
        axis = axis[0][0]
        min_lat, max_lat = 10000, -10000
        min_lon, max_lon = 10000, -10000
        for track in self.data:
            min_lat, max_lat = min(min_lat, np.min(track.lat.values)), \
                               max(max_lat, np.max(track.lat.values))
            min_lon, max_lon = min(min_lon, np.min(track.lon.values)), \
                               max(max_lon, np.max(track.lon.values))
        axis.set_extent(([min_lon-deg_border, max_lon+deg_border,
                          min_lat-deg_border, max_lat+deg_border]))
        plot.add_shapes(axis)
        if title:
            axis.set_title(title)
        for track in self.data:
            if track.orig_event_flag:
                color = 'b'
            else:
                color = 'k'
            axis.plot(track.lon.values, track.lat.values, c=color)
        return fig, axis

    def wind_decay(self):
        """Apply wind decay due to landfall in synthetic tracks."""
        raise NotImplementedError

    @property
    def size(self):
        """ Get longitude from coord array """
        return len(self.data)

    def _read_one_csv(self, file_name):
        """Read IBTrACS track file.

            Parameters:
                file_name (str): file name containing one IBTrACS track to read
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

        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', datetimes)
        tr_ds.coords['lat'] = ('time', lat)
        tr_ds.coords['lon'] = ('time', lon)
        tr_ds['time_step'] = ('time', dfr['tint'].values)
        tr_ds['radius_max_wind'] = ('time', dfr['rmax'].values)
        tr_ds['max_sustained_wind'] = ('time', max_sus_wind)
        tr_ds['central_pressure'] = ('time', cen_pres)
        tr_ds['environmental_pressure'] = ('time', dfr['penv'].values)
        tr_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
        tr_ds.attrs['central_pressure_unit'] = 'mb'
        tr_ds.attrs['name'] = name
        tr_ds.attrs['orig_event_flag'] = bool(dfr['original_data']. values[0])
        tr_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
        tr_ds.attrs['basin'] = dfr['gen_basin'].values[0]
        tr_ds.attrs['id_no'] = float(name.replace('N', '0'). replace('S', '1'))
        tr_ds.attrs['category'] = _set_category(max_sus_wind, \
                   max_sus_wind_unit)

        self.data.append(tr_ds)

class TropCyclone(Hazard):
    """Contains tropical cyclone events obtained from a csv IBTrACS file.
    """
    intensity_thres = 17.5
    """ intensity threshold for storage in m/s """

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

    def set_from_tracks(self, tracks, centroids=None, description='',
                        model='H08'):
        """Clear and model tropical cyclone from input csv IBTrACS file.
        Parallel process.

        Parameters:
            tracks (TCTracks): tracks of events
            centroids (Centroids, optional): Centroids where to model TC.
                Default: global centroids.
            description (str, optional): description of the events
            model (str, optional): model to compute gust. Default Holland2008.

        Raises:
            ValueError
        """
        num_tracks = tracks.size

        if centroids is None:
            centroids = Centroids(GLB_CENTROIDS_MAT, 'Global centroids')
        # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
        coastal_centr = coastal_centr_idx(centroids)
        centr_list = to_list(num_tracks, centroids, 'centroids')
        coast_list = to_list(num_tracks, coastal_centr, 'coast centroids')

        chunksize = 1
        if num_tracks > 1000:
            chunksize = 250

        for tc_haz in Pool().map(self._tc_from_track, tracks.data, centr_list,
                                 coast_list,
                                 itertools.repeat(model, num_tracks),
                                 chunksize=chunksize):
            self.append(tc_haz)

        self._set_frequency(tracks.data)
        self.tag.description = description

    @staticmethod
    def _tc_from_track(track, centroids, coastal_centr, model='H08'):
        """ Set hazard from input file. If centroids are not provided, they are
        read from the same file.

        Parameters:
            track (xr.Dataset): tropical cyclone track.
            centroids (Centroids): Centroids instance. Use global
                centroids if not provided.
            coastal_centr (np.array): indeces of centroids close to coast.
            model (str, optional): model to compute gust. Default Holland2008.

        Raises:
            ValueError, KeyError

        Returns:
            TropCyclone
        """
        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, 'IBTrACS: ' + track.name)
        new_haz.intensity = gust_from_track(track, centroids, coastal_centr,
                                            model)
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
        return new_haz

    def _set_frequency(self, tracks):
        """Set hazard frequency from tracks data.

        Parameters:
            tracks (list(xr.Dataset))
        """
        delta_time = \
            np.max([np.max(track.time.dt.year.values) \
                    for track in tracks]) - \
            np.min([np.min(track.time.dt.year.values) \
                    for track in tracks]) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

def coastal_centr_idx(centroids, lat_max=61):
    """ Compute centroids indices which are inside INLAND_MAX_DIST_KM and
    with lat < lat_max.

    Parameters:
        lat_max (float, optional): Maximum latitude to consider. Default: 61.

    Returns:
        np.array
    """
    if centroids.dist_coast.size == 0:
        centroids.calc_dist_to_coast()
    return np.logical_and(centroids.dist_coast < INLAND_MAX_DIST_KM,
                          centroids.lat < lat_max).nonzero()[0]

def gust_from_track(track, centroids, coastal_centr=None, model='H08'):
    """ Compute wind gusts at centroids from track. Track is interpolated to
    configured time step.

    Parameters:
        track (xr.Dataset): track infomation
        centroids (Centroids): centroids where gusts are computed
        coastal_centr (np.array): indices of centroids which are close to coast
        model (str, optional): model to compute gust. Default Holland2008

    Returns:
        sparse.csr_matrix
    """
    if coastal_centr is None:
        coastal_centr = coastal_centr_idx(centroids)
    # Compute wind gusts
    intensity = sparse.lil_matrix((1, centroids.id.size))
    intensity[0, coastal_centr] = _windfield_holland(track, \
             centroids.coord[coastal_centr, :], model)
    return sparse.csr_matrix(intensity)

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
    min_wind_threshold = TropCyclone.intensity_thres

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
        close_centr = np.argwhere(r_arr < CENTR_NODE_MAX_DIST_KM).reshape(-1,)
        r_arr = r_arr[close_centr]

        # m/s
        v_trans, v_trans_corr = _vtrans_holland(track, i_node, \
            centroids[close_centr, :], r_arr, ureg)
        v_ang = _vang_holland(track, i_node, r_arr, v_trans, model)

        v_full = v_trans_corr + v_ang
        v_full[np.isnan(v_full)] = 0
        v_full[v_full < min_wind_threshold] = 0

        # keep maximum instantaneous wind
        intensity[0, close_centr] = np.maximum(
            intensity[0, close_centr].todense(), v_full)

        # keep maximum instantaneous wind
        intensity[0, close_centr] = np.maximum(
            intensity[0, close_centr].todense(), v_full)

    return intensity

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

def _vtrans_holland(track, i_node, close_centr, r_arr, ureg):
    """ Compute Hollands translation wind corrections. Returns factor.

    Parameters:
        track (xr.Dataset): contains TC track information
        i_node (int): track node (point) to compute
        close_centr (np.array): each row is a centroid [lat, lon] that is close
        r_arr (np.array): distance between coastal centroids and track node
        ureg (UnitRegistry): units handler

    Returns:
        v_trans (float), v_trans corrected (np.array)
    """
    v_trans = _vtrans(track, i_node, ureg)

    # we use the scalar product of the track forward vector and the vector
    # towards each centroid to figure the angle between and hence whether
    # the translational wind needs to be added (on the right side of the
    # track for Northern hemisphere) and to which extent (100% exactly 90
    # to the right of the track, zero in front of the track)

    # hence, rotate track forward vector 90 degrees clockwise, i.e.
    node_dx = 0
    node_dy = 0
    if i_node < track.lon.size - 1:
        node_dy = (-track.lon[i_node + 1] + track.lon[i_node]).values
        node_dx = (track.lat[i_node + 1] - track.lat[i_node]).values
    node_dlen = LA.norm([node_dx, node_dy])

    # the vector towards each centroid
    centroids_dlon = close_centr[:, 1] - track.lon[i_node].values
    centroids_dlat = close_centr[:, 0] - track.lat[i_node].values

    # scalar product, a*b=|a|*|b|*cos(phi), phi angle between vectors
    with np.errstate(invalid='print'):
        cos_phi = (centroids_dlon * node_dx + centroids_dlat * node_dy) / \
            LA.norm([centroids_dlon, centroids_dlat], axis=0) / node_dlen

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
    r_arr_mult = 0.5 * 1000 * r_arr * f_val
    # units are m/s
    with np.errstate(all='ignore'):
        r_max_norm = (r_max/r_arr)**hol_b
        return np.sqrt(100 * hol_b / rho * r_max_norm * (penv - pcen) *
                       np.exp(-r_max_norm) + r_arr_mult**2) - r_arr_mult

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
    penv = track.environmental_pressure.values[i_node]
    pcen = track.central_pressure.values[i_node]
    ycoord = track.lat.values[i_node]

    hol_xx = 0.6 * (1. - (penv - pcen) / 215)
    if model == 'H08':
        # adjust pressure at previous track point
        pre_pcen = track.central_pressure.values[i_node - 1]
        if pre_pcen < 850:
            pre_pcen = track.central_pressure.values[i_node]
        hol_b = _bs_value(v_trans, penv, pcen, pre_pcen, \
                           ycoord, hol_xx, track.time_step.values[i_node])
    else:
        # TODO H80: b=b_value(v_trans,vmax,penv,pcen,rho);
        raise NotImplementedError

    return _stat_holland(r_arr, track.radius_max_wind.values[i_node],
                         hol_b, penv, pcen, ycoord)
