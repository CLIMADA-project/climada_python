"""
Define TropCyclone class.
"""

__all__ = ['TropCyclone']

import logging
import datetime as dt
import itertools
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from pint import UnitRegistry
from pathos.multiprocessing import ProcessingPool as Pool

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.util.files_handler import to_list
from climada.util.constants import GLB_CENTROIDS_MAT, ONE_LAT_KM
from climada.util.interpolation import dist_sqr_approx

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
""" Hazard type acronym for Tropical Cyclone """

INLAND_MAX_DIST_KM = 1000
""" Maximum inland distance of the centroids in km """

CENTR_NODE_MAX_DIST_KM = 300
""" Maximum distance between centroid and TC track node in km """

class TropCyclone(Hazard):
    """Contains tropical cyclone events.

    Attributes:
        category (np.array(int)): for every event, the TC category using the
            Saffir-Simpson scale:
                -1 tropical depression
                 0 tropical storm
                 1 Hurrican category 1
                 2 Hurrican category 2
                 3 Hurrican category 3
                 4 Hurrican category 4
                 5 Hurrican category 5
    """
    intensity_thres = 17.5
    """ intensity threshold for storage in m/s """

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.category = np.array([], int)

    def set_from_tracks(self, tracks, centroids=None, description='',
                        model='H08'):
        """Clear and model tropical cyclone from input IBTrACS tracks.
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
        new_haz.category = np.array([track.category])
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
