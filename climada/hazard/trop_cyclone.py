"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define TropCyclone class.
"""

__all__ = ['TropCyclone']

import itertools
import logging
import copy
import time
import datetime as dt
import numpy as np
from numpy import linalg as LA
from scipy import sparse
import matplotlib.animation as animation
from pint import UnitRegistry
from numba import jit
from tqdm import tqdm

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.tc_clim_change import get_knutson_criterion, calc_scale_knutson
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT
from climada.util.interpolation import dist_approx
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
""" Hazard type acronym for Tropical Cyclone """

INLAND_MAX_DIST_KM = 1000
""" Maximum inland distance of the centroids in km """

CENTR_NODE_MAX_DIST_KM = 300
""" Maximum distance between centroid and TC track node in km """

MODEL_VANG = {'H08': 0
             }
""" Enumerate different symmetric wind field calculation."""

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
        basin (list(str)): basin where every event starts
            'NA' North Atlantic
            'EP' Eastern North Pacific
            'WP' Western North Pacific
            'NI' North Indian
            'SI' South Indian
            'SP' Southern Pacific
            'SA' South Atlantic
    """
    intensity_thres = 17.5
    """ intensity threshold for storage in m/s """

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.category = np.array([], int)
        self.basin = list()
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_from_tracks(self, tracks, centroids=None, description='',
                        model='H08', ignore_distance_to_coast=False):
        """Clear and model tropical cyclone from input IBTrACS tracks.
        Parallel process.
        Parameters:
            tracks (TCTracks): tracks of events
            centroids (Centroids, optional): Centroids where to model TC.
                Default: global centroids.
            description (str, optional): description of the events
            model (str, optional): model to compute gust. Default Holland2008.
            ignore_distance_to_coast (boolean, optional): if True, centroids
                far from coast are not ignored. Default False
        Raises:
            ValueError
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids()
            centroids.read_mat(GLB_CENTROIDS_MAT)
        if ignore_distance_to_coast: # Select centroids with lat < 61
            coastal_idx = np.logical_and(centroids.lat < 61, True).nonzero()[0]
        else:  # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
            coastal_idx = coastal_centr_idx(centroids)
        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        LOGGER.info('Mapping %s tracks to %s centroids.', str(tracks.size),
                    str(centroids.size))
        if self.pool:
            chunksize = min(num_tracks//self.pool.ncpus, 1000)
            tc_haz = self.pool.map(self._tc_from_track, tracks.data,
                                   itertools.repeat(centroids, num_tracks),
                                   itertools.repeat(coastal_idx, num_tracks),
                                   itertools.repeat(model, num_tracks),
                                   chunksize=chunksize)
        else:
            tc_haz = list()
            for track in tracks.data:
                tc_haz.append(self._tc_from_track(track, centroids, coastal_idx,
                                                  model))
        LOGGER.debug('Append events.')
        self._append_all(tc_haz)
        LOGGER.debug('Compute frequency.')
        self._set_frequency(tracks.data)
        self.tag.description = description

    def set_climate_scenario_knu(self, ref_year=2050, rcp_scenario=45):
        """ Compute future events for given RCP scenario and year. RCP 4.5
        from Knutson et al 2015.
        Parameters:
            ref_year (int): year between 2000 ad 2100. Default: 2050
            rcp_scenario (int):  26 for RCP 2.6, 45 for RCP 4.5 (default),
                60 for RCP 6.0 and 85 for RCP 8.5.
        Returns:
            TropCyclone
        """
        criterion = get_knutson_criterion()
        scale = calc_scale_knutson(ref_year, rcp_scenario)
        haz_cc = self._apply_criterion(criterion, scale)
        haz_cc.tag.description = 'climate change scenario for year %s and RCP %s '\
        'from Knutson et al 2015.' % (str(ref_year), str(rcp_scenario))
        return haz_cc

    @staticmethod
    def video_intensity(track_name, tracks, centroids, file_name=None,
                        writer=animation.PillowWriter(bitrate=500),
                        **kwargs):
        """ Generate video of TC wind fields node by node and returns its
        corresponding TropCyclone instances and track pieces.

        Parameters:
            track_name (str): name of the track contained in tracks to record
            tracks (TCTracks): tracks
            centroids (Centroids): centroids where wind fields are mapped
            file_name (str, optional): file name to save video, if provided
            writer = (matplotlib.animation.*, optional): video writer. Default:
                pillow with bitrate=500
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            list(TropCyclone), list(np.array)

        Raises:
            ValueError
        """
        # initialization
        track = tracks.get_track(track_name)
        if not track:
            LOGGER.error('%s not found in track data.', track_name)
            raise ValueError
        idx_plt = np.argwhere(np.logical_and(np.logical_and(np.logical_and( \
            track.lon.values < centroids.total_bounds[2] + 1, \
            centroids.total_bounds[0] - 1 < track.lon.values), \
            track.lat.values < centroids.total_bounds[3] + 1), \
            centroids.total_bounds[1] - 1 < track.lat.values)).reshape(-1)

        tc_list = []
        tr_coord = {'lat':[], 'lon':[]}
        for node in range(idx_plt.size-2):
            tr_piece = track.sel(time=slice(track.time.values[idx_plt[node]], \
                track.time.values[idx_plt[node+2]]))
            tr_piece.attrs['n_nodes'] = 2 # plot only one node
            tr_sel = TCTracks()
            tr_sel.append(tr_piece)
            tr_coord['lat'].append(tr_sel.data[0].lat.values[:-1])
            tr_coord['lon'].append(tr_sel.data[0].lon.values[:-1])

            tc_tmp = TropCyclone()
            tc_tmp.set_from_tracks(tr_sel, centroids)
            tc_tmp.event_name = [track.name + ' ' + time.strftime("%d %h %Y %H:%M", \
                time.gmtime(tr_sel.data[0].time[1].values.astype(int)/1000000000))]
            tc_list.append(tc_tmp)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'Greys'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.array([tc_.intensity.min() for tc_ in tc_list]).min()
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.array([tc_.intensity.max() for tc_ in tc_list]).max()

        def run(node):
            tc_list[node].plot_intensity(1, axis=axis, **kwargs)
            axis.plot(tr_coord['lon'][node], tr_coord['lat'][node], 'k')
            axis.set_title(tc_list[node].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info('Generating video %s', file_name)
            fig, axis = u_plot.make_map()
            pbar = tqdm(total=idx_plt.size-2)
            ani = animation.FuncAnimation(fig, run, frames=idx_plt.size-2,
                                          interval=500, blit=False)
            ani.save(file_name, writer=writer)
            pbar.close()
        return tc_list, tr_coord

    def _set_frequency(self, tracks):
        """Set hazard frequency from tracks data.
        Parameters:
            tracks (list(xr.Dataset))
        """
        if not tracks:
            return
        delta_time = np.max([np.max(track.time.dt.year.values) \
            for track in tracks]) - np.min([np.min(track.time.dt.year.values) \
            for track in tracks]) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @staticmethod
    @jit
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
        new_haz.event_name = [track.sid]
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1)
        # store date of start
        new_haz.date = np.array([dt.datetime(
            track.time.dt.year[0], track.time.dt.month[0],
            track.time.dt.day[0]).toordinal()])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [track.basin]
        return new_haz

    def _apply_criterion(self, criterion, scale):
        """ Apply changes defined in criterion with a given scale
        Parameters:
            criterion (list(dict)): list of criteria
            scale (float): scale parameter because of chosen year and RCP
        Returns:
            TropCyclone
        """
        haz_cc = copy.deepcopy(self)
        for chg in criterion:
            # filter criteria
            select = np.ones(haz_cc.size, bool)
            for var_name, cri_val in chg['criteria'].items():
                var_val = getattr(haz_cc, var_name)
                if isinstance(var_val, list):
                    var_val = np.array(var_val)
                tmp_select = np.logical_or.reduce([var_val == val for val in cri_val])
                select = np.logical_and(select, tmp_select)
            if chg['function'] == np.multiply:
                change = 1 + (chg['change'] - 1) * scale
            elif chg['function'] == np.add:
                change = chg['change'] * scale
            if select.any():
                new_val = getattr(haz_cc, chg['variable'])
                new_val[select] *= change
                setattr(haz_cc, chg['variable'], new_val)
        return haz_cc

def coastal_centr_idx(centroids, lat_max=61):
    """ Compute centroids indices which are inside INLAND_MAX_DIST_KM and
    with lat < lat_max.
    Parameters:
        lat_max (float, optional): Maximum latitude to consider. Default: 61.
    Returns:
        np.array
    """
    if not centroids.dist_coast.size:
        centroids.set_dist_coast()
    return np.logical_and(centroids.dist_coast < INLAND_MAX_DIST_KM*1000,
                          centroids.lat < lat_max).nonzero()[0]

def gust_from_track(track, centroids, coastal_idx=None, model='H08'):
    """ Compute wind gusts at centroids from track. Track is interpolated to
    configured time step.
    Parameters:
        track (xr.Dataset): track infomation
        centroids (Centroids): centroids where gusts are computed
        coastal_idx (np.array): indices of centroids which are close to coast
        model (str, optional): model to compute gust. Default Holland2008
    Returns:
        sparse.csr_matrix
    """
    if coastal_idx is None:
        coastal_idx = coastal_centr_idx(centroids)
    try:
        mod_id = MODEL_VANG[model]
    except KeyError:
        LOGGER.error('Not implemented model %s.', model)
        raise ValueError
    # Compute wind gusts
    intensity = _windfield(track, centroids.coord, coastal_idx, mod_id)
    return sparse.csr_matrix(intensity)

@jit
def _windfield(track, centroids, coastal_idx, model):
    """ Compute windfields (in m/s) in centroids using Holland model 08.

    Parameters:
        track (xr.Dataset): track infomation
        centroids (2d np.array): each row is a centroid [lat, lon]
        coastal_idx (1d np.array): centroids indices that are close to coast
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        np.array
    """
    np.warnings.filterwarnings('ignore')
    # Make sure that CentralPressure never exceeds EnvironmentalPressure
    up_pr = np.argwhere(track.central_pressure.values >
                        track.environmental_pressure.values)
    track.central_pressure.values[up_pr] = \
        track.environmental_pressure.values[up_pr]

    # Extrapolate RadiusMaxWind from pressure if not given
    ureg = UnitRegistry()
    track['radius_max_wind'] = ('time', _extra_rad_max_wind( \
        track.central_pressure.values, track.radius_max_wind.values, ureg))

    # Track translational speed at every node
    v_trans = _vtrans(track.lat.values, track.lon.values,
                      track.time_step.values, ureg)

    # Compute windfield
    intensity = np.zeros((centroids.shape[0], ))
    intensity[coastal_idx] = _wind_per_node(centroids[coastal_idx, :], track,
                                            v_trans, model)

    return intensity

@jit
def _vtrans(t_lat, t_lon, t_tstep, ureg):
    """ Translational spped at every track node.

    Parameters:
        t_lat (np.array): track latitudes
        t_lon (np.array): track longitudes
        t_tstep (np.array): track time steps
        ureg (UnitRegistry): units handler

    Returns:
        np.array
    """
    v_trans = dist_approx(t_lat[:-1], t_lon[:-1],
                          np.cos(np.radians(t_lat[:-1])), t_lat[1:],
                          t_lon[1:]) / t_tstep[1:]
    v_trans = (v_trans * ureg.km/ureg.hour).to(ureg.meter/ureg.second).magnitude

    # nautical miles/hour, limit to 30 nmph
    v_max = (30*ureg.knot).to(ureg.meter/ureg.second).magnitude
    v_trans[v_trans > v_max] = v_max
    return v_trans

@jit
def _extra_rad_max_wind(t_cen, t_rad, ureg):
    """ Extrapolate RadiusMaxWind from pressure and change to km.

    Parameters:
        t_cen (np.array): track central pressures
        t_rad (np.array): track radius of maximum wind
        ureg (UnitRegistry): units handler

    Returns:
        np.array
    """
    # TODO: always extrapolate???!!!
    # rmax thresholds in nm
    rmax_1, rmax_2, rmax_3 = 15, 25, 50
    # pressure in mb
    pres_1, pres_2, pres_3 = 950, 980, 1020
    t_rad[t_cen <= pres_1] = rmax_1

    to_change = np.logical_and(t_cen > pres_1, t_cen <= pres_2).nonzero()[0]
    t_rad[to_change] = (t_cen[to_change] - pres_1) * \
        (rmax_2 - rmax_1)/(pres_2 - pres_1) + rmax_1

    to_change = np.argwhere(t_cen > pres_2).squeeze()
    t_rad[to_change] = (t_cen[to_change] - pres_2) * \
        (rmax_3 - rmax_2)/(pres_3 - pres_2) + rmax_2

    return (t_rad * ureg.nautical_mile).to(ureg.kilometer).magnitude

@jit(parallel=True)
def _wind_per_node(coastal_centr, track, v_trans, model):
    """ Compute sustained winds at each centroid.

    Parameters:
        coastal_centr (2d np.array): centroids
        track (xr.Dataset): track latitudes
        v_trans (np.array): track translational velocity
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        2d np.array
    """

    t_lat, t_lon = track.lat.values, track.lon.values
    t_rad, t_env = track.radius_max_wind.values, track.environmental_pressure.values
    t_cen, t_tstep = track.central_pressure.values, track.time_step.values

    centr_cos_lat = np.cos(np.radians(coastal_centr[:, 0]))
    intensity = np.zeros((coastal_centr.shape[0],))

    n_nodes = t_lat.size
    if 'n_nodes' in track.attrs:
        n_nodes = track.attrs['n_nodes']

    for i_node in range(1, n_nodes):
        # compute distance to all centroids
        r_arr = dist_approx(coastal_centr[:, 0], coastal_centr[:, 1], \
            centr_cos_lat, t_lat[i_node], t_lon[i_node])

        # Choose centroids that are close enough
        close_centr = np.argwhere(r_arr < CENTR_NODE_MAX_DIST_KM).reshape(-1,)
        r_arr = r_arr[close_centr]

        # translational component
        if i_node < t_lat.size-1:
            v_trans_corr = _vtrans_correct(t_lat[i_node:i_node+2], \
                t_lon[i_node:i_node+2], t_rad[i_node], \
                coastal_centr[close_centr, :], r_arr)
        else:
            v_trans_corr = np.zeros((r_arr.size,))

        # angular component
        v_ang = _vang_sym(t_env[i_node], t_cen[i_node-1:i_node+1],
                          t_lat[i_node], t_tstep[i_node], t_rad[i_node],
                          r_arr, v_trans[i_node-1], model)

        v_full = v_trans[i_node-1] * v_trans_corr + v_ang
        v_full[np.isnan(v_full)] = 0
        v_full[v_full < TropCyclone.intensity_thres] = 0

        # keep maximum instantaneous wind
        intensity[close_centr] = np.maximum(intensity[close_centr], v_full)

    return intensity

@jit
def _vtrans_correct(t_lats, t_lons, t_rad, close_centr, r_arr):
    """ Compute Hollands translational wind corrections. Returns factor.

    Parameters:
        t_lats (tuple): current and next latitude
        t_lats (tuple): current and next longitude
        t_rad (float): current radius of maximum wind
        close_centr (np.array): centroids
        r_arr (np.array): distance from current node to all centroids

    Returns:
        np.array
    """
    # we use the scalar product of the track forward vector and the vector
    # towards each centroid to figure the angle between and hence whether
    # the translational wind needs to be added (on the right side of the
    # track for Northern hemisphere) and to which extent (100% exactly 90
    # to the right of the track, zero in front of the track)
    lon, nex_lon = t_lons
    lat, nex_lat = t_lats

    # hence, rotate track forward vector 90 degrees clockwise, i.e.
    node_dy = -nex_lon + lon
    node_dx = nex_lat - lat

    # the vector towards each centroid
    centroids_dlon = close_centr[:, 1] - lon
    centroids_dlat = close_centr[:, 0] - lat

    # scalar product, a*b=|a|*|b|*cos(phi), phi angle between vectors
    cos_phi = (centroids_dlon * node_dx + centroids_dlat * node_dy) / \
        LA.norm([centroids_dlon, centroids_dlat], axis=0) / LA.norm([node_dx, node_dy])

    # southern hemisphere
    if lat < 0:
        cos_phi = -cos_phi

    # calculate v_trans wind field array assuming that
    # - effect of v_trans decreases with distance from eye (r_arr_normed)
    # - v_trans is added 100% to the right of the track, 0% in front (cos_phi)
    r_arr_normed = t_rad / r_arr
    r_arr_normed[r_arr_normed > 1] = 1

    return np.multiply(r_arr_normed, cos_phi)

@jit(['f8(f8, f8, f8, f8, f8, f8, f8)'], nopython=True)
def _bs_hol08(v_trans, penv, pcen, prepcen, lat, hol_xx, tint):
    """ Halland's 2008 b value computation.

    Parameters:
        v_trans (float): translational wind (m/s)
        penv (float): environmental pressure (hPa)
        pcen (float): central pressure (hPa)
        prepcen (float): previous central pressure (hPa)
        lat (float): latitude (degrees)
        hol_xx (float): Holland's xx value
        tint (float): time step (h)

    Returns:
        float
    """
    return -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv-pcen) + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0

@jit(nopython=True)
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
    f_val = 2 * 0.0000729 * np.sin(np.radians(np.abs(ycoord)))
    r_arr_mult = 0.5 * 1000 * r_arr * f_val
    # units are m/s
    r_max_norm = (r_max/r_arr)**hol_b
    return np.sqrt(100 * hol_b / rho * r_max_norm * (penv - pcen) *
                   np.exp(-r_max_norm) + r_arr_mult**2) - r_arr_mult

@jit(nopython=True)
def _vang_sym(t_env, t_cens, t_lat, t_step, t_rad, r_arr, v_trans, model):
    """ Compute symmetric and static wind field (in m/s) filed (angular
    wind component.

    Parameters:
        t_env (float): environmental pressures
        t_cens (tuple): previous and current central pressures
        t_lat (float): latitude
        t_tstep (float): time steps
        t_rad (float): radius of maximum wind
        r_arr (np.array): distance from current node to all centroids
        v_trans (float): translational wind field
        model (int): Holland model to use, default 2008.

    Returns:
        np.array
    """
    # data for windfield calculation
    prev_pres, pres = t_cens
    hol_xx = 0.6 * (1. - (t_env - pres) / 215)
    if model == 0:
        # adjust pressure at previous track point
        if prev_pres < 850:
            prev_pres = pres
        hol_b = _bs_hol08(v_trans, t_env, pres, prev_pres, t_lat, hol_xx, t_step)
    else:
        # TODO H80: b=b_value(v_trans,vmax,penv,pcen,rho);
        raise NotImplementedError

    return _stat_holland(r_arr, t_rad, hol_b, t_env, pres, t_lat)
