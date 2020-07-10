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
from numba import jit
from tqdm import tqdm

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.tc_tracks import TCTracks, estimate_rmw
from climada.hazard.tc_clim_change import get_knutson_criterion, calc_scale_knutson
from climada.hazard.centroids.centr import Centroids
from climada.util import ureg
from climada.util.coordinates import dist_approx
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
"""Hazard type acronym for Tropical Cyclone"""

INLAND_MAX_DIST_KM = 1000
"""Maximum inland distance of the centroids in km"""

CENTR_NODE_MAX_DIST_KM = 300
"""Maximum distance between centroid and TC track node in km"""

CENTR_NODE_MAX_DIST_DEG = 5.5
"""Maximum distance between centroid and TC track node in degrees"""

MODEL_VANG = { 'H08': 0 }
"""Enumerate different symmetric wind field calculation."""

KMH_TO_MS = (1.0 * ureg.km/ureg.hour).to(ureg.meter/ureg.second).magnitude
KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter/ureg.second).magnitude
NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

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
    """intensity threshold for storage in m/s"""

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self, pool=None):
        """Empty constructor."""
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
        """Clear and fill with windfields from specified tracks.

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
            centroids = Centroids.from_base_grid(res_as=360, land=False)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        if ignore_distance_to_coast:
            # Select centroids with lat < 61
            coastal_idx = (np.abs(centroids.lat) < 61).nonzero()[0]
        else:
            # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
            if not centroids.dist_coast.size:
                centroids.set_dist_coast()
            coastal_idx = ((centroids.dist_coast < INLAND_MAX_DIST_KM * 1000)
                           & (np.abs(centroids.lat) < 61)).nonzero()[0]

        LOGGER.info('Mapping %s tracks to %s centroids.', str(tracks.size),
                    str(coastal_idx.size))
        if self.pool:
            chunksize = min(num_tracks//self.pool.ncpus, 1000)
            tc_haz = self.pool.map(self._tc_from_track, tracks.data,
                                   itertools.repeat(centroids, num_tracks),
                                   itertools.repeat(coastal_idx, num_tracks),
                                   itertools.repeat(model, num_tracks),
                                   chunksize=chunksize)
        else:
            tc_haz = [self._tc_from_track(track, centroids, coastal_idx, model)
                      for track in tracks.data]
        LOGGER.debug('Append events.')
        self.concatenate(tc_haz)
        LOGGER.debug('Compute frequency.')
        self._set_frequency(tracks.data)
        self.tag.description = description

    def set_climate_scenario_knu(self, ref_year=2050, rcp_scenario=45):
        """Compute future events for given RCP scenario and year. RCP 4.5
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
        """Generate video of TC wind fields node by node and returns its
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
            tracks (list of xarray.Dataset)
        """
        if not tracks: return
        year_max = np.amax([t.time.dt.year.values.max() for t in tracks])
        year_min = np.amax([t.time.dt.year.values.min() for t in tracks])
        year_delta = year_max - year_min + 1
        num_orig = self.orig.nonzero()[0].size
        ens_size = (self.event_id.size / num_orig) if num_orig > 0 else 1
        self.frequency = np.ones(self.event_id.size) / (year_delta * ens_size)

    def _tc_from_track(self, track, centroids, coastal_centr, model='H08'):
        """Generate windfield hazard from a single track dataset

        Parameters:
            track (xr.Dataset): single tropical cyclone track.
            centroids (Centroids): Centroids instance.
            coastal_centr (np.array): Indices of centroids close to coast.
            model (str, optional): Windfield model. Default: H08.

        Raises:
            ValueError, KeyError

        Returns:
            TropCyclone
        """
        try:
            mod_id = MODEL_VANG[model]
        except KeyError:
            LOGGER.error('Model not implemented: %s.', model)
            raise ValueError
        intensity = np.zeros(centroids.coord.shape[0])
        intensity[coastal_centr] = _windfield(track,
            centroids.coord[coastal_centr], mod_id)
        intensity[intensity < self.intensity_thres] = 0

        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, 'Name: ' + track.name)
        new_haz.intensity = sparse.csr_matrix(intensity)
        new_haz.units = 'm/s'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1)
        # store first day of track as date
        new_haz.date = np.array([
            dt.datetime(track.time.dt.year[0],
                        track.time.dt.month[0],
                        track.time.dt.day[0]).toordinal()
        ])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [track.basin]
        return new_haz

    def _apply_criterion(self, criterion, scale):
        """Apply changes defined in criterion with a given scale
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

def _windfield(track, centroids, model):
    """Compute windfields (in m/s) in centroids using Holland model 08.

    Parameters:
        track (xr.Dataset): track infomation
        centroids (2d np.array): each row is a centroid [lat, lon]
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        np.array
    """
    # initialize intensity
    intensity = np.zeros(centroids.shape[0])

    # shorthands for track data
    t_lat, t_lon = track.lat.values, track.lon.values
    t_rad, t_env = track.radius_max_wind.values, track.environmental_pressure.values
    t_cen, t_tstep = track.central_pressure.values, track.time_step.values

    if t_lon.size < 2:
        return intensity

    # never use longitudes at -180 degrees or below
    t_lon[t_lon <= -180] += 360

    # only use longitudes above 180, if 180 degree border is crossed
    if t_lon.min() > 180: t_lon -= 360

    # restrict to centroids in rectangular bounding box around track
    track_centr_msk = _close_centroids(t_lat, t_lon, centroids)
    track_centr = centroids[track_centr_msk]

    if track_centr.shape[0] == 0:
        return intensity

    # compute distances and vectors to all centroids
    v_centr = [ar[0] for ar in dist_approx(t_lat[None], t_lon[None],
        track_centr[None,:, 0], track_centr[None,:, 1],
        log=True, method="geosphere")]
    d_centr = v_centr[0]

    # exclude centroids that are too far from or too close to the eye
    close_centr = (d_centr < CENTR_NODE_MAX_DIST_KM) & (d_centr > 1e-5)

    if not np.any(close_centr):
        return intensity

    # make sure that central pressure never exceeds environmental pressure
    msk = (t_cen > t_env)
    t_cen[msk] = t_env[msk]

    # extrapolate radius of max wind from pressure if not given
    t_rad[:] = estimate_rmw(t_rad, t_lat, t_cen) * NM_TO_KM

    # translational speed of track at every node
    v_trans = _vtrans(t_lat, t_lon, t_tstep)
    v_trans_corr = _vtrans_correct(v_centr, v_trans, t_rad, t_lat, close_centr)
    v_trans_norm = v_trans[0]

    # adjust pressure at previous track point
    prev_pres = t_cen[:-1].copy()
    msk = (prev_pres < 850)
    prev_pres[msk] = t_cen[1:][msk]

    # compute b-value
    if model == 0:
        hol_xx = 0.6 * (1. - (t_env - t_cen) / 215)
        hol_b = _bs_hol08(v_trans_norm[:-1], t_env[1:], t_cen[1:], prev_pres,
            t_lat[1:], hol_xx[1:], t_tstep[1:])
    else:
        #TODO: H80 with hol_b = b_value(v_trans, vmax, penv, pcen, rho)
        raise NotImplementedError

    # derive angular velocity
    v_ang = _stat_holland(d_centr[1:], t_rad[1:], hol_b, t_env[1:],
                          t_cen[1:], t_lat[1:], close_centr[1:])

    v_full = v_trans_norm[:-1,None] * v_trans_corr[1:] + v_ang
    v_full[np.isnan(v_full)] = 0
    intensity[track_centr_msk] = v_full.max(axis=0)
    return intensity

def _close_centroids(t_lat, t_lon, centroids):
    """Choose centroids within padded rectangular region around track

    Parameters:
        t_lat (np.array): latitudinal coordinates of track points
        t_lon (np.array): longitudinal coordinates of track points
        centroids (np.array): coordinates of centroids to check

    Returns:
        np.array (mask)
    """
    if (t_lon < -170).any() and (t_lon > 170).any():
        # crosses 180 degrees east/west -> use positive degrees east
        t_lon[t_lon < 0] += 360

    track_bounds = np.array([t_lon.min(), t_lat.min(), t_lon.max(), t_lat.max()])
    track_bounds[:2] -= CENTR_NODE_MAX_DIST_DEG
    track_bounds[2:] += CENTR_NODE_MAX_DIST_DEG
    if track_bounds[2] > 180:
        # crosses 180 degrees East/West
        track_bounds[2] -= 360

    centr_lat, centr_lon = centroids[:, 0], centroids[:, 1]
    msk_lat = (track_bounds[1] < centr_lat) & (centr_lat < track_bounds[3])
    if track_bounds[2] < track_bounds[0]:
        # crosses 180 degrees East/West
        msk_lon = (track_bounds[0] < centr_lon) | (centr_lon < track_bounds[2])
    else:
        msk_lon = (track_bounds[0] < centr_lon) & (centr_lon < track_bounds[2])
    return (msk_lat & msk_lon)

def _vtrans(t_lat, t_lon, t_tstep):
    """Translational vector and velocity at each track node.

    Parameters
    ----------
    t_lat : np.array
        track latitudes
    t_lon : np.array
        track longitudes
    t_tstep : np.array
        track time steps

    Returns
    -------
    v_trans_norm : np.array
        Same shape as input, the last velocity is always 0.
    v_trans : np.array
        Directional vectors of velocity.
    """
    v_trans = np.zeros((t_lat.size, 2))
    v_trans_norm = np.zeros((t_lat.size,))
    norm, vec = dist_approx(t_lat[:-1,None], t_lon[:-1,None],
        t_lat[1:,None], t_lon[1:,None], log=True, method="geosphere")
    v_trans[:-1,:] = vec[:,0,0]
    v_trans[:-1,:] *= KMH_TO_MS / t_tstep[1:,None]
    v_trans_norm[:-1] = norm[:,0,0]
    v_trans_norm[:-1] *= KMH_TO_MS / t_tstep[1:]

    # limit to 30 nautical miles per hour
    msk = (v_trans_norm > 30 * KN_TO_MS)
    fact = 30 * KN_TO_MS / v_trans_norm[msk]
    v_trans[msk,:] *= fact[:,None]
    v_trans_norm[msk] *= fact
    return v_trans_norm, v_trans

def _vtrans_correct(v_centr, v_trans, t_rad, t_lat, close_centr):
    """Hollands translational wind corrections

    Use the angle between the track forward vector and the vector towards each
    centroid to decide whether the translational wind needs to be added (on the
    right side of the track for Northern hemisphere) and to which extent (100%
    exactly 90 degree to the right of the track, zero in front of the track).

    Parameters:
        v_centr (tuple): distance and vector from track points to centroids
        v_trans (tuple): distance and vector between consecutive track points
        t_rad (float): radius of maximum wind at each track point
        t_lat (float): latitude coordinates of track points
        close_centr (np.array): mask indicating which centroids are close enough

    Returns:
        np.array
    """
    # exclude stationary points
    msk = close_centr & (v_trans[0][:,None] > 0)

    # rotate track forward vector 90 degrees clockwise
    trans_orth = np.array([-1.0, 1.0])[...,:] * v_trans[1][...,::-1]
    trans_orth_bc = np.broadcast_arrays(v_centr[1], trans_orth[:,None,:])[1]

    # scalar product, a*b = |a|*|b|*cos(phi), phi angle between vectors
    norm = v_centr[0] * v_trans[0][:,None]
    cos_phi = np.zeros_like(v_centr[0])
    cos_phi[msk] = (v_centr[1][msk,:] * trans_orth_bc[msk,:]).sum(axis=-1) / norm[msk]
    cos_phi = np.clip(cos_phi, -1, 1)

    # inverse orientation on southern hemisphere
    cos_phi[t_lat < 0,:] *= -1

    t_rad_bc = np.broadcast_arrays(t_rad[:,None], v_centr[0])[0]
    v_trans_corr = np.zeros_like(v_centr[0])
    v_trans_corr[msk] = np.fmin(1, t_rad_bc[msk] / v_centr[0][msk]) * cos_phi[msk]
    return v_trans_corr

def _bs_hol08(v_trans, penv, pcen, prepcen, lat, hol_xx, tint):
    """Holland's 2008 b value computation.

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
    return -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv - pcen) + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0

def _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, close_centr):
    """Holland symmetric and static wind field (in m/s) according to
    Holland1980 or Holland2008m depending on hol_b parameter.

    Parameters:
        d_centr (2d np.array): distance between coastal centroids and track node
        r_max (1d np.array): radius_max_wind along track
        hol_b (1d np.array): Holland's b parameter along track
        penv (1d np.array): environmental pressure along track
        pcen (1d np.array): central pressure along track
        lat (1d np.array): latitude along track
        close_centr (2d np.array): mask

    Returns:
        np.array
    """
    v_ang = np.zeros_like(d_centr)
    r_max, hol_b, lat, penv, pcen, d_centr = [ar[close_centr]
        for ar in np.broadcast_arrays(r_max[:,None], hol_b[:,None], lat[:,None],
                                      penv[:,None], pcen[:,None], d_centr)]
    rho = 1.15
    f_val = 2 * 0.0000729 * np.sin(np.radians(np.abs(lat)))
    d_centr_mult = 0.5 * 1000 * d_centr * f_val
    # units are m/s
    msk = (d_centr > 1e-5) & (hol_b > 0) & (hol_b < 5)
    r_max_norm = np.zeros_like(d_centr)
    r_max_norm[msk] = (r_max[msk] / d_centr[msk])**hol_b[msk]
    sqrt_term = 100 * hol_b / rho * r_max_norm * (penv - pcen) \
                * np.exp(-r_max_norm) + d_centr_mult**2
    v_ang[close_centr] = np.sqrt(np.fmax(0, sqrt_term)) - d_centr_mult
    return v_ang
