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
from scipy import sparse
import matplotlib.animation as animation
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

MODEL_VANG = {'H08': 0}
"""Enumerate different symmetric wind field calculation."""

KMH_TO_MS = (1.0 * ureg.km / ureg.hour).to(ureg.meter / ureg.second).magnitude
KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
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
                        model='H08', ignore_distance_to_coast=False,
                        store_windfields=False):
        """Clear and fill with windfields from specified tracks.

        Parameters:
            tracks (TCTracks): tracks of events
            centroids (Centroids, optional): Centroids where to model TC.
                Default: global centroids.
            description (str, optional): description of the events
            model (str, optional): model to compute gust. Default Holland2008.
            ignore_distance_to_coast (boolean, optional): if True, centroids
                far from coast are not ignored. Default False
            store_windfields (boolean, optional): If True, the Hazard object
                gets a list `windfields` of sparse matrices. For each track,
                the full velocity vectors at each centroid and track position
                are stored in a sparse matrix of shape
                (npositions,  ncentroids * 2), that can be reshaped to a full
                ndarray of shape (npositions, ncentroids, 2). Default: False.

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
            chunksize = min(num_tracks // self.pool.ncpus, 1000)
            tc_haz = self.pool.map(
                self._tc_from_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(coastal_idx, num_tracks),
                itertools.repeat(model, num_tracks),
                itertools.repeat(store_windfields, num_tracks),
                chunksize=chunksize)
        else:
            last_perc = 0
            tc_haz = []
            for track in tracks.data:
                perc = 100 * len(tc_haz) / len(tracks.data)
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                tc_haz.append(
                    self._tc_from_track(track, centroids, coastal_idx,
                                        model=model,
                                        store_windfields=store_windfields))
        LOGGER.debug('Append events.')
        self.concatenate(tc_haz)
        LOGGER.debug('Compute frequency.')
        self.frequency_from_tracks(tracks.data)
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
        idx_plt = np.argwhere(
            (track.lon.values < centroids.total_bounds[2] + 1)
            & (centroids.total_bounds[0] - 1 < track.lon.values)
            & (track.lat.values < centroids.total_bounds[3] + 1)
            & (centroids.total_bounds[1] - 1 < track.lat.values)
        ).reshape(-1)

        tc_list = []
        tr_coord = {'lat': [], 'lon': []}
        for node in range(idx_plt.size - 2):
            tr_piece = track.sel(
                time=slice(track.time.values[idx_plt[node]],
                           track.time.values[idx_plt[node + 2]]))
            tr_piece.attrs['n_nodes'] = 2  # plot only one node
            tr_sel = TCTracks()
            tr_sel.append(tr_piece)
            tr_coord['lat'].append(tr_sel.data[0].lat.values[:-1])
            tr_coord['lon'].append(tr_sel.data[0].lon.values[:-1])

            tc_tmp = TropCyclone()
            tc_tmp.set_from_tracks(tr_sel, centroids)
            tc_tmp.event_name = [
                track.name + ' ' + time.strftime(
                    "%d %h %Y %H:%M",
                    time.gmtime(tr_sel.data[0].time[1].values.astype(int)
                                / 1000000000)
                )
            ]
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
            pbar = tqdm(total=idx_plt.size - 2)
            ani = animation.FuncAnimation(fig, run, frames=idx_plt.size - 2,
                                          interval=500, blit=False)
            ani.save(file_name, writer=writer)
            pbar.close()
        return tc_list, tr_coord

    def frequency_from_tracks(self, tracks):
        """Set hazard frequency from tracks data.

        Parameters:
            tracks (list of xarray.Dataset)
        """
        if not tracks:
            return
        year_max = np.amax([t.time.dt.year.values.max() for t in tracks])
        year_min = np.amin([t.time.dt.year.values.min() for t in tracks])
        year_delta = year_max - year_min + 1
        num_orig = np.count_nonzero(self.orig)
        ens_size = (self.event_id.size / num_orig) if num_orig > 0 else 1
        self.frequency = np.ones(self.event_id.size) / (year_delta * ens_size)

    def _tc_from_track(self, track, centroids, coastal_idx, model='H08',
                       store_windfields=False):
        """Generate windfield hazard from a single track dataset

        Parameters:
            track (xr.Dataset): single tropical cyclone track.
            centroids (Centroids): Centroids instance.
            coastal_idx (np.array): Indices of centroids close to coast.
            model (str, optional): Windfield model. Default: H08.
            store_windfields (boolean, optional): If True, store windfields.
                Default: False.

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
        ncentroids = centroids.coord.shape[0]
        coastal_centr = centroids.coord[coastal_idx]
        windfields = compute_windfields(track, coastal_centr, mod_id)
        npositions = windfields.shape[0]
        intensity = np.zeros(ncentroids)
        intensity[coastal_idx] = np.linalg.norm(windfields, axis=-1)\
                                                .max(axis=0)
        intensity[intensity < self.intensity_thres] = 0

        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, 'Name: ' + track.name)
        new_haz.intensity = sparse.csr_matrix(intensity.reshape(1, -1))
        if store_windfields:
            wf_full = np.zeros((npositions, ncentroids, 2))
            wf_full[:, coastal_idx, :] = windfields
            new_haz.windfields = [
                sparse.csr_matrix(wf_full.reshape(npositions, -1))]
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
                select = select & tmp_select
            if chg['function'] == np.multiply:
                change = 1 + (chg['change'] - 1) * scale
            elif chg['function'] == np.add:
                change = chg['change'] * scale
            if select.any():
                new_val = getattr(haz_cc, chg['variable'])
                new_val[select] *= change
                setattr(haz_cc, chg['variable'], new_val)
        return haz_cc

def compute_windfields(track, centroids, model):
    """Compute 1-minute sustained winds (in m/s) at 10 meters above ground

    Parameters:
        track (xr.Dataset): track infomation
        centroids (2d np.array): each row is a centroid [lat, lon]
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        np.array
    """
    # copies of track data
    t_lat, t_lon, t_tstep, t_rad, t_env, t_cen = [
        track[ar].values.copy() for ar in ['lat', 'lon', 'time_step', 'radius_max_wind',
                                           'environmental_pressure', 'central_pressure']
    ]

    ncentroids = centroids.shape[0]
    npositions = t_lat.shape[0]
    windfields = np.zeros((npositions, ncentroids, 2))

    if t_lon.size < 2:
        return windfields

    # never use longitudes at -180 degrees or below
    t_lon[t_lon <= -180] += 360

    # only use longitudes above 180, if 180 degree border is crossed
    if t_lon.min() > 180:
        t_lon -= 360

    # restrict to centroids in rectangular bounding box around track
    track_centr_msk = _close_centroids(t_lat, t_lon, centroids)
    track_centr_idx = track_centr_msk.nonzero()[0]
    track_centr = centroids[track_centr_msk]

    if track_centr.shape[0] == 0:
        return windfields

    # compute distances and vectors to all centroids
    d_centr, v_centr = [ar[0] for ar in dist_approx(
        t_lat[None], t_lon[None],
        track_centr[None, :, 0], track_centr[None, :, 1],
        log=True, method="geosphere")]

    # exclude centroids that are too far from or too close to the eye
    close_centr = (d_centr < CENTR_NODE_MAX_DIST_KM) & (d_centr > 1e-2)
    if not np.any(close_centr):
        return windfields
    v_centr_normed = np.zeros_like(v_centr)
    v_centr_normed[close_centr] = v_centr[close_centr] / d_centr[close_centr, None]

    # make sure that central pressure never exceeds environmental pressure
    pres_exceed_msk = (t_cen > t_env)
    t_cen[pres_exceed_msk] = t_env[pres_exceed_msk]

    # extrapolate radius of max wind from pressure if not given
    t_rad[:] = estimate_rmw(t_rad, t_cen) * NM_TO_KM

    # translational speed of track at every node
    v_trans = _vtrans(t_lat, t_lon, t_tstep)
    v_trans_norm = v_trans[0]

    # adjust pressure at previous track point
    prev_pres = t_cen[:-1].copy()
    msk = (prev_pres < 850)
    prev_pres[msk] = t_cen[1:][msk]

    # compute b-value
    if model == 0:
        hol_b = _bs_hol08(v_trans_norm[1:], t_env[1:], t_cen[1:], prev_pres,
                          t_lat[1:], t_tstep[1:])
    else:
        raise NotImplementedError

    # derive angular velocity
    v_ang_norm = _stat_holland(d_centr[1:], t_rad[1:], hol_b, t_env[1:],
                               t_cen[1:], t_lat[1:], close_centr[1:])
    hemisphere = 'N'
    if np.count_nonzero(t_lat < 0) > np.count_nonzero(t_lat > 0):
        hemisphere = 'S'
    v_ang_rotate = [1.0, -1.0] if hemisphere == 'N' else [-1.0, 1.0]
    v_ang_dir = np.array(v_ang_rotate)[..., :] * v_centr_normed[1:, :, ::-1]
    v_ang = np.zeros_like(v_ang_dir)
    v_ang[close_centr[1:]] = v_ang_norm[close_centr[1:], None] \
                             * v_ang_dir[close_centr[1:]]

    # Influence of translational speed decreases with distance from eye.
    # The "absorbing factor" is according to the following paper (see Fig. 7):
    #
    #   Mouton, F., & Nordbeck, O. (1999). Cyclone Database Manager. A tool
    #   for converting point data from cyclone observations into tracks and
    #   wind speed profiles in a GIS. UNED/GRID-Geneva.
    #   https://unepgrid.ch/en/resource/19B7D302
    #
    t_rad_bc = np.broadcast_arrays(t_rad[:, None], d_centr)[0]
    v_trans_corr = np.zeros_like(d_centr)
    v_trans_corr[close_centr] = np.fmin(1, t_rad_bc[close_centr] / d_centr[close_centr])

    # add angular and corrected translational velocity vectors
    v_full = v_trans[1][1:, None, :] * v_trans_corr[1:, :, None] + v_ang
    v_full[np.isnan(v_full)] = 0

    windfields[1:, track_centr_idx, :] = v_full
    return windfields

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
    return msk_lat & msk_lon

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
        Same shape as input, the first velocity is always 0.
    v_trans : np.array
        Directional vectors of velocity.
    """
    v_trans = np.zeros((t_lat.size, 2))
    v_trans_norm = np.zeros((t_lat.size,))
    norm, vec = dist_approx(t_lat[:-1, None], t_lon[:-1, None],
                            t_lat[1:, None], t_lon[1:, None],
                            log=True, method="geosphere")
    v_trans[1:, :] = vec[:, 0, 0]
    v_trans[1:, :] *= KMH_TO_MS / t_tstep[1:, None]
    v_trans_norm[1:] = norm[:, 0, 0]
    v_trans_norm[1:] *= KMH_TO_MS / t_tstep[1:]

    # limit to 30 nautical miles per hour
    msk = (v_trans_norm > 30 * KN_TO_MS)
    fact = 30 * KN_TO_MS / v_trans_norm[msk]
    v_trans[msk, :] *= fact[:, None]
    v_trans_norm[msk] *= fact
    return v_trans_norm, v_trans

def _bs_hol08(v_trans, penv, pcen, prepcen, lat, tint):
    """Holland's 2008 b-value computation for sustained surface winds

    The parameter applies to 1-minute sustained winds at 10 meters above ground.
    It is taken from equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    b_s = -4.4 * 1e-5 * (penv - pcen)^2 + 0.01 * (penv - pcen)
          + 0.03 * (dp/dt) - 0.014 * |lat| + 0.15 * (v_trans)^hol_xx + 1.0

    where `dp/dt` is the time derivative of central pressure and `hol_xx` is
    Holland's x parameter: hol_xx = 0.6 * (1 - (penv - pcen) / 215)

    Parameters:
        v_trans (float): translational wind (m/s)
        penv (float): environmental pressure (hPa)
        pcen (float): central pressure (hPa)
        prepcen (float): previous central pressure (hPa)
        lat (float): latitude (degrees)
        tint (float): time step (h)

    Returns:
        float
    """
    hol_xx = 0.6 * (1. - (penv - pcen) / 215)
    hol_b = -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv - pcen) + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0
    return np.clip(hol_b, 1, 2.5)

def _stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, close_centr):
    """Holland symmetric and static wind field (in m/s)

    Because recorded winds are less reliable than pressure, recorded wind speeds
    are not used, but 1-min sustained surface winds are estimated from central
    pressure using the formula `v_m = ((b_s / (rho * e)) * (penv - pcen))^0.5`,
    see equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    Depending on the hol_b parameter, the resulting model is according to
    Holland (1980), which models gradient winds, or Holland (2008), which models
    surface winds at 10 meters above ground.

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
    r_max, hol_b, lat, penv, pcen, d_centr = [
        ar[close_centr] for ar in np.broadcast_arrays(
            r_max[:, None], hol_b[:, None], lat[:, None],
            penv[:, None], pcen[:, None], d_centr)
    ]

    # air density
    rho = 1.15

    # Coriolis force parameter
    f_val = 2 * 0.0000729 * np.sin(np.radians(np.abs(lat)))

    # d_centr is in km, convert to m and apply Coriolis force factor
    d_centr_mult = 0.5 * 1000 * d_centr * f_val

    # the factor 100 is from conversion between mbar and pascal
    r_max_norm = (r_max / d_centr)**hol_b
    sqrt_term = 100 * hol_b / rho * r_max_norm * (penv - pcen) \
                * np.exp(-r_max_norm) + d_centr_mult**2

    v_ang[close_centr] = np.sqrt(np.fmax(0, sqrt_term)) - d_centr_mult
    return v_ang
