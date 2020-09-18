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

Generate synthetic tropical cyclone tracks from real ones
"""

import array
import itertools
import logging
import matplotlib.cm as cm_mp
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from numba import jit
import numpy as np

from climada.util.config import CONFIG
import climada.util.coordinates
import climada.hazard.tc_tracks

LOGGER = logging.getLogger(__name__)


def calc_random_walk(tracks,
                     ens_size=9,
                     ens_amp0=1.5,
                     ens_amp=0.1,
                     max_angle=np.pi / 10,
                     seed=CONFIG['trop_cyclone']['random_seed'],
                     decay=True):
    """
    Generate synthetic tracks based on directed random walk. An ensemble of
    tracks is computed for every track contained.
    Please note that there is a bias towards higher latitudes in the random
    wiggle. The wiggles are applied for each timestep. Please consider using
    equal_timestep() for unification before generating synthetic tracks.
    Be careful when changing ens_amp and max_angle and test changes of the
    parameter values before application.

    The object is mutated in-place.

    Parameters:
        tracks (TCTracks): See `climada.hazard.tc_tracks`.
        ens_size (int, optional): number of ensemble members per track.
            Default 9.
        ens_amp0 (float, optional): amplitude of max random starting point
            shift in decimal degree (longitude and latitude). Default: 1.5
        ens_amp (float, optional): amplitude of random walk wiggles in
            decimal degree (longitude and latitude). Default: 0.1
        max_angle (float, optional): maximum angle of variation. Default: pi/10.
            - max_angle=pi results in undirected random change with
                no change in direction;
            - max_angle=0 (or very close to 0) is not recommended. It results
                in non-random synthetic tracks with constant shift to higher latitudes;
            - for 0<max_angle<pi/2, the change in latitude is always toward
                higher latitudes, i.e. poleward,
            - max_angle=pi/4 results in random angles within one quadrant,
                also with a poleward bias;
            - decreasing max_angle starting from pi will gradually increase
                the spread of the synthetic tracks until they start converging
                towards the result of max_angle=0 at a certain value (depending
                on length of timesteps and ens_amp).
        seed (int, optional): random number generator seed for replicability
            of random walk. Put negative value if you don't want to use it.
            Default: configuration file
        decay (bool, optional): compute land decay in probabilistic tracks.
            Default: True
    """
    LOGGER.info('Computing %s synthetic tracks.', ens_size * tracks.size)

    if max_angle == 0:
        LOGGER.warning('max_angle=0 is not recommended. It results in non-random \
                     synthetic tracks with a constant shift to higher latitudes.')
    if seed >= 0:
        np.random.seed(seed)

    random_vec = [np.random.uniform(size=ens_size * (2 + track.time.size))
                  for track in tracks.data]

    if tracks.pool:
        chunksize = min(tracks.size // tracks.pool.ncpus, 1000)
        new_ens = tracks.pool.map(_one_rnd_walk, tracks.data,
                                  itertools.repeat(ens_size, tracks.size),
                                  itertools.repeat(ens_amp0, tracks.size),
                                  itertools.repeat(ens_amp, tracks.size),
                                  itertools.repeat(max_angle, tracks.size),
                                  random_vec, chunksize=chunksize)
    else:
        new_ens = [_one_rnd_walk(track, ens_size, ens_amp0, ens_amp,
                                 max_angle, rand)
                   for track, rand in zip(tracks.data, random_vec)]

    tracks.data = sum(new_ens, [])

    if decay:
        hist_tracks = [track for track in tracks.data if track.orig_event_flag]
        if hist_tracks:
            try:
                extent = tracks.get_extent()
                land_geom = climada.util.coordinates.get_land_geometry(
                    extent=extent, resolution=10
                )
                v_rel, p_rel = _calc_land_decay(hist_tracks, land_geom,
                                                pool=tracks.pool)
                tracks.data = _apply_land_decay(tracks.data, v_rel, p_rel,
                                                land_geom, pool=tracks.pool)
            except ValueError:
                LOGGER.info('No land decay coefficients could be applied.')
        else:
            LOGGER.error('No historical tracks contained. '
                         'Historical tracks are needed for land decay.')


@jit(parallel=True)
def _one_rnd_walk(track, ens_size, ens_amp0, ens_amp, max_angle, rnd_vec):
    """Interpolate values of one track.

    Parameters:
        track (xr.Dataset): track data

    Returns:
        list(xr.Dataset)
    """
    ens_track = list()
    n_dat = track.time.size
    xy_ini = ens_amp0 * (rnd_vec[:2 * ens_size].reshape((2, ens_size)) - 0.5)
    tmp_ang = np.cumsum(2 * max_angle * rnd_vec[2 * ens_size:] - max_angle)
    coord_xy = np.empty((2, ens_size * n_dat))
    coord_xy[0] = np.cumsum(ens_amp * np.sin(tmp_ang))
    coord_xy[1] = np.cumsum(ens_amp * np.cos(tmp_ang))

    ens_track.append(track)
    for i_ens in range(ens_size):
        i_track = track.copy(True)

        d_xy = coord_xy[:, i_ens * n_dat: (i_ens + 1) * n_dat] - \
            np.expand_dims(coord_xy[:, i_ens * n_dat], axis=1)
        # change sign of latitude change for southern hemishpere:
        d_xy = np.sign(track.lat.values[0]) * d_xy

        d_lat_lon = d_xy + np.expand_dims(xy_ini[:, i_ens], axis=1)

        i_track.lon.values = i_track.lon.values + d_lat_lon[0, :]
        i_track.lat.values = i_track.lat.values + d_lat_lon[1, :]
        i_track.attrs['orig_event_flag'] = False
        i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + str(i_ens + 1)
        i_track.attrs['sid'] = i_track.attrs['sid'] + '_gen' + str(i_ens + 1)
        i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens + 1) / 100

        ens_track.append(i_track)

    return ens_track


def _calc_land_decay(hist_tracks, land_geom, s_rel=True, check_plot=False,
                     pool=None):
    """Compute wind and pressure decay coefficients from historical events

    Decay is calculated for every TC category according to the formulas:

        - wind decay = exp(-x*A)
        - pressure decay = S-(S-1)*exp(-x*B)

    Parameters:
        hist_tracks (list): List of xarray Datasets describing TC tracks.
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool, optional): use environmental presure to calc S value
            (true) or central presure (false)
        check_plot (bool, optional): visualize computed coefficients.
            Default: False

    Returns:
        v_rel (dict(category: A)), p_rel (dict(category: (S, B)))
    """
    # Key is Saffir-Simpson scale
    # values are lists of wind/wind at landfall
    v_lf = dict()
    # values are tuples with first value the S parameter, second value
    # list of central pressure/central pressure at landfall
    p_lf = dict()
    # x-scale values to compute landfall decay
    x_val = dict()

    if pool:
        dec_val = pool.map(_decay_values, hist_tracks, itertools.repeat(land_geom),
                           itertools.repeat(s_rel),
                           chunksize=min(len(hist_tracks) // pool.ncpus, 1000))
    else:
        dec_val = [_decay_values(track, land_geom, s_rel) for track in hist_tracks]

    for (tv_lf, tp_lf, tx_val) in dec_val:
        for key in tv_lf.keys():
            v_lf.setdefault(key, []).extend(tv_lf[key])
            p_lf.setdefault(key, ([], []))
            p_lf[key][0].extend(tp_lf[key][0])
            p_lf[key][1].extend(tp_lf[key][1])
            x_val.setdefault(key, []).extend(tx_val[key])

    v_rel, p_rel = _decay_calc_coeff(x_val, v_lf, p_lf)
    if check_plot:
        _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel)

    return v_rel, p_rel


def _apply_land_decay(tracks, v_rel, p_rel, land_geom, s_rel=True,
                      check_plot=False, pool=None):
    """Compute wind and pressure decay due to landfall in synthetic tracks.

    Parameters:
        v_rel (dict): {category: A}, where wind decay = exp(-x*A)
        p_rel (dict): (category: (S, B)}, where pressure decay
            = S-(S-1)*exp(-x*B)
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool, optional): use environmental presure to calc S value
            (true) or central presure (false)
        check_plot (bool, optional): visualize computed changes
    """
    sy_tracks = [track for track in tracks if not track.orig_event_flag]
    if not sy_tracks:
        LOGGER.error('No synthetic tracks contained. Synthetic tracks'
                     ' are needed.')
        raise ValueError

    if not v_rel or not p_rel:
        LOGGER.info('No decay coefficients.')
        return

    if check_plot:
        orig_wind, orig_pres = [], []
        for track in sy_tracks:
            orig_wind.append(np.copy(track.max_sustained_wind.values))
            orig_pres.append(np.copy(track.central_pressure.values))

    if pool:
        chunksize = min(len(tracks) // pool.ncpus, 1000)
        tracks = pool.map(_apply_decay_coeffs, tracks,
                          itertools.repeat(v_rel), itertools.repeat(p_rel),
                          itertools.repeat(land_geom), itertools.repeat(s_rel),
                          chunksize=chunksize)
    else:
        tracks = [_apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel)
                  for track in tracks]

    if check_plot:
        _check_apply_decay_plot(tracks, orig_wind, orig_pres)
    return tracks


def _decay_values(track, land_geom, s_rel):
    """Compute wind and pressure relative to landafall values.

    Parameters
    ----------
    track : xr.Dataset
        track
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        land geometry
    s_rel : bool
        use environmental presure for S value (true) or central presure (false)

    Returns
    -------
    v_lf : dict
        key is Saffir-Simpson scale, values are arrays of wind/wind at landfall
    p_lf : dict
        key is Saffir-Simpson scale, values are tuples with first value array
        of S parameter, second value array of central pressure/central pressure
        at landfall
    x_val : dict
        key is Saffir-Simpson scale, values are arrays with the values used as
        "x" in the coefficient fitting, the distance since landfall
    """
    v_lf = dict()
    p_lf = dict()
    x_val = dict()

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    # Index in land that comes from previous sea index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
        for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
            v_landfall = track.max_sustained_wind[sea_land - 1].values
            scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
            ss_scale_idx = np.where(v_landfall < scale_thresholds)[0][0] + 1

            v_land = track.max_sustained_wind[sea_land - 1:land_sea].values
            if v_land[0] > 0:
                v_land = (v_land[1:] / v_land[0]).tolist()
            else:
                v_land = v_land[1:].tolist()

            p_landfall = float(track.central_pressure[sea_land - 1].values)
            p_land = track.central_pressure[sea_land - 1:land_sea].values
            p_land = (p_land[1:] / p_land[0]).tolist()

            p_land_s = _calc_decay_ps_value(
                track, p_landfall, land_sea - 1, s_rel)
            p_land_s = len(p_land) * [p_land_s]

            if ss_scale_idx not in v_lf:
                v_lf[ss_scale_idx] = array.array('f', v_land)
                p_lf[ss_scale_idx] = (array.array('f', p_land_s),
                                      array.array('f', p_land))
                x_val[ss_scale_idx] = array.array('f',
                                                  track.dist_since_lf[sea_land:land_sea])
            else:
                v_lf[ss_scale_idx].extend(v_land)
                p_lf[ss_scale_idx][0].extend(p_land_s)
                p_lf[ss_scale_idx][1].extend(p_land)
                x_val[ss_scale_idx].extend(track.dist_since_lf[sea_land:land_sea])
    return v_lf, p_lf, x_val


def _decay_calc_coeff(x_val, v_lf, p_lf):
    """From track's relative velocity and pressure, compute the decay
    coefficients.
    - wind decay = exp(-x*A)
    - pressure decay = S-(S-1)*exp(-x*A)

    Parameters:
        x_val (dict): key is Saffir-Simpson scale, values are lists with
            the values used as "x" in the coefficient fitting, the
            distance since landfall
        v_lf (dict): key is Saffir-Simpson scale, values are lists of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value the S parameter, second value list of central
            pressure/central pressure at landfall

    Returns:
        v_rel (dict()), p_rel (dict())
    """
    np.warnings.filterwarnings('ignore')
    v_rel = dict()
    p_rel = dict()
    for ss_scale, val_lf in v_lf.items():
        x_val_ss = np.array(x_val[ss_scale])

        y_val = np.array(val_lf)
        v_coef = _solve_decay_v_function(y_val, x_val_ss)
        v_coef = v_coef[np.isfinite(v_coef)]
        v_coef = np.mean(v_coef)

        ps_y_val = np.array(p_lf[ss_scale][0])
        y_val = np.array(p_lf[ss_scale][1])
        y_val[ps_y_val <= y_val] = np.nan
        y_val[ps_y_val <= 1] = np.nan
        valid_p = np.isfinite(y_val)
        ps_y_val = ps_y_val[valid_p]
        y_val = y_val[valid_p]
        p_coef = _solve_decay_p_function(ps_y_val, y_val, x_val_ss[valid_p])
        ps_y_val = np.mean(ps_y_val)
        p_coef = np.mean(p_coef)

        if np.isfinite(v_coef) and np.isfinite(ps_y_val) and np.isfinite(ps_y_val):
            v_rel[ss_scale] = v_coef
            p_rel[ss_scale] = (ps_y_val, p_coef)

    scale_fill = np.array(list(p_rel.keys()))
    if not scale_fill.size:
        LOGGER.info('No historical track with landfall.')
        return v_rel, p_rel
    for ss_scale in range(1, len(climada.hazard.tc_tracks.SAFFIR_SIM_CAT) + 1):
        if ss_scale not in p_rel:
            close_scale = scale_fill[np.argmin(np.abs(scale_fill - ss_scale))]
            LOGGER.debug('No historical track of category %s with landfall. '
                         'Decay parameters from category %s taken.',
                         climada.hazard.tc_tracks.CAT_NAMES[ss_scale - 2],
                         climada.hazard.tc_tracks.CAT_NAMES[close_scale - 2])
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]

    return v_rel, p_rel


def _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel):
    """Generate one graph with wind decay and an other with central pressure
    decay, true and approximated."""
    # One graph per TC category
    for track_cat, color in zip(v_lf.keys(),
                                cm_mp.rainbow(np.linspace(0, 1, len(v_lf)))):
        _, axes = plt.subplots(2, 1)
        x_eval = np.linspace(0, np.max(x_val[track_cat]), 20)

        axes[0].set_xlabel('Distance from landfall (km)')
        axes[0].set_ylabel('Max sustained wind relative to landfall')
        axes[0].set_title('Wind')
        axes[0].plot(x_val[track_cat], v_lf[track_cat], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat - 2])
        axes[0].plot(x_eval, _decay_v_function(v_rel[track_cat], x_eval),
                     '-', c=color)

        axes[1].set_xlabel('Distance from landfall (km)')
        axes[1].set_ylabel('Central pressure relative to landfall')
        axes[1].set_title('Pressure')
        axes[1].plot(x_val[track_cat], p_lf[track_cat][1], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat - 2])
        axes[1].plot(
            x_eval,
            _decay_p_function(p_rel[track_cat][0], p_rel[track_cat][1], x_eval),
            '-', c=color)


def _apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel):
    """Change track's max sustained wind and central pressure using the land
    decay coefficients.

    Parameters:
        track (xr.Dataset): TC track
        v_rel (dict): {category: A}, where wind decay = exp(-x*A)
        p_rel (dict): (category: (S, B)},
            where pressure decay = S-(S-1)*exp(-x*B)
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool): use environmental presure for S value (true) or
            central presure (false)

    Returns:
        xr.Dataset
    """
    # return if historical track
    if track.orig_event_flag:
        return track

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    # Index in land that comes from previous sea index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if not sea_land_idx.size or land_sea_idx.size > sea_land_idx.size:
        return track
    for idx, (sea_land, land_sea) \
            in enumerate(zip(sea_land_idx, land_sea_idx)):
        v_landfall = track.max_sustained_wind[sea_land - 1].values
        p_landfall = float(track.central_pressure[sea_land - 1].values)
        scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
        try:
            ss_scale_idx = np.where(v_landfall < scale_thresholds)[0][0] + 1
        except IndexError:
            continue
        if land_sea - sea_land == 1:
            continue
        p_decay = _calc_decay_ps_value(track, p_landfall, land_sea - 1, s_rel)
        p_decay = _decay_p_function(p_decay, p_rel[ss_scale_idx][1],
                                    track.dist_since_lf[sea_land:land_sea].values)
        # dont applay decay if it would decrease central pressure
        p_decay[p_decay < 1] = track.central_pressure[sea_land:land_sea][p_decay < 1] / p_landfall
        track.central_pressure[sea_land:land_sea] = p_landfall * p_decay

        v_decay = _decay_v_function(v_rel[ss_scale_idx],
                                    track.dist_since_lf[sea_land:land_sea].values)
        # dont applay decay if it would increas wind speeds
        v_decay[v_decay > 1] = (track.max_sustained_wind[sea_land:land_sea][v_decay > 1]
                                / v_landfall)
        track.max_sustained_wind[sea_land:land_sea] = v_landfall * v_decay

        # correct values of sea between two landfalls
        if land_sea < track.time.size and idx + 1 < sea_land_idx.size:
            rndn = 0.1 * float(np.abs(np.random.normal(size=1) * 5) + 6)
            r_diff = track.central_pressure[land_sea].values - \
                track.central_pressure[land_sea - 1].values + rndn
            track.central_pressure[land_sea:sea_land_idx[idx + 1]] += - r_diff

            rndn = rndn * 10  # mean value 10
            r_diff = track.max_sustained_wind[land_sea].values - \
                track.max_sustained_wind[land_sea - 1].values - rndn
            track.max_sustained_wind[land_sea:sea_land_idx[idx + 1]] += - r_diff

    # correct limits
    np.warnings.filterwarnings('ignore')
    cor_p = track.central_pressure.values > track.environmental_pressure.values
    track.central_pressure[cor_p] = track.environmental_pressure[cor_p]
    track.max_sustained_wind[track.max_sustained_wind < 0] = 0
    track.attrs['category'] = climada.hazard.tc_tracks.set_category(
        track.max_sustained_wind.values, track.max_sustained_wind_unit)
    return track


def _check_apply_decay_plot(all_tracks, syn_orig_wind, syn_orig_pres):
    """Plot wind and presure before and after correction for synthetic tracks.
    Plot wind and presure for unchanged historical tracks."""
    # Plot synthetic tracks
    sy_tracks = [track for track in all_tracks if not track.orig_event_flag]
    graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a = \
        _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                    syn_orig_pres)

    # Plot historic tracks
    hist_tracks = [track for track in all_tracks if track.orig_event_flag]
    graph_hv, graph_hp, graph_hpd_a, graph_hped_a = \
        _check_apply_decay_hist_plot(hist_tracks)

    # Put legend and fix size
    scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
    leg_lines = [Line2D([0], [0], color=climada.hazard.tc_tracks.CAT_COLORS[i_col], lw=2)
                 for i_col in range(len(scale_thresholds))]
    leg_lines.append(Line2D([0], [0], color='k', lw=2))
    leg_names = [climada.hazard.tc_tracks.CAT_NAMES[i_col]
                 for i_col in sorted(climada.hazard.tc_tracks.CAT_NAMES.keys())]
    leg_names.append('Sea')
    all_gr = [graph_v_a, graph_v_b, graph_p_a, graph_p_b, graph_ped_a,
              graph_pd_a, graph_hv, graph_hp, graph_hpd_a, graph_hped_a]
    for graph in all_gr:
        graph.axs[0].legend(leg_lines, leg_names)
        fig, _ = graph.get_elems()
        fig.set_size_inches(18.5, 10.5)


def _calc_decay_ps_value(track, p_landfall, pos, s_rel):
    if s_rel:
        p_land_s = track.environmental_pressure[pos].values
    else:
        p_land_s = track.central_pressure[pos].values
    return float(p_land_s / p_landfall)


def _decay_v_function(a_coef, x_val):
    """Decay function used for wind after landfall."""
    return np.exp(-a_coef * x_val)


def _solve_decay_v_function(v_y, x_val):
    """Solve decay function used for wind after landfall. Get A coefficient."""
    return -np.log(v_y) / x_val


def _decay_p_function(s_coef, b_coef, x_val):
    """Decay function used for pressure after landfall."""
    return s_coef - (s_coef - 1) * np.exp(-b_coef * x_val)


def _solve_decay_p_function(ps_y, p_y, x_val):
    """Solve decay function used for pressure after landfall.
    Get B coefficient."""
    return -np.log((ps_y - p_y) / (ps_y - 1.0)) / x_val


def _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                syn_orig_pres):
    """Plot winds and pressures of synthetic tracks before and after
    correction."""
    _, graph_v_b = plt.subplots()
    graph_v_b.set_title('Wind before land decay correction')
    graph_v_b.set_xlabel('Node number')
    graph_v_b.set_ylabel('Max sustained wind (kn)')

    _, graph_v_a = plt.subplots()
    graph_v_a.set_title('Wind after land decay correction')
    graph_v_a.set_xlabel('Node number')
    graph_v_a.set_ylabel('Max sustained wind (kn)')

    _, graph_p_b = plt.subplots()
    graph_p_b.set_title('Pressure before land decay correctionn')
    graph_p_b.set_xlabel('Node number')
    graph_p_b.set_ylabel('Central pressure (mb)')

    _, graph_p_a = plt.subplots()
    graph_p_a.set_title('Pressure after land decay correctionn')
    graph_p_a.set_xlabel('Node number')
    graph_p_a.set_ylabel('Central pressure (mb)')

    _, graph_pd_a = plt.subplots()
    graph_pd_a.set_title('Relative pressure after land decay correction')
    graph_pd_a.set_xlabel('Distance from landfall (km)')
    graph_pd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_ped_a = plt.subplots()
    graph_ped_a.set_title(
        'Environmental - central pressure after land decay correction')
    graph_ped_a.set_xlabel('Distance from landfall (km)')
    graph_ped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track, orig_wind, orig_pres in \
            zip(sy_tracks, syn_orig_wind, syn_orig_pres):
        # Index in land that comes from previous sea index
        sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
        # Index in sea that comes from previous land index
        land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
        if track.on_land[-1]:
            land_sea_idx = np.append(land_sea_idx, track.time.size)
        if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                v_lf = track.max_sustained_wind[sea_land - 1].values
                p_lf = track.central_pressure[sea_land - 1].values
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                ss_scale = np.where(v_lf < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_v_a.plot(on_land, track.max_sustained_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])
                graph_v_b.plot(on_land, orig_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])
                graph_p_a.plot(on_land, track.central_pressure[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])
                graph_p_b.plot(on_land, orig_pres[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])
                graph_pd_a.plot(track.dist_since_lf[on_land],
                                track.central_pressure[on_land] / p_lf,
                                'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])
                graph_ped_a.plot(track.dist_since_lf[on_land],
                                 track.environmental_pressure[on_land] -
                                 track.central_pressure[on_land],
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale])

            on_sea = np.arange(track.time.size)[~track.on_land]
            graph_v_a.plot(on_sea, track.max_sustained_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_v_b.plot(on_sea, orig_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_p_a.plot(on_sea, track.central_pressure[on_sea],
                           'o', c='k', markersize=5)
            graph_p_b.plot(on_sea, orig_pres[on_sea],
                           'o', c='k', markersize=5)

    return graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a


def _check_apply_decay_hist_plot(hist_tracks):
    """Plot winds and pressures of historical tracks."""
    _, graph_hv = plt.subplots()
    graph_hv.set_title('Historical wind')
    graph_hv.set_xlabel('Node number')
    graph_hv.set_ylabel('Max sustained wind (kn)')

    _, graph_hp = plt.subplots()
    graph_hp.set_title('Historical pressure')
    graph_hp.set_xlabel('Node number')
    graph_hp.set_ylabel('Central pressure (mb)')

    _, graph_hpd_a = plt.subplots()
    graph_hpd_a.set_title('Historical relative pressure')
    graph_hpd_a.set_xlabel('Distance from landfall (km)')
    graph_hpd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_hped_a = plt.subplots()
    graph_hped_a.set_title('Historical environmental - central pressure')
    graph_hped_a.set_xlabel('Distance from landfall (km)')
    graph_hped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track in hist_tracks:
        # Index in land that comes from previous sea index
        sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
        # Index in sea that comes from previous land index
        land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
        if track.on_land[-1]:
            land_sea_idx = np.append(land_sea_idx, track.time.size)
        if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                scale = np.where(track.max_sustained_wind[sea_land - 1].values
                                 < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_hv.add_curve(on_land, track.max_sustained_wind[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[scale])
                graph_hp.add_curve(on_land, track.central_pressure[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[scale])
                graph_hpd_a.plot(track.dist_since_lf[on_land],
                                 track.central_pressure[on_land]
                                 / track.central_pressure[sea_land - 1].values,
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[scale])
                graph_hped_a.plot(track.dist_since_lf[on_land],
                                  track.environmental_pressure[on_land] -
                                  track.central_pressure[on_land],
                                  'o', c=climada.hazard.tc_tracks.CAT_COLORS[scale])

            on_sea = np.arange(track.time.size)[~track.on_land]
            graph_hp.plot(on_sea, track.central_pressure[on_sea],
                          'o', c='k', markersize=5)
            graph_hv.plot(on_sea, track.max_sustained_wind[on_sea],
                          'o', c='k', markersize=5)

    return graph_hv, graph_hp, graph_hpd_a, graph_hped_a
