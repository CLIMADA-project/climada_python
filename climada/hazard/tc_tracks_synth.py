"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Generate synthetic tropical cyclone tracks from real ones
"""

import array
import itertools
import logging
from typing import Dict
import os
import warnings
import matplotlib.cm as cm_mp
import matplotlib.pyplot as plt
import numba
import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import scipy.interpolate
from matplotlib.lines import Line2D
from pathos.abstract_launcher import AbstractWorkerPool
from shapely.geometry.multipolygon import MultiPolygon
from pathlib import Path
from copy import deepcopy
import statsmodels.api as sm

import climada.hazard.tc_tracks
import climada.util.coordinates
from climada import CONFIG

LOGGER = logging.getLogger(__name__)

LANDFALL_DECAY_V = {
    -1: 0.00012859077693295416,
    0: 0.0017226346292718126,
    1: 0.002309772914350468,
    2: 0.0025968221565522698,
    3: 0.002626252944053856,
    4: 0.002550639312763181,
    5: 0.003788695795963695
}
"""Global landfall decay parameters for wind speed by TC category.

Keys are TC categories with -1='TD', 0='TS', 1='Cat 1', ..., 5='Cat 5'.

It is v_rel as derived from:

>>> tracks = TCTracks.from_ibtracs_netcdf(year_range=(1980,2019), estimate_missing=True)
>>> extent = tracks.get_extent()
>>> land_geom = climada.util.coordinates.get_land_geometry(
...     extent=extent, resolution=10
... )
>>> v_rel, p_rel = _calc_land_decay(tracks.data, land_geom, pool=tracks.pool)
"""

LANDFALL_DECAY_P = {
    -1: (1.0088807492745373, 0.002117478217863062),
    0: (1.0192813768091684, 0.003068578025845065),
    1: (1.0362982218631644, 0.003620816186262243),
    2: (1.0468630800617038, 0.004067381088015585),
    3: (1.0639055205005432, 0.003708174876364079),
    4: (1.0828373148889825, 0.003997492773076179),
    5: (1.1088615145002092, 0.005224331234796362)}
"""Global landfall decay parameters for pressure by TC category.

Keys are TC categories with -1='TD', 0='TS', 1='Cat 1', ..., 5='Cat 5'.

It is p_rel as derived from:

>>> tracks = TCTracks.from_ibtracs_netcdf(year_range=(1980,2019), estimate_missing=True)
>>> extent = tracks.get_extent()
>>> land_geom = climada.util.coordinates.get_land_geometry(
...     extent=extent, resolution=10
... )
>>> v_rel, p_rel = _calc_land_decay(tracks.data, land_geom, pool=tracks.pool)
"""

DATA_DIR = Path(__file__).parent.joinpath('data')
RANDOM_WALK_DATA_INTENSIFICATION = pd.read_csv(DATA_DIR.joinpath('tc_intensification_params.csv'))
RANDOM_WALK_DATA_DURATION = pd.read_csv(DATA_DIR.joinpath('tc_peak_params.csv'), na_values='', keep_default_na=False)
RANDOM_WALK_DATA_SEA_DECAY = pd.read_csv(DATA_DIR.joinpath('tc_decay_params.csv'), na_values='', keep_default_na=False)

TIME_MODEL_BEFORE_HIST_LF_H = 4
"""Time (in hours) before a historical landfall from which intensity (central
pressure) is assumed to be affected by the upcoming landfall"""

RANDOM_WALK_DATA_CAT_STR = {
    -1: 'TD-TS',
    0: 'TD-TS',
    1: "Cat 1",
    2: "Cat 2",
    3: "Cat 3",
    4: "Cat 4",
    5: "Cat 5"
}

FIT_TRACK_VARS_RANGE = {
    'max_sustained_wind': {
        # minimum value in IBTrACS
        'min': 5,
        # maximum in IBTrACS USA is 185 kts. But cannot exclude more powerful
        # (basic theory says 175 kts, which has already been exceeded; also,
        # hypothetical hypercanes)
        'max': 200
    },
    'radius_max_wind': {
        # range in IBTrACS
        'min': 10,
        'max': 650
    },
    'radius_oci': {
        # radius of outermost closed isobar: range in IBTrACS
        'min': 4,
        'max': 300
    }
}
"""Dict of track variables to estimate from track-specific fit to central_pressure after
modelling intensity, containing 'min' and 'max' allowed values."""

FIT_TIME_ADJUST_HOUR = 6
"""Number of hours before track intensity is modelled for which to estimate
track parameters"""

NEGLECT_LANDFALL_DURATION_HOUR = 4.5
"""Minimum landfall duration in hours from which to correct intensity. Perturbed systems 
spending less time than this over land will not be adjusted for landfall effects"""

TRACK_EXTENSION_PARS = {
    'max_shift_ini': 0,
    'max_dspeed_rel': 0.05,
    'max_ddirection': np.radians(3.5),
    'autocorr_ddirection': 0.2,
    'autocorr_dspeed': 0,
    'decay_ddirection_hourly': 0
}
"""Trajectory perturbations parameters for track extension"""
# - bearing angle change has mean 0 deg/h and std 3.35 deg/h, autocorr=0.21
# - relative change in translation speed has a mean of 0% /hour and std 4.2%/hr,
#   autocorr=0

for envvar in [
    'TRACKGEN_TEST_DECAY_AT_EXTREME_LATS',
    'TRACKGEN_TEST_DECAY_BEYOND_ADVISORIES',
    'TRACKGEN_TEST_KILL_BEYOND_ADVISORIES',
    'TRACKGEN_TEST_KILL_BEYOND_EXTRATROPICAL',
    'TRACKGEN_TEST_TARGET_PRESSURE_OVER_SEA'
]:
    if os.getenv(envvar) is None:
        os.environ[envvar] = '1'

if int(os.getenv('TRACKGEN_TEST_DECAY_AT_EXTREME_LATS')):
    MAX_WIND_BY_LAT_RANGE = {
        'EP': {
            'lat_min': {'lat': [1, 2.5, 7.5], 'max_wind': [0, 72, 115]},
            'lat_max' : {'lat': [37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 65], 'max_wind': [95, 75, 75, 67, 55, 30, 0]}
        },
        'NA': {
            'lat_min': {'lat': [5, 7.5, 12.5], 'max_wind': [0, 97, 150]},
            'lat_max': {'lat': [57.5, 62.5, 67.5, 72.5, 75], 'max_wind': [75, 65, 50, 30, 0]}
        },
        'NI': {
            'lat_min': {'lat': [1, 2.5, 7.5], 'max_wind': [0, 35, 90]},
            'lat_max': {'lat': [22.5, 27.5, 30], 'max_wind': [140, 45, 0]}
        },
        'SA': {
            'lat_min': {'lat': [-40], 'max_wind': [45]},
            'lat_max': {'lat': [15], 'max_wind': [30]}
        },
        'SI': {
            'lat_min': {'lat': [-50, -47,5, -42.5, -37.5], 'max_wind': [0, 57, 74, 86]},
            'lat_max': {'lat': [-7.5, -2.5, -1], 'max_wind': [155, 49, 0]}
        },
        'SP': {
            'lat_min': {'lat': [-70, -67.5, -62.5, -57.5, -52.5, -47.5, -42.5], 'max_wind': [0, 63, 63, 63, 63, 85, 90]},
            'lat_max': {'lat': [-7.5, -2.5, -1], 'max_wind': [103, 34, 0]}
        },
        'WP': {
            'lat_min': {'lat': [1, 2.5, 7.5], 'max_wind': [0, 125, 165]},
            'lat_max': {'lat': [47.5, 52.5, 57.5, 62.5, 65], 'max_wind': [85, 75, 42, 40, 0]}
        }
    }
    """Maximum max_sustained_wind value found in each basin for latitudinal value ranges."""
else:
    MAX_WIND_BY_LAT_RANGE = {
        'EP': {
            'lat_min': {'lat': [2.5, 7.5], 'max_wind': [72, 115]},
            'lat_max' : {'lat': [37.5, 57.5], 'max_wind': [75, 55]}
        },
        'NA': {
            'lat_min': {'lat': [7.5, 12.5], 'max_wind': [97, 150]},
            'lat_max': {'lat': [47.5, 57.5, 67.5], 'max_wind': [105, 75, 50]}
        },
        'NI': {
            'lat_min': {'lat': [2.5, 7.5], 'max_wind': [65, 90]},
            'lat_max': {'lat': [27.5], 'max_wind': [65]}
        },
        'SA': {
            'lat_min': {'lat': [-40], 'max_wind': [65]},
            'lat_max': {'lat': [-15], 'max_wind': [40]}
        },
        'SI': {
            'lat_min': {'lat': [-42.5, -37.5, -32.5], 'max_wind': [55, 80, 100]},
            'lat_max': {'lat': [-7.5, -2.5], 'max_wind': [155, 45]}
        },
        'SP': {
            'lat_min': {'lat': [-52.5, -42.5, -32.5, -27.5], 'max_wind': [45, 60, 80, 120]},
            'lat_max': {'lat': [-7.5, -2.5], 'max_wind': [120, 30]}
        },
        'WP': {
            'lat_min': {'lat': [2.5, 7.5], 'max_wind': [115, 165]},
            'lat_max': {'lat': [37.5, 42.5, 47.2], 'max_wind': [110, 90, 70]}
        }
    }
    """Maximum max_sustained_wind value found in each basin for latitudinal value ranges."""


def calc_perturbed_trajectories(
    tracks,
    nb_synth_tracks: int = 9,
    max_shift_ini: float = 0.75,
    max_dspeed_rel: float = 0.3,
    max_ddirection: float = np.pi / 360,
    autocorr_dspeed: float = 0.85,
    autocorr_ddirection: float = 0.5,
    decay_ddirection_hourly: float = 1/(2.5*24),
    seed: int = CONFIG.hazard.trop_cyclone.random_seed.int(),
    adjust_intensity: str = 'explicit',
    central_pressure_pert: float = 7.5,
    decay: bool = None,
    use_global_decay_params: bool = True,
    extend_track: bool = True,
    pool: AbstractWorkerPool = None,
):
    """
    Generate synthetic tracks based on directed random walk. An ensemble of nb_synth_tracks
    synthetic tracks is computed for every track contained in self.

    The methodology perturbs the tracks locations, and if decay is True it additionally
    includes decay of wind speed and central pressure drop after landfall. No other track
    parameter is perturbed.
    The track starting point location is perturbed by random uniform values of
    magnitude up to max_shift_ini in both longitude and latitude. Then, each segment
    between two consecutive points is perturbed in direction and distance (i.e.,
    translational speed). These perturbations can be correlated in time, i.e.,
    the perturbation in direction applied to segment i is correlated with the perturbation
    in direction applied to segment i-1 (and similarly for the perturbation in translational
    speed).
    Perturbations in track direction and temporal auto-correlations in perturbations are
    on an hourly basis, and the perturbations in translational speed is relative.
    Hence, the parameter values are relatively insensitive to the temporal
    resolution of the tracks. Note however that all tracks should be at the same
    temporal resolution, which can be achieved using equal_timestep().
    max_dspeed_rel and autocorr_dspeed control the spread along the track ('what distance
    does the track run for'), while max_ddirection and autocorr_ddirection control the spread
    perpendicular to the track movement ('how does the track diverge in direction').
    max_dspeed_rel and max_ddirection control the amplitude of perturbations at each track
    timestep but perturbations may tend to compensate each other over time, leading to
    a similar location at the end of the track, while autocorr_dspeed and autocorr_ddirection
    control how these perturbations persist in time and hence the amplitude of the
    perturbations towards the end of the track.

    Note that the default parameter values have been only roughly calibrated so that
    the frequency of tracks in each 5x5degree box remains approximately constant.
    This is not an in-depth calibration and should be treated as such.
    The object is mutated in-place.

    Parameters
    ----------
    tracks : climada.hazard.TCTracks
        Tracks data.
    nb_synth_tracks : int, optional
        Number of ensemble members per track. Default: 9.
    max_shift_ini : float, optional
        Amplitude of max random starting point shift in decimal degree
        (up to +/-max_shift_ini for longitude and latitude). Default: 0.75.
    max_dspeed_rel : float, optional
        Amplitude of translation speed perturbation in relative terms
        (e.g., 0.2 for +/-20%). Default: 0.3.
    max_ddirection : float, optional
        Amplitude of track direction (bearing angle) perturbation
        per hour, in radians. Default: pi/360.
    autocorr_dspeed : float, optional
        Temporal autocorrelation in translation speed perturbation
        at a lag of 1 hour. Default: 0.85.
    autocorr_ddirection : float, optional
        Temporal autocorrelation of track direction perturbation
        at a lag of 1 hour. Default: 0.5.
    decay_ddirection_hourly : float, optional
        Exponential decay parameter applied to reduce the track direction
        perturbation with track time, in units of "per hour". Set to larger than
        0 to prevent long tracks to deviate too much from their historical
        counterpart. Default: 1/(2.5*24) (i.e. per 2.5 days).
    seed : int, optional
        Random number generator seed for replicability of random walk.
        Put negative value if you don't want to use it. Default: configuration file.
    adjust_intensity : str, optional
        Whether and how tropical cyclone intensity (central pressure,
        max_sustained_wind) should be modelled. One of 'explicit',
        'legacy_decay', or 'none'.
        For 'explicit', intensity as well as radius_oci (outermost closed isobar)
        and radius_max_wind are statistically modelled depending on landfalls in 
        historical and synthetic tracks (track intensification, peak intensity 
        duration as well as intensity decay over the ocean and over land are 
        explicitly modelled).
        For 'legacy_decay', a landfall decay is applied when a synthetic track
        reached land; however when a synthetic track is over the ocean while its
        historical counterpart was over land, intensity will be underestimated.
        For 'none', the track intensity will be the same as the historical track
        independently on whether both tracks are over land or the ocean.
        For None, will be set to 'explicit'.
        Default: None (i.e., 'explicit').
    central_pressure_pert : float, optional
        Magnitude of the intensity perturbation (in mbar). This value is used to
        perturb the maximum intensity (lowest central_pressure value) when
        adjust_intensity is True. Perturbations are generated from a normal
        distribution with mean=0 and sd=central_pressure_pert/2 constrained
        between -central_pressure_pert and +central_pressure_pert. Default: 7.5
        (corresponds to about 10 kn).
    decay : bool, optional
        Deprecated, for backward compatibility only. If True, equivalent to
        setting 'adjust_intensity' to 'legacy_decay'. If False, equivalent to
        setting 'adjust_intensity' to 'none'. Default: None (i.e. rely on the
        value of 'adjust_intensity').
    use_global_decay_params : bool, optional
        Whether to use precomputed global parameter values for landfall decay
        obtained from IBTrACS (1980-2019). If False, parameters are fitted
        using historical tracks in input parameter 'tracks', in which case the
        landfall decay applied depends on the tracks passed as an input and may
        not be robust if few historical tracks make landfall in this object.
        Default: True.
    pool : pathos.pool, optional
        Pool that will be used for parallel computation when applicable. If not given, the
        pool attribute of `tracks` will be used. Default: None
    """
    if decay is not None:
        LOGGER.warning("`decay` is deprecated. "
                        "Use `adjust_intensity` instead.")
        if adjust_intensity == 'explicit':
            raise ValueError(
                'Set `adjust_intensity` to "legacy_decay" or `decay` to False.'
            )
        if decay:
            LOGGER.warning('decay is set to True - this sets adjust_intensity to "legacy_decay"')
            adjust_intensity = 'legacy_decay'
        else:
            LOGGER.warning('decay is set to False - this sets adjust_intensity to "none" (as a string)')
            adjust_intensity = 'none'            
    if adjust_intensity is None:
        adjust_intensity = 'explicit'
    if adjust_intensity not in ['explicit', 'legacy_decay', 'none']:
        raise ValueError("adjust_intensity should be one of 'explicit', 'legacy_decay', 'none', or None")
    
    LOGGER.info('Computing %s synthetic tracks.', nb_synth_tracks * tracks.size)

    pool = tracks.pool if pool is None else pool

    if seed >= 0:
        np.random.seed(seed)

    # ensure tracks have constant time steps
    time_step_h = np.unique(np.concatenate([np.unique(x['time_step']) for x in tracks.data]))
    if not np.allclose(time_step_h, time_step_h[0]):
        raise ValueError('Tracks have different temporal resolution. '
                         'Please ensure constant time steps by applying equal_timestep beforehand')
    time_step_h = time_step_h[0]
    if 0 < NEGLECT_LANDFALL_DURATION_HOUR < time_step_h:
        LOGGER.warning('A higher temporal resolution is recommended to resolve shorter landfalls.')

    # ensure we're not making synths from synths
    if not sum(1 for t in tracks.data if t.orig_event_flag) == tracks.size:
        raise ValueError(
            "Not all tracks are original; refusing to compute perturbed "
            "trajectories on perturbed trajectories."
        )

    # get variables and attributes to keep at the end
    track_vars_attrs = (
        set(tracks.data[0].variables).union(['on_land', 'dist_since_lf']),
        set(tracks.data[0].attrs.keys())
    )

    LOGGER.debug('Generating random number for locations perturbations...')
    random_vec = _get_random_trajectories_perts(tracks,
                                                nb_synth_tracks,
                                                time_step_h,
                                                max_shift_ini,
                                                max_dspeed_rel,
                                                max_ddirection,
                                                autocorr_ddirection,
                                                autocorr_dspeed,
                                                decay_ddirection_hourly)

    if adjust_intensity == 'explicit':
        # to assign land parameters to historical tracks for use in synthetic tracks later
        land_geom_hist = climada.util.coordinates.get_land_geometry(
            extent=tracks.get_extent(deg_buffer=0.1), resolution=10
        )
    else:
        land_geom_hist = None

    LOGGER.debug('Applying locations perturbations...')
    if pool:
        chunksize = max(min(tracks.size // pool.ncpus, 1000), 1)
        new_ens = pool.map(_one_rnd_walk, tracks.data,
                           itertools.repeat(nb_synth_tracks, tracks.size),
                           itertools.repeat(land_geom_hist, tracks.size),
                           itertools.repeat(central_pressure_pert, tracks.size),
                           random_vec, chunksize=chunksize)
    else:
        new_ens = [_one_rnd_walk(track, nb_synth_tracks,
                                 land_geom_hist, central_pressure_pert, rand)
                   for track, rand in zip(tracks.data, random_vec)]

    if adjust_intensity == "explicit":
        track_fits_warnings = ', '.join([x[3] for x in new_ens if x[3] is not None])
        if len(track_fits_warnings) > 0:
            LOGGER.warning('Positive slope in wind-pressure relationship for the '
                           'following storms (max amplitude, max potentially '
                           'affected category, fitted relationship type): %s',
                           track_fits_warnings)
    else:
        # log tracks that have been cut-off at high latitudes
        cutoff_track_ids_tc = [x[1] for x in new_ens]
        cutoff_track_ids_tc = sum(cutoff_track_ids_tc, [])
        cutoff_track_ids_ts = [x[2] for x in new_ens]
        cutoff_track_ids_ts = sum(cutoff_track_ids_ts, [])
        if len(cutoff_track_ids_tc) > 0:
            LOGGER.info('The following generated synthetic tracks moved beyond '
                        'the range of [-70, 70] degrees latitude. Cut out '
                        'at TC category >1: %s.',
                        ', '.join(cutoff_track_ids_tc))
        if len(cutoff_track_ids_ts) > 0:
            LOGGER.debug('The following generated synthetic tracks moved beyond '
                        'the range of [-70, 70] degrees latitude. Cut out '
                        'at TC category <= 1: %s.',
                        ', '.join(cutoff_track_ids_ts))
    new_ens = [x[0] for x in new_ens]
    tracks.data = sum(new_ens, [])

    if adjust_intensity == 'explicit':
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=tracks.get_extent(deg_buffer=0.1), resolution=10
        )

        LOGGER.debug('Identifying tracks chunks...')
        if pool:
            chunksize = min(tracks.size // pool.ncpus, 2000)
            tracks_with_id_chunks = pool.map(
                _add_id_synth_chunks_shift_init,
                tracks.data,
                itertools.repeat(time_step_h, tracks.size),
                itertools.repeat(land_geom, tracks.size),
                itertools.repeat(True, tracks.size),
                chunksize=chunksize
            )
        else:
            # returns a list of tuples (track, no_sea_chunks, no_land_chunks, track_ext)
            tracks_with_id_chunks = [
                _add_id_synth_chunks_shift_init(track, time_step_h, land_geom, shift_values_init=True)
                for track in tracks.data
            ]
        # on_land and dist_since_lf calculated except for track extension

        # track extension when shifted
        if extend_track:
            # get random numbers: track_ext.time.size-2 for each track_ext
            LOGGER.debug('Extending tracks after shift')
            random_traj_extension = [
                _get_random_trajectory_ext(track_ext, time_step_h)
                for _,_,_,track_ext in tracks_with_id_chunks
            ]
            # create the tracks with extension - this is very fast
            LOGGER.debug('create the tracks with extension')
            sid_extended_tracks_shift = [track_ext.sid for _, _, _, track_ext in tracks_with_id_chunks if track_ext is not None]
            tracks_with_id_chunks = [
                (_create_track_from_ext(track, track_ext, rnd_tpl), -1, -1)
                if track_ext is not None
                else (track, no_chunks_sea, no_chunks_land)
                for (track, no_chunks_sea, no_chunks_land, track_ext),rnd_tpl in zip(tracks_with_id_chunks, random_traj_extension)
            ]
            # get new land_geom
            extent = climada.util.coordinates.latlon_bounds(
                np.concatenate([t[0].lat.values for t in tracks_with_id_chunks]),
                np.concatenate([t[0].lon.values for t in tracks_with_id_chunks]),
                buffer=0.1
            )
            extent = (extent[0], extent[2], extent[1], extent[3])
            land_geom = climada.util.coordinates.get_land_geometry(
                extent=extent, resolution=10
            )
            # on_land still True for id_chunk NA
            # extend id_chunk and get number of chunks
            LOGGER.debug('extend id_chunk and get number of chunks')
            no_chunks = [
                _track_ext_id_chunk(track, land_geom)
                if 'id_chunk' in track.variables and np.any(np.isnan(track['id_chunk'].values))
                else (no_chunks_sea, no_chunks_land)
                for track, no_chunks_sea, no_chunks_land in tracks_with_id_chunks
            ]
            tracks_list = [track for track, _, _ in tracks_with_id_chunks]
        else:
            sid_extended_tracks_shift = []
            no_chunks = [
                (no_chunks_sea, no_chunks_land)
                for _, no_chunks_sea, no_chunks_land, _ in tracks_with_id_chunks
            ]
            tracks_list = [track for track, _, _, _ in tracks_with_id_chunks]

        # FOR EACH CHUNK OVER THE OCEAN, WE NEED 4 RANDOM VALUES: intensification
        # target perturbation, intensification shape, peak duration, decay
        LOGGER.debug('Generating random number for intensity perturbations...')
        random_vec_intensity = [np.random.uniform(size=nb_cnk[0] * 4)
                                for nb_cnk in no_chunks]

        # calculate landfall decay parameters
        if use_global_decay_params:
            v_rel = LANDFALL_DECAY_V
            p_rel = LANDFALL_DECAY_P
        else:
            hist_tracks = [track for track in tracks.data if track.orig_event_flag]
            if hist_tracks:
                try:
                    v_rel, p_rel = _calc_land_decay(hist_tracks, land_geom, pool=pool)
                except ValueError as verr:
                    raise ValueError('Landfall decay parameters could not be calculated.') from verr
            else:
                raise ValueError('No historical tracks found. Historical'
                                 ' tracks are needed for land decay calibration'
                                 ' if use_global_decay_params=False.')

        LOGGER.debug('Modelling TC intensities...')
        (
            ocean_modelled_tracks,
            sid_extended_tracks_intensity,
            sid_outside_lat_range, 
            sid_outside_intensity_range
        ) = _model_synth_tc_intensity(
            tracks_list=tracks_list,
            random_vec_intensity=random_vec_intensity,
            time_step_h=time_step_h,
            track_vars_attrs=track_vars_attrs,
            extend_track=extend_track,
            pool=pool,
            central_pressure_pert = central_pressure_pert,
            v_rel=v_rel,
            p_rel=p_rel,
            s_rel=True
        )
        if extend_track:
            nb_extended_tracks = np.unique(np.concatenate([sid_extended_tracks_shift, sid_extended_tracks_intensity])).size
        elif len(sid_extended_tracks_intensity) > 0:
            raise ValueError('Track extension, why did it happen?')
        else:
            nb_extended_tracks = 0
        if len(sid_outside_lat_range):
            LOGGER.debug('Outside latitude range for following events - applied land decay: %s',
                         ', '.join(sid_outside_lat_range))
        if int(os.getenv('TRACKGEN_TEST_KILL_BEYOND_ADVISORIES')):
            if len(sid_outside_intensity_range):
                LOGGER.debug('Tracks exceeding final advisory pressure - killed: %s',
                            ', '.join(sid_outside_intensity_range))
        elif int(os.getenv('TRACKGEN_TEST_DECAY_BEYOND_ADVISORIES')):
            if len(sid_outside_intensity_range):
                LOGGER.debug('Tracks exceeding final advisory pressure - applied land decay: %s',
                            ', '.join(sid_outside_intensity_range))
        LOGGER.debug(
            f"Extended {nb_extended_tracks} synthetic tracks that ended at Tropical Storm or above category. "
            f"Adapted intensity on {len(ocean_modelled_tracks)} tracks for a total of "
            f"{sum(no_sea_chunks for no_sea_chunks, _ in no_chunks)} ocean and "
            f"{sum(no_land_chunks for _, no_land_chunks in no_chunks)} land chunks."
        )
        tracks.data = ocean_modelled_tracks

    elif adjust_intensity == 'legacy_decay':
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=tracks.get_extent(deg_buffer=0.1), resolution=10
        )

        if use_global_decay_params:
            tracks.data = _apply_land_decay(tracks.data, LANDFALL_DECAY_V,
                                            LANDFALL_DECAY_P, land_geom, pool=pool)
        else:
            # fit land decay coefficients based on historical tracks
            hist_tracks = [track for track in tracks.data if track.orig_event_flag]
            if hist_tracks:
                try:
                    v_rel, p_rel = _calc_land_decay(hist_tracks, land_geom, pool=pool)
                    tracks.data = _apply_land_decay(
                        tracks.data, v_rel, p_rel, land_geom, pool=pool)
                except ValueError as verr:
                    raise ValueError('Landfall decay could not be applied.') from verr
            else:
                raise ValueError('No historical tracks found. Historical'
                                 ' tracks are needed for land decay calibration'
                                 ' if use_global_decay_params=False.')


def _one_rnd_walk(track,
                  nb_synth_tracks,
                  land_geom,
                  central_pressure_pert,
                  rnd_tpl):
    """
    Apply random walk to one track.

    Parameters
    ----------
    track : xr.Dataset
        Track data.
    nb_synth_tracks : int, optional
        Number of ensemble members per track. Default: 9.
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        Land geometry. Required to model intensity (central pressure,
        max_sustained_wind, as well as radius_oci and radius_max_wind) depending
        on landfalls in historical and synthetic tracks. If provided, variable
        'on_land' is calculated for the historical track and renamed to
        'on_land_hist' in synthetic tracks, and variable
        'target_central_pressure' is created as the lowest central pressure from
        each time step to the end of the track.
    central_pressure_pert : float, optional
        Maximum perturbations in central pressure. Used to determine the range
        of pressure values over which to fit data. This argment must be provided
        if 'land_geom' is provided.
    rnd_tpl : np.ndarray of shape (2 * nb_synth_tracks * track.time.size),)
        Vector of random perturbations.

    Returns
    -------
    ens_track : list(xr.Dataset)
        List of the track and the generated synthetic tracks.
    cutoff_track_ids_tc : List of str
        List containing information about the tracks that were cut off at high
        latitudes with wind speed of TC category 2-5.
    cutoff_track_ids_ts : List of str
        List containing information about the tracks that were cut off at high
        latitudes with a wind speed up to TC category 1.
    """

    # on_land parameters of historical track will be required for each synthetic track
    if land_geom is not None:
        climada.hazard.tc_tracks.track_land_params(track, land_geom, land_params='on_land')

    ens_track = list()
    ens_track.append(track.copy(True))

    # calculate historical track values that are used for synthetic track modelling
    if land_geom is not None:
        # compute minimum pressure occurring on or after each timestep
        if int(os.getenv('TRACKGEN_TEST_TARGET_PRESSURE_OVER_SEA')):
            over_sea_pressure = np.array([
                pres if not land
                else track.central_pressure.values[-1]
                for pres, land in zip(track.central_pressure.values, track.on_land.values)
            ])
            target_pressure = np.flip(np.minimum.accumulate(
                np.flip(over_sea_pressure)
            ))
        else:
            target_pressure = np.flip(np.minimum.accumulate(
                np.flip(track.central_pressure.values)
            ))
        # compute linear regression onto intensification and decay
        track_fits_warning = _add_fits_to_track(track, central_pressure_pert)
        # track are not to be cut-off. Latitudinal tresholds are applied later
        # in the model.
        cutoff_track_ids_tstc = None
    else:
        # to keep track of cut-off tracks
        cutoff_track_ids_tstc = {'tc': {}, 'ts': {}}
        track_fits_warning = None

    # generate tracks with perturbed trajectories
    for i_ens in range(nb_synth_tracks):
        i_track = track.copy(True)

        # adjust attributes
        i_track.attrs['orig_event_flag'] = False
        i_track.attrs['name'] = f"{i_track.attrs['name']}_gen{i_ens + 1}"
        i_track.attrs['sid'] = f"{i_track.attrs['sid']}_gen{i_ens + 1}"
        i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens + 1) / 100

        # apply locations perturbations
        last_idx = _apply_random_walk_pert(
            i_track,
            rnd_tpl=rnd_tpl[i_ens],
            cutoff_track_ids_tstc=cutoff_track_ids_tstc
        )

        if land_geom is not None:
            i_track = i_track.assign(
                {
                    "target_central_pressure": ("time", target_pressure),
                    # "on_land_hist" : ("time", i_track["on_land"]),
                }
            )
            i_track = i_track.rename({"on_land": "on_land_hist"})

        # remove 'on_land' and 'dist_since_lf' if present since not correct for synthetic track
        vars_to_drop = [v for v in ['on_land', 'dist_since_lf'] if v in list(i_track.variables)]
        i_track = i_track.drop_vars(vars_to_drop)
        if last_idx < i_track.time.size:
            i_track = i_track.isel(time=slice(None, last_idx))

        ens_track.append(i_track)

    if land_geom is None:
        cutoff_track_ids_tc = [f"{k} ({v})" for k,v in cutoff_track_ids_tstc['tc'].items()]
        cutoff_track_ids_ts = [f"{k} ({v})" for k,v in cutoff_track_ids_tstc['ts'].items()]
    else:
        cutoff_track_ids_tc = []
        cutoff_track_ids_ts = []

    return ens_track, cutoff_track_ids_tc, cutoff_track_ids_ts, track_fits_warning

def _get_random_trajectories_perts(tracks,
                                    nb_synth_tracks,
                                    time_step_h,
                                    max_shift_ini,
                                    max_dspeed_rel,
                                    max_ddirection,
                                    autocorr_ddirection,
                                    autocorr_dspeed,
                                    decay_ddirection_hourly):
    """Generate random numbers for random walk
    
    Parameters
    ----------
    tracks : List
        List containing tracks data (i.e. the 'data' attribute of a
        climada.hazard.tc_tracks.TCTracks object).
    nb_synth_tracks : int
        Number of ensemble members per track.
    time_step_h : float
        Temporal resolution of the time series, in hour.
    max_shift_ini : float
        Amplitude of max random starting point shift in decimal degree
        (up to +/-max_shift_ini for longitude and latitude).
    max_dspeed_rel : float
        Amplitude of translation speed perturbation in relative terms
        (e.g., 0.2 for +/-20%).
    max_ddirection : float
        Amplitude of track direction (bearing angle) perturbation
        per hour, in radians.
    autocorr_dspeed : float
        Temporal autocorrelation in translation speed perturbation
        at a lag of 1 hour.
    autocorr_ddirection : float
        Temporal autocorrelation of translational direction perturbation
        at a lag of 1 hour
    decay_ddirection_hourly : float
        Exponential decay parameter applied to reduce the track direction
        perturbation with track time, in units of "per hour". Set to larger than
        0 to prevent long tracks to deviate too much from their historical
        counterpart. Default: 1/(2.5*24) (i.e. per 2.5 days).

    Returns
    -------
    random_vec : Tuple
        Tuple of tuples of tuples: random_vec[i][j] contains random track
        locations perturbations for the j-th synthetic track generated based on
        historical track i. random_vec[i][j] is a tuple of length 3 which
        contains random numbers for the perturbation of (1) the track starting
        point (2 values), (2) the track direction at each track
        node (1 values per track segment), and (3) the track
        translation speed at each track node (1 value per track segment). For tracks containing a single
        time step or if 'autocorr_ddirection' or 'autocorr_dspeed' are equal to
        0 the Tuples will only contain the first element.
    """
    # number of random value per synthetic track:
    # 2 for starting points (lon, lat)
    # (track.time.size-1) for angle and same for translation perturbation
    # hence sum is nb_synth_tracks * (2 + 2*(size-1)) = nb_synth_tracks * 2 * size
    if autocorr_ddirection == 0 and autocorr_dspeed == 0:
        random_vec = tuple(
            # one element for each historical track
            tuple(
                # one element for each synthetic track to create for that historical track
                tuple(
                    # ini pert: random, uncorrelated, within [-max_shift_ini, +max_shift_ini]
                    [max_shift_ini * (2 * np.random.uniform(size=2) - 1)]
                    # further random parameters are not required
                )
                for i in range(nb_synth_tracks)
            )
            if track.time.size > 1
            else tuple(tuple(np.random.uniform(size=nb_synth_tracks * 2)))
            for track in tracks.data
        )
    else:
        random_vec = tuple(
            # one element for each historical track
            tuple(
                # one element for each synthetic track to create for that
                # historical track
                tuple(
                    [
                        # ini pert: random, uncorrelated, within [-max_shift_ini, +max_shift_ini]
                        max_shift_ini * (2 * np.random.uniform(size=2) -1),
                        # ddirection pert: autocorrelated hourly pert, used as angle
                        # and scale
                        np.cumsum(time_step_h * np.degrees(
                            max_ddirection * (
                                2 * _random_uniform_ac(track.time.size - 1,
                                                    autocorr_ddirection,
                                                    time_step_h) - 1
                            )
                        )) * np.exp(-decay_ddirection_hourly*np.arange(0, time_step_h*(track.time.size-1), time_step_h)),
                        1 + max_dspeed_rel * (
                            2 *_random_uniform_ac(track.time.size - 1,
                                                autocorr_dspeed,
                                                time_step_h) - 1
                        )
                    ]
                )
                for i in range(nb_synth_tracks)
            )
            if track.time.size > 1
            else tuple(tuple(tuple([max_shift_ini * (2 * np.random.uniform(size=2) -1), None, None])) for i in range(nb_synth_tracks))
            for track in tracks.data
        )
    return random_vec

def _apply_random_walk_pert(track: xr.Dataset,
                            rnd_tpl: tuple,
                            cutoff_track_ids_tstc: dict):
    """
    Perturb track locations using provided random numbers and parameters.

    The inputs track and cutoff_track_ids_tstc are modified in place.

    TODO complete docstring
    """
    # synth_track = track.copy(True)
    new_lon = np.zeros_like(track.lon.values)
    new_lat = np.zeros_like(track.lat.values)
    n_seg = track.time.size - 1
    last_idx = track.time.size

    # get perturbations
    xy_ini, ang_pert_cum, trans_pert = rnd_tpl

    # get bearings and angular distance for the original track
    bearings = _get_bearing_angle(track.lon.values, track.lat.values)
    angular_dist = climada.util.coordinates.dist_approx(track.lat.values[:-1, None],
                                                        track.lon.values[:-1, None],
                                                        track.lat.values[1:, None],
                                                        track.lon.values[1:, None],
                                                        method="geosphere",
                                                        units="degree")[:, 0, 0]

    # apply initial location shift
    new_lon[0] = track.lon.values[0] + xy_ini[0]
    new_lat[0] = track.lat.values[0] + xy_ini[1]
    
    # apply perturbations along the track segments
    for i in range(n_seg):
        new_lon[i + 1], new_lat[i + 1] = \
            _get_destination_points(new_lon[i], new_lat[i],
                                    bearings[i] + ang_pert_cum[i],
                                    trans_pert[i] * angular_dist[i])
        if cutoff_track_ids_tstc is not None:
            # if track crosses latitudinal thresholds (+-70°),
            # keep up to this segment (i+1), set i+2 as last point,
            # and discard all further points > i+2.
            if i+2 < last_idx and (new_lat[i + 1] > 70 or new_lat[i + 1] < -70):
                last_idx = i + 2
                # end the track here
                max_wind_end = track.max_sustained_wind.values[last_idx:]
                ss_scale_end = climada.hazard.tc_tracks.set_category(max_wind_end,
                        track.max_sustained_wind_unit)
                # TC category at ending point should not be higher than 1
                if ss_scale_end > 1:
                    cutoff_track_ids_tstc['tc'][track.attrs['name']] = climada.hazard.tc_tracks.CAT_NAMES[ss_scale_end]
                else:
                    cutoff_track_ids_tstc['ts'][track.attrs['name']] = climada.hazard.tc_tracks.CAT_NAMES[ss_scale_end]
                break
    # make sure longitude values are within (-180, 180)
    climada.util.coordinates.lon_normalize(new_lon, center=0.0)

    track.lon.values = new_lon
    track.lat.values = new_lat
    return last_idx

def _random_uniform_ac(n_ts, autocorr, time_step_h):
    """
    Generate a series of autocorrelated uniformly distributed random numbers.

    This implements the algorithm described here to derive a uniformly distributed
    series with specified autocorrelation (here at a lag of 1 hour):
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number
    Autocorrelation is specified at a lag of 1 hour. To get a time series at a
    different temporal resolution (time_step_h), an hourly time series is generated
    and resampled (using linear interpolation) to the target resolution.

    Parameters
    ----------
    n_ts : int
        Length of the series.
    autocorr : float
        Autocorrelation (between -1 and 1) at hourly time scale.
    time_step_h : float
        Temporal resolution of the time series, in hour.

    Returns
    -------
    x_ts : numpy.ndarray of shape (n_ts,)
        n values at time_step_h intervals that are uniformly distributed and with
            the requested temporal autocorrelation at a scale of 1 hour.
    """
    # generate autocorrelated 1-hourly perturbations, so first create hourly
    #   time series of perturbations
    n_ts_hourly_exact = n_ts * time_step_h
    n_ts_hourly = int(np.ceil(n_ts_hourly_exact))
    x = np.random.normal(size=n_ts_hourly)
    theta = np.arccos(autocorr)
    for i in range(1, len(x)):
        x[i] = _h_ac(x[i - 1], x[i], theta)
    # scale x to have magnitude [0,1]
    x = (x + np.sqrt(3)) / (2 * np.sqrt(3))
    # resample at target time step
    x_ts = np.interp(np.arange(start=0, stop=n_ts_hourly_exact, step=time_step_h),
                     np.arange(n_ts_hourly), x)
    return x_ts


@numba.njit
def _h_ac(x, y, theta):
    """
    Generate next random number from current number for autocorrelated uniform series

    Implements function h defined in:
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number

    Parameters
    ----------
    x : float
        Previous random number.
    y : float
        Random Standard Normal.
    theta : float
        arccos of autocorrelation.

    Returns
    -------
    x_next : float
        Next value in the series.
    """
    gamma = np.abs(np.mod(theta, np.pi) - \
                   np.floor((np.mod(theta, np.pi) / (np.pi / 2)) + 0.5) * np.pi / 2)
    x_next = 2 * np.sqrt(3) * (_f_ac(np.cos(theta) * x + np.sin(theta) * y, gamma) - 1 / 2)
    return x_next


@numba.njit
def _f_ac(z, theta):
    """
    F transform for autocorrelated random uniform series generation

    Implements function F defined in:
    https://stats.stackexchange.com/questions/48086/
        algorithm-to-produce-autocorrelated-uniformly-distributed-number
    i.e., the CDF of Y.

    Parameters
    ----------
    z : float
        Value.
    theta : float
        arccos of autocorrelation.

    Returns
    -------
        res : float
            CDF at value z
    """
    c = np.cos(theta)
    s = np.sin(theta)
    if z >= np.sqrt(3) * (c + s):
        res = 1
    elif z > np.sqrt(3) * (c - s):
        res = 1 / 12 / np.sin(2 * theta) * \
              (-3 - z ** 2 + 2 * np.sqrt(3) * z * (c + s) + 9 * np.sin(2 * theta))
    elif z > np.sqrt(3) * (-c + s):
        res = 1 / 6 * (3 + np.sqrt(3) * z / c)
    elif z > -np.sqrt(3) * (c + s):
        res = 1 / 12 / np.sin(2 * theta) * \
              (z ** 2 + 2 * np.sqrt(3) * z * (c + s) + 3 * (1 + np.sin(2 * theta)))
    else:
        res = 0
    return res


@numba.njit
def _get_bearing_angle(lon, lat):
    """
    Compute bearing angle of great circle paths defined by consecutive points

    Returns initial bearing (also called forward azimuth) of the n-1 great circle
    paths define by n consecutive longitude/latitude points. The bearing is the angle
    (clockwise from North) which if followed in a straight line along a great-circle
    arc will take you from the start point to the end point. See also:
    http://www.movable-type.co.uk/scripts/latlong.html
    Here, the bearing of each pair of consecutive points is computed.

    Parameters
    ----------
    lon : numpy.ndarray of shape (n,)
        Longitude coordinates of consecutive point, in decimal degrees.
    lat : numpy.ndarray of shape (n,)
        Latitude coordinates of consecutive point, in decimal degrees.

    Returns
    -------
        earth_ang_fix : numpy.ndarray of shape (n-1,)
            Bearing angle for each segment, in decimal degrees
    """
    lon, lat = map(np.radians, [lon, lat])
    # Segments between all point (0 -> 1, 1 -> 2, ..., n-1 -> n)
    # starting points
    lat_1 = lat[:-1]
    lon_1 = lon[:-1]
    # ending points
    lat_2 = lat[1:]
    lon_2 = lon[1:]
    delta_lon = lon_2 - lon_1
    # what to do with the points that don't move?
    #   i.e. where lat_2=lat_1 and lon_2=lon_1? The angle does not matter in
    # that case because angular distance will be 0.
    earth_ang_fix = np.arctan2(np.sin(delta_lon) * np.cos(lat_2),
                               np.cos(lat_1) * np.sin(lat_2) - \
                               np.sin(lat_1) * np.cos(lat_2) * np.cos(delta_lon))
    return np.degrees(earth_ang_fix)


@numba.njit
def _get_destination_points(lon, lat, bearing, angular_distance):
    """
    Get coordinates of endpoints from a given locations with the provided bearing and distance

    Parameters
    ----------
    lon : numpy.ndarray of shape (n,)
        Longitude coordinates of each starting point, in decimal degrees.
    lat : numpy.ndarray of shape (n,)
        Latitude coordinates of each starting point, in decimal degrees.
    bearing : numpy.ndarray of shape (n,)
        Bearing to follow for each starting point (direction Northward, clockwise).
    angular_distance : numpy.ndarray of shape (n,)
        Angular distance to travel for each starting point, in decimal degrees.

    Returns
    -------
        lon_2 : numpy.ndarray of shape (n,)
            Longitude coordinates of each ending point, in decimal degrees.
        lat_2 : numpy.ndarray of shape (n,)
            Latitude coordinates of each ending point, in decimal degrees.
    """
    lon, lat = map(np.radians, [lon, lat])
    bearing = np.radians(bearing)
    angular_distance = np.radians(angular_distance)
    lat_2 = np.arcsin(np.sin(lat) * np.cos(angular_distance) + np.cos(lat) * \
                      np.sin(angular_distance) * np.cos(bearing))
    lon_2 = lon + np.arctan2(np.sin(bearing) * np.sin(angular_distance) * np.cos(lat),
                             np.cos(angular_distance) - np.sin(lat) * np.sin(lat_2))
    return np.degrees(lon_2), np.degrees(lat_2)

def _create_track_from_ext(track, track_ext, rnd_tpl):
    """Create the actual track with extension.

    For shifted track values initially: estimate_track_params is False, pass
    values to keep as values_df (all values)

    For tracks where an extension is needed after intensity modelling:
    estimate_track_params is True, pass only pcen (and vmax where relevant) in
    values_df - other variables are modelled.

    # TODO test all of this
    """
    # model track locations
    _project_trajectory_track_ext(track_ext, rnd_tpl)
    miss_vars_ext = set(list(track.variables)) - set(list(track_ext.variables))
    if len(miss_vars_ext) > 0:
        LOGGER.debug('Variables in track but not in track_ext: %s (%s)',
                     miss_vars_ext, track.sid)
    miss_vars = set(list(track_ext.variables)) - set(list(track.variables))
    if len(miss_vars) > 0:
        LOGGER.debug('Variables in track_ext but not in track: %s (%s)',
                     miss_vars, track.sid)
    if track.time.values[-1] != track_ext.time.values[1]:
        raise ValueError('Time of track and track_ext do not match')
    track_new = xr.concat([track, track_ext.isel(time=slice(2, None))], "time")
    return track_new

def _get_shift_idx_start(on_land_hist, on_land_synth, time_step_h, shift_values_init=True, sid=None):
    if np.all(on_land_synth) or np.all(on_land_hist):
        raise ValueError('Cannot deal with tracks fully over land')

    nts = on_land_hist.size
    # get the default transitions
    transitions_hist = _get_transitions(on_land_hist, time_step_h, exclude_short_landfalls=True)
    transitions_synth = _get_transitions(on_land_synth, time_step_h, exclude_short_landfalls=True)
    transitions_hist_all = _get_transitions(on_land_hist, time_step_h, exclude_short_landfalls=False)
    transitions_synth_all = _get_transitions(on_land_synth, time_step_h, exclude_short_landfalls=False)

    # identify landfalls
    idx_lf_hist = _get_landfall_idx(transitions_hist)
    # idx_lf_hist_all = _get_landfall_idx(transitions_hist_all)
    idx_lf_synth = _get_landfall_idx(transitions_synth)
    # idx_lf_synth_all = _get_landfall_idx(transitions_synth_all)
    # idx_lf_hist and idx_lf_hist_all are index based on hist time - TODO shift later

    # if a track starts over the ocean, allow shift if reaches land in the next 6 hours
    nts_in_next_lf = int(np.floor(6/time_step_h))
    # if a track starts over ocean and does not move over land within nts_in_next_lf,
    #   allow shifting the other track
    nts_in_next_ocean = int(np.floor(18/time_step_h))

    # case when any track starts over land, what is the shift between the first point
    # over the ocean in modelled vs historical track?
    if shift_values_init and (on_land_synth[0] or on_land_hist[0]):
        # Here we already know that both tracks move to the ocean later on and
        #   at least on of them starts over land
        first_sea_hist = np.where(~on_land_hist)[0][0]
        first_sea_synth = np.where(~on_land_synth)[0][0]
        if on_land_synth[0] and on_land_hist[0]:
            
            # 0) Both tracks start over land: shift values such as to match first point over the ocean.
            
            shift_first_sea = first_sea_synth - first_sea_hist
            # no modelling needed before first synth landfall or first hist landfall.
            idx_start_model_hist = idx_lf_hist[0]
            idx_start_model_synth = idx_lf_synth[0]
            # This applies if both tracks start over land or one track does not
            # reach land anytime soon.
        elif ~on_land_synth[0]:

            # 1) synth starts over the ocean: shift_first_sea <= 0, i.e. shift
            #    values back in time

            if np.any(on_land_synth[:nts_in_next_lf + 1]):
                # a) synth reaches land in the next 6 hours: shift to end of
                # that landfall, i.e. first land-to-sea transition
                # i.e. match
                if np.any(transitions_synth_all == 1):
                    # i) synth moves back to sea after end of first landfall i.e. transition==1
                    first_sea_synth = np.where(transitions_synth_all == 1)[0][0]
                    shift_first_sea = first_sea_synth - first_sea_hist
                    # From hist perspective: Model latest from next hist
                    # landfall (i.e. first sea-to-land transition; by definition
                    # hist starts over land)
                    idx_start_model_hist = idx_lf_hist[0]
                    # From synth perspective, model latest from next landfall
                    # (i.e. second landfall)
                    idx_start_model_synth = idx_lf_synth[1] if idx_lf_synth.size > 1 else nts
                else:
                    # ii) synth track never moves back to the sea after first
                    # landfall: model from the start of the synth landfall
                    shift_first_sea = first_sea_synth - first_sea_hist
                    # no modelling needed before first synth landfall or first hist landfall.
                    idx_start_model_hist = idx_lf_hist[0]
                    idx_start_model_synth = idx_lf_synth[0]

            elif np.all(on_land_hist[:nts_in_next_ocean + 1]):

                # b) synth track not over land soon AND hist track NOT soon over the
                #   ocean: model intensity explicitly from the start
                shift_first_sea = 0
                idx_start_model_hist = 0
                idx_start_model_synth = 0
            else:

                # c) synth track not over land soon AND hist track soon over the ocean: shift as by default: first
                # hist track over the ocean as first track point in synth
                shift_first_sea = first_sea_synth - first_sea_hist
                # no modelling needed before first synth landfall or first hist landfall
                idx_start_model_hist = idx_lf_hist[0]
                idx_start_model_synth = idx_lf_synth[0]

        else:

            # 2) hist starts over the ocean: shift_first_sea > 0, i.e. shift
            #    values forward in time

            if np.any(on_land_hist[:nts_in_next_lf + 1]):
                # a) hist reaches land in the next 6 hours: shift to end of
                # that landfall, if any, i.e. first land-to-sea transition for both

                if np.any(transitions_hist_all == 1):
                    # hist moves back to sea after end of first landfall
                    first_sea_hist = np.where(transitions_hist_all == 1)[0][0]
                    shift_first_sea = first_sea_synth - first_sea_hist
                    # From hist perpective: model latest from next landfall
                    # (i.e. second landfall)
                    idx_start_model_hist = idx_lf_hist[1] if idx_lf_hist.size >= 2 else nts
                    # From synth perspective: model latest from next landfall
                    # (first landfall)
                    idx_start_model_synth = idx_lf_synth[0]
                else:
                    # hist track never moves back to sea after first landfall
                    # hence need to model from the start
                    shift_first_sea = 0
                    idx_start_model_synth = 0
                    idx_start_model_hist = 0

            elif np.all(on_land_synth[:nts_in_next_ocean + 1]):

                # b) Hist track not over land soon AND synth track NOT soon over the ocean
                # -> model intensity explicitly from the start
                shift_first_sea = 0
                idx_start_model_synth = 0
                idx_start_model_hist = 0
            else:
                # c) hist track not over land soon AND synth track soon over the ocean
                # -> shift as normal
                shift_first_sea = first_sea_synth - first_sea_hist
                # from hist perspective: model latest from first landfall
                idx_start_model_hist = idx_lf_hist[0]
                # from synth perspective: model latest from first landfall
                idx_start_model_synth = idx_lf_synth[0]

    else:
        # case without shift
        shift_first_sea = 0
        # model from first landfall in either track
        idx_start_model_hist = idx_lf_hist[0]
        idx_start_model_synth = idx_lf_synth[0]
        # transitions_hist = _get_transitions(on_land_hist, time_step_h)
        # _remove_landstart(transitions_hist)
        # ftr_hist = _first_transition(transitions_hist)
        # # when to start modelling intensity according to first historical landfall
        # if ftr_hist == transitions_hist.size:
        #     idx_start_model = on_land_synth.size
        # else:
        #     idx_start_model = ftr_hist -
        #     np.floor(TIME_MODEL_BEFORE_HIST_LF_H/time_step_h)
    
    # account for shift and buffer for historical landfall
    if idx_start_model_hist < nts:
        idx_start_model_hist = max(0, min(
            nts,
            idx_start_model_hist + shift_first_sea - np.floor(TIME_MODEL_BEFORE_HIST_LF_H/time_step_h)
        ))
    # take the first idx
    idx_start_model = int(max(0, min(idx_start_model_hist, idx_start_model_synth)))
    return shift_first_sea, idx_start_model

def _track_ext_id_chunk(track, land_geom):
    if track.orig_event_flag:
        return 0, 0
    check_id_chunk(track.id_chunk.values, track.sid, allow_missing=True)
    # up to this point, 'on_land' was set to True for all extension points
    # and dist_since_lf was not correct
    climada.hazard.tc_tracks.track_land_params(track, land_geom=land_geom)
    if np.any(np.isnan(track['id_chunk'].values)):
        # case of extending due to a shift in values
        na_start_idx = np.where(np.isnan(track['id_chunk']))[0][0]
        # na_start_idx cannot be 0, otherwise it should fail in check_id_chunk above
        id_chunk_full = track['id_chunk'].values[:na_start_idx]
        # re-compute id_chunk for the full track
        transitions = _get_transitions(track.on_land.values, track.time_step.values[0])
        if np.any(id_chunk_full != 0):
            idx_start_model = np.where(id_chunk_full != 0)[0][0]
        else:
            idx_start_model = id_chunk_full.size
        _set_first_transition(transitions, idx_start_model, track.on_land.values)
        id_chunk, no_chunks_sea, no_chunks_land = _get_id_chunk(transitions)
        # check that newly computed id_chunk values match the original ones
        if np.any(id_chunk[:na_start_idx] != track['id_chunk'][:na_start_idx].values.astype(int)):
            # due to a short landfall at the end of track before missing values.
            # check if calculating id_chunk from subset
            transitions_orig = _get_transitions(track.on_land.values[:na_start_idx], track.time_step.values[0])
            if np.any(id_chunk_full != 0):
                idx_start_model = np.where(id_chunk_full != 0)[0][0]
            else:
                idx_start_model = id_chunk_full.size
            _set_first_transition(transitions_orig, idx_start_model, track.on_land.values)
            id_chunk_orig, _, _ = _get_id_chunk(transitions_orig)
            min_n_ts = np.ceil(NEGLECT_LANDFALL_DURATION_HOUR/track.time_step.values[0]).astype(int)
            if np.any(id_chunk_orig != track['id_chunk'][:na_start_idx].values):
                raise ValueError('id_chunk values mismatch: %s' % track.sid)
            elif np.any(id_chunk[:max(0, na_start_idx-min_n_ts-1)] != track['id_chunk'][:max(0, na_start_idx-min_n_ts-1)].values.astype(int)):
                raise ValueError('id_chunk values mismatch (2): %s' % track.sid)
        # now id_chunk should be valid and not contain any missing value
        check_id_chunk(id_chunk, track.sid, allow_missing=False)
        if id_chunk[na_start_idx] != id_chunk[na_start_idx - 1] and id_chunk[na_start_idx-1] != 0:
            if track.on_land.values[na_start_idx] == track.on_land.values[na_start_idx - 1]:
                # TODO remove this check as should not be possible now
                raise ValueError('Issue1: %s', track.sid)
        track['id_chunk'] = ('time', id_chunk)
    no_chunks_sea = int(max(0, track['id_chunk'].values.max()))
    no_chunks_land = int(np.abs(min(0, track['id_chunk'].values.min())))
    return no_chunks_sea, no_chunks_land

def _create_raw_track_extension(track,
                                nb_time_steps,
                                time_step_h,
                                values_df: pd.DataFrame=None):
    """Append new time steps to a track.
    
    For a TC track, create a new track starting at the end of the original
    track, with a given number of time steps and optionally some variables
    values set. Longitude/Latitude are all constant and set to the last values
    in track.
    For variables for which not values are provided, the last values in track
    will be repeated until the end.

    Note that the returned track contains the two last time steps of the input
    track as its first time steps, followed by an additional 'nb_time_steps' time steps.
    allows applying trajectory modelling.
    """
    time_append = np.array([
        track.time.values[-1] + np.timedelta64(np.round(3600*dt).astype(int), 's')
        for dt in np.arange(time_step_h, nb_time_steps*time_step_h+1, time_step_h)
    ])
    lon_append = track.lon.values[-1]*np.ones_like(time_append, dtype="float")
    lat_append = track.lat.values[-1]*np.ones_like(time_append, dtype="float")
    def _build_constant_var(x):
        return ('time', np.repeat([x], nb_time_steps + 2))
    # get the values where applicable
    vars_set_missing = ['on_land', 'dist_since_lf', 'id_chunk']
    vars_not_copy = ['time', 'lat', 'lon'] + vars_set_missing
    if values_df is not None and values_df.shape[0] > 0:
        vars_values = {
            k:('time', np.concatenate([track[k].values[-2:], v.values])) for k,v in values_df.items()
        }
        vars_set_constant = [v for v in track.variables
                        if v not in list(values_df.columns) + vars_not_copy]
    else:
        vars_values = {}
        vars_set_constant = [v for v in track.variables if v not in vars_not_copy]
    for v in vars_set_constant:
        vars_values[v] = _build_constant_var(track[v].values[-1])
    for v in vars_set_missing:
        if v in track.variables:
            if v == 'on_land':
                # to keep bool type
                append_vals = np.repeat(True, nb_time_steps)
            else:
                append_vals = np.repeat([np.nan], nb_time_steps)
            vars_values[v] = ('time', np.concatenate([
                track[v].values[-2:],
                append_vals
            ]))
    track_ext = xr.Dataset(
        vars_values,
        coords={
            'time': np.concatenate([track.time.values[-2:], time_append]),
            'lat': ('time', np.concatenate([track.lat.values[-2:], lat_append])),
            'lon': ('time', np.concatenate([track.lon.values[-2:], lon_append])),
        },
        attrs=track.attrs
    )
    return track_ext

def _project_trajectory_track_ext(track_ext, rnd_tpl):
    """"Determine track trajectory for track extensions"""
    nb_time_steps = track_ext.time.size - 2
    # get first bearing and translation speed in the extension
    last_bearing = _get_bearing_angle(track_ext.lon.values[:2], track_ext.lat.values[:2])[0]
    last_ang_dist = climada.util.coordinates.dist_approx(track_ext.lat.values[:1, None],
                                                        track_ext.lon.values[:1, None],
                                                        track_ext.lat.values[1:2, None],
                                                        track_ext.lon.values[1:2, None],
                                                        method="geosphere",
                                                        units="degree")[0, 0, 0]
    # generate random perturbations
    # according to get_changes_for_rnd_walk:
    # - bearing angle change has mean 0 deg/h and std 3.35 deg/h, autocorr=0.21
    # - relative change in translation speed has a mean of 0% /hour and std 4.2%/hr, autocorr=0
    # Therefore: Assume a straight line, then apply these perturbations

    # get perturbations
    ang_pert_cum, trans_pert = rnd_tpl

    # Create a straight line from last point
    lon_append = track_ext.lon.values[1]*np.ones([nb_time_steps+1], dtype="float")
    lat_append = track_ext.lat.values[1]*np.ones([nb_time_steps+1], dtype="float")
    for i in range(nb_time_steps):
        lon_append[i + 1], lat_append[i + 1] = \
            _get_destination_points(lon_append[i], lat_append[i],
                                    bearing=last_bearing + ang_pert_cum[i],
                                    angular_distance=last_ang_dist * trans_pert[i])
    # make sure longitude values are within (-180, 180)
    climada.util.coordinates.lon_normalize(lon_append, center=0.0)
    track_ext['lon'][2:] = lon_append[1:]
    track_ext['lat'][2:] = lat_append[1:]

def _get_random_trajectory_ext(track_ext, time_step_h):
    if track_ext is None:
        return tuple()
    random_tpl = tuple(
        tuple(
            [
                np.cumsum(time_step_h * np.degrees(
                    TRACK_EXTENSION_PARS['max_ddirection'] * (
                        2 * _random_uniform_ac(track_ext.time.size - 2,
                                            TRACK_EXTENSION_PARS['autocorr_ddirection'],
                                            time_step_h) - 1
                    )
                )) * np.exp(-TRACK_EXTENSION_PARS['decay_ddirection_hourly']*np.arange(0, time_step_h*(track_ext.time.size-2), time_step_h)),
                1 + TRACK_EXTENSION_PARS['max_dspeed_rel'] * (
                    2 *_random_uniform_ac(track_ext.time.size - 1,
                                        TRACK_EXTENSION_PARS['autocorr_dspeed'],
                                        time_step_h) - 1
                )
            ]
        )
    )
    return random_tpl

def _calc_land_decay(hist_tracks, land_geom, s_rel=True, check_plot=False,
                     pool=None):
    """Compute wind and pressure decay coefficients from historical events

    Decay is calculated for every TC category according to the formulas:

        - wind decay = exp(-x*A)
        - pressure decay = S-(S-1)*exp(-x*B)

    Parameters
    ----------
    hist_tracks : list
        List of xarray Datasets describing TC tracks.
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        land geometry
    s_rel : bool, optional
        use environmental presure to calc S value
        (true) or central presure (false)
    check_plot : bool, optional
        visualize computed coefficients.
        Default: False

    Returns
    -------
    v_rel : dict(category: A)
    p_rel : dict(category: (S, B))
    """

    if len(hist_tracks) < 100:
        LOGGER.warning('For the calibration of the landfall decay '
                       'it is recommended to provide as many historical '
                       'tracks as possible, but only %s historical tracks '
                       'were provided. '
                       'For a more robust calculation consider using '
                       'a larger number of tracks or set '
                       '`use_global_decay_params` to True', len(hist_tracks))

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
                           chunksize=max(min(len(hist_tracks) // pool.ncpus, 1000), 1))
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


def _apply_land_decay(
    tracks, v_rel: Dict, p_rel: Dict, land_geom, s_rel=True, check_plot=False, pool=None
):
    """Compute wind and pressure decay due to landfall in synthetic tracks.

    Parameters
    ----------
    v_rel : dict
        {category: A}, where wind decay = exp(-x*A)
    p_rel : dict
        (category: (S, B)}, where pressure decay
        = S-(S-1)*exp(-x*B)
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        land geometry
    s_rel : bool, optional
        use environmental presure to calc S value
        (true) or central presure (false)
    check_plot : bool, optional
        visualize computed changes
    """
    sy_tracks = [track for track in tracks if not track.orig_event_flag]
    if not sy_tracks:
        raise ValueError('No synthetic tracks contained. Synthetic tracks'
                         ' are needed.')

    if not v_rel or not p_rel:
        LOGGER.info('No decay coefficients.')
        return

    if check_plot:
        orig_wind, orig_pres = [], []
        for track in sy_tracks:
            orig_wind.append(np.copy(track.max_sustained_wind.values))
            orig_pres.append(np.copy(track.central_pressure.values))

    if pool:
        chunksize = max(min(len(tracks) // pool.ncpus, 1000), 1)
        # TODO @benoit did you actually mean to map over `tracks` instead of `sy_tracks`?
        tracks = pool.map(_apply_decay_coeffs, tracks,
                          itertools.repeat(v_rel), itertools.repeat(p_rel),
                          itertools.repeat(land_geom), itertools.repeat(s_rel),
                          chunksize=chunksize)
    else:
        tracks = [_apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel)
                  for track in tracks]

    for track in tracks:
        if track.orig_event_flag:
            climada.hazard.tc_tracks.track_land_params(track, land_geom)
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
    # pylint: disable=protected-access
    v_lf = dict()
    p_lf = dict()
    x_val = dict()

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
    if sea_land_idx.size:
        for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
            v_landfall = track.max_sustained_wind[sea_land - 1].values
            ss_scale = climada.hazard.tc_tracks.set_category(v_landfall,
                                                             track.max_sustained_wind_unit)

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

            if ss_scale not in v_lf:
                v_lf[ss_scale] = array.array('f', v_land)
                p_lf[ss_scale] = (array.array('f', p_land_s),
                                      array.array('f', p_land))
                x_val[ss_scale] = array.array('f',
                                                  track.dist_since_lf[sea_land:land_sea])
            else:
                v_lf[ss_scale].extend(v_land)
                p_lf[ss_scale][0].extend(p_land_s)
                p_lf[ss_scale][1].extend(p_land)
                x_val[ss_scale].extend(track.dist_since_lf[sea_land:land_sea])
    return v_lf, p_lf, x_val


def _decay_calc_coeff(x_val, v_lf, p_lf):
    """From track's relative velocity and pressure, compute the decay
    coefficients.
    - wind decay = exp(-x*A)
    - pressure decay = S-(S-1)*exp(-x*A)

    Parameters
    ----------
    x_val : dict
        key is Saffir-Simpson scale, values are lists with
        the values used as "x" in the coefficient fitting, the
        distance since landfall
    v_lf : dict
        key is Saffir-Simpson scale, values are lists of
        wind/wind at landfall
    p_lf : dict
        key is Saffir-Simpson scale, values are tuples with
        first value the S parameter, second value list of central
        pressure/central pressure at landfall

    Returns
    -------
    v_rel : dict
    p_rel : dict
    """
    warnings.filterwarnings('ignore')
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
    for ss_scale, ss_name in climada.hazard.tc_tracks.CAT_NAMES.items():
        if ss_scale not in p_rel:
            close_scale = scale_fill[np.argmin(np.abs(scale_fill - ss_scale))]
            close_name = climada.hazard.tc_tracks.CAT_NAMES[close_scale]
            LOGGER.debug('No historical track of category %s with landfall. '
                         'Decay parameters from category %s taken.',
                         ss_name, close_name)
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]
        elif v_rel[ss_scale] < 0:
            raise ValueError('The calibration of landfall decay for wind speed resulted in'
                             f' a wind speed increase for TC category {ss_name}.'
                             ' This behaviour is unphysical. Please use a larger number of tracks'
                             ' or use global paramaters by setting `use_global_decay_params` to'
                             ' `True`')
        elif p_rel[ss_scale][0] < 0 or p_rel[ss_scale][1] < 0:
            raise ValueError('The calibration of landfall decay for central pressure resulted in'
                             f' a pressure decrease for TC category {ss_name}.'
                             ' This behaviour is unphysical. Please use a larger number of tracks'
                             ' or use global paramaters by setting `use_global_decay_params` to'
                             ' `True`')

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
        axes[0].set_ylabel('Max sustained wind\nrelative to landfall')
        axes[0].set_title(f'Wind, TC cat {climada.hazard.tc_tracks.CAT_NAMES[track_cat]}')
        axes[0].plot(x_val[track_cat], v_lf[track_cat], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[0].plot(x_eval, _decay_v_function(v_rel[track_cat], x_eval),
                     '-', c=color)

        axes[1].set_xlabel('Distance from landfall (km)')
        axes[1].set_ylabel('Central pressure\nrelative to landfall')
        axes[1].set_title(f'Pressure, TC cat {climada.hazard.tc_tracks.CAT_NAMES[track_cat]}')
        axes[1].plot(x_val[track_cat], p_lf[track_cat][1], '*', c=color,
                     label=climada.hazard.tc_tracks.CAT_NAMES[track_cat])
        axes[1].plot(
            x_eval,
            _decay_p_function(p_rel[track_cat][0], p_rel[track_cat][1], x_eval),
            '-', c=color)

def _add_fits_to_track(track: xr.Dataset, central_pressure_pert: float):
    """Calculate fit of variables to be modelled as a function of central
    pressure for the intensification and the decay phase, and add the results as
    attributes to track.

    The input 'track' is modified in-place!
    
    The following variables are fitted: Maximum sustained wind, radius of
    maximum winds and radius of outmost closed isobar.

    max_sustained_wind: linear regression
    radius_max_wind,radius_oci: piecewise linear regression (up to 4 segments)

    Parameters
    ----------
    track : xr.Dataset
        A single TC track.
    central_pressure_pert : float
        Maximum perturbations in central pressure. Used to determine the range
        of pressure values over which to fit data.    

    Returns
    -------
    track : xr.Dataset 
        Same as input parameter track but with additional attributes 'fit_intens' and
        'fit_decay' (see _get_fit_single_phase). If intensification and/or decay
        consist of less than 3 data points, the corresponding attribute is not set.
    warning_slope : str
        A warning string if a slope has unexpected sign.
    """
    pcen = track.central_pressure.values
    where_max_intensity = np.where(pcen == pcen.min())[0]
    warning_slope = []
    if where_max_intensity[0] > 3:
        track_intens = track[dict(time=slice(None,where_max_intensity[0] + 1))]
        fit_attrs_intens, warning_slope_intens = _get_fit_single_phase(track_intens, central_pressure_pert)
        track.attrs['fit_intens'] = fit_attrs_intens
        if warning_slope_intens is not None:
            warning_slope += ["%s intensification %s" % (track.sid, warning_slope_intens)]
    if where_max_intensity[-1] < len(pcen) - 4:    
        track_decay = track[dict(time=slice(where_max_intensity[-1],None))]
        fit_attrs_decay, warning_slope_decay = _get_fit_single_phase(track_decay, central_pressure_pert)
        track.attrs['fit_decay'] = fit_attrs_decay
        if warning_slope_decay is not None:
            warning_slope += ["%s decay %s" % (track.sid, warning_slope_decay)]
    if len(warning_slope) > 0:
        warning_slope = ', '.join(warning_slope)
    else:
        warning_slope = None
    return warning_slope

def _get_fit_single_phase(track_sub, central_pressure_pert):
    """Calculate order and fit of variables to be modelled for a temporal subset
    of a track.
    
    The following variables are fitted: Maximum sustained wind, radius of
    maximum winds and radius of outmost closed isobar.

    max_sustained_wind: linear regression
    radius_max_wind,radius_oci: piecewise linear regression (up to 4 segments)

    Parameters
    ----------
    track_sub : xr.Dataset
        A temporal subset of a single TC track.
    central_pressure_pert : float
        Maximum perturbations in central pressure. Used to determine the range
        of pressure values over which to fit data.    

    Returns
    -------
    fit_output : Dict
        Dictionary of dictionaries:
        {'var': {
            'order': (int or tuple),
            'fit': statsmodels.regression.linear_model.RegressionResultsWrapper
        }}
        where 'var' is the fitted variables (e.g., 'max_sustained_wind'), 'order'
        is an int or Tuple, and 'fit' is a statsmodels.regression.linear_model.RegressionResultsWrapper
        object.
    """
    pcen = track_sub['central_pressure'].values
    fit_output = {}
    warning_str = None
    for var in FIT_TRACK_VARS_RANGE.keys():
        order = _get_fit_order(pcen, central_pressure_pert)
        d_explanatory = _prepare_data_piecewise(pcen, order)
        d_explained = pd.DataFrame({
            var: track_sub[var].values
        })
        sm_results = sm.OLS(d_explained, d_explanatory).fit()
        fit_output[var] = {
            'order': order,
            'fit': sm_results
        }
        # for wind, check that slope is negative
        if var == 'max_sustained_wind':
            # check that fit lead to monotonous negative function - or at
            # least not impacting more than 2.5 kts
            kts_margin = 2.5
            if isinstance(order, tuple):
                interp_at_res = sm_results.predict(
                    _prepare_data_piecewise(
                        np.array(order),
                        order
                    )
                )
                if np.any(np.diff(interp_at_res.values) > kts_margin):
                    idx_pos = np.where(np.diff(interp_at_res.values) > kts_margin)[0]
                    max_wind = [interp_at_res[i+1] for i in idx_pos]
                    warning_str = "(%s kts, %s, piecewise linear)" % (
                        np.round(np.max(np.diff(interp_at_res.values)), 1),
                        climada.hazard.tc_tracks.CAT_NAMES[
                            climada.hazard.tc_tracks.set_category(max_wind, track_sub.max_sustained_wind_unit)
                        ]
                    )
            else:
                pcen_range = np.diff([pcen.min()-central_pressure_pert, 1010])[0]
                if sm_results.params.values[0]*pcen_range > kts_margin:
                    max_wind = sm_results.predict(
                        _prepare_data_piecewise(
                            np.array([pcen.min()-central_pressure_pert]),
                            order
                        )
                    )
                    warning_str = "(%s kts, %s, linear)" % (
                        np.round(sm_results.params.values[0]*pcen_range, 1),
                        climada.hazard.tc_tracks.CAT_NAMES[
                            climada.hazard.tc_tracks.set_category(max_wind, track_sub.max_sustained_wind_unit)
                        ]
                    )
    return fit_output, warning_str

def _get_fit_order(pcen, central_pressure_pert):
    """Get the order of the data fit.

    Data is split into up to 4 bins, requiring bin width >= 10 mbar and at least 5
    data points per bin. The bins will cover the range from the minimum central
    pressure minus the maximum possible perturbation up to 1021 mbar.

    Parameters
    ----------
    pcen : np.array
        Central pressure values.
    central_pressure_pert : float
        Maximum perturbations in central pressure. Used to determine the range
        of pressure values over which to fit data.

    Returns
    -------
    order : int or Tuple
        Order of the statistical model. Either a tuple with breakpoints for a
        piecewise linear regression, or 1 (int) for a single linear regression.
    """
    # max number of bins to get at least 5 points per bin
    min_bin_pts = 5
    min_bin_width = 10
    n_bins_max = np.fmin(
        4,
        np.fmin(
            np.floor(len(pcen) / min_bin_pts).astype(int),
            np.floor((pcen.max()-pcen.min())/min_bin_width).astype(int)
        )
    )
    # if var_name == 'max_sustained_wind' or len(pcen) <= 5:
    if n_bins_max <= 1:
        order = 1
    else:
        # split the central pressure domain into up to 4 bins
        pcen_min = pcen.min() - central_pressure_pert - 1
        # cover the full range of possible values
        # bins_range = [pcen_min -1, max(pcen.max()+1, 1021)]
        for n_bins in np.arange(n_bins_max, 1, -1):
            order = np.linspace(pcen.min()-0.1, pcen.max(), num=n_bins+1)
            # count valuer per bin
            val_bins = np.digitize(pcen, order, right=True)
            if np.all(np.bincount(val_bins)[1:] >= min_bin_pts):
                break
        order[0] = pcen_min
        order[-1] = max(1021, pcen.max() + 0.1)
        if len(order) == 2:
            # single bin: back to standard linear regression
            order = 1
        else:
            order = tuple(order)
    return order

def _prepare_data_piecewise(pcen, order):
    """Prepare data for statistical modelling of track parameters.

    Parameters
    ----------
    pcen : np.array
        Central pressure values.
    order : int or Tuple
        Order of the statistical model.

    Returns
    -------
    d_explanatory : pandas.DataFrame
        Dataframe with one column per order.
    """
    if isinstance(order, int):
        if order != 1:
            raise ValueError('Unexpected value in order')
        d_explanatory = pd.DataFrame({
            'pcen': pcen,
            'const': [1.0] * len(pcen)
        })
    else:
        # piecewise linear regression
        d_explanatory = dict()
        for i, order_i in enumerate(order):
            col = f'var{order_i}'
            slope_0 = 1. / (order_i - order[i - 1]) if i > 0 else 0
            slope_1 = 1. / (order[i + 1] - order_i) if i + 1 < len(order) else 0
            d_explanatory[col] = np.fmax(0, (1 - slope_0 * np.fmax(0, order_i - pcen)
                                                - slope_1 * np.fmax(0, pcen - order_i)))
        d_explanatory = pd.DataFrame(d_explanatory)
    return d_explanatory

def _estimate_vars_chunk(track: xr.Dataset,
                         phase: str,
                         idx):
    """Estimate a synthetic track's parameters from central pressure based
    on relationships fitted on that track's historical values for a subset of
    the track as specified by the phase (intensification, peak, decay).

    The input 'track' is modified in place!

    The variables estimated from central pressure are 'max_sustained_wind',
    'radius_max_wind 'radius_oci'.

    Parameters
    ----------
    track : xarray.Datasset
        Track data.
    phase : str
        Track phase to be modelled. One of 'intens', 'peak', or 'decay'.
    idx : numpy.array
        Indices in time corresponding to the phase to be modelled.

    Returns
    -------
    Nothing.
    """
    pcen = track.central_pressure.values[idx]
    if phase in ['intens', 'decay']:
        if 'fit_' + phase in track.attrs.keys():
            fit_data = track.attrs['fit_' + phase]
            for var in FIT_TRACK_VARS_RANGE.keys():
                track[var][idx] = np.fmax(
                    FIT_TRACK_VARS_RANGE[var]['min'],
                    np.fmin(
                        FIT_TRACK_VARS_RANGE[var]['max'],
                        fit_data[var]['fit'].predict(
                            _prepare_data_piecewise(pcen, fit_data[var]['order'])
                        ).values
                    )
                )
        else:
            # apply global fit
            track['max_sustained_wind'][idx] = climada.hazard.tc_tracks._estimate_vmax(
                np.repeat(np.nan, pcen.shape), track.lat[idx], track.lon[idx], pcen)
            track['radius_max_wind'][idx] = climada.hazard.tc_tracks.estimate_rmw(
                np.repeat(np.nan, pcen.shape), pcen)
            track['radius_oci'][idx] = climada.hazard.tc_tracks.estimate_roci(
                np.repeat(np.nan, pcen.shape), pcen)
        return
    elif phase != "peak":
        raise ValueError("'phase' should be one of 'intens', 'decay' or 'peak'")

    # for peak: interpolate between estimated first and last value
    if 'fit_intens' in track.attrs.keys():
        fit_data = track.attrs['fit_intens']
        start_vals = {}
        for var in FIT_TRACK_VARS_RANGE.keys():
            start_vals[var] = np.fmax(
                FIT_TRACK_VARS_RANGE[var]['min'],
                np.fmin(
                    FIT_TRACK_VARS_RANGE[var]['max'],
                    fit_data[var]['fit'].predict(
                        _prepare_data_piecewise(np.array([pcen[0]]), fit_data[var]['order'])
                    ).values[0]
                )
            )
    else:
        # take previous values or global fit
        if idx[0] > 0:
            start_vals = {var: track[var].values[idx[0]-1] for var in FIT_TRACK_VARS_RANGE.keys()}
        else:
            start_vals = {}
            # TODO does not work for a single value...
            start_vals['max_sustained_wind'] = climada.hazard.tc_tracks._estimate_vmax(
                np.array([np.nan]), np.array([track.lat[0]]),
                np.array([track.lon[0]]), np.array([pcen[0]]))[0]
            start_vals['radius_max_wind'] = climada.hazard.tc_tracks.estimate_rmw(
                np.array([np.nan]), np.array([pcen[0]]))[0]
            start_vals['radius_oci'] = climada.hazard.tc_tracks.estimate_roci(
                np.array([np.nan]), np.array([pcen[0]]))[0]
    if len(idx) == 1:
        # no need to get the decay fit, just use the intens one
        for var in FIT_TRACK_VARS_RANGE.keys():
            track[var][idx] = start_vals[var]
        return
    if 'fit_decay' in track.attrs.keys():
        fit_data = track.attrs['fit_decay']
        end_vals = {}
        for var in FIT_TRACK_VARS_RANGE.keys():
            end_vals[var] = np.fmax(
                FIT_TRACK_VARS_RANGE[var]['min'],
                np.fmin(
                    FIT_TRACK_VARS_RANGE[var]['max'],
                    fit_data[var]['fit'].predict(
                        _prepare_data_piecewise(np.array([pcen[-1]]), fit_data[var]['order'])
                    ).values[0]
                )
            )
    else:
        # take next values or global fit
        if idx[-1] < len(pcen) - 1:
            end_vals = {var: track[var][idx[-1]+1] for var in FIT_TRACK_VARS_RANGE.keys()}
        else:
            end_vals = {}
            end_vals['max_sustained_wind'] = climada.hazard.tc_tracks._estimate_vmax(
                np.array([np.nan]), np.array([track.lat[idx[-1]]]),
                np.array([track.lon[idx[-1]]]), np.array([pcen[-1]]))[0]
            end_vals['radius_max_wind'] = climada.hazard.tc_tracks.estimate_rmw(
                np.array([np.nan]), np.array([pcen[-1]]))[0]
            end_vals['radius_oci'] = climada.hazard.tc_tracks.estimate_roci(
                np.array([np.nan]), np.array([pcen[-1]]))[0]

    for var in FIT_TRACK_VARS_RANGE.keys():
        track[var][idx] = np.interp(
            idx, np.array([idx[0], idx[-1]]), np.array([start_vals[var], end_vals[var]])
        )
    return track

def check_id_chunk(id_chunk: np.ndarray, sid: str, allow_missing: bool = False, raise_error: bool = True):
    """Check if id_chunk values are valid

    Raises an error if id_chunk values are not valid.

    Parameters
    ----------
    id_chunk : np.ndarray
        Id chunk values.
    sid : str
        Track ID, used for the error message.
    allow_missing : bool
        Whether missing values (np.nan) are allowed in id_chunk.
    raise_error : bool
        Whether an error should be thrown id id_chunk is not valid

    Returns
    -------
    valid : bool
        True if id_chunk is valid, False otherwise.
    """
    valid = True
    # check missing values in id_chunk
    if np.all(np.isnan(id_chunk)):
        LOGGER.debug('All id_chunk values are missing.')
        valid = False
    elif np.any(np.isnan(id_chunk)):
        valid = allow_missing
        # nans are all at the end
        if not np.all(np.isnan(id_chunk[np.where(np.isnan(id_chunk))[0][0]:])):
            LOGGER.debug('id_chunk contains missing values before a non-missing value.')
            valid = False
        if np.isnan(id_chunk[0]):
            LOGGER.debug('id_chunk starts with a missing value.')
            valid = False
        # remove NA values for further checks
        id_chunk = id_chunk[~np.isnan(id_chunk)]
    elif id_chunk.dtype != 'int':
        LOGGER.debug('id_chunk should be integer if no missing values is present.')
        valid = False
    if not valid:
        if raise_error:
            raise ValueError('Invalid id_chunk values: %s.' % sid)
        return valid
    if np.all(id_chunk == 0):
        return valid

    # from here onwards there are not missing values in id_chunk
    if np.any(id_chunk == 0):
        # Any zero after first non-zero value?
        if np.any(id_chunk[np.flatnonzero(id_chunk)[0]:] == 0):
            LOGGER.debug('id_chunk contains zeros after a non-zero.')
            valid = False
    if np.any(np.diff(id_chunk[id_chunk >= 0]) < 0):
        LOGGER.debug('Positive id_chunk values have negative increment.')
        valid = False
    if np.any(np.diff(id_chunk[id_chunk <= 0]) > 0):
        LOGGER.debug('Negative id_chunk values have positive increment.')
        valid = False
    if np.any(np.diff(np.abs(id_chunk)) < 0):
        LOGGER.debug('Absolute value of id_chunk decreases in time.')
        valid = False
    if np.abs(id_chunk[id_chunk != 0][0]) != 1:
        LOGGER.debug('First id_chunk is not 1 or -1.')
        valid = False
    # complete check
    for id in np.unique(id_chunk):
        idx = np.where(id_chunk == id)[0]
        start_idx,end_idx = idx[0],idx[-1]
        if np.any(id_chunk[start_idx:end_idx+1] != id):
            LOGGER.debug('Non-continuous id_chunks.')
            valid = False
        if start_idx > 0:
            if np.sign(id) == np.sign(id_chunk[start_idx-1]):
                LOGGER.debug('Sign of id_chunk issue.')
                valid = False
        if end_idx < id_chunk.size-1:
            if np.sign(id) == np.sign(id_chunk[end_idx+1]):
                LOGGER.debug('Sign of id_chunk issue.')
                valid = False
    # check number of chunks
    no_chunks_sea = int(max(0, id_chunk.max()))
    no_chunks_land = int(np.abs(min(0, id_chunk.min())))
    if np.abs(no_chunks_sea - no_chunks_land) > 1:
        LOGGER.debug('Number of sea and land chunks differ by more than 1.')
        valid = False

    if not valid:
        if raise_error:
            raise ValueError('Invalid id_chunk values: %s.' % sid)
    return valid

def _get_transitions(on_land, time_step_h, exclude_short_landfalls=True):
    """Get sea-land and land_sea transitions

    Transitions are coded as -1 for the first point over land after being over the
    ocean, and +1 for the first point over the ocean after having been on land.
    0 Elsewhere.
    Landfalls during less than NEGLECT_LANDFALL_DURATION_HOUR hour are ignored
    by default.

    Parameters
    ----------
    on_land : np.ndarray (bool)
        True for points over land, False over the ocean.
    time_step_h : float
        Temporal resolution of the on_land array, in hour.
    exclude_short_landfall : bool
        Whether short landfall should be excluded.

    Returns
    -------
    transitions : np.ndarray (int)
        -1 for first point over land if previous point is over the ocean. +1 for
        first point over the ocean if previous point is over land. 0 Otherwise.
    """
    transitions = np.append(0, np.diff((~on_land).astype(int)))
    if exclude_short_landfalls:
        # remove short landfalls
        min_n_ts = NEGLECT_LANDFALL_DURATION_HOUR/time_step_h
        for idx in np.where(transitions == -1)[0]:
            transitions_end = transitions[idx:]
            if np.any(transitions_end == 1):
                n_ts = np.where(transitions_end == 1)[0][0]
                if n_ts < min_n_ts:
                    transitions[idx] = 0
                    transitions[idx+n_ts] = 0
    return transitions

def _first_transition(transitions):
    """Get index of first transition.

    Parameters
    ----------
    transitions : np.ndarray (int)
        -1 for first point over land if previous point is over the ocean. +1 for
        first point over the ocean if previous point is over land. 0 Otherwise.

    Returns
    -------
    tnz : int
        Index of the first non-zero value in transitions. If all values in
        transitions are 0, then transitions.size is returned.
    """
    tnz = np.flatnonzero(transitions)
    return tnz[0] if tnz.size > 0 else transitions.size

def _set_first_transition(transitions_synth, idx_start_model, on_land_synth):
    """Remove any transition before idx_start_model and set the approriate first
    transition there. The input 'transitions_synth' is modified in place.
    """
    if idx_start_model != _first_transition(transitions_synth):
        # adjust transitions according to idx_start_model
        if transitions_synth[idx_start_model] == 0:
            # set a transition at idx_start_model
            trans_idx = np.flatnonzero(transitions_synth)
            if np.any(trans_idx < idx_start_model):
                # take last transition before
                idx = trans_idx[trans_idx < idx_start_model][-1]
                transitions_synth[idx_start_model] = transitions_synth[idx]
            elif np.any(trans_idx > idx_start_model):
                # take opposite of next transition
                idx = trans_idx[trans_idx > idx_start_model][0]
                transitions_synth[idx_start_model] = -transitions_synth[idx]
            elif on_land_synth.all():
                # no transition. If all points over land, must be -1.
                transitions_synth[idx_start_model] = -1
            else:
                # no transition. If any point over ocean, must be +1.
                transitions_synth[idx_start_model] = 1
        # remove transitions before
        transitions_synth[:idx_start_model] = 0

def _get_landfall_idx(transitions):
    if np.any(transitions == -1):
        tnz = np.where(transitions == -1)[0]
    else:
        tnz = np.array([transitions.size])
    return tnz

def _remove_landstart(transitions):
    """Removes first transition if track starts over land.

    The input 'transitions' is modified in place. If the first non-zero element
    is 1, it is set to 0. Nothing else is done.

    Parameters
    ----------
    transitions : np.ndarray (int)
        -1 for first point over land if previous point is over the ocean. +1 for
        first point over the ocean if previous point is over land. 0 Otherwise.
    """
    ftr = _first_transition(transitions)
    # if the first transition is from land to sea, drop it
    if ftr < transitions.size and transitions[ftr] == 1:
        transitions[ftr] = 0

def _get_id_chunk(transitions_synth):
    """Get chunks of track to be modelled and number of land and ocean chunks.

    Parameters
    ----------
    transitions_synth : np.ndarray (int)
        See _get_transitions.

    Returns
    -------
    id_chunks : np.ndarray (int)
        0 before track intensity needs to be modelled, -1 to -no_chunks_land for
        chunks over land and +1 to +no_chunks_sea for chunks over the ocean.
    no_chunks_sea : int
        Number of ocean chunks.
    no_chunks_land : int
        Number of land chunks.
    """
    to_sea = np.where(transitions_synth > 0, transitions_synth, 0)
    no_chunks_sea = np.count_nonzero(to_sea)
    to_land = np.where(transitions_synth < 0, transitions_synth, 0)
    no_chunks_land = np.count_nonzero(to_land)
    # filter out short landfalls from on_land_synth
    def get_on_land_from_transition(transitions):
        on_land_rec = np.zeros_like(transitions)
        trans_non0 = np.append(np.where(transitions != 0)[0], transitions.size)
        for idx_start,idx_end in zip(trans_non0[:-1], trans_non0[1:]):
            if transitions[idx_start] == -1:
                on_land_rec[idx_start:idx_end] = True
        return on_land_rec
    id_chunks = np.where(
        get_on_land_from_transition(transitions_synth),
        to_land.cumsum(),
        to_sea.cumsum()
    ).astype(int)
    return id_chunks, no_chunks_sea, no_chunks_land

def _add_id_synth_chunks_shift_init(track: xr.Dataset,
                                    time_step_h: float = None,
                                    land_geom: MultiPolygon = None,
                                    shift_values_init: bool = True):
    """Identify track chunks for which intensity is to be modelled, and shift
    track parameter value in case the track starts over land.

    The track chunks are coded as follows:
    * -1 to -n when a track point is overland, decreasing by 1 with each landfall.
    * 1 to n when a track point is over the ocean, increasing by 1 with each land-to-ocean
      transition
    * 0 for any point that does not neet any intensity modelling, i.e.:
            * If the track starts on land, the first points on land are set to 0
            * If both historical and synthetic tracks have the same points on land, return all 0
            * If both historical and synthetic tracks have fewer than 6 hours on land, return all 0
    * To capture the intensification of a non-landfalling synthetic track versus its historical
      counterpart, the non-landfalling points on the synthetic track are counted as if after a
      land-to-ocean transition.

    If the synthetic track and/or its historical counterparts starts over land, track parameters
    are additionally shifted in time such as to have both starting over the ocean with the same
    parameter values.

    Parameters
    ----------
    track : xr.Dataset
        A single TC track.
    land_geom : shapely.geometry.multipolygon.MultiPolygon
        Land geometry to assign land parameters.
    shift_values_init : bool
        Whether to shift track parameters (central pressure, maximum sustained
        wind, radii, environmental pressure) in time if the first point over the
        ocean is not the same in on_land and on_land_hist.

    Returns
    -------
    track : xr.Dataset 
        as input parameter track but with additional variable 'id_chunk'
        (ID of chunks value per track point) and, if shift_values_init is True,
        with variables shifted in time if the first track point over the ocean
        is not the same in synthetic vs historical tracks (on_land vs
        on_land_hist variables)
    no_chunks_sea : int
        number of chunks that need intensity modulation over sea
    no_chunks_land : int
        number of chunks that need intensity modulation over land
    track_end_shift : xarray.Dataset
        Subset of track for which the track needs to be extended
    """
    track_end_shift = None
    if track.orig_event_flag:
        return track, 0, 0, track_end_shift

    # default assignment
    track = track.assign({
        "id_chunk" : ("time", np.zeros_like(track.central_pressure, dtype='int')),
    })
    if track.time.size == 1:
        return track, 0, 0, track_end_shift

    if 'on_land' not in list(track.data_vars):
        if land_geom is None:
            raise ValueError('Track %s is missing land params. Argument land_geom should be provided.' % track.sid)
        climada.hazard.tc_tracks.track_land_params(track, land_geom)
    if time_step_h is None:
        time_step_h = track['time_step'].values[0]

    on_land_synth = track.on_land.values.copy()
    on_land_hist = track.on_land_hist.values.copy()

    n_frames_threshold = int(np.floor(6 / track.time_step[0]))   # 6 hours over land
    below_threshold = np.sum(on_land_synth) <= n_frames_threshold and np.sum(on_land_hist) <= n_frames_threshold
    all_equal = np.all(on_land_synth == on_land_hist)
    if below_threshold or all_equal:
        return track, 0, 0, track_end_shift

    # synth track fully over land: apply decay when hist track over ocean
    if np.all(on_land_synth):
        if np.any(~on_land_hist):
            # all over land but historical went over the ocean: prevent intensification
            track['id_chunk'][:] = np.where(np.cumsum(~on_land_hist) > 0, -1, 0).astype(int)
            check_id_chunk(track['id_chunk'].values, track.sid)
        return track, 0, 1, track_end_shift
    # hist track fully over land: model from first point over the ocean.
    if np.all(on_land_hist):
        transitions_synth = _get_transitions(on_land_synth, time_step_h, exclude_short_landfalls=True)
        id_chunk, no_chunks_sea, no_chunks_land = _get_id_chunk(transitions_synth)
        check_id_chunk(id_chunk, track.sid)
        track['id_chunk'][:] = id_chunk
        return track, no_chunks_sea, no_chunks_land, track_end_shift

    shift_first_sea, idx_start_model = _get_shift_idx_start(on_land_hist, on_land_synth, time_step_h, shift_values_init, track.sid)
    if shift_first_sea != 0:
        if shift_first_sea > 0:
            track_shift_values = track.isel(
                time=slice(-shift_first_sea, None)
            ).to_dataframe().reset_index().drop(['time', 'lat', 'lon'], axis=1)
            track_end_shift = _create_raw_track_extension(
                track,
                nb_time_steps=shift_first_sea,
                time_step_h=time_step_h,
                values_df=track_shift_values
            )
        params_fixed = ['time_step', 'basin', 'on_land', 'dist_since_lf']
        params_avail = list(track.data_vars)
        for tc_var in list(set(params_avail) - set(params_fixed)):
            if shift_first_sea < 0:
                track[tc_var].values[:shift_first_sea] = track[tc_var].values[-shift_first_sea:]
            else:
                track[tc_var].values[shift_first_sea:] = track[tc_var].values[:-shift_first_sea]

    if idx_start_model == on_land_synth.size:
        # intensity not to be modelled.
        return track, 0, 0, track_end_shift

    transitions_synth = _get_transitions(on_land_synth, time_step_h)
    _set_first_transition(transitions_synth, idx_start_model, on_land_synth)

    id_chunk, no_chunks_sea, no_chunks_land = _get_id_chunk(transitions_synth)
    check_id_chunk(id_chunk, track.sid)
    track['id_chunk'][:] = id_chunk
    return track, no_chunks_sea, no_chunks_land, track_end_shift

def _one_model_synth_tc_intensity(track: xr.Dataset,
                                  v_rel: dict,
                                  p_rel: dict,
                                  s_rel: bool,
                                  central_pressure_pert: float,
                                  rnd_pars: np.ndarray):
    """Models a synthetic track's intensity evolution

    Sequentially moves over each unique track["id_chunk"] and applies the following:
    * If id_chunk is negative, i.e. over land, applies landfall decay logic
    * if id_chunk is positive, i.e. over sea, applies intensification/peak duration/decay logic
    
    # TODO update docstring

    Parameters
    ----------
    track : xarray.Dataset
        A synthetic track object
    v_rel : Dict
        A dict of form {category : A} for MSW decay of the form exp(-x*A)
    p_rel : Dict
        A dict of form {category : (S, B)} for pressure decay of the form S-(S-1)*exp(-x*B)
    s_rel : bool, optional
        use environmental presure to calc S value
        (true) or central presure (false)
    central_pressure_pert : float
        Magnitude of the intensity perturbation (in mbar). This value is used to
        perturb the maximum intensity (lowest central_pressure value) when
        adjust_intensity is True. Perturbations are generated from a normal
        distribution with mean=0 and sd=central_pressure_pert/2 constrained
        between -central_pressure_pert and +central_pressure_pert.
    rnd_pars: np.ndarray
        Array of 4 random values within (0,1] to be used for perturbing track
        intensity over the ocean.
    """
    # if no land point in either track -> return track
    values_ext_df = None
    if track.time.size == 1 or track.orig_event_flag:
        return track, values_ext_df
    check_id_chunk(track['id_chunk'].values, track.sid, allow_missing=False)
    if np.all(track['id_chunk'] == 0):
        return track, values_ext_df
    
    # organise chunks to be processed in temporal sequence
    chunk_index = np.unique(track.id_chunk.values, return_index=True)[1]
    id_chunk_sorted = [track.id_chunk.values[index] for index in sorted(chunk_index)]
    # track_orig = track.copy()

    last_pcen = track.central_pressure.values[-1]
    track_orig = track.copy(deep=True)
    
    for id_chunk in id_chunk_sorted:
        if id_chunk == 0:
            continue
        elif id_chunk > 0:
            pcen_extend = intensity_evolution_sea(track, id_chunk, central_pressure_pert, rnd_pars[4*(id_chunk-1):4*id_chunk])
            v_extend = None
        else:
            pcen_extend, v_extend = intensity_evolution_land(track, id_chunk, v_rel, p_rel, s_rel)

    # make sure track ends meaningfully (low intensity)
    if pcen_extend is not None:
        values_ext_df = {
            'central_pressure': pcen_extend
        }
        if v_extend is not None:
            values_ext_df['max_sustained_wind'] = v_extend
        values_ext_df = pd.DataFrame(values_ext_df)

    return track, values_ext_df

def _model_synth_tc_intensity(tracks_list,
                              random_vec_intensity,
                              time_step_h,
                              track_vars_attrs,
                              extend_track,
                              pool,
                              central_pressure_pert,
                              v_rel,
                              p_rel,
                              s_rel):
    # model track intensity
    sid_extended_tracks = []
    sid_outside_lat_range = []
    sid_outside_intensity_range = []
    if pool:
        chunksize = min(len(tracks_list) // pool.ncpus, 1000)
        tracks_intensified = pool.map(
            _one_model_synth_tc_intensity,
            tracks_list,
            itertools.repeat(v_rel, len(tracks_list)),
            itertools.repeat(p_rel, len(tracks_list)),
            itertools.repeat(s_rel, len(tracks_list)),
            itertools.repeat(central_pressure_pert, len(tracks_list)),
            random_vec_intensity,
            chunksize=chunksize
        )
    else:
        tracks_intensified = [
            _one_model_synth_tc_intensity(track, v_rel, p_rel, s_rel,
                                        central_pressure_pert, rnd_pars=random_intensity)
            for track, random_intensity in zip(tracks_list, random_vec_intensity)
        ]
    if not extend_track:
        for (track,_) in tracks_intensified:
            if not track.orig_event_flag:
                if np.max(np.diff(track.max_sustained_wind.values)) > 20:
                    print(f"BEFORE PROBLEM: {track.sid}")
                _estimate_params_track(track)
                track.attrs['category'] = climada.hazard.tc_tracks.set_category(
                    track.max_sustained_wind.values, track.max_sustained_wind_unit)
                if np.max(np.diff(track.max_sustained_wind.values)) > 20:
                    print(f"AFTER PROBLEM: {track.sid}")
        new_tracks_list = [
            drop_temporary_variables(track, track_vars_attrs)
            for (track,_) in tracks_intensified
        ]
        return (
            new_tracks_list, 
            sid_extended_tracks,
            sid_outside_lat_range,
            sid_outside_intensity_range
            )

    LOGGER.debug('Extending tracks after intensity modelling')
    # create extension and trajectory
    tracks_intensified_new = []
    LOGGER.debug('create track extensions')
    for i,(track,values_df) in enumerate(tracks_intensified):
        if values_df is not None and values_df.shape[0] == 0:
            LOGGER.warning('values_df has 0 rows, please check: %s', track.sid)
        if values_df is None or values_df.shape[0] == 0:
            tracks_intensified_new.append(track)
            continue
        sid_extended_tracks.append(track.sid)
        track_ext = _create_raw_track_extension(
            track,
            nb_time_steps=values_df.shape[0],
            time_step_h=time_step_h,
            values_df=values_df
        )
        rnd_ext = _get_random_trajectory_ext(track_ext, time_step_h)
        tracks_intensified_new.append(_create_track_from_ext(track, track_ext, rnd_ext))
    # get new land geometry
    extent = climada.util.coordinates.latlon_bounds(
        np.concatenate([t.lat.values for t in tracks_intensified_new]),
        np.concatenate([t.lon.values for t in tracks_intensified_new]),
        buffer=0.1
    )
    extent = (extent[0], extent[2], extent[1], extent[3])
    land_geom = climada.util.coordinates.get_land_geometry(
        extent=extent, resolution=10
    )
    # any sea-to-land transition? If so apply landfall decay thereafter
    LOGGER.debug('apply decay after sea-to-land transition')
    tracks_intensified_new2 = []
    for i,track in enumerate(tracks_intensified_new):
        if track.orig_event_flag:
            tracks_intensified_new2.append(track)
            continue
        
        if np.sum(np.isnan(track['id_chunk'].values)) == 0:
            # no track extension
            _estimate_params_track(track)
            track.attrs['category'] = climada.hazard.tc_tracks.set_category(
                track.max_sustained_wind.values, track.max_sustained_wind_unit)
            tracks_intensified_new2.append(track)
            continue
        climada.hazard.tc_tracks.track_land_params(track, land_geom=land_geom)
        sea_land = np.where(
            np.isnan(track['id_chunk']),
            np.append(0, np.diff(track.on_land.astype(int))) == 1,
            False
        )
        _estimate_params_track(track)
        # apply decay (landfall) if track moves too much poleward or equatorward
        latrange_out_idx = _get_outside_lat_idx(track)
        if latrange_out_idx < track.time.size:
            if np.all(~sea_land) or np.where(sea_land)[0][0] > latrange_out_idx:
                sid_outside_lat_range.append(track.sid)
                sea_land[latrange_out_idx] = True
        # apply decay (landfall) if track falls below the intensity of the final
        # point in ibtracs
        intensityrange_out_idx = _get_finalintensity_idx(track, original_track=tracks_list[i])
        if intensityrange_out_idx < track.time.size:
            if int(os.getenv('TRACKGEN_TEST_DECAY_BEYOND_ADVISORIES')):
                if np.all(~sea_land) or np.where(sea_land)[0][0] > intensityrange_out_idx:
                    sid_outside_intensity_range.append(track.sid)
                    sea_land[intensityrange_out_idx] = True
        if np.any(sea_land):
            # apply landfall decay thereafter
            sea_land_idx = np.where(sea_land)[0][0]
            track['id_chunk'] = ('time', np.concatenate([
                np.repeat([0], sea_land_idx),
                np.repeat([-1], track.time.size - sea_land_idx)
            ]).astype(int))
            check_id_chunk(track['id_chunk'].values, track.sid, allow_missing=False)
            # assign fake dist_since_lf
            track['on_land'][sea_land_idx:] = True
            track['dist_since_lf'] = ('time', climada.hazard.tc_tracks._dist_since_lf(track))
            pcen_extend, v_extend = intensity_evolution_land(track,
                                                             id_chunk=-1,
                                                             v_rel=v_rel,
                                                             p_rel=p_rel,
                                                             s_rel=s_rel)
            if v_extend is not None:
                LOGGER.warning(
                    f'Track extension not over owing to a landfall for {track.sid}. '
                    f'Central pressure: {pcen_extend[-1]}, Max sustained wind: {v_extend[-1]}.'
                )
            elif pcen_extend is not None:
                LOGGER.warning(
                    f'Track extension not over owing to a landfall for {track.sid}. '
                    f'Central pressure: {pcen_extend[-1]}.'
                )
            _estimate_params_track(track)
        
        # cutoff track end?
        extended_cat = np.array([
            climada.hazard.tc_tracks.set_category(
                track['max_sustained_wind'].values[idx],
                wind_unit=track.max_sustained_wind_unit
            )
            for idx in range(track.time.size)
        ])

        if np.any(extended_cat >= 0):  
            # Kill the extended track 12 hours after dropping below Cat 1
            buffer_frames = int(np.ceil(12/time_step_h))
            cutoff_idx = min(np.where(extended_cat >= 0)[0][-1] + buffer_frames, extended_cat.size-1)
            track = track.isel(time=slice(None, cutoff_idx))

        # If the parent storm finished as extratropical, kill any extension frames at times beyond the original track end
        original_size = tracks_list[i].time.size
        if hasattr(track, 'nature'):
            advisory_range_et_idx = original_size if tracks_list[i].nature.values[-1] == 'ET' else track.time.size
            if advisory_range_et_idx < track.time.size:
                if int(os.getenv('TRACKGEN_TEST_KILL_BEYOND_EXTRATROPICAL')):
                    track = track.isel(time=slice(None, advisory_range_et_idx))
                    sid_outside_intensity_range.append(track.sid)
                    sid_outside_intensity_range = list(set(sid_outside_intensity_range))
        
        # If the child storm weakens below the parent's final strength, kill it
        intensityrange_out_idx = _get_finalintensity_idx(track, original_track=tracks_list[i])
        intensityrange_out_idx = max(original_size, intensityrange_out_idx)
        if intensityrange_out_idx < track.time.size:
            if int(os.getenv('TRACKGEN_TEST_KILL_BEYOND_ADVISORIES')):
                track = track.isel(time=slice(None, intensityrange_out_idx))
                sid_outside_intensity_range.append(track.sid)
                sid_outside_intensity_range = list(set(sid_outside_intensity_range))
        
        tracks_intensified_new2.append(track)
        tracks_intensified_new2[i].attrs['category'] = climada.hazard.tc_tracks.set_category(
            tracks_intensified_new2[i].max_sustained_wind.values,
            tracks_intensified_new2[i].max_sustained_wind_unit
        )
        
    LOGGER.debug('dropping temporary variables')
    new_tracks_list = [
        drop_temporary_variables(track, track_vars_attrs)
        for track in tracks_intensified_new2
    ]
    return new_tracks_list, sid_extended_tracks, sid_outside_lat_range, sid_outside_intensity_range

def drop_temporary_variables(track : xr.Dataset, track_vars_attrs):
    # TODO docstring
    vars_to_drop = set(track.variables) - track_vars_attrs[0]
    attrs_to_drop = set(track.attrs.keys()) - track_vars_attrs[1]
    for attr in attrs_to_drop:
        del(track.attrs[attr])
    for v in ['central_pressure', 'max_sustained_wind', 'radius_max_wind', 'environmental_pressure',
            'time', 'lat', 'lon', 'on_land']:
        if v in track.variables:
            if np.any(np.isnan(track[v].values)):
                raise ValueError('Missing values in %s: %s', v, track.sid)
    # check time
    time_steps_h = np.diff(track.time.values) / np.timedelta64(1, 'h')
    if track.time.size > 1 and not np.allclose(time_steps_h, time_steps_h[0]):
        LOGGER.debug('time steps in hour: %s' % ','.join(time_steps_h.astype(str)))
        raise ValueError('Temporal resolution not constant: %s', track.sid)
    return track.drop_vars(vars_to_drop)

def _get_cat_from_pcen(track_chunk, phase, idx=None, pcen=None):
    if pcen is not None:
        if isinstance(pcen, float):
            pcen = np.array([pcen], dtype = 'float')
        if idx is not None:
            raise ValueError('only one of pcen and idx can be provided')
        track_peak = track_chunk.copy(True).isel(time = slice(None, pcen.size))
        track_peak['central_pressure'][:] = pcen
    else:
        track_peak = track_chunk.copy(True).isel(time = [idx])
    _estimate_vars_chunk(track_peak, phase=phase, idx=np.zeros(1).astype(int))
    peak_vmax = track_peak['max_sustained_wind'].values
    peak_cat = climada.hazard.tc_tracks.set_category(peak_vmax, wind_unit=track_chunk.max_sustained_wind_unit)
    return peak_cat

def _get_peak_duration(track_chunk, peak_pcen_i, pcen, rnd_par):
    peak_cat = _get_cat_from_pcen(track_chunk, "peak", pcen=pcen)
    peak_cat = RANDOM_WALK_DATA_CAT_STR[peak_cat]
    peak_basin = track_chunk.basin.values[peak_pcen_i]
    duration_i = np.logical_and(
        RANDOM_WALK_DATA_DURATION['basin'] == peak_basin,
        RANDOM_WALK_DATA_DURATION['category'] == peak_cat
    )
    lognorm_pars = RANDOM_WALK_DATA_DURATION.loc[duration_i, ['meanlog', 'sdlog']].to_dict('records')[0]
    # sample from that distribution
    peak_duration_days = np.exp(scipy.stats.norm.ppf(rnd_par,
                                                        loc=lognorm_pars['meanlog'],
                                                        scale=lognorm_pars['sdlog']))
    return peak_duration_days

def _get_decay_k(basin, rnd_par):
    weibull_pars = RANDOM_WALK_DATA_SEA_DECAY.loc[
        RANDOM_WALK_DATA_SEA_DECAY['basin'] == basin
    ].to_dict('records')[0]
    peak_decay_k = scipy.stats.weibull_min.ppf(rnd_par,
                                                c = weibull_pars['shape'],
                                                scale = weibull_pars['scale'])
    return peak_decay_k

def intensity_evolution_sea(track, id_chunk, central_pressure_pert, rnd_pars_i):
    if id_chunk <= 0:
        raise ValueError('id_chunk should be positive')
    # TODO docstring
    in_chunk = np.where(track.id_chunk.values == id_chunk)[0]
    pcen_extend = None
    # if a single point over the ocean, do not adjust intensity - keep constant
    if len(in_chunk) == 1:
        # TODO if last point, any need to model?
        if in_chunk[-1] == track.time.size - 1:
            peak_cat = _get_cat_from_pcen(track, "decay", idx = in_chunk[-1])
            if peak_cat > 0:
                LOGGER.warning('Track ends on a single ocean point with cat %s' % RANDOM_WALK_DATA_CAT_STR[peak_cat])
        if in_chunk[0] > 0:
            track['max_sustained_wind'][in_chunk] = track['max_sustained_wind'].values[in_chunk-1]
            track['central_pressure'][in_chunk] = track['central_pressure'].values[in_chunk-1]
        return pcen_extend
    track_stage_end = 'intens'
    
    # taking last value before the chunk as a starting point from where to model intensity
    if in_chunk[0] > 0:
        in_chunk = np.append(in_chunk[0]-1, in_chunk)
    
    track_chunk = track.isel(time = in_chunk)
    pcen = track_chunk.central_pressure.values
    time_days = np.append(0, np.cumsum(track_chunk.time_step.values[1:] / 24))
    time_step_h = track_chunk.time_step.values[0]

    # perturb target central pressure: truncated normal distribution
    target_peak_pert = central_pressure_pert / 2 * scipy.stats.truncnorm.ppf(rnd_pars_i[0], -2, 2)
    target_peak = track_chunk.target_central_pressure.values[0] + target_peak_pert

    if pcen[0] <= target_peak:
        # already at target, no intensification needed - keep at current intensity
        pcen[:] = pcen[0]
        # ensure central pressure < environmental pressure
        pcen = np.fmin(pcen, track_chunk.environmental_pressure.values)
    else:
        # intensification parameters
        # was there any landfall prior to then?
        after_lf = int(np.any(np.diff(track.on_land[:in_chunk[1]].values.astype(int)) == 1))
        # filter to current case
        inten_i = (after_lf == RANDOM_WALK_DATA_INTENSIFICATION['afterLF']) & \
            (rnd_pars_i[1] >= RANDOM_WALK_DATA_INTENSIFICATION['cumfreqlow']) & \
            (rnd_pars_i[1] < RANDOM_WALK_DATA_INTENSIFICATION['cumfreqhigh']) & \
            (pcen[0] - target_peak >= RANDOM_WALK_DATA_INTENSIFICATION['pcen_low_bnds'])
        # keep the one with highest pcen_low_bnds
        inten_pars = RANDOM_WALK_DATA_INTENSIFICATION.loc[
            inten_i, ['a', 'b', 'pcen_low_bnds']
        ].sort_values('pcen_low_bnds', ascending=False).iloc[0].loc[['a', 'b']].to_dict()
        # apply intensification
        pcen = np.fmax(pcen[0] + inten_pars['a']*(1-np.exp(inten_pars['b']*time_days)), np.array([target_peak]))
        # ensure central pressure < environmental pressure
        pcen = np.fmin(pcen, track_chunk.environmental_pressure.values)
        track_chunk['central_pressure'][:] = pcen

    # peak duration
    # defined as the time difference between the first point after peak
    # intensity and first point within peak intensity.

    # Peak is reached if target is reached within 5 mbar
    is_peak = pcen - target_peak <= 5
    if np.sum(is_peak) > 0:

        # if the chunks starts at peak, account for previous time steps for
        # peak duration
        if is_peak[0]:
            # account for previous time steps for peak duration
            pcen_before_peak_rev = np.flip(track.central_pressure.values[:in_chunk[0]])
            idx_before_peak_rev = np.where(pcen_before_peak_rev - target_peak > 5)[0]
            time_before_peak_rev = np.flip(track.time.values[:in_chunk[0]])
            if idx_before_peak_rev.shape[0] > 0:
                idx_start_peak_rev = idx_before_peak_rev[0] -1
                # time of first point within peak
                peak_start_time = time_before_peak_rev[idx_start_peak_rev]
                # time spent in peak until chunk start
                time_already_in_peak = track_chunk.time.values[0] - peak_start_time
                time_already_in_peak_days = time_already_in_peak / np.timedelta64(1, 'D')
            else:
                time_already_in_peak_days = 0
        else:
            time_already_in_peak_days = 0

        # apply duration: as a function of basin and category
        peak_pcen_i = np.where(pcen == np.min(pcen))[0][0]
        # since we do not know if we are in the intensification or decay
        # phase of the whole TC, take the average Vmax estimate from both
        # fits to determine TC category at peak
        peak_duration_days = _get_peak_duration(track_chunk, peak_pcen_i, np.min(pcen), rnd_pars_i[2])
        # last peak point
        time_in_peak = time_days - time_days[np.where(is_peak)[0][0]] + time_already_in_peak_days
        if np.any(time_in_peak <= peak_duration_days):
            end_peak = np.where(time_in_peak <= peak_duration_days)[0][-1]
            if end_peak > 2:
                # add pcen variations during peak
                if is_peak[0]:
                    # starting at peak: linear decrease to peak value +2.5mbar
                    # from peak center
                    peak_start_idx = np.where(pcen == np.min(pcen))[0][0]
                    mid_peak_idx = np.floor((end_peak+1)/2).astype(int)
                    pcen[mid_peak_idx:end_peak + 1] = np.interp(
                        np.arange(mid_peak_idx, end_peak+1),
                        np.array([mid_peak_idx, end_peak]),
                        np.array([pcen[mid_peak_idx], target_peak+2.5])
                    )
                elif end_peak - np.where(is_peak)[0][0] >= 2:
                    # model peak as a quadratic function
                    peak_start_idx = np.where(is_peak)[0][0]
                    mid_peak_idx = (peak_start_idx + np.floor((end_peak-peak_start_idx+1)/2)).astype(int)
                    if pcen[peak_start_idx] == pcen[mid_peak_idx]:
                        pcen[peak_start_idx] = pcen[peak_start_idx] + 2.5
                    interp_fun = scipy.interpolate.interp1d(
                        x=np.array([peak_start_idx, mid_peak_idx, end_peak]),
                        y=np.array([pcen[peak_start_idx], pcen[mid_peak_idx], pcen[end_peak]+2.5]),
                        kind='quadratic'
                    )
                    pcen[peak_start_idx:end_peak+1] = interp_fun(np.arange(peak_start_idx, end_peak+1))
        else:
            # peak already ends
            end_peak = 0

        # decay required?
        if end_peak < len(time_days) - 1:
            peak_decay_k = _get_decay_k(track_chunk.basin.values[end_peak], rnd_pars_i[3])
            time_in_decay = time_days[end_peak:] - time_days[end_peak]
            # decay: p(t) = p_env(t) + (p(t=0) - p_env(t)) * exp(-k*t)
            p_drop = (pcen[end_peak] - track_chunk.environmental_pressure.values[end_peak:])
            p_drop = p_drop * np.exp(-peak_decay_k * time_in_decay)
            pcen[end_peak:] = track_chunk.environmental_pressure.values[end_peak:] + p_drop

            # ensure central pressure < environmental pressure
            pcen = np.fmin(pcen, track_chunk.environmental_pressure.values)
            track_stage_end = 'decay'

        else:
            track_stage_end = 'peak'

    if track.id_chunk.values[-1] == id_chunk:
        # Add additional data points if ends at high category
        # peak not reached but end of track
        end_cat = _get_cat_from_pcen(track_chunk, track_stage_end, pcen=pcen[-1])
        if end_cat >= 0:
            # ending as a Tropical Storm or above: extend track
            # when will peak be reached?

            # INTENSIFICATION EXTENSION
            if track_stage_end == 'intens':
                nb_days_extend_peak = np.log(1-(target_peak-pcen[0])/inten_pars['a'])/inten_pars['b']
                nts_extend_peak = int(np.ceil(24*nb_days_extend_peak/time_step_h)) - len(pcen) + 1
                time_days_extend = time_days[-1] + np.cumsum(np.repeat([time_step_h/24], nts_extend_peak))
                pcen_extend = np.fmax(pcen[0] + inten_pars['a']*(1-np.exp(inten_pars['b']*time_days_extend)), np.array([target_peak]))
                pcen_extend = np.fmin(pcen_extend, track_chunk.environmental_pressure.values[-1])
                # track_stage_end = 'peak'
            else:
                pcen_extend = np.array([], dtype='float')

            # PEAK EXTENSION
            if track_stage_end in ['intens', 'peak']:
                track_chunk_peak = track_chunk.copy(True).isel(time=[-1])
                peak_duration_days = _get_peak_duration(track_chunk_peak, -1, target_peak, rnd_pars_i[2])
                peak_nts = np.floor(24*peak_duration_days/time_step_h)
                if track_stage_end == 'intens':
                    # peak starts during extension
                    track_chunk_peak['central_pressure'][:] = pcen_extend[-1]
                    peak_start_idx = np.where(pcen_extend <= target_peak + 5)[0][0]
                else:
                    # peak starts before extension
                    track_chunk_peak['central_pressure'][:] = pcen[-1]
                    peak_start_idx = np.where(pcen <= target_peak + 5)[0][0] - pcen.size
                peak_end_idx = int(peak_start_idx + peak_nts)
                if peak_end_idx >= 0 and peak_end_idx < pcen_extend.size:
                    pcen_extend = pcen_extend[:peak_end_idx]
                elif peak_end_idx >= pcen_extend.size:
                    pcen_extend = np.concatenate([pcen_extend, np.repeat([target_peak], peak_end_idx-pcen_extend.size)])
                pcen_extend = np.fmin(pcen_extend, track_chunk.environmental_pressure.values[-1])
                # if peak_end_idx is negative: peak occurs before extension
            else:
                # if extension starts at decay, set negative peak_end_idx
                peak_end_idx = end_peak - pcen.size

            # TODO add pcen variations during peak?

            # DECAY EXTENSION - done in all cases
            # then get decay up to below TS
            peak_decay_k = _get_decay_k(track_chunk.basin.values[-1], rnd_pars_i[3])
            if peak_end_idx >= 0:
                if len(pcen_extend) > 0:
                    p_drop = pcen_extend[-1] - track_chunk.environmental_pressure.values[-1]
                else:
                    # happens if track_stage_end was 'peak' but peak duration was already reached.
                    p_drop = pcen[-1] - track_chunk.environmental_pressure.values[-1]
            else:
                p_drop = pcen[peak_end_idx] - track_chunk.environmental_pressure.values[-1]
            # p_drop >= 0 implies pcen >= penv i.e. no extension to be applied.
            # Hence, decay extension is to be applied only if p_drop < 0
            if p_drop < 0:
                target_decay_pres = track_chunk.environmental_pressure.values[-1] - 5
                p_drop_rel = (target_decay_pres - track_chunk.environmental_pressure.values[-1])/p_drop
                nb_days_decay = -np.log(p_drop_rel) / peak_decay_k
                nts_extend_decay = int(np.ceil(24*nb_days_decay/time_step_h))
                if nts_extend_decay > 0:
                    time_in_decay = np.cumsum(np.repeat([time_step_h/24], nts_extend_decay))
                    p_drop2 = p_drop * np.exp(-peak_decay_k * time_in_decay)
                    pcen_decay = track_chunk.environmental_pressure.values[-1] + p_drop2
                    # p_drop = (pcen[end_peak] - track_chunk.environmental_pressure.values[end_peak:])
                    # p_drop = p_drop * np.exp(-peak_decay_k * time_in_decay)
                    # pcen[end_peak:] = track_chunk.environmental_pressure.values[end_peak:] + p_drop
                    if peak_end_idx < 0:
                        # peak ended during existing chunk, therefore first
                        # abs(peak_end_idx) time steps already in pcen: exclude those
                        pcen_decay = pcen_decay[-(peak_end_idx+1):]
                    pcen_extend = np.concatenate([pcen_extend, pcen_decay])
            pcen_extend = np.fmin(pcen_extend, track_chunk.environmental_pressure.values[-1])

            # finally, truncate track to keep only 4 hours after entering TD category
            extended_cat = np.array([
                _get_cat_from_pcen(track_chunk, phase='decay', pcen=pcen_extend[idx])
                for idx in range(pcen_extend.size)
            ])
            n_buffer_frames = int(np.ceil(12 / time_step_h))
            if np.any(extended_cat >= 0):
                cutoff_idx = min(np.where(extended_cat >= 0)[0][-1] + n_buffer_frames, extended_cat.size-1)
                pcen_extend = pcen_extend[:cutoff_idx]
            if pcen_extend.size == 0:
                # in case in the end there is nothing to extend (0 time steps)
                pcen_extend = None

    # Now central pressure has been modelled
    track_chunk['central_pressure'][:] = pcen
    track['central_pressure'][in_chunk] = track_chunk['central_pressure'][:]
    return pcen_extend

def intensity_evolution_land(track, id_chunk, v_rel, p_rel, s_rel):
    # TODO docstring
    if id_chunk >= 0:
        raise ValueError('id_chunk should be negative')
    if track.time.size < 2:
        raise ValueError('track intensity can only be modelled if track consists of at least 2 points')
    # taking last value before the chunk as a starting point from where to model intensity
    in_chunk = np.where(track.id_chunk.values == id_chunk)[0]

    if in_chunk[0] > 0:
        if track.id_chunk.values[in_chunk[0]-1] != 0:
            # need to estimate wind speed
            _estimate_params_track(track)
        # values just before landfall
        p_landfall = float(track.central_pressure[in_chunk[0]-1].values)
        v_landfall = float(track.max_sustained_wind[in_chunk[0]-1].values)
    else:
        p_landfall = float(track.central_pressure[in_chunk[0]].values)
        v_landfall = float(track.max_sustained_wind[in_chunk[0]].values)
    track_chunk = track.isel(time = in_chunk)
    # implement that part
    ss_scale = climada.hazard.tc_tracks.set_category(v_landfall,
                                                        track.max_sustained_wind_unit)
    # position of the last point on land: last item in chunk
    S = _calc_decay_ps_value(track_chunk, p_landfall, len(in_chunk)-1, s_rel)
    if S <= 1:
        # central_pressure at start of landfall > env_pres after landfall:
        # set central_pressure to environmental pressure during whole lf
        track_chunk.central_pressure[:] = track_chunk.environmental_pressure.values
    else:
        p_decay = _decay_p_function(S, p_rel[ss_scale][1],
                                    track_chunk.dist_since_lf.values)
        # dont apply decay if it would decrease central pressure
        if np.any(p_decay < 1):
            LOGGER.info('Landfall decay would decrease pressure for '
                        'track id %s, leading to an intensification '
                        'of the Tropical Cyclone. This behaviour is '
                        'unphysical and therefore landfall decay is not '
                        'applied in this case.',
                        track_chunk.sid)
            p_decay[p_decay < 1] = (track_chunk.central_pressure[p_decay < 1]
                                    / p_landfall)
        track_chunk['central_pressure'][:] = p_landfall * p_decay

    v_decay = _decay_v_function(v_rel[ss_scale],
                                track_chunk.dist_since_lf.values)
    # dont apply decay if it would increase wind speeds
    if np.any(v_decay > 1):
        # should not happen unless v_rel is negative
        LOGGER.info('Landfall decay would increase wind speed for '
                    'track id %s. This behaviour is unphysical and '
                    'therefore landfall decay is not applied in this '
                    'case.',
                    track_chunk.sid)
        v_decay[v_decay > 1] = (track_chunk.max_sustained_wind[v_decay > 1]
                                / v_landfall)
    track_chunk['max_sustained_wind'][:] = v_landfall * v_decay

    # Any need to extend the track?
    pcen_extend, v_extend = None, None
    if track.id_chunk.values[-1] == id_chunk:
        end_cat = _get_cat_from_pcen(track_chunk, "decay", track_chunk.time.size-1)
        if end_cat > 0:
            target_wind_speed = 25
            target_v_decay = target_wind_speed / v_landfall
            target_dist_since_lf = -np.log(target_v_decay)/v_rel[ss_scale]
            # approximate the distance between points in the extension with the
            #   distance between the last two existing points
            if track_chunk.lon.values.size == 1:
                # this chunk is the end and track.time.size>1 hence in_chunk[0]>0
                lon_last_but_one = track.lon.values[in_chunk[0]-1:in_chunk[0], None]
                lat_last_but_one = track.lat.values[in_chunk[0]-1:in_chunk[0], None]
            else:
                lon_last_but_one = track_chunk.lon.values[-2:-1, None]
                lat_last_but_one = track_chunk.lat.values[-2:-1, None]
            d_dist_since_lf = climada.util.coordinates.dist_approx(
                lat_last_but_one,
                lon_last_but_one,
                track_chunk.lat.values[-1:, None],
                track_chunk.lon.values[-1:, None],
                method="geosphere",
                units="km"
            )[0, 0, 0]
            if d_dist_since_lf < 1:
                LOGGER.debug('Tiny distance since landfall found, adjusting up for track '
                             'extension calculations: %s',  str(d_dist_since_lf))
                d_dist_since_lf = 1

            dist_since_lf_extend = np.arange(
                track_chunk.dist_since_lf.values[-1],
                target_dist_since_lf,
                d_dist_since_lf
            ) + d_dist_since_lf

            if dist_since_lf_extend.size > 0:
                v_decay_extend = _decay_v_function(v_rel[ss_scale], dist_since_lf_extend)
                v_extend = v_landfall * v_decay_extend
                p_decay_extend = _decay_p_function(S, p_rel[ss_scale][1],
                                            dist_since_lf_extend)
                pcen_extend = p_landfall * p_decay_extend

    # correct limits
    np.warnings.filterwarnings('ignore')
    cor_p = track_chunk.central_pressure.values > track_chunk.environmental_pressure.values
    track_chunk.central_pressure[cor_p] = track_chunk.environmental_pressure[cor_p]
    track_chunk.max_sustained_wind[track_chunk.max_sustained_wind < 0] = 0

    # assign output
    track['central_pressure'][in_chunk] = track_chunk['central_pressure'][:]
    return pcen_extend, v_extend

def _estimate_params_track(track):
    """Estimate a synthetic track's parameters from central pressure based
    on relationships fitted on that track's historical values.

    # TODO review docstring
    The input 'track' is modified in place!

    The variables estimated from central pressure are 'max_sustained_wind',
    'radius_max_wind 'radius_oci'. The track is split into 3 phases:
    intensification (up to maximum intensity +/-5 mbar), peak (set of
    subsequent values around the minimum central pressure value, all within
    5 mbar of that minimum value), and decay (thereafter).

    Parameters
    ----------
    track : xarray.Datasset
        Track data.
    """
    if np.all(track.id_chunk.values == 0):
        return
    pcen = track.central_pressure.values
    def _get_peak_idx(pcen):
        peak_idx = np.where(pcen == pcen.min())[0]
        if np.all(pcen - pcen.min() <= 5):
            return 0, len(pcen) - 1
        peak_idx = peak_idx[0]
        # detect end of peak
        if peak_idx < len(pcen) - 1:
            idx_out = np.where(np.diff((pcen[peak_idx:] - pcen.min() <= 5).astype(int)) == -1)[0]
            if idx_out.shape[0] == 0:
                peak_end_idx = len(pcen) - 1
            else:
                peak_end_idx = peak_idx + idx_out[0]
        else:
            peak_end_idx = peak_idx
        # detect start of peak
        if peak_idx > 0:
            idx_out = np.where(np.diff((np.flip(pcen[:peak_idx+1]) - pcen.min() <= 5).astype(int)) == -1)[0]
            if idx_out.shape[0] == 0:
                peak_start_idx = 0
            else:
                peak_start_idx = peak_idx - [0]
        else:
            peak_start_idx = 0
        return peak_start_idx, peak_end_idx

    # identify phases
    peak_start_idx, peak_end_idx = _get_peak_idx(pcen)
    # data point from which to re-estimate variables
    first_idx = np.where(track.id_chunk.values != 0)[0][0]
    interp_idx = None
    if first_idx > 0:
        # how many time steps to go back
        time_step_h = np.unique(track['time_step'].values)[0]
        nb_timesteps_adjust = np.ceil(FIT_TIME_ADJUST_HOUR/time_step_h).astype(int)
        # where to adjust previous time steps
        if nb_timesteps_adjust > 0:
            # copy original values
            track_orig = track.copy(deep = True)
            # indices where to interpolate between original and estimated values
            interp_idx = np.arange(max(0, first_idx - nb_timesteps_adjust), first_idx)
        # where to estimate each chunk: not before first_idx - nb_timesteps_adjust
        intens_idx = np.arange(max(0, first_idx - nb_timesteps_adjust), peak_start_idx)
        peak_idx = np.arange(max(peak_start_idx, first_idx - nb_timesteps_adjust), peak_end_idx + 1)
        decay_idx = np.arange(max(peak_end_idx + 1, first_idx - nb_timesteps_adjust), len(pcen))
    else:
        intens_idx = np.arange(peak_start_idx)
        peak_idx = np.arange(peak_start_idx, peak_end_idx + 1)
        decay_idx = np.arange(peak_end_idx + 1, len(pcen))

    # apply adjustments
    if len(intens_idx) > 0:
        _estimate_vars_chunk(track, 'intens', intens_idx)
    if len(decay_idx) > 0:
        _estimate_vars_chunk(track, 'decay', decay_idx)
    if len(peak_idx) > 0:
        _estimate_vars_chunk(track, 'peak', peak_idx)

    # mediate adjustments
    if interp_idx is not None:
        # interpolate between new and old values
        weights_idx = (np.arange(len(interp_idx)) + 1) / (len(interp_idx) + 1)
        for var in FIT_TRACK_VARS_RANGE.keys():
            track[var][interp_idx] = weights_idx * track[var][interp_idx] + (1-weights_idx) * track_orig[var][interp_idx]

def _get_outside_lat_idx(track):
    """Get the index of the first track point located too poleward or equatorward
    
    A track point with unrealistically high wind speed for its latitude is
    considered to be too poleward or too equatorward.

    Parameters
    ----------
    track : xr.Dataset
        TC track

    Returns
    -------
    latrange_out_idx : int
        Index of the first point with unrealistic wind speed for its latitude.
        If no such point if found in the track, set to track.time.size.
    """
    mid_basin = track.basin.values[int(np.floor(track.basin.size/2))]
    latrange_out_idx = track.time.size
    for lat_max,max_wind in zip(
        MAX_WIND_BY_LAT_RANGE[mid_basin]['lat_max']['lat'],
        MAX_WIND_BY_LAT_RANGE[mid_basin]['lat_max']['max_wind']
    ):
        outside_range = np.logical_and(
            track.lat.values > lat_max,
            track.max_sustained_wind.values > 1.1*max_wind
        )
        if np.sum(outside_range) > 3:
            latrange_out_idx = min(np.where(outside_range)[0][0], latrange_out_idx)
    for lat_min,max_wind in zip(
        MAX_WIND_BY_LAT_RANGE[mid_basin]['lat_min']['lat'],
        MAX_WIND_BY_LAT_RANGE[mid_basin]['lat_min']['max_wind']
    ):
        outside_range = np.logical_and(
            track.lat.values < lat_min,
            track.max_sustained_wind.values > 1.1*max_wind
        )
        if np.sum(outside_range) > 3:
            latrange_out_idx = min(np.where(outside_range)[0][0], latrange_out_idx)
    return latrange_out_idx

def _get_finalintensity_idx(track, original_track):
    '''Get the index of the frame where the synthetic track's central 
    pressure first goes above source track's final central pressure (or 
    the source track's maximum pressure reacheed after its minimum pressure, 
    whichever is higher). This is usually the last frame.'''
    original_pres = original_track.central_pressure.values
    min_idx = np.where(original_pres == np.min(original_pres))[0][-1]
    max_pres = np.max(original_pres[min_idx:])

    idx = track.central_pressure.values[min_idx:] >= max_pres + 2
    return min_idx + np.where(idx)[0][0] if idx.any() else track.time.size


def _apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel):
    """Change track's max sustained wind and central pressure using the land
    decay coefficients.

    Parameters
    ----------
    track : xr.Dataset
        TC track

    Returns
    -------
    dist : np.array
        Distances in km, points on water get nan values.
    """
    # pylint: disable=protected-access
    # return if historical track
    if track.orig_event_flag:
        return track

    climada.hazard.tc_tracks.track_land_params(track, land_geom)
    sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
    if not sea_land_idx.size:
        return track
    for idx, (sea_land, land_sea) \
            in enumerate(zip(sea_land_idx, land_sea_idx)):
        v_landfall = track.max_sustained_wind[sea_land - 1].values
        p_landfall = float(track.central_pressure[sea_land - 1].values)
        ss_scale = climada.hazard.tc_tracks.set_category(v_landfall,
                                                         track.max_sustained_wind_unit)
        if land_sea - sea_land == 1:
            continue
        S = _calc_decay_ps_value(track, p_landfall, land_sea - 1, s_rel)
        if S <= 1:
            # central_pressure at start of landfall > env_pres after landfall:
            # set central_pressure to environmental pressure during whole lf
            track.central_pressure[sea_land:land_sea] = track.environmental_pressure[sea_land:land_sea]
        else:
            p_decay = _decay_p_function(S, p_rel[ss_scale][1],
                                        track.dist_since_lf[sea_land:land_sea].values)
            # dont apply decay if it would decrease central pressure
            if np.any(p_decay < 1):
                LOGGER.info('Landfall decay would decrease pressure for '
                            'track id %s, leading to an intensification '
                            'of the Tropical Cyclone. This behaviour is '
                            'unphysical and therefore landfall decay is not '
                            'applied in this case.',
                            track.sid)
                p_decay[p_decay < 1] = (track.central_pressure[sea_land:land_sea][p_decay < 1]
                                        / p_landfall)
            track.central_pressure[sea_land:land_sea] = p_landfall * p_decay

        v_decay = _decay_v_function(v_rel[ss_scale],
                                    track.dist_since_lf[sea_land:land_sea].values)
        # dont apply decay if it would increase wind speeds
        if np.any(v_decay > 1):
            # should not happen unless v_rel is negative
            LOGGER.info('Landfall decay would increase wind speed for '
                        'track id %s. This behaviour is unphysical and '
                        'therefore landfall decay is not applied in this '
                        'case.',
                        track.sid)
            v_decay[v_decay > 1] = (track.max_sustained_wind[sea_land:land_sea][v_decay > 1]
                                    / v_landfall)
        track.max_sustained_wind[sea_land:land_sea] = v_landfall * v_decay

        # correct values of sea after a landfall (until next landfall, if any)
        if land_sea < track.time.size:
            if idx + 1 < sea_land_idx.size:
                # if there is a next landfall, correct until last point before
                # reaching land again
                end_cor = sea_land_idx[idx]
            else:
                # if there is no further landfall, correct until the end of
                # the track
                end_cor = track.time.size
            rndn = 0.1 * float(np.abs(np.random.normal(size=1) * 5) + 6)
            r_diff = track.central_pressure[land_sea].values - \
                     track.central_pressure[land_sea - 1].values + rndn
            track.central_pressure[land_sea:end_cor] += - r_diff

            rndn = rndn * 10  # mean value 10
            r_diff = track.max_sustained_wind[land_sea].values - \
                     track.max_sustained_wind[land_sea - 1].values - rndn
            track.max_sustained_wind[land_sea:end_cor] += - r_diff

        # correct limits
        warnings.filterwarnings('ignore')
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
    # pylint: disable=protected-access
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
        sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
        if sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                v_lf = track.max_sustained_wind[sea_land - 1].values
                p_lf = track.central_pressure[sea_land - 1].values
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                ss_scale_idx = np.where(v_lf < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_v_a.plot(on_land, track.max_sustained_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_v_b.plot(on_land, orig_wind[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_p_a.plot(on_land, track.central_pressure[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_p_b.plot(on_land, orig_pres[on_land],
                               'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_pd_a.plot(track.dist_since_lf[on_land],
                                track.central_pressure[on_land] / p_lf,
                                'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_ped_a.plot(track.dist_since_lf[on_land],
                                 track.environmental_pressure[on_land] -
                                 track.central_pressure[on_land],
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])

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
    # pylint: disable=protected-access
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
        sea_land_idx, land_sea_idx = climada.hazard.tc_tracks._get_landfall_idx(track)
        if sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                scale_thresholds = climada.hazard.tc_tracks.SAFFIR_SIM_CAT
                ss_scale_idx = np.where(track.max_sustained_wind[sea_land - 1].values
                                 < scale_thresholds)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_hv.add_curve(on_land, track.max_sustained_wind[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hp.add_curve(on_land, track.central_pressure[on_land],
                                   'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hpd_a.plot(track.dist_since_lf[on_land],
                                 track.central_pressure[on_land]
                                 / track.central_pressure[sea_land - 1].values,
                                 'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])
                graph_hped_a.plot(track.dist_since_lf[on_land],
                                  track.environmental_pressure[on_land] -
                                  track.central_pressure[on_land],
                                  'o', c=climada.hazard.tc_tracks.CAT_COLORS[ss_scale_idx])

            on_sea = np.arange(track.time.size)[~track.on_land]
            graph_hp.plot(on_sea, track.central_pressure[on_sea],
                          'o', c='k', markersize=5)
            graph_hv.plot(on_sea, track.max_sustained_wind[on_sea],
                          'o', c='k', markersize=5)

    return graph_hv, graph_hp, graph_hpd_a, graph_hped_a
