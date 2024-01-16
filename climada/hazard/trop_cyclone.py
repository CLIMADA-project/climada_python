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

Define TC wind hazard (TropCyclone class).
"""

__all__ = ['TropCyclone']

import copy
import datetime as dt
import itertools
import logging
import time
from typing import Optional, Tuple, List, Union

import numpy as np
from scipy import sparse
import matplotlib.animation as animation
from tqdm import tqdm
import pathos.pools
import xarray as xr

from climada.hazard.base import Hazard
from climada.hazard.tc_tracks import TCTracks, estimate_rmw
from climada.hazard.tc_clim_change import get_knutson_criterion, calc_scale_knutson
from climada.hazard.centroids.centr import Centroids
from climada.util import ureg
import climada.util.constants as u_const
import climada.util.coordinates as u_coord
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
"""Hazard type acronym for Tropical Cyclone"""

DEF_MAX_DIST_EYE_KM = 300
"""Default value for the maximum distance (in km) of a centroid to the TC center at which wind
speed calculations are done."""

DEF_INTENSITY_THRES = 17.5
"""Default value for the threshold below which wind speeds (in m/s) are stored as 0."""

DEF_MAX_MEMORY_GB = 8
"""Default value of the memory limit (in GB) for windfield computations (in each thread)."""

MODEL_VANG = {'H08': 0, 'H1980': 1, 'H10': 2, 'ER11': 3}
"""Enumerate different symmetric wind field models."""

RHO_AIR = 1.15
"""Air density. Assumed constant, following Holland 1980."""

GRADIENT_LEVEL_TO_SURFACE_WINDS = 0.9
"""Gradient-to-surface wind reduction factor according to the 90%-rule:

Franklin, J.L., Black, M.L., Valde, K. (2003): GPS Dropwindsonde Wind Profiles in Hurricanes and
Their Operational Implications. Weather and Forecasting 18(1): 32–44.
https://doi.org/10.1175/1520-0434(2003)018<0032:GDWPIH>2.0.CO;2
"""

KMH_TO_MS = (1.0 * ureg.km / ureg.hour).to(ureg.meter / ureg.second).magnitude
KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
KM_TO_M = (1.0 * ureg.kilometer).to(ureg.meter).magnitude
H_TO_S = (1.0 * ureg.hours).to(ureg.seconds).magnitude
MBAR_TO_PA = (1.0 * ureg.millibar).to(ureg.pascal).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

V_ANG_EARTH = 7.29e-5
"""Earth angular velocity (in radians per second)"""

class TropCyclone(Hazard):
    """
    Contains tropical cyclone events.

    Attributes
    ----------
    category : np.ndarray of ints
        for every event, the TC category using the Saffir-Simpson scale:

        * -1 tropical depression
        *  0 tropical storm
        *  1 Hurrican category 1
        *  2 Hurrican category 2
        *  3 Hurrican category 3
        *  4 Hurrican category 4
        *  5 Hurrican category 5
    basin : list of str
        Basin where every event starts:

        * 'NA' North Atlantic
        * 'EP' Eastern North Pacific
        * 'WP' Western North Pacific
        * 'NI' North Indian
        * 'SI' South Indian
        * 'SP' Southern Pacific
        * 'SA' South Atlantic
    windfields : list of csr_matrix
        For each event, the full velocity vectors at each centroid and track position in a sparse
        matrix of shape (npositions, ncentroids * 2) that can be reshaped to a full ndarray of
        shape (npositions, ncentroids, 2).
    """
    intensity_thres = DEF_INTENSITY_THRES
    """intensity threshold for storage in m/s"""

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(
        self,
        category: Optional[np.ndarray] = None,
        basin: Optional[List] = None,
        windfields: Optional[List[sparse.csr_matrix]] = None,
        **kwargs,
    ):
        """Initialize values.

        Parameters
        ----------
        category : np.ndarray of int, optional
            For every event, the TC category using the Saffir-Simpson scale:
                -1 tropical depression
                0 tropical storm
                1 Hurrican category 1
                2 Hurrican category 2
                3 Hurrican category 3
                4 Hurrican category 4
                5 Hurrican category 5
        basin : list of str, optional
            Basin where every event starts:
                'NA' North Atlantic
                'EP' Eastern North Pacific
                'WP' Western North Pacific
                'NI' North Indian
                'SI' South Indian
                'SP' Southern Pacific
                'SA' South Atlantic
        windfields : list of csr_matrix, optional
            For each event, the full velocity vectors at each centroid and track position in a
            sparse matrix of shape (npositions,  ncentroids * 2) that can be reshaped to a full
            ndarray of shape (npositions, ncentroids, 2).
        **kwargs : Hazard properties, optional
            All other keyword arguments are passed to the Hazard constructor.
        """
        kwargs.setdefault('haz_type', HAZ_TYPE)
        Hazard.__init__(self, **kwargs)
        self.category = category if category is not None else np.array([], int)
        self.basin = basin if basin is not None else []
        self.windfields = windfields if windfields is not None else []

    def set_from_tracks(self, *args, **kwargs):
        """This function is deprecated, use TropCyclone.from_tracks instead."""
        LOGGER.warning("The use of TropCyclone.set_from_tracks is deprecated."
                       "Use TropCyclone.from_tracks instead.")
        if "intensity_thres" not in kwargs:
            # some users modify the threshold attribute before calling `set_from_tracks`
            kwargs["intensity_thres"] = self.intensity_thres
        if self.pool is not None and 'pool' not in kwargs:
            kwargs['pool'] = self.pool
        self.__dict__ = TropCyclone.from_tracks(*args, **kwargs).__dict__

    @classmethod
    def from_tracks(
        cls,
        tracks: TCTracks,
        centroids: Optional[Centroids] = None,
        pool: Optional[pathos.pools.ProcessPool] = None,
        model: str = 'H08',
        ignore_distance_to_coast: bool = False,
        store_windfields: bool = False,
        metric: str = "equirect",
        intensity_thres: float = DEF_INTENSITY_THRES,
        max_latitude: float = 61,
        max_dist_inland_km: float = 1000,
        max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
        max_memory_gb: float = DEF_MAX_MEMORY_GB,
    ):
        """
        Create new TropCyclone instance that contains windfields from the specified tracks.

        This function sets the `intensity` attribute to contain, for each centroid,
        the maximum wind speed (1-minute sustained winds at 10 meters above ground) experienced
        over the whole period of each TC event in m/s. The wind speed is set to 0 if it doesn't
        exceed the threshold `intensity_thres`.

        The `category` attribute is set to the value of the `category`-attribute
        of each of the given track data sets.

        The `basin` attribute is set to the genesis basin for each event, which
        is the first value of the `basin`-variable in each of the given track data sets.

        Optionally, the time dependent, vectorial winds can be stored using the `store_windfields`
        function parameter (see below).

        Parameters
        ----------
        tracks : climada.hazard.TCTracks
            Tracks of storm events.
        centroids : Centroids, optional
            Centroids where to model TC. Default: global centroids at 360 arc-seconds resolution.
        pool : pathos.pool, optional
            Pool that will be used for parallel computation of wind fields. Default: None
        description : str, optional
            Description of the event set. Default: "".
        model : str, optional
            Parametric wind field model to use: one of "H1980" (the prominent Holland 1980 model),
            "H08" (Holland 1980 with b-value from Holland 2008), "H10" (Holland et al. 2010), or
            "ER11" (Emanuel and Rotunno 2011).
            Default: "H08".
        ignore_distance_to_coast : boolean, optional
            If True, centroids far from coast are not ignored. Default: False.
        store_windfields : boolean, optional
            If True, the Hazard object gets a list `windfields` of sparse matrices. For each track,
            the full velocity vectors at each centroid and track position are stored in a sparse
            matrix of shape (npositions,  ncentroids * 2) that can be reshaped to a full ndarray
            of shape (npositions, ncentroids, 2). Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances:

            * "equirect": Distance according to sinusoidal projection. Fast, but inaccurate for
              large distances and high latitudes.
            * "geosphere": Exact spherical distance. Much more accurate at all distances, but slow.

            Default: "equirect".
        intensity_thres : float, optional
            Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
        max_latitude : float, optional
            No wind speed calculation is done for centroids with latitude larger than this
            parameter. Default: 61
        max_dist_inland_km : float, optional
            No wind speed calculation is done for centroids with a distance (in km) to the coast
            larger than this parameter. Default: 1000
        max_dist_eye_km : float, optional
            No wind speed calculation is done for centroids with a distance (in km) to the TC
            center ("eye") larger than this parameter. Default: 300
        max_memory_gb : float, optional
            To avoid memory issues, the computation is done for chunks of the track sequentially.
            The chunk size is determined depending on the available memory (in GB). Note that this
            limit applies to each thread separately if a `pool` is used. Default: 8

        Raises
        ------
        ValueError

        Returns
        -------
        TropCyclone
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids.from_base_grid(res_as=360, land=False)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        if ignore_distance_to_coast:
            # Select centroids with lat <= max_latitude
            coastal_idx = (np.abs(centroids.lat) <= max_latitude).nonzero()[0]
        else:
            # Select centroids which are inside max_dist_inland_km and lat <= max_latitude
            if not centroids.dist_coast.size:
                centroids.set_dist_coast()
            coastal_idx = ((centroids.dist_coast <= max_dist_inland_km * 1000)
                           & (np.abs(centroids.lat) <= max_latitude)).nonzero()[0]

        # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
        # Later, there will be another filtering step with more accurate distances in km.
        max_dist_eye_deg = max_dist_eye_km / (
            u_const.ONE_LAT_KM * np.cos(np.radians(max_latitude))
        )

        # Restrict to coastal centroids within reach of any of the tracks
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tracks.get_bounds(deg_buffer=max_dist_eye_deg)
        t_mid_lon = 0.5 * (t_lon_min + t_lon_max)
        coastal_centroids = centroids.coord[coastal_idx]
        u_coord.lon_normalize(coastal_centroids[:, 1], center=t_mid_lon)
        coastal_idx = coastal_idx[((t_lon_min <= coastal_centroids[:, 1])
                                   & (coastal_centroids[:, 1] <= t_lon_max)
                                   & (t_lat_min <= coastal_centroids[:, 0])
                                   & (coastal_centroids[:, 0] <= t_lat_max))]

        LOGGER.info('Mapping %s tracks to %s coastal centroids.', str(tracks.size),
                    str(coastal_idx.size))
        if pool:
            chunksize = max(min(num_tracks // pool.ncpus, 1000), 1)
            tc_haz_list = pool.map(
                cls.from_single_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(coastal_idx, num_tracks),
                itertools.repeat(model, num_tracks),
                itertools.repeat(store_windfields, num_tracks),
                itertools.repeat(metric, num_tracks),
                itertools.repeat(intensity_thres, num_tracks),
                itertools.repeat(max_dist_eye_km, num_tracks),
                itertools.repeat(max_memory_gb, num_tracks),
                chunksize=chunksize)
        else:
            last_perc = 0
            tc_haz_list = []
            for track in tracks.data:
                perc = 100 * len(tc_haz_list) / len(tracks.data)
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                tc_haz_list.append(
                    cls.from_single_track(track, centroids, coastal_idx,
                                          model=model, store_windfields=store_windfields,
                                          metric=metric, intensity_thres=intensity_thres,
                                          max_dist_eye_km=max_dist_eye_km,
                                          max_memory_gb=max_memory_gb))
            if last_perc < 100:
                LOGGER.info("Progress: 100%")

        LOGGER.debug('Concatenate events.')
        haz = cls.concat(tc_haz_list)
        haz.pool = pool
        haz.intensity_thres = intensity_thres
        LOGGER.debug('Compute frequency.')
        haz.frequency_from_tracks(tracks.data)
        return haz

    def apply_climate_scenario_knu(
        self,
        ref_year: int = 2050,
        rcp_scenario: int = 45
    ):
        """
        From current TC hazard instance, return new hazard set with
        future events for a given RCP scenario and year based on the
        parametrized values derived from Table 3 in Knutson et al 2015.
        https://doi.org/10.1175/JCLI-D-15-0129.1 . The scaling for different
        years and RCP scenarios is obtained by linear interpolation.

        Note: The parametrized values are derived from the overall changes
        in statistical ensemble of tracks. Hence, this method should only be
        applied to sufficiently large tropical cyclone event sets that
        approximate the reference years 1981 - 2008 used in Knutson et. al.

        The frequency and intensity changes are applied independently from
        one another. The mean intensity factors can thus slightly deviate
        from the Knutson value (deviation was found to be less than 1%
        for default IBTrACS event sets 1980-2020 for each basin).

        Parameters
        ----------
        ref_year : int
            year between 2000 ad 2100. Default: 2050
        rcp_scenario : int
            26 for RCP 2.6, 45 for RCP 4.5, 60 for RCP 6.0 and 85 for RCP 8.5.
            The default is 45.

        Returns
        -------
        haz_cc : climada.hazard.TropCyclone
            Tropical cyclone with frequencies and intensity scaled according
            to the Knutson criterion for the given year and RCP. Returns
            a new instance of climada.hazard.TropCyclone, self is not
            modified.
        """
        chg_int_freq = get_knutson_criterion()
        scale_rcp_year  = calc_scale_knutson(ref_year, rcp_scenario)
        haz_cc = self._apply_knutson_criterion(chg_int_freq, scale_rcp_year)
        return haz_cc

    def set_climate_scenario_knu(self, *args, **kwargs):
        """This function is deprecated, use TropCyclone.apply_climate_scenario_knu instead."""
        LOGGER.warning("The use of TropCyclone.set_climate_scenario_knu is deprecated."
                       "Use TropCyclone.apply_climate_scenario_knu instead.")
        return self.apply_climate_scenario_knu(*args, **kwargs)

    @classmethod
    def video_intensity(
        cls,
        track_name: str,
        tracks: TCTracks,
        centroids: Centroids,
        file_name: Optional[str] = None,
        writer: animation = animation.PillowWriter(bitrate=500),
        figsize: Tuple[float, float] = (9, 13),
        adapt_fontsize: bool = True,
        **kwargs
    ):
        """
        Generate video of TC wind fields node by node and returns its
        corresponding TropCyclone instances and track pieces.

        Parameters
        ----------
        track_name : str
            name of the track contained in tracks to record
        tracks : climada.hazard.TCTracks
            tropical cyclone tracks
        centroids : climada.hazard.Centroids
            centroids where wind fields are mapped
        file_name : str, optional
            file name to save video (including full path and file extension)
        writer : matplotlib.animation.*, optional
            video writer. Default is pillow with bitrate=500
        figsize : tuple, optional
            figure size for plt.subplots
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.
        kwargs : optional
            arguments for pcolormesh matplotlib function used in event plots

        Returns
        -------
        tc_list, tc_coord : list(TropCyclone), list(np.ndarray)

        Raises
        ------
        ValueError

        """
        # initialization
        track = tracks.get_track(track_name)
        if not track:
            raise ValueError(f'{track_name} not found in track data.')
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

            tc_tmp = cls.from_tracks(tr_sel, centroids=centroids)
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
            fig, axis, _fontsize = u_plot.make_map(figsize=figsize, adapt_fontsize=adapt_fontsize)
            pbar = tqdm(total=idx_plt.size - 2)
            ani = animation.FuncAnimation(fig, run, frames=idx_plt.size - 2,
                                          interval=500, blit=False)
            fig.tight_layout()
            ani.save(file_name, writer=writer)
            pbar.close()
        return tc_list, tr_coord

    def frequency_from_tracks(self, tracks: List):
        """
        Set hazard frequency from tracks data.

        Parameters
        ----------
        tracks : list of xarray.Dataset
        """
        if not tracks:
            return
        year_max = np.amax([t.time.dt.year.values.max() for t in tracks])
        year_min = np.amin([t.time.dt.year.values.min() for t in tracks])
        year_delta = year_max - year_min + 1
        num_orig = np.count_nonzero(self.orig)
        ens_size = (self.event_id.size / num_orig) if num_orig > 0 else 1
        self.frequency = np.ones(self.event_id.size) / (year_delta * ens_size)

    @classmethod
    def from_single_track(
        cls,
        track: xr.Dataset,
        centroids: Centroids,
        coastal_idx: np.ndarray,
        model: str = 'H08',
        store_windfields: bool = False,
        metric: str = "equirect",
        intensity_thres: float = DEF_INTENSITY_THRES,
        max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
        max_memory_gb: float = DEF_MAX_MEMORY_GB,
    ):
        """
        Generate windfield hazard from a single track dataset

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : Centroids
            Centroids instance.
        coastal_idx : np.ndarray
            Indices of centroids close to coast.
        model : str, optional
            Parametric wind field model, one of "H1980" (the prominent Holland 1980 model),
            "H08" (Holland 1980 with b-value from Holland 2008), "H10" (Holland et al. 2010), or
            "ER11" (Emanuel and Rotunno 2011).
            Default: "H08".
        store_windfields : boolean, optional
            If True, store windfields. Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances: "equirect" (faster) or
            "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
            Default: "equirect".
        intensity_thres : float, optional
            Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
        max_dist_eye_km : float, optional
            No wind speed calculation is done for centroids with a distance (in km) to the TC
            center ("eye") larger than this parameter. Default: 300
        max_memory_gb : float, optional
            To avoid memory issues, the computation is done for chunks of the track sequentially.
            The chunk size is determined depending on the available memory (in GB). Default: 8

        Raises
        ------
        ValueError, KeyError

        Returns
        -------
        haz : TropCyclone
        """
        intensity_sparse, windfields_sparse = _compute_windfields_sparse(
            track=track,
            centroids=centroids,
            coastal_idx=coastal_idx,
            model=model,
            store_windfields=store_windfields,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

        new_haz = cls(haz_type=HAZ_TYPE)
        new_haz.intensity_thres = intensity_thres
        new_haz.intensity = intensity_sparse
        if store_windfields:
            new_haz.windfields = [windfields_sparse]
        new_haz.units = 'm/s'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = sparse.csr_matrix(new_haz.intensity.shape)
        # store first day of track as date
        new_haz.date = np.array([
            dt.datetime(track.time.dt.year.values[0],
                        track.time.dt.month.values[0],
                        track.time.dt.day.values[0]).toordinal()
        ])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        # users that pickle TCTracks objects might still have data with the legacy basin attribute,
        # so we have to deal with it here
        new_haz.basin = [track.basin if isinstance(track.basin, str)
                         else str(track.basin.values[0])]
        return new_haz

    def _apply_knutson_criterion(
        self,
        chg_int_freq: List,
        scaling_rcp_year: float
    ):
        """
        Apply changes to intensities and cumulative frequencies.

        Parameters
        ----------
        chg_int_freq : list(dict))
            list of criteria from climada.hazard.tc_clim_change
        scaling_rcp_year : float
            scale parameter because of chosen year and RCP

        Returns
        -------
        tc_cc : climada.hazard.TropCyclone
            Tropical cyclone with frequency and intensity scaled inspired by
            the Knutson criterion. Returns a new instance of TropCyclone.
        """

        tc_cc = copy.deepcopy(self)

        # Criterion per basin
        for basin in np.unique(tc_cc.basin):

            bas_sel = np.array(tc_cc.basin) == basin

            # Apply intensity change
            inten_chg = [chg
                         for chg in chg_int_freq
                         if (chg['variable'] == 'intensity' and
                             chg['basin'] == basin)
                         ]
            for chg in inten_chg:
                sel_cat_chg = np.isin(tc_cc.category, chg['category']) & bas_sel
                inten_scaling = 1 + (chg['change'] - 1) * scaling_rcp_year
                tc_cc.intensity = sparse.diags(
                    np.where(sel_cat_chg, inten_scaling, 1)
                    ).dot(tc_cc.intensity)

            # Apply frequency change
            freq_chg = [chg
                        for chg in chg_int_freq
                        if (chg['variable'] == 'frequency' and
                            chg['basin'] == basin)
                        ]
            freq_chg.sort(reverse=False, key=lambda x: len(x['category']))

            # Scale frequencies by category
            cat_larger_list = []
            for chg in freq_chg:
                cat_chg_list = [cat
                                for cat in chg['category']
                                if cat not in cat_larger_list
                                ]
                sel_cat_chg = np.isin(tc_cc.category, cat_chg_list) & bas_sel
                if sel_cat_chg.any():
                    freq_scaling = 1 + (chg['change'] - 1) * scaling_rcp_year
                    tc_cc.frequency[sel_cat_chg] *= freq_scaling
                cat_larger_list += cat_chg_list

        if (tc_cc.frequency < 0).any():
            raise ValueError("The application of the given climate scenario"
                             "resulted in at least one negative frequency.")

        return tc_cc

def _compute_windfields_sparse(
    track: xr.Dataset,
    centroids: Centroids,
    coastal_idx: np.ndarray,
    model: str = 'H08',
    store_windfields: bool = False,
    metric: str = "equirect",
    intensity_thres: float = DEF_INTENSITY_THRES,
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Version of `compute_windfields` that returns sparse matrices and limits memory usage

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : Centroids
        Centroids instance.
    coastal_idx : np.ndarray
        Indices of centroids close to coast.
    model : str, optional
        Parametric wind field model, one of "H1980" (the prominent Holland 1980 model),
        "H08" (Holland 1980 with b-value from Holland 2008), "H10" (Holland et al. 2010), or
        "ER11" (Emanuel and Rotunno 2011).
        Default: "H08".
    store_windfields : boolean, optional
        If True, store windfields. Default: False.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    intensity_thres : float, optional
        Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
    max_dist_eye_km : float, optional
        No wind speed calculation is done for centroids with a distance (in km) to the TC
        center ("eye") larger than this parameter. Default: 300
    max_memory_gb : float, optional
        To avoid memory issues, the computation is done for chunks of the track sequentially.
        The chunk size is determined depending on the available memory (in GB). Default: 8

    Raises
    ------
    ValueError

    Returns
    -------
    intensity : csr_matrix
        Maximum wind speed in each centroid over the whole storm life time.
    windfields : csr_matrix or None
        If store_windfields is True, the full velocity vectors at each centroid and track position
        are stored in a sparse matrix of shape (npositions,  ncentroids * 2) that can be reshaped
        to a full ndarray of shape (npositions, ncentroids, 2).
        If store_windfields is False, `None` is returned.
    """
    try:
        mod_id = MODEL_VANG[model]
    except KeyError as err:
        raise ValueError(f'Model not implemented: {model}.') from err

    ncentroids = centroids.coord.shape[0]
    coastal_centr = centroids.coord[coastal_idx]
    npositions = track.sizes["time"]
    windfields_shape = (npositions, ncentroids * 2)
    intensity_shape = (1, ncentroids)

    # start with the assumption that no centroids are within reach
    windfields_sparse = (
        sparse.csr_matrix(([], ([], [])), shape=windfields_shape)
        if store_windfields else None
    )
    intensity_sparse = sparse.csr_matrix(([], ([], [])), shape=intensity_shape)

    # The wind field model requires at least two track positions because translational speed
    # as well as the change in pressure (in case of H08) are required.
    if npositions < 2:
        return intensity_sparse, windfields_sparse

    # convert track variables to SI units
    si_track = tctrack_to_si(track, metric=metric)
    t_lat, t_lon = si_track["lat"].values, si_track["lon"].values

    # normalize longitudinal coordinates of centroids
    u_coord.lon_normalize(coastal_centr[:, 1], center=si_track.attrs["mid_lon"])

    # Restrict to the bounding box of the whole track first (this can already reduce the number of
    # centroids that are considered by a factor larger than 30).
    max_dist_eye_lat = max_dist_eye_km / u_const.ONE_LAT_KM
    max_dist_eye_lon = max_dist_eye_km / (
        u_const.ONE_LAT_KM * np.cos(np.radians(np.abs(coastal_centr[:, 0]) + max_dist_eye_lat))
    )
    coastal_idx = coastal_idx[
        (t_lat.min() - coastal_centr[:, 0] <= max_dist_eye_lat)
        & (coastal_centr[:, 0] - t_lat.max() <= max_dist_eye_lat)
        & (t_lon.min() - coastal_centr[:, 1] <= max_dist_eye_lon)
        & (coastal_centr[:, 1] - t_lon.max() <= max_dist_eye_lon)
    ]
    coastal_centr = centroids.coord[coastal_idx]

    # After the previous filtering step, finding and storing the reachable centroids is not a
    # memory bottle neck and can be done before chunking.
    track_centr_msk = get_close_centroids(
        t_lat, t_lon, coastal_centr, max_dist_eye_km, metric=metric,
    )
    coastal_idx = coastal_idx[track_centr_msk.any(axis=0)]
    coastal_centr = centroids.coord[coastal_idx]
    nreachable = coastal_centr.shape[0]
    if nreachable == 0:
        return intensity_sparse, windfields_sparse

    # the total memory requirement in GB if we compute everything without chunking:
    # 8 Bytes per entry (float64), 10 arrays
    total_memory_gb = npositions * nreachable * 8 * 10 / 1e9
    if total_memory_gb > max_memory_gb and npositions > 2:
        # If the number of positions is down to 2 already, we cannot split any further. In that
        # case, we just take the risk and try to do the computation anyway. It might still work
        # since we have only computed an upper bound for the number of affected centroids.

        # Split the track into chunks, compute the result for each chunk, and combine:
        return _compute_windfields_sparse_chunked(
            track_centr_msk,
            track,
            centroids,
            coastal_idx,
            model=model,
            store_windfields=store_windfields,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

    windfields, reachable_centr_idx = _compute_windfields(
        si_track, coastal_centr, mod_id, metric=metric, max_dist_eye_km=max_dist_eye_km,
    )
    reachable_coastal_centr_idx = coastal_idx[reachable_centr_idx]
    npositions = windfields.shape[0]

    intensity = np.linalg.norm(windfields, axis=-1).max(axis=0)
    intensity[intensity < intensity_thres] = 0
    intensity_sparse = sparse.csr_matrix(
        (intensity, reachable_coastal_centr_idx, [0, intensity.size]),
        shape=intensity_shape)
    intensity_sparse.eliminate_zeros()

    windfields_sparse = None
    if store_windfields:
        n_reachable_coastal_centr = reachable_coastal_centr_idx.size
        indices = np.zeros((npositions, n_reachable_coastal_centr, 2), dtype=np.int64)
        indices[:, :, 0] = 2 * reachable_coastal_centr_idx[None]
        indices[:, :, 1] = 2 * reachable_coastal_centr_idx[None] + 1
        indices = indices.ravel()
        indptr = np.arange(npositions + 1) * n_reachable_coastal_centr * 2
        windfields_sparse = sparse.csr_matrix((windfields.ravel(), indices, indptr),
                                              shape=windfields_shape)
        windfields_sparse.eliminate_zeros()

    return intensity_sparse, windfields_sparse

def _compute_windfields_sparse_chunked(
    track_centr_msk: np.ndarray,
    track: xr.Dataset,
    *args,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
    **kwargs,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Call `_compute_windfields_sparse` for chunks of the track and re-assemble the results

    Parameters
    ----------
    track_centr_msk : np.ndarray
        Each row is a mask that indicates the centroids within reach for one track position.
    track : xr.Dataset
        Single tropical cyclone track.
    max_memory_gb : float, optional
        Maximum memory requirements (in GB) for the computation of a single chunk of the track.
        Default: 8
    args, kwargs :
        The remaining arguments are passed on to `_compute_windfields_sparse`.

    Returns
    -------
    intensity, windfields :
        See `_compute_windfields_sparse` for a description of the return values.
    """
    npositions = track.sizes["time"]
    # The memory requirements for each track position are estimated for the case of 10 arrays
    # containing `nreachable` float64 (8 Byte) values each. The chunking is only relevant in
    # extreme cases with a very high temporal and/or spatial resolution.
    max_nreachable = max_memory_gb * 1e9 / (8 * 10 * npositions)
    split_pos = [0]
    chunk_size = 2
    while split_pos[-1] + chunk_size < npositions:
        chunk_size += 1
        # create overlap between consecutive chunks
        chunk_start = max(0, split_pos[-1] - 1)
        chunk_end = chunk_start + chunk_size
        nreachable = track_centr_msk[chunk_start:chunk_end].any(axis=0).sum()
        if nreachable > max_nreachable:
            split_pos.append(chunk_end - 1)
            chunk_size = 2
    split_pos.append(npositions)

    intensity = []
    windfields = []
    for prev_chunk_end, chunk_end in zip(split_pos[:-1], split_pos[1:]):
        chunk_start = max(0, prev_chunk_end - 1)
        inten, win = _compute_windfields_sparse(
            track.isel(time=slice(chunk_start, chunk_end)), *args,
            max_memory_gb=max_memory_gb, **kwargs,
        )
        intensity.append(inten)
        windfields.append(win)

    intensity = sparse.csr_matrix(sparse.vstack(intensity).max(axis=0))
    if windfields[0] is not None:
        # eliminate the overlap between consecutive chunks
        windfields = [windfields[0]] + [win[1:, :] for win in windfields[1:]]
        windfields = sparse.vstack(windfields, format="csr")
    return intensity, windfields

def _compute_windfields(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    model: int,
    metric: str = "equirect",
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1-minute sustained winds (in m/s) at 10 meters above ground

    In a first step, centroids within reach of the track are determined so that wind fields will
    only be computed and returned for those centroids. Still, since computing the distance of
    the storm center to the centroids is computationally expensive, make sure to pre-filter the
    centroids and call this function only for those centroids that are potentially affected.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. Which data variables are used in the computation of the wind
        speeds depends on the selected model.
    centroids : np.ndarray with two dimensions
        Each row is a centroid [lat, lon].
        Centroids that are not within reach of the track are ignored.
    model : int
        Wind profile model selection according to MODEL_VANG.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    max_dist_eye_km : float, optional
        No wind speed calculation is done for centroids with a distance (in km) to the TC center
        ("eye") larger than this parameter. Default: 300

    Returns
    -------
    windfields : np.ndarray of shape (npositions, nreachable, 2)
        Directional wind fields for each track position on those centroids within reach
        of the TC track. Note that the wind speeds at the first position are all zero because
        the discrete time derivatives involved in the process are implemented using backward
        differences. However, the first position is usually not relevant for impact calculations
        since it is far off shore.
    reachable_centr_idx : np.ndarray of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    # start with the assumption that no centroids are within reach
    npositions = si_track.sizes["time"]
    reachable_centr_idx = np.zeros((0,), dtype=np.int64)
    windfields = np.zeros((npositions, 0, 2), dtype=np.float64)

    # compute distances (in m) and vectors to all centroids
    [d_centr], [v_centr_normed] = u_coord.dist_approx(
        si_track["lat"].values[None], si_track["lon"].values[None],
        centroids[None, :, 0], centroids[None, :, 1],
        log=True, normalize=False, method=metric, units="m")

    # exclude centroids that are too far from or too close to the eye
    close_centr_msk = (d_centr <= max_dist_eye_km * KM_TO_M) & (d_centr > 1)
    if not np.any(close_centr_msk):
        return windfields, reachable_centr_idx

    # restrict to the centroids that are within reach of any of the positions
    track_centr_msk = close_centr_msk.any(axis=0)
    close_centr_msk = close_centr_msk[:, track_centr_msk]
    d_centr = d_centr[:, track_centr_msk]
    v_centr_normed = v_centr_normed[:, track_centr_msk, :]

    # normalize the vectors pointing from the eye to the centroids
    v_centr_normed[~close_centr_msk] = 0
    v_centr_normed[close_centr_msk] /= d_centr[close_centr_msk, None]

    # derive (absolute) angular velocity from parametric wind profile
    v_ang_norm = compute_angular_windspeeds(
        si_track, d_centr, close_centr_msk, model, cyclostrophic=False,
    )


    # Influence of translational speed decreases with distance from eye.
    # The "absorbing factor" is according to the following paper (see Fig. 7):
    #
    #   Mouton, F. & Nordbeck, O. (2005). Cyclone Database Manager. A tool
    #   for converting point data from cyclone observations into tracks and
    #   wind speed profiles in a GIS. UNED/GRID-Geneva.
    #   https://unepgrid.ch/en/resource/19B7D302
    #
    t_rad_bc = np.broadcast_to(si_track["rad"].values[:, None], d_centr.shape)
    v_trans_corr = np.zeros_like(d_centr)
    v_trans_corr[close_centr_msk] = np.fmin(
        1, t_rad_bc[close_centr_msk] / d_centr[close_centr_msk])

    if model in [MODEL_VANG['H08'], MODEL_VANG['H10']]:
        # In these models, v_ang_norm already contains vtrans_norm, so subtract it first, before
        # converting to vectors and then adding (vectorial) vtrans again. Make sure to apply the
        # "absorbing factor" in both steps:
        vtrans_norm_bc = np.broadcast_to(si_track["vtrans_norm"].values[:, None], d_centr.shape)
        v_ang_norm[close_centr_msk] -= (
                vtrans_norm_bc[close_centr_msk] * v_trans_corr[close_centr_msk]
        )

    # vectorial angular velocity
    windfields = (
            si_track.attrs["latsign"] * np.array([1.0, -1.0])[..., :] * v_centr_normed[:, :, ::-1]
    )
    windfields[close_centr_msk] *= v_ang_norm[close_centr_msk, None]

    # add angular and corrected translational velocity vectors
    windfields[1:] += si_track["vtrans"].values[1:, None, :] * v_trans_corr[1:, :, None]
    windfields[np.isnan(windfields)] = 0
    windfields[0, :, :] = 0
    [reachable_centr_idx] = track_centr_msk.nonzero()
    return windfields, reachable_centr_idx

def tctrack_to_si(
    track: xr.Dataset,
    metric: str = "equirect",
) -> xr.Dataset:
    """Convert track variables to SI units and prepare for wind field computation

    In addition to unit conversion, the variable names are shortened, the longitudinal coordinates
    are normalized and additional variables are defined:

    * cp (coriolis parameter)
    * vtrans (translational velocity vectors)
    * vtrans_norm (absolute value of translational speed)

    Furthermore, some scalar variables are stored as attributes:

    * latsign (1.0 if the track is located on the northern and -1.0 if on southern hemisphere)
    * mid_lon (the central longitude that was used to normalize the longitudinal coordinates)

    Finally, some corrections are applied to variables:

    * clip central pressure values so that environmental pressure values are never exceeded
    * extrapolate radius of max wind from pressure if missing

    Parameters
    ----------
    track : xr.Dataset
        Track information.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".

    Returns
    -------
    xr.Dataset
    """
    si_track = track[["lat", "lon", "time"]].copy()
    si_track["tstep"] = track["time_step"] * H_TO_S
    si_track["env"] = track["environmental_pressure"] * MBAR_TO_PA

    # we support some non-standard unit names
    unit_replace = {"mb": "mbar", "kn": "knots"}
    configs = [
        ("central_pressure", "cen", "Pa"),
        ("max_sustained_wind", "vmax", "m/s"),
    ]
    for long_name, var_name, si_unit in configs:
        unit = track.attrs[f"{long_name}_unit"]
        unit = unit_replace.get(unit, unit)
        try:
            conv_factor = ureg(unit).to(si_unit).magnitude
        except Exception as ex:
            raise ValueError(
                f"The {long_name}_unit '{unit}' in the provided track is not supported."
             ) from ex
        si_track[var_name] = track[long_name] * conv_factor

    # normalize longitudinal coordinates
    si_track.attrs["mid_lon"] = 0.5 * sum(u_coord.lon_bounds(si_track["lon"].values))
    u_coord.lon_normalize(si_track["lon"].values, center=si_track.attrs["mid_lon"])

    # make sure that central pressure never exceeds environmental pressure
    pres_exceed_msk = (si_track["cen"] > si_track["env"]).values
    si_track["cen"].values[pres_exceed_msk] = si_track["env"].values[pres_exceed_msk]

    # extrapolate radius of max wind from pressure if not given
    si_track["rad"] = track["radius_max_wind"].copy()
    si_track["rad"].values[:] = estimate_rmw(
        si_track["rad"].values, si_track["cen"].values / MBAR_TO_PA,
    )
    si_track["rad"] *= NM_TO_KM * KM_TO_M

    hemisphere = 'N'
    if np.count_nonzero(si_track["lat"] < 0) > np.count_nonzero(si_track["lat"] > 0):
        hemisphere = 'S'
    si_track.attrs["latsign"] = 1.0 if hemisphere == 'N' else -1.0

    # add translational speed of track at every node (in m/s)
    _vtrans(si_track, metric=metric)

    # convert surface winds to gradient winds without translational influence
    si_track["vgrad"] = (
        np.fmax(0, si_track["vmax"] - si_track["vtrans_norm"]) / GRADIENT_LEVEL_TO_SURFACE_WINDS
    )

    si_track["cp"] = ("time", _coriolis_parameter(si_track["lat"].values))

    return si_track

def compute_angular_windspeeds(si_track, d_centr, close_centr_msk, model, cyclostrophic=False):
    """Compute (absolute) angular wind speeds according to a parametric wind profile

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. Which data variables are used in the computation of the wind
        profile depends on the selected model.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    close_centr_msk : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
    """
    if model == MODEL_VANG['H1980']:
        _B_holland_1980(si_track)
    elif model in [MODEL_VANG['H08'], MODEL_VANG['H10']]:
        _bs_holland_2008(si_track)

    if model in [MODEL_VANG['H1980'], MODEL_VANG['H08']]:
        result = _stat_holland_1980(
            si_track, d_centr, close_centr_msk, cyclostrophic=cyclostrophic,
        )
        if model == MODEL_VANG['H1980']:
            result *= GRADIENT_LEVEL_TO_SURFACE_WINDS
    elif model == MODEL_VANG['H10']:
        # this model is always cyclostrophic
        _v_max_s_holland_2008(si_track)
        hol_x = _x_holland_2010(si_track, d_centr, close_centr_msk)
        result = _stat_holland_2010(si_track, d_centr, close_centr_msk, hol_x)
    elif model == MODEL_VANG['ER11']:
        result = _stat_er_2011(si_track, d_centr, close_centr_msk, cyclostrophic=cyclostrophic)
    else:
        raise NotImplementedError

    result[0, :] *= 0

    return result

def get_close_centroids(
    t_lat: np.ndarray,
    t_lon: np.ndarray,
    centroids: np.ndarray,
    buffer_km: float,
    metric: str = "equirect",
) -> np.ndarray:
    """Check whether centroids lay within a buffer around track positions

    The longitudinal coordinates are assumed to be normalized around a central longitude. This
    makes sure that the buffered bounding box around the track doesn't cross the antimeridian.

    The only hypothetical problem occurs when a TC track is travelling so far in longitude that
    adding a buffer exceeds 360 degrees (i.e. crosses the antimeridian).
    Of course, this case is physically impossible.

    Parameters
    ----------
    t_lat : np.ndarray of shape (npositions,)
        Latitudinal coordinates of track positions.
    t_lon : np.ndarray of shape (npositions,)
        Longitudinal coordinates of track positions, normalized around a central longitude.
    centroids : np.ndarray of shape (ncentroids, 2)
        Coordinates of centroids, each row is a pair [lat, lon].
    buffer_km : float
        Size of the buffer (in km). The buffer is converted to a lat/lon buffer, rescaled in
        longitudinal direction according to the t_lat coordinates.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".

    Returns
    -------
    mask : np.ndarray of shape (npositions, ncentroids)
        Mask that is True for close centroids and False for other centroids.
    """
    npositions = t_lat.size
    ncentroids = centroids.shape[0]
    centr_lat, centr_lon = centroids[:, 0], centroids[:, 1]
    buffer_lat = buffer_km / u_const.ONE_LAT_KM
    buffer_lon = buffer_km / (u_const.ONE_LAT_KM * np.cos(np.radians(
        np.fmin(89.999, np.abs(t_lat[:, None]) + buffer_lat)
    )))
    # check for each track position which centroids are within rectangular buffers
    [idx_rects] = (
        (t_lat[:, None] - buffer_lat <= centr_lat[None])
        & (t_lat[:, None] + buffer_lat >= centr_lat[None])
        & (t_lon[:, None] - buffer_lon <= centr_lon[None])
        & (t_lon[:, None] + buffer_lon >= centr_lon[None])
    ).any(axis=0).nonzero()

    # We do the distance computation for chunks of the track since computing the distance requires
    # npositions*ncentroids*8*3 Bytes of memory. For example, Hurricane FAITH's life time was more
    # than 500 hours. At 0.5-hourly resolution and 1,000,000 centroids, that's 24 GB of memory for
    # FAITH. With a chunk size of 10, this figure is down to 360 MB. The final mask will require
    # 1.0 GB of memory.
    chunk_size = 10
    chunks = np.split(np.arange(t_lat.size), np.arange(chunk_size, t_lat.size, chunk_size))
    dist_mask_rects = np.concatenate([
        (
            u_coord.dist_approx(
                t_lat[None, chunk], t_lon[None, chunk],
                centr_lat[None, idx_rects], centr_lon[None, idx_rects],
                normalize=False, method=metric, units="km",
            )[0] <= buffer_km
        ) for chunk in chunks
    ], axis=0)
    mask = np.zeros((npositions, ncentroids), dtype=bool)
    mask[:, idx_rects] = dist_mask_rects
    return mask

def _vtrans(si_track: xr.Dataset, metric: str = "equirect"):
    """Translational vector and velocity (in m/s) at each track node.

    The track dataset is modified in place, with the following variables added:

    * vtrans (directional vectors of velocity, in meters per second)
    * vtrans_norm (absolute velocity in meters per second; the first velocity is always 0)

    The meridional component (v) of the vectors is listed first.

    Parameters
    ----------
    si_track : xr.Dataset
        Track information as returned by `tctrack_to_si`. The data variables used by this function
        are "lat", "lon", and "tstep". The results are stored in place as new data
        variables "vtrans" and "vtrans_norm".
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".
    """
    npositions = si_track.sizes["time"]
    si_track["vtrans_norm"] = (["time"], np.zeros((npositions,)))
    si_track["vtrans"] = (["time", "component"], np.zeros((npositions, 2)))
    si_track["component"] = ("component", ["v", "u"])

    t_lat, t_lon = si_track["lat"].values, si_track["lon"].values
    norm, vec = u_coord.dist_approx(t_lat[:-1, None], t_lon[:-1, None],
                                    t_lat[1:, None], t_lon[1:, None],
                                    log=True, normalize=False, method=metric, units="m")
    si_track["vtrans"].values[1:, :] = vec[:, 0, 0] / si_track["tstep"].values[1:, None]
    si_track["vtrans_norm"].values[1:] = norm[:, 0, 0] / si_track["tstep"].values[1:]

    # limit to 30 nautical miles per hour
    msk = si_track["vtrans_norm"].values > 30 * KN_TO_MS
    fact = 30 * KN_TO_MS / si_track["vtrans_norm"].values[msk]
    si_track["vtrans"].values[msk, :] *= fact[:, None]
    si_track["vtrans_norm"].values[msk] *= fact

def _coriolis_parameter(lat: np.ndarray) -> np.ndarray:
    """Compute the Coriolis parameter from latitude.

    Parameters
    ----------
    lat : np.ndarray
        Latitude (degrees).

    Returns
    -------
    cp : np.ndarray of same shape as input
        Coriolis parameter.
    """
    return 2 * V_ANG_EARTH * np.sin(np.radians(np.abs(lat)))

def _bs_holland_2008(si_track: xr.Dataset):
    """Holland's 2008 b-value estimate for sustained surface winds.

    The result is stored in place as a new data variable "hol_b".

    Unlike the original 1980 formula (see `_B_holland_1980`), this approach does not require any
    wind speed measurements, but is based on the more reliable pressure information.

    The parameter applies to 1-minute sustained winds at 10 meters above ground.
    It is taken from equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    b_s = -4.4 * 1e-5 * (penv - pcen)^2 + 0.01 * (penv - pcen)
          + 0.03 * (dp/dt) - 0.014 * |lat| + 0.15 * (v_trans)^hol_xx + 1.0

    where `dp/dt` is the time derivative of central pressure and `hol_xx` is Holland's x
    parameter: hol_xx = 0.6 * (1 - (penv - pcen) / 215)

    The equation for b_s has been fitted statistically using hurricane best track records for
    central pressure and maximum wind. It therefore performs best in the North Atlantic.

    Furthermore, b_s has been fitted under the assumption of a "cyclostrophic" wind field which
    means that the influence from Coriolis forces is assumed to be small. This is reasonable close
    to the radius of maximum wind where the Coriolis term (r*f/2) is small compared to the rest
    (see `_stat_holland_1980`). More precisely: At the radius of maximum wind speeds, the typical
    order of the Coriolis term is 1 while wind speed is 50 (which changes away from the
    radius of maximum winds and as the TC moves away from the equator).

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. The data variables used by this function are "lat", "tstep",
        "vtrans_norm", "cen", and "env". The result is stored in place as a new data
        variable "hol_b".
    """
    # adjust pressure at previous track point
    prev_cen = np.zeros_like(si_track["cen"].values)
    prev_cen[1:] = si_track["cen"].values[:-1].copy()
    msk = prev_cen < 850 * MBAR_TO_PA
    prev_cen[msk] = si_track["cen"].values[msk]

    # The formula assumes that pressure values are in millibar (hPa) instead of SI units (Pa),
    # and time steps are in hours instead of seconds, but translational wind speed is still
    # expected to be in m/s.
    pdelta = (si_track["env"] - si_track["cen"]) / MBAR_TO_PA
    hol_xx = 0.6 * (1. - pdelta / 215)
    si_track["hol_b"] = (
        -4.4e-5 * pdelta**2 + 0.01 * pdelta
        + 0.03 * (si_track["cen"] - prev_cen) / si_track["tstep"] * (H_TO_S / MBAR_TO_PA)
        - 0.014 * abs(si_track["lat"])
        + 0.15 * si_track["vtrans_norm"]**hol_xx + 1.0
    )
    si_track["hol_b"] = np.clip(si_track["hol_b"], 1, 2.5)

def _v_max_s_holland_2008(si_track: xr.Dataset):
    """Compute maximum surface winds from pressure according to Holland 2008.

    The result is stored in place as a data variable "vmax". If a variable of that name already
    exists, its values are overwritten.

    This function implements equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    v_ms = [b_s / (rho * e) * (penv - pcen)]^0.5

    where `b_s` is Holland b-value (see `_bs_holland_2008`), e is Euler's number, rho is the
    density of air, `penv` is environmental, and `pcen` is central pressure.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si` with "hol_b" variable (see _bs_holland_2008). The data variables
        used by this function are "env", "cen", and "hol_b". The results are stored in place as
        a new data variable "vmax". If a variable of that name already exists, its values are
        overwritten.
    """
    pdelta = si_track["env"] - si_track["cen"]
    si_track["vmax"] = np.sqrt(si_track["hol_b"] / (RHO_AIR * np.exp(1)) * pdelta)

def _B_holland_1980(si_track: xr.Dataset):  # pylint: disable=invalid-name
    """Holland's 1980 B-value computation for gradient-level winds.

    The result is stored in place as a new data variable "hol_b".

    The parameter applies to gradient-level winds (about 1000 metres above the earth's surface).
    The formula for B is derived from equations (5) and (6) in the following paper:

    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure Profiles
    in Hurricanes. Monthly Weather Review 108(8): 1212–1218.
    https://doi.org/10.1175/1520-0493(1980)108<1212:AAMOTW>2.0.CO;2

    For reference, inserting (6) into (5) and solving for B at r = RMW yields:

    B = v^2 * e * rho / (penv - pcen)

    where v are maximum gradient-level winds `gradient_winds`, e is Euler's number, rho is the
    density of air, `penv` is environmental, and `pcen` is central pressure.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si` with "vgrad" variable (see _vgrad). The data variables
        used by this function are "vgrad", "env", and "cen". The results are stored in place as
        a new data variable "hol_b".
    """
    pdelta = si_track["env"] - si_track["cen"]
    si_track["hol_b"] = si_track["vgrad"]**2 * np.exp(1) * RHO_AIR / np.fmax(np.spacing(1), pdelta)
    si_track["hol_b"] = np.clip(si_track["hol_b"], 1, 2.5)

def _x_holland_2010(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr: np.ndarray,
    v_n: Union[float, np.ndarray] = 17.0,
    r_n_km: Union[float, np.ndarray] = 300.0,
) -> np.ndarray:
    """Compute exponent for wind model according to Holland et al. 2010.

    This function implements equation (10) from the following paper:

    Holland et al. (2010): A Revised Model for Radial Profiles of Hurricane Winds. Monthly
    Weather Review 138(12): 4393–4401. https://doi.org/10.1175/2010MWR3317.1

    For reference, it reads

    x = 0.5  [for r < r_max]
    x = 0.5 + (r - r_max) * (x_n - 0.5) / (r_n - r_max)  [for r >= r_max]

    The peripheral exponent x_n is adjusted to fit the peripheral observation of wind speeds `v_n`
    at radius `r_n`.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si` with "hol_b" variable (see _bs_holland_2008). The data variables
        used by this function are "rad", "vmax", and "hol_b".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    v_n : np.ndarray of shape (nnodes,) or float, optional
        Peripheral wind speeds (in m/s) at radius `r_n` outside of radius of maximum winds `r_max`.
        In absence of a second wind speed measurement, this value defaults to 17 m/s following
        Holland et al. 2010 (at a radius of 300 km).
    r_n_km : np.ndarray of shape (nnodes,) or float, optional
        Radius (in km) where the peripheral wind speed `v_n` is measured (or assumed).
        In absence of a second wind speed measurement, this value defaults to 300 km following
        Holland et al. 2010.

    Returns
    -------
    hol_x : np.ndarray of shape (nnodes, ncentroids)
        Exponents according to Holland et al. 2010.
    """
    hol_x = np.zeros_like(d_centr)
    r_max, v_max_s, hol_b, d_centr, v_n, r_n = [
        np.broadcast_to(ar, d_centr.shape)[close_centr]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["vmax"].values[:, None],
            si_track["hol_b"].values[:, None],
            d_centr,
            np.atleast_1d(v_n)[:, None],
            np.atleast_1d(r_n_km)[:, None],
        ]
    ]

    # convert to SI units
    r_n *= KM_TO_M

    # compute peripheral exponent from second measurement
    r_max_norm = (r_max / r_n)**hol_b
    x_n = np.log(v_n / v_max_s) / np.log(r_max_norm * np.exp(1 - r_max_norm))

    # linearly interpolate between max exponent and peripheral exponent
    x_max = 0.5
    hol_x[close_centr] = x_max + np.fmax(0, d_centr - r_max) * (x_n - x_max) / (r_n - r_max)

    # Negative hol_x values appear when v_max_s is very close to or even lower than v_n (which
    # should never happen in theory). In those cases, wind speeds might decrease outside of the eye
    # wall and increase again towards the peripheral radius (which is actually unphysical).
    # We clip hol_x to 0, otherwise wind speeds keep increasing indefinitely away from the eye:
    hol_x[close_centr] = np.fmax(hol_x[close_centr], 0.0)

    return hol_x

def _stat_holland_2010(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr: np.ndarray,
    hol_x: Union[float, np.ndarray],
) -> np.ndarray:
    """Symmetric and static surface wind fields (in m/s) according to Holland et al. 2010

    This function applies the cyclostrophic surface wind model expressed in equation (6) from

    Holland et al. (2010): A Revised Model for Radial Profiles of Hurricane Winds. Monthly
    Weather Review 138(12): 4393–4401. https://doi.org/10.1175/2010MWR3317.1

    More precisely, this function implements the following equation:

    V(r) = v_max_s * [(r_max / r)^b_s * e^(1 - (r_max / r)^b_s)]^x

    In terms of this function's arguments, b_s is `hol_b` and r is `d_centr`.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si` with "hol_b" (see _bs_holland_2008) data variables. The data
        variables used by this function are "vmax", "rad", and "hol_b".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    hol_x : np.ndarray of shape (nnodes, ncentroids) or float
        The exponent according to `_x_holland_2010`.

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (in m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    v_max_s, r_max, hol_b, d_centr, hol_x = [
        np.broadcast_to(ar, d_centr.shape)[close_centr]
        for ar in [
            si_track["vmax"].values[:, None],
            si_track["rad"].values[:, None],
            si_track["hol_b"].values[:, None],
            d_centr,
            hol_x,
        ]
    ]

    r_max_norm = (r_max / np.fmax(1, d_centr))**hol_b
    v_ang[close_centr] = v_max_s * (r_max_norm * np.exp(1 - r_max_norm))**hol_x
    return v_ang

def _stat_holland_1980(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr: np.ndarray,
    cyclostrophic: bool = False
) -> np.ndarray:
    """Symmetric and static wind fields (in m/s) according to Holland 1980.

    This function applies the gradient wind model expressed in equation (4) (combined with
    equation (6)) from

    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure Profiles in Hurricanes.
    Monthly Weather Review 108(8): 1212–1218.

    More precisely, this function implements the following equation:

    V(r) = [(B/rho) * (r_max/r)^B * (penv - pcen) * e^(-(r_max/r)^B) + (r*f/2)^2]^0.5 - (r*f/2)

    In terms of this function's arguments, B is `hol_b` and r is `d_centr`.
    The air density rho is assumed to be constant while the Coriolis parameter f is computed
    from the latitude `lat` using the constant rotation rate of the earth.

    Even though the equation has been derived originally for gradient winds (when combined with the
    output of `_B_holland_1980`), it can be used for surface winds by adjusting the parameter
    `hol_b` (see function `_bs_holland_2008`).

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si` with "hol_b" (see, e.g., _B_holland_1980) data variable. The data
        variables used by this function are "lat", "cp", "rad", "cen", "env", and "hol_b".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    r_max, hol_b, penv, pcen, coriolis_p, d_centr = [
        np.broadcast_to(ar, d_centr.shape)[close_centr]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["hol_b"].values[:, None],
            si_track["env"].values[:, None],
            si_track["cen"].values[:, None],
            si_track["cp"].values[:, None],
            d_centr,
        ]
    ]

    r_coriolis = 0
    if not cyclostrophic:
        r_coriolis = 0.5 * d_centr * coriolis_p

    r_max_norm = (r_max / np.fmax(1, d_centr))**hol_b
    sqrt_term = hol_b / RHO_AIR * r_max_norm * (penv - pcen) * np.exp(-r_max_norm) + r_coriolis**2
    v_ang[close_centr] = np.sqrt(np.fmax(0, sqrt_term)) - r_coriolis
    return v_ang

def _stat_er_2011(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr: np.ndarray,
    cyclostrophic: bool = False,
) -> np.ndarray:
    """Symmetric and static wind fields (in m/s) according to Emanuel and Rotunno 2011

    Emanuel, K., Rotunno, R. (2011): Self-Stratification of Tropical Cyclone Outflow. Part I:
    Implications for Storm Structure. Journal of the Atmospheric Sciences 68(10): 2236–2249.
    https://dx.doi.org/10.1175/JAS-D-10-05024.1

    The wind speeds `v_ang` are extracted from the momentum via the relationship M = v_ang * r,
    where r corresponds to `d_centr`. On the other hand, the momentum is derived from the momentum
    at the peak wind position using equation (36) from Emanuel and Rotunno 2011 with Ck == Cd:

    M = M_max * [2 * (r / r_max)^2 / (1 + (r / r_max)^2)].

    The momentum at the peak wind position is

    M_max = r_max * v_max + 0.5 * f * r_max**2,

    where the Coriolis parameter f is computed from the latitude `lat` using the constant rotation
    rate of the earth.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. The data variables used by this function are "lat", "cp", "rad",
        and "vmax".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0) in
        the computation of M_max. Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    r_max, v_max, coriolis_p, d_centr = [
        np.broadcast_to(ar, d_centr.shape)[close_centr]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["vmax"].values[:, None],
            si_track["cp"].values[:, None],
            d_centr,
        ]
    ]

    # compute the momentum at the maximum
    momentum_max = r_max * v_max

    if not cyclostrophic:
        # add the influence of the Coriolis force
        momentum_max += 0.5 * coriolis_p * r_max**2

    # rescale the momentum using formula (36) in Emanuel and Rotunno 2011 with Ck == Cd
    r_max_norm = (d_centr / r_max)**2
    momentum = momentum_max * 2 * r_max_norm / (1 + r_max_norm)

    # extract the velocity from the rescaled momentum through division by r
    v_ang[close_centr] = np.fmax(0, momentum / (d_centr + 1e-11))
    return v_ang
