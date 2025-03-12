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

__all__ = ["TropCyclone"]

import copy
import datetime as dt
import itertools
import logging
import time
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import numpy as np
import pathos.pools
import xarray as xr
from scipy import sparse
from tqdm import tqdm

import climada.util.constants as u_const
import climada.util.coordinates as u_coord
import climada.util.plot as u_plot
from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
from climada.hazard.tc_clim_change import get_knutson_scaling_factor
from climada.hazard.tc_tracks import TCTracks

from .trop_cyclone_windfields import (
    DEF_INTENSITY_THRES,
    DEF_MAX_DIST_EYE_KM,
    DEF_MAX_MEMORY_GB,
    compute_windfields_sparse,
)

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = "TC"
"""Hazard type acronym for Tropical Cyclone"""


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

    vars_opt = Hazard.vars_opt.union({"category"})
    """Name of the variables that are not needed to compute the impact."""

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

            * '-1 tropical depression
            * '0 tropical storm
            * '1 Hurrican category 1
            * '2 Hurrican category 2
            * '3 Hurrican category 3
            * '4 Hurrican category 4
            * '5 Hurrican category 5

        basin : list of str, optional
            Basin where every event starts:

            * 'NA' North Atlantic
            * 'EP' Eastern North Pacific
            * 'WP' Western North Pacific
            * 'NI' North Indian
            * 'SI' South Indian
            * 'SP' Southern Pacific
            * 'SA' South Atlantic

        windfields : list of csr_matrix, optional
            For each event, the full velocity vectors at each centroid and track position in a
            sparse matrix of shape (npositions,  ncentroids * 2) that can be reshaped to a full
            ndarray of shape (npositions, ncentroids, 2).
        **kwargs : Hazard properties, optional
            All other keyword arguments are passed to the Hazard constructor.
        """
        kwargs.setdefault("haz_type", HAZ_TYPE)
        Hazard.__init__(self, **kwargs)
        self.category = category if category is not None else np.array([], int)
        self.basin = basin if basin is not None else []
        self.windfields = windfields if windfields is not None else []

    def set_from_tracks(self, *args, **kwargs):
        """This function is deprecated, use TropCyclone.from_tracks instead."""
        LOGGER.warning(
            "The use of TropCyclone.set_from_tracks is deprecated."
            "Use TropCyclone.from_tracks instead."
        )
        if "intensity_thres" not in kwargs:
            # some users modify the threshold attribute before calling `set_from_tracks`
            kwargs["intensity_thres"] = self.intensity_thres
        if self.pool is not None and "pool" not in kwargs:
            kwargs["pool"] = self.pool
        self.__dict__ = TropCyclone.from_tracks(*args, **kwargs).__dict__

    @classmethod
    def from_tracks(
        cls,
        tracks: TCTracks,
        centroids: Centroids,
        pool: Optional[pathos.pools.ProcessPool] = None,
        model: str = "H08",
        model_kwargs: Optional[dict] = None,
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

        This function sets the ``intensity`` attribute to contain, for each centroid,
        the maximum wind speed (1-minute sustained winds at 10 meters above ground)
        experienced over the whole period of each TC event in m/s. The wind speed is set
        to 0 if it doesn't exceed the threshold ``intensity_thres``.

        The ``category`` attribute is set to the value of the ``category``-attribute
        of each of the given track data sets.

        The ``basin`` attribute is set to the genesis basin for each event, which
        is the first value of the ``basin``-variable in each of the given track data sets.

        Optionally, the time dependent, vectorial winds can be stored using the
        ``store_windfields`` function parameter (see below).

        Parameters
        ----------
        tracks : climada.hazard.TCTracks
            Tracks of storm events.
        centroids : Centroids, optional
            Centroids where to model TC. Default: global centroids at 360 arc-seconds
            resolution.
        pool : pathos.pool, optional
            Pool that will be used for parallel computation of wind fields. Default:
            None
        model : str, optional
            Parametric wind field model to use. Default: "H08".

            * ``"H1980"`` (the prominent Holland 1980 model) from the paper:
                    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure
                    Profiles in Hurricanes. Monthly Weather Review 108(8): 1212–1218.
                    ``https://doi.org/10.1175/1520-0493(1980)108<1212:AAMOTW>2.0.CO;2``
            * ``"H08"`` (Holland 1980 with b-value from Holland 2008) from the paper:
                    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
                    Weather Review, 136(9), 3432–3445.
                    https://doi.org/10.1175/2008MWR2395.1
            * ``"H10"`` (Holland et al. 2010) from the paper:
                    Holland et al. (2010): A Revised Model for Radial Profiles of
                    Hurricane Winds. Monthly Weather Review 138(12): 4393–4401.
                    https://doi.org/10.1175/2010MWR3317.1
            * ``"ER11"`` (Emanuel and Rotunno 2011) from the paper:
                    Emanuel, K., Rotunno, R. (2011): Self-Stratification of Tropical
                    Cyclone Outflow. Part I: Implications for Storm Structure. Journal
                    of the Atmospheric Sciences 68(10): 2236–2249.
                    https://dx.doi.org/10.1175/JAS-D-10-05024.1

        model_kwargs : dict, optional
            If given, forward these kwargs to the selected wind model. None of the
            parameters is currently supported by the ER11 model. Default: None.
            The Holland models support the following parameters, in alphabetical order:

            gradient_to_surface_winds : float, optional
                The gradient-to-surface wind reduction factor to use. In H1980, the wind
                profile is computed on the gradient level, and wind speeds are converted
                to the surface level using this factor. In H08 and H10, the wind profile
                is computed on the surface level, but the clipping interval of the
                B-value depends on this factor. Default: 0.9
            rho_air_const : float or None, optional
                The constant value for air density (in kg/m³) to assume in the formulas
                from Holland 1980. By default, the constant value suggested in Holland
                1980 is used. If set to None, the air density is computed from pressure
                following equation (9) in Holland et al. 2010. Default: 1.15
            vmax_from_cen : boolean, optional
                Only used in H10. If True, replace the recorded value of vmax along the
                track by an estimate from pressure, following equation (8) in Holland et
                al. 2010. Default: True
            vmax_in_brackets : bool, optional
                Only used in H10. Specifies which of the two formulas in equation (6) of
                Holland et al. 2010 to use. If False, the formula with vmax outside of
                the brackets is used. Note that, a side-effect of the formula with vmax
                inside of the brackets is that the wind speed maximum is attained a bit
                farther away from the center than according to the recorded radius of
                maximum winds (RMW). Default: False
            cyclostrophic : bool, optional
                If True, do not apply the influence of the Coriolis force (set the Coriolis
                terms to 0).
                Default: True for H10 model, False otherwise.

        ignore_distance_to_coast : boolean, optional
            If True, centroids far from coast are not ignored.
            If False, the centroids' distances to the coast are calculated with the
            `Centroids.get_dist_coast()` method (unless there is "dist_coast" column in
            the centroids' GeoDataFrame) and centroids far from coast are ignored.
            Default: False.
        store_windfields : boolean, optional
            If True, the Hazard object gets a list ``windfields`` of sparse matrices.
            For each track, the full velocity vectors at each centroid and track
            position are stored in a sparse matrix of shape (npositions,
            ncentroids * 2) that can be reshaped to a full ndarray of shape (npositions,
            ncentroids, 2). Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances:

            * "equirect": Distance according to sinusoidal projection. Fast, but
              inaccurate for large distances and high latitudes.
            * "geosphere": Exact spherical distance. Much more accurate at all
              distances, but slow.

            Default: "equirect".
        intensity_thres : float, optional
            Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
        max_latitude : float, optional
            No wind speed calculation is done for centroids with latitude larger than
            this parameter. Default: 61
        max_dist_inland_km : float, optional
            No wind speed calculation is done for centroids with a distance (in km) to
            the coast larger than this parameter. Default: 1000
        max_dist_eye_km : float, optional
            No wind speed calculation is done for centroids with a distance (in km) to
            the TC center ("eye") larger than this parameter. Default: 300
        max_memory_gb : float, optional
            To avoid memory issues, the computation is done for chunks of the track
            sequentially. The chunk size is determined depending on the available memory
            (in GB). Note that this limit applies to each thread separately if a
            ``pool`` is used. Default: 8

        Raises
        ------
        ValueError

        Returns
        -------
        TropCyclone
        """
        num_tracks = tracks.size

        if ignore_distance_to_coast:
            # Select centroids with lat <= max_latitude
            [idx_centr_filter] = (np.abs(centroids.lat) <= max_latitude).nonzero()
        else:
            # Select centroids which are inside max_dist_inland_km and lat <= max_latitude
            if "dist_coast" not in centroids.gdf.columns:
                dist_coast = centroids.get_dist_coast()
            else:
                dist_coast = centroids.gdf["dist_coast"].values
            [idx_centr_filter] = (
                (dist_coast <= max_dist_inland_km * 1000)
                & (np.abs(centroids.lat) <= max_latitude)
            ).nonzero()

        # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
        # Later, there will be another filtering step with more accurate distances in km.
        max_dist_eye_deg = max_dist_eye_km / (
            u_const.ONE_LAT_KM * np.cos(np.radians(max_latitude))
        )

        # Restrict to coastal centroids within reach of any of the tracks
        t_lon_min, t_lat_min, t_lon_max, t_lat_max = tracks.get_bounds(
            deg_buffer=max_dist_eye_deg
        )
        t_mid_lon = 0.5 * (t_lon_min + t_lon_max)
        filtered_centroids = centroids.coord[idx_centr_filter]
        u_coord.lon_normalize(filtered_centroids[:, 1], center=t_mid_lon)
        idx_centr_filter = idx_centr_filter[
            (t_lon_min <= filtered_centroids[:, 1])
            & (filtered_centroids[:, 1] <= t_lon_max)
            & (t_lat_min <= filtered_centroids[:, 0])
            & (filtered_centroids[:, 0] <= t_lat_max)
        ]

        # prepare keyword arguments to pass to `from_single_track`
        kwargs_from_single_track = dict(
            centroids=centroids,
            idx_centr_filter=idx_centr_filter,
            model=model,
            model_kwargs=model_kwargs,
            store_windfields=store_windfields,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

        LOGGER.info(
            "Mapping %d tracks to %d coastal centroids.",
            num_tracks,
            idx_centr_filter.size,
        )
        if pool:
            chunksize = max(min(num_tracks // pool.ncpus, 1000), 1)
            kwargs_repeated = [
                itertools.repeat(val, num_tracks)
                for val in kwargs_from_single_track.values()
            ]
            tc_haz_list = pool.map(
                cls.from_single_track,
                tracks.data,
                *kwargs_repeated,
                chunksize=chunksize,
            )
        else:
            last_perc = 0
            tc_haz_list = []
            for track in tracks.data:
                perc = 100 * len(tc_haz_list) / len(tracks.data)
                if perc - last_perc >= 10:
                    LOGGER.info("Progress: %d%%", perc)
                    last_perc = perc
                tc_haz_list.append(
                    cls.from_single_track(track, **kwargs_from_single_track)
                )
            if last_perc < 100:
                LOGGER.info("Progress: 100%")

        LOGGER.debug("Concatenate events.")
        haz = cls.concat(tc_haz_list)
        haz.pool = pool
        haz.intensity_thres = intensity_thres
        LOGGER.debug("Compute frequency.")
        haz.frequency_from_tracks(tracks.data)
        return haz

    def apply_climate_scenario_knu(
        self,
        percentile: str = "50",
        scenario: str = "4.5",
        target_year: int = 2050,
        **kwargs,
    ):
        """
        From current TC hazard instance, return new hazard set with future events
        for a given RCP scenario and year based on the parametrized values derived
        by Jewson 2021 (https://doi.org/10.1175/JAMC-D-21-0102.1) based on those
        published by Knutson 2020 (https://doi.org/10.1175/BAMS-D-18-0194.1). The
        scaling for different years and RCP scenarios is obtained by linear
        interpolation.

        Note: Only frequency changes are applied as suggested by Jewson 2022
        (https://doi.org/10.1007/s00477-021-02142-6). Applying only frequency anyway
        changes mean intensities and most importantly avoids possible inconsistencies
        (including possible double-counting) that may arise from the application of both
        frequency and intensity changes, as the relationship between these two is non
        trivial to resolve.

        Parameters
        ----------
        percentile: str
            percentiles of Knutson et al. 2020 estimates, representing the mode
            uncertainty in future changes in TC activity. These estimates come from
            a review of state-of-the-art literature and models. For the 'cat05' variable
            (i.e. frequency of all tropical cyclones) the 5th, 25th, 50th, 75th and 95th
            percentiles are provided. For 'cat45' and 'intensity', the provided percentiles
            are the 10th, 25th, 50th, 75th and 90th. Please refer to the mentioned publications
            for more details.
            possible percentiles:

            * '5/10' either the 5th or 10th percentile depending on variable (see text above)
            * '25' for the 25th percentile
            * '50' for the 50th percentile
            * '75' for the 75th percentile
            * '90/95' either the 90th or 95th percentile depending on variable  (see text above)

            Default: '50'
        scenario : str
            possible scenarios:

            * '2.6' for RCP 2.6
            * '4.5' for RCP 4.5
            * '6.0' for RCP 6.0
            * '8.5' for RCP 8.5

        target_year : int
            future year to be simulated, between 2000 and 2100. Default: 2050.
        Returns
        -------
        haz_cc : climada.hazard.TropCyclone
            Tropical cyclone with frequencies and intensity scaled according
            to the Knutson criterion for the given year, RCP and percentile.
            Returns a new instance of climada.hazard.TropCyclone, self is not
            modified.
        """

        if self.category.size == 0:
            LOGGER.warning(
                "Tropical cyclone categories are missing and"
                "no effect of climate change can be modelled."
                "The original event set is returned"
            )
            return self

        tc_cc = copy.deepcopy(self)

        sel_cat05 = np.isin(tc_cc.category, [0, 1, 2, 3, 4, 5])
        sel_cat03 = np.isin(tc_cc.category, [0, 1, 2, 3])
        sel_cat45 = np.isin(tc_cc.category, [4, 5])

        years = np.array([dt.datetime.fromordinal(date).year for date in self.date])

        for basin in np.unique(tc_cc.basin):
            scale_year_rcp_05, scale_year_rcp_45 = [
                get_knutson_scaling_factor(
                    percentile=percentile,
                    variable=variable,
                    basin=basin,
                    baseline=(np.min(years), np.max(years)),
                    **kwargs,
                ).loc[target_year, scenario]
                for variable in ["cat05", "cat45"]
            ]

            bas_sel = np.array(tc_cc.basin) == basin

            cat_05_freqs_change = scale_year_rcp_05 * np.sum(
                tc_cc.frequency[sel_cat05 & bas_sel]
            )
            cat_45_freqs_change = scale_year_rcp_45 * np.sum(
                tc_cc.frequency[sel_cat45 & bas_sel]
            )
            cat_03_freqs = np.sum(tc_cc.frequency[sel_cat03 & bas_sel])

            scale_year_rcp_03 = (
                cat_05_freqs_change - cat_45_freqs_change
            ) / cat_03_freqs

            tc_cc.frequency[sel_cat03 & bas_sel] *= 1 + scale_year_rcp_03 / 100
            tc_cc.frequency[sel_cat45 & bas_sel] *= 1 + scale_year_rcp_45 / 100

            if any(tc_cc.frequency) < 0:
                raise ValueError(
                    " The application of the climate scenario leads to "
                    " negative frequencies. One solution - if appropriate -"
                    " could be to use a less extreme percentile."
                )

        return tc_cc

    def set_climate_scenario_knu(self, *args, **kwargs):
        """This function is deprecated, use TropCyclone.apply_climate_scenario_knu instead."""
        LOGGER.warning(
            "The use of TropCyclone.set_climate_scenario_knu is deprecated."
            "Use TropCyclone.apply_climate_scenario_knu instead."
        )
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
        **kwargs,
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
            raise ValueError(f"{track_name} not found in track data.")
        idx_plt = np.argwhere(
            (track["lon"].values < centroids.total_bounds[2] + 1)
            & (centroids.total_bounds[0] - 1 < track["lon"].values)
            & (track["lat"].values < centroids.total_bounds[3] + 1)
            & (centroids.total_bounds[1] - 1 < track["lat"].values)
        ).reshape(-1)

        tc_list = []
        tr_coord = {"lat": [], "lon": []}
        for node in range(idx_plt.size - 2):
            tr_piece = track.sel(
                time=slice(
                    track["time"].values[idx_plt[node]],
                    track["time"].values[idx_plt[node + 2]],
                )
            )
            tr_piece.attrs["n_nodes"] = 2  # plot only one node
            tr_sel = TCTracks()
            tr_sel.append(tr_piece)
            tr_coord["lat"].append(tr_sel.data[0]["lat"].values[:-1])
            tr_coord["lon"].append(tr_sel.data[0]["lon"].values[:-1])

            tc_tmp = cls.from_tracks(tr_sel, centroids=centroids)
            tc_tmp.event_name = [
                track["name"]
                + " "
                + time.strftime(
                    "%d %h %Y %H:%M",
                    time.gmtime(
                        tr_sel.data[0]["time"][1].values.astype(int) / 1000000000
                    ),
                )
            ]
            tc_list.append(tc_tmp)

        if "cmap" not in kwargs:
            kwargs["cmap"] = "Greys"
        if "vmin" not in kwargs:
            kwargs["vmin"] = np.array([tc_.intensity.min() for tc_ in tc_list]).min()
        if "vmax" not in kwargs:
            kwargs["vmax"] = np.array([tc_.intensity.max() for tc_ in tc_list]).max()

        def run(node):
            tc_list[node].plot_intensity(1, axis=axis, **kwargs)
            axis.plot(tr_coord["lon"][node], tr_coord["lat"][node], "k")
            axis.set_title(tc_list[node].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info("Generating video %s", file_name)
            fig, axis, _fontsize = u_plot.make_map(
                figsize=figsize, adapt_fontsize=adapt_fontsize
            )
            pbar = tqdm(total=idx_plt.size - 2)
            ani = animation.FuncAnimation(
                fig, run, frames=idx_plt.size - 2, interval=500, blit=False
            )
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
        year_max = np.amax([t["time"].dt.year.values.max() for t in tracks])
        year_min = np.amin([t["time"].dt.year.values.min() for t in tracks])
        year_delta = year_max - year_min + 1
        num_orig = np.count_nonzero(self.orig)
        ens_size = (self.event_id.size / num_orig) if num_orig > 0 else 1
        self.frequency = np.ones(self.event_id.size) / (year_delta * ens_size)

    @classmethod
    def from_single_track(
        cls,
        track: xr.Dataset,
        centroids: Centroids,
        idx_centr_filter: np.ndarray,
        model: str = "H08",
        model_kwargs: Optional[dict] = None,
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
        idx_centr_filter : np.ndarray
            Indices of centroids to restrict to (e.g. sufficiently close to coast).
        model : str, optional
            Parametric wind field model, one of "H1980" (the prominent Holland 1980 model),
            "H08" (Holland 1980 with b-value from Holland 2008), "H10" (Holland et al. 2010), or
            "ER11" (Emanuel and Rotunno 2011).
            Default: "H08".
        model_kwargs: dict, optional
            If given, forward these kwargs to the selected model. Default: None
        store_windfields : boolean, optional
            If True, store windfields. Default: False.
        metric : str, optional
            Specify an approximation method to use for earth distances: "equirect" (faster) or
            "geosphere" (more accurate). See ``dist_approx`` function in
            ``climada.util.coordinates``.
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
        intensity_sparse, windfields_sparse = compute_windfields_sparse(
            track=track,
            centroids=centroids,
            idx_centr_filter=idx_centr_filter,
            model=model,
            model_kwargs=model_kwargs,
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
        new_haz.units = "m/s"
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.attrs["sid"]]
        new_haz.fraction = sparse.csr_matrix(new_haz.intensity.shape)
        # store first day of track as date
        new_haz.date = np.array(
            [
                dt.datetime(
                    track["time"].dt.year.values[0],
                    track["time"].dt.month.values[0],
                    track["time"].dt.day.values[0],
                ).toordinal()
            ]
        )
        new_haz.orig = np.array([track.attrs["orig_event_flag"]])
        new_haz.category = np.array([track.attrs["category"]])
        # users that pickle TCTracks objects might still have data with the legacy basin attribute,
        # so we have to deal with it here
        new_haz.basin = [
            (
                track["basin"]
                if isinstance(track["basin"], str)
                else str(track["basin"].values[0])
            )
        ]
        return new_haz

    def _apply_knutson_criterion(self, chg_int_freq: List, scaling_rcp_year: float):
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
            inten_chg = [
                chg
                for chg in chg_int_freq
                if (chg["variable"] == "intensity" and chg["basin"] == basin)
            ]
            for chg in inten_chg:
                sel_cat_chg = np.isin(tc_cc.category, chg["category"]) & bas_sel
                inten_scaling = 1 + (chg["change"] - 1) * scaling_rcp_year
                tc_cc.intensity = sparse.diags(
                    np.where(sel_cat_chg, inten_scaling, 1)
                ).dot(tc_cc.intensity)

            # Apply frequency change
            freq_chg = [
                chg
                for chg in chg_int_freq
                if (chg["variable"] == "frequency" and chg["basin"] == basin)
            ]
            freq_chg.sort(reverse=False, key=lambda x: len(x["category"]))

            # Scale frequencies by category
            cat_larger_list = []
            for chg in freq_chg:
                cat_chg_list = [
                    cat for cat in chg["category"] if cat not in cat_larger_list
                ]
                sel_cat_chg = np.isin(tc_cc.category, cat_chg_list) & bas_sel
                if sel_cat_chg.any():
                    freq_scaling = 1 + (chg["change"] - 1) * scaling_rcp_year
                    tc_cc.frequency[sel_cat_chg] *= freq_scaling
                cat_larger_list += cat_chg_list

        if (tc_cc.frequency < 0).any():
            raise ValueError(
                "The application of the given climate scenario"
                "resulted in at least one negative frequency."
            )

        return tc_cc
