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
from climada.hazard.tag import Tag as TagHazard
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
    basin : list(str)
        Basin where every event starts:

        * 'NA' North Atlantic
        * 'EP' Eastern North Pacific
        * 'WP' Western North Pacific
        * 'NI' North Indian
        * 'SI' South Indian
        * 'SP' Southern Pacific
        * 'SA' South Atlantic
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
            For each event
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
        description: str = '',
        model: str = 'H08',
        ignore_distance_to_coast: bool = False,
        store_windfields: bool = False,
        metric: str = "equirect",
        intensity_thres: float = DEF_INTENSITY_THRES,
        max_latitude: float = 61,
        max_dist_inland_km: float = 1000,
        max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
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
            chunksize = min(num_tracks // pool.ncpus, 1000)
            tc_haz_list = pool.map(
                cls.from_single_track, tracks.data,
                itertools.repeat(centroids, num_tracks),
                itertools.repeat(coastal_idx, num_tracks),
                itertools.repeat(model, num_tracks),
                itertools.repeat(store_windfields, num_tracks),
                itertools.repeat(metric, num_tracks),
                itertools.repeat(intensity_thres, num_tracks),
                itertools.repeat(max_dist_eye_km, num_tracks),
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
                                          max_dist_eye_km=max_dist_eye_km))
            if last_perc < 100:
                LOGGER.info("Progress: 100%")

        LOGGER.debug('Concatenate events.')
        haz = cls.concat(tc_haz_list)
        haz.pool = pool
        haz.intensity_thres = intensity_thres
        LOGGER.debug('Compute frequency.')
        haz.frequency_from_tracks(tracks.data)
        haz.tag.description = description
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
        haz_cc.tag.description = 'climate change scenario for year %s and RCP %s '\
        'from Knutson et al 2015.' % (str(ref_year), str(rcp_scenario))
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
            raise ValueError('%s not found in track data.' % track_name)
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

        Raises
        ------
        ValueError, KeyError

        Returns
        -------
        haz : TropCyclone
        """
        try:
            mod_id = MODEL_VANG[model]
        except KeyError as err:
            raise ValueError(f'Model not implemented: {model}.') from err
        ncentroids = centroids.coord.shape[0]
        coastal_centr = centroids.coord[coastal_idx]
        windfields, reachable_centr_idx = compute_windfields(
            track, coastal_centr, mod_id, metric=metric, max_dist_eye_km=max_dist_eye_km)
        reachable_coastal_centr_idx = coastal_idx[reachable_centr_idx]
        npositions = windfields.shape[0]

        intensity = np.linalg.norm(windfields, axis=-1).max(axis=0)
        intensity[intensity < intensity_thres] = 0
        intensity_sparse = sparse.csr_matrix(
            (intensity, reachable_coastal_centr_idx, [0, intensity.size]),
            shape=(1, ncentroids))
        intensity_sparse.eliminate_zeros()

        new_haz = cls()
        new_haz.tag = TagHazard(HAZ_TYPE, 'Name: ' + track.name)
        new_haz.intensity_thres = intensity_thres
        new_haz.intensity = intensity_sparse
        if store_windfields:
            n_reachable_coastal_centr = reachable_coastal_centr_idx.size
            indices = np.zeros((npositions, n_reachable_coastal_centr, 2), dtype=np.int64)
            indices[:, :, 0] = 2 * reachable_coastal_centr_idx[None]
            indices[:, :, 1] = 2 * reachable_coastal_centr_idx[None] + 1
            indices = indices.ravel()
            indptr = np.arange(npositions + 1) * n_reachable_coastal_centr * 2
            windfields_sparse = sparse.csr_matrix((windfields.ravel(), indices, indptr),
                                                  shape=(npositions, ncentroids * 2))
            windfields_sparse.eliminate_zeros()
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

            bas_sel = (np.array(tc_cc.basin) == basin)

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


def compute_windfields(
    track: xr.Dataset,
    centroids: np.ndarray,
    model: int,
    metric: str = "equirect",
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1-minute sustained winds (in m/s) at 10 meters above ground

    In a first step, centroids within reach of the track are determined so that wind fields will
    only be computed and returned for those centroids.

    Parameters
    ----------
    track : xr.Dataset
        Track infomation.
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
        of the TC track.
    reachable_centr_idx : np.ndarray of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    # copies of track data (note that max wind records are not used in all wind field models)
    t_lat, t_lon, t_tstep, t_rad, t_env, t_cen = [
        track[ar].values.copy() for ar in ['lat', 'lon', 'time_step', 'radius_max_wind',
                                           'environmental_pressure', 'central_pressure']
    ]

    # start with the assumption that no centroids are within reach
    npositions = t_lat.shape[0]
    reachable_centr_idx = np.zeros((0,), dtype=np.int64)
    windfields = np.zeros((npositions, 0, 2), dtype=np.float64)

    # the wind field model requires at least two track positions because translational speed
    # as well as the change in pressure are required
    if npositions < 2:
        return windfields, reachable_centr_idx

    # normalize longitude values (improves performance of `dist_approx` and `_close_centroids`)
    mid_lon = 0.5 * sum(u_coord.lon_bounds(t_lon))
    u_coord.lon_normalize(t_lon, center=mid_lon)
    u_coord.lon_normalize(centroids[:, 1], center=mid_lon)

    # Filter early with a larger threshold, but inaccurate (lat/lon) distances.
    # There is another filtering step with more accurate distances in km later.
    max_dist_eye_deg = max_dist_eye_km / (
        u_const.ONE_LAT_KM * np.cos(np.radians(np.abs(t_lat).max()))
    )

    # restrict to centroids within rectangular bounding boxes around track positions
    track_centr_msk = _close_centroids(t_lat, t_lon, centroids, max_dist_eye_deg)
    track_centr = centroids[track_centr_msk]
    nreachable = track_centr.shape[0]
    if nreachable == 0:
        return windfields, reachable_centr_idx

    # compute distances (in km) and vectors to all centroids
    [d_centr], [v_centr_normed] = u_coord.dist_approx(
        t_lat[None], t_lon[None], track_centr[None, :, 0], track_centr[None, :, 1],
        log=True, normalize=False, method=metric)

    # exclude centroids that are too far from or too close to the eye
    close_centr_msk = (d_centr <= max_dist_eye_km) & (d_centr > 1e-2)
    if not np.any(close_centr_msk):
        return windfields, reachable_centr_idx
    v_centr_normed[~close_centr_msk] = 0
    v_centr_normed[close_centr_msk] /= d_centr[close_centr_msk, None]

    # make sure that central pressure never exceeds environmental pressure
    pres_exceed_msk = (t_cen > t_env)
    t_cen[pres_exceed_msk] = t_env[pres_exceed_msk]

    # extrapolate radius of max wind from pressure if not given (and convert to km)
    t_rad[:] = estimate_rmw(t_rad, t_cen) * NM_TO_KM

    # translational speed of track at every node (in m/s)
    [v_trans_norm, v_trans] = _vtrans(t_lat, t_lon, t_tstep, metric=metric)

    # adjust pressure at previous track point
    prev_pres = t_cen[:-1].copy()
    msk = (prev_pres < 850)
    prev_pres[msk] = t_cen[1:][msk]

    # derive (absolute) angular velocity from parametric wind profile
    v_ang_norm = np.zeros((npositions, nreachable), dtype=np.float64)
    if model == MODEL_VANG['H1980']:
        # convert surface winds (in m/s) to gradient winds without translational influence
        t_vmax = track.max_sustained_wind.values.copy() * KN_TO_MS
        t_gradient_winds = np.fmax(0, t_vmax - v_trans_norm) / GRADIENT_LEVEL_TO_SURFACE_WINDS
        hol_b = _B_holland_1980(t_gradient_winds[1:], t_env[1:], t_cen[1:])
        v_ang_norm[1:] = _stat_holland_1980(d_centr[1:], t_rad[1:], hol_b, t_env[1:],
                                            t_cen[1:], t_lat[1:], close_centr_msk[1:])
        v_ang_norm *= GRADIENT_LEVEL_TO_SURFACE_WINDS
    elif model == MODEL_VANG['H08']:
        # this model doesn't use the recorded surface winds
        hol_b = _bs_holland_2008(v_trans_norm[1:], t_env[1:], t_cen[1:], prev_pres,
                                 t_lat[1:], t_tstep[1:])
        v_ang_norm[1:] = _stat_holland_1980(d_centr[1:], t_rad[1:], hol_b, t_env[1:],
                                            t_cen[1:], t_lat[1:], close_centr_msk[1:])
    elif model == MODEL_VANG['H10']:
        # this model doesn't use the recorded surface winds
        hol_b = _bs_holland_2008(v_trans_norm[1:], t_env[1:], t_cen[1:], prev_pres,
                                 t_lat[1:], t_tstep[1:])
        t_vmax = _v_max_s_holland_2008(t_env[1:], t_cen[1:], hol_b)
        hol_x = _x_holland_2010(d_centr[1:], t_rad[1:], t_vmax, hol_b, close_centr_msk[1:])
        v_ang_norm[1:] = _stat_holland_2010(d_centr[1:], t_vmax, t_rad[1:], hol_b,
                                            close_centr_msk[1:], hol_x)
    elif model == MODEL_VANG['ER11']:
        t_vmax = track.max_sustained_wind.values.copy() * KN_TO_MS
        v_ang_norm[:] = _stat_er_2011(d_centr, t_vmax, t_rad, t_lat)
    else:
        raise NotImplementedError

    # vectorial angular velocity
    hemisphere = 'N'
    if np.count_nonzero(t_lat < 0) > np.count_nonzero(t_lat > 0):
        hemisphere = 'S'
    v_ang_rotate = [1.0, -1.0] if hemisphere == 'N' else [-1.0, 1.0]
    windfields = np.array(v_ang_rotate)[..., :] * v_centr_normed[:, :, ::-1]
    windfields[close_centr_msk] *= v_ang_norm[close_centr_msk, None]

    # Influence of translational speed decreases with distance from eye.
    # The "absorbing factor" is according to the following paper (see Fig. 7):
    #
    #   Mouton, F. & Nordbeck, O. (2005). Cyclone Database Manager. A tool
    #   for converting point data from cyclone observations into tracks and
    #   wind speed profiles in a GIS. UNED/GRID-Geneva.
    #   https://unepgrid.ch/en/resource/19B7D302
    #
    t_rad_bc = np.broadcast_arrays(t_rad[:, None], d_centr)[0]
    v_trans_corr = np.zeros_like(d_centr)
    v_trans_corr[close_centr_msk] = np.fmin(
        1, t_rad_bc[close_centr_msk] / d_centr[close_centr_msk])

    # add angular and corrected translational velocity vectors
    windfields[1:] += v_trans[1:, None, :] * v_trans_corr[1:, :, None]
    windfields[np.isnan(windfields)] = 0
    windfields[0, :, :] = 0
    [reachable_centr_idx] = track_centr_msk.nonzero()
    return windfields, reachable_centr_idx

def _close_centroids(
    t_lat: np.ndarray,
    t_lon: np.ndarray,
    centroids: np.ndarray,
    buffer: float,
) -> np.ndarray:
    """Check whether centroids lay within a rectangular buffer around track positions

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
    buffer : float
        Size of the buffer (in degrees).

    Returns
    -------
    mask : np.ndarray of shape (ncentroids,)
        Mask that is True for close centroids and False for other centroids.
    """
    centr_lat, centr_lon = centroids[:, 0], centroids[:, 1]
    # check for each track position which centroids are within buffer, uses NumPy's broadcasting
    mask = ((t_lat[:, None] - buffer <= centr_lat[None])
            & (centr_lat[None] <= t_lat[:, None] + buffer)
            & (t_lon[:, None] - buffer <= centr_lon[None])
            & (centr_lon[None] <= t_lon[:, None] + buffer))
    # for each centroid, check whether it is in the buffer for any of the track positions
    return mask.any(axis=0)

def _vtrans(
    t_lat: np.ndarray,
    t_lon: np.ndarray,
    t_tstep: np.ndarray,
    metric: str = "equirect"
) -> Tuple[np.ndarray, np.ndarray]:
    """Translational vector and velocity (in m/s) at each track node.

    Parameters
    ----------
    t_lat : np.ndarray
        track latitudes (in degrees)
    t_lon : np.ndarray
        track longitudes (in degrees)
    t_tstep : np.ndarray
        track time steps (in hours)
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See `dist_approx` function in `climada.util.coordinates`.
        Default: "equirect".

    Returns
    -------
    v_trans_norm : np.ndarray of same shape as input
        Absolute velocity in meters per second. The first velocity is always 0.
    v_trans : np.ndarray
        Directional vectors of velocity (in meters per second).
    """
    v_trans = np.zeros((t_lat.size, 2))
    v_trans_norm = np.zeros((t_lat.size,))
    norm, vec = u_coord.dist_approx(t_lat[:-1, None], t_lon[:-1, None],
                                    t_lat[1:, None], t_lon[1:, None],
                                    log=True, normalize=False, method=metric)
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


def _bs_holland_2008(
    v_trans: np.ndarray,
    penv: np.ndarray,
    pcen: np.ndarray,
    prepcen: np.ndarray,
    lat: np.ndarray,
    tint: np.ndarray
) -> np.ndarray:
    """Holland's 2008 b-value estimate for sustained surface winds.

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
    v_trans : np.ndarray
        Translational wind (in m/s).
    penv : np.ndarray
        Environmental pressure (in hPa).
    pcen : np.ndarray
        Central pressure (in hPa).
    prepcen : np.ndarray
        Central pressure (in hPa) at previous track position.
    lat : np.ndarray
        Latitude (in degrees).
    tint : np.ndarray
        Time step (in h).

    Returns
    -------
    b_s : np.ndarray
        Holland b-value
    """
    pdelta = penv - pcen
    hol_xx = 0.6 * (1. - pdelta / 215)
    hol_b = -4.4e-5 * pdelta**2 + 0.01 * pdelta + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0
    return np.clip(hol_b, 1, 2.5)

def _v_max_s_holland_2008(
    penv: np.ndarray,
    pcen: np.ndarray,
    b_s: np.ndarray,
) -> np.ndarray:
    """Compute maximum surface winds from pressure according to Holland 2008.

    This function implements equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    v_ms = [b_s / (rho * e) * (penv - pcen)]^0.5

    where `b_s` is Holland b-value (see `_bs_holland_2008`), e is Euler's number, rho is the
    density of air, `penv` is environmental, and `pcen` is central pressure.

    Parameters
    ----------
    penv : np.ndarray
        Environmental pressure (in hPa).
    pcen : np.ndarray
        Central pressure (in hPa).
    b_s : np.ndarray
        Holland's b-parameter according to `_bs_holland_2008`.

    Returns
    -------
    v_max_s : np.ndarray
        Maximum surface winds (in m/s).
    """
    # the factor 100 is from conversion between mbar (hPa) and pascal (Pa)
    v_squared = b_s / (RHO_AIR * np.exp(1)) * 100 * (penv - pcen)
    return np.sqrt(v_squared)

def _B_holland_1980(
    gradient_winds: np.ndarray,
    penv: np.ndarray,
    pcen: np.ndarray,
) -> np.ndarray:  # pylint: disable=invalid-name
    """Holland's 1980 B-value computation for gradient-level winds.

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
    gradient_winds : np.ndarray
        Maximum gradient-level wind speeds (m/s) of the tropical cyclone. If your data are maximum
        surface wind speeds (e.g. from IBTrACS), make sure to subtract translational wind speed and
        convert to gradient-level winds first.
    penv : np.ndarray
        Environmental pressure (hPa).
    pcen : np.ndarray
        Central pressure (hPa).

    Returns
    -------
    B : np.ndarray
        Holland b-value
    """
    # the factor 100 is from conversion between mbar and pascal
    pdelta = 100 * (penv - pcen)
    hol_b = gradient_winds**2 * np.exp(1) * RHO_AIR / np.fmax(np.spacing(1), pdelta)
    return np.clip(hol_b, 1, 2.5)

def _x_holland_2010(
    d_centr: np.ndarray,
    r_max: np.ndarray,
    v_max_s: np.ndarray,
    hol_b: np.ndarray,
    close_centr: np.ndarray,
    v_n: Union[float, np.ndarray] = 17.0,
    r_n: Union[float, np.ndarray] = 300
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
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in km) between centroids and track nodes.
    r_max : np.ndarray of shape (nnodes,)
        Radius (in km) of maximum winds at each track node.
    v_max_s : np.ndarray of shape (nnodes,)
        Maximum surface winds (in m/s) at each track node.
    hol_b : np.ndarray of shape (nnodes,)
        Holland's b parameter at each track node.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    v_n : np.ndarray of shape (nnodes,) or float, optional
        Peripheral wind speeds (in m/s) at radius `r_n` outside of radius of maximum winds `r_max`.
        In absence of a second wind speed measurement, this value defaults to 17 m/s following
        Holland et al. 2010 (at a radius of 300 km).
    r_n : np.ndarray of shape (nnodes,) or float, optional
        Radius (in km) where the peripheral wind speed `v_n` is measured (or assumed).
        In absence of a second wind speed measurement, this value defaults to 300 km following
        Holland et al. 2010.

    Returns
    -------
    x : np.ndarray of shape (nnodes, ncentroids)
        Exponents according to Holland et al. 2010.
    """
    x = np.zeros_like(d_centr)
    r_max, v_max_s, hol_b, d_centr, v_n, r_n = [
        ar[close_centr] for ar in np.broadcast_arrays(
            r_max[:, None], v_max_s[:, None], hol_b[:, None], d_centr,
            np.atleast_1d(v_n)[:, None], np.atleast_1d(r_n)[:, None])
    ]

    # compute peripheral exponent from second measurement
    r_max_norm = (r_max / r_n)**hol_b
    x_n = np.log(v_n / v_max_s) / np.log(r_max_norm * np.exp(1 - r_max_norm))

    # linearly interpolate between max exponent and peripheral exponent
    x_max = 0.5
    x[close_centr] = x_max + np.fmax(0, d_centr - r_max) * (x_n - x_max) / (r_n - r_max)
    x[close_centr] = np.clip(x[close_centr], 0.0, 0.5)
    return x


def _stat_holland_2010(
    d_centr: np.ndarray,
    v_max_s: np.ndarray,
    r_max: np.ndarray,
    hol_b: np.ndarray,
    close_centr: np.ndarray,
    x: Union[float, np.ndarray]
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
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in km) between centroids and track nodes.
    v_max_s : np.ndarray of shape (nnodes,)
        Maximum surface wind speeds (in m/s) of the tropical cyclone according to
        `_v_max_s_holland_2008`.
    r_max : np.ndarray of shape (nnodes,)
        Radius (in km) of maximum winds at each track node.
    hol_b : np.ndarray of shape (nnodes,)
        Holland's b parameter at each track node according to `_bs_holland_2008`.
    close_centr : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    x : np.ndarray of shape (nnodes, ncentroids) or float, optional
        The exponent according to `_x_holland_2010`.

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (in m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    d_centr, v_max_s, r_max, hol_b, x = [
        ar[close_centr] for ar in np.broadcast_arrays(
            d_centr, v_max_s[:, None], r_max[:, None], hol_b[:, None], x)
    ]

    r_max_norm = (r_max / d_centr)**hol_b
    v_ang[close_centr] = v_max_s * (r_max_norm * np.exp(1 - r_max_norm))**x
    return v_ang

def _stat_holland_1980(
    d_centr: np.ndarray,
    r_max: np.ndarray,
    hol_b: np.ndarray,
    penv: np.ndarray,
    pcen: np.ndarray,
    lat: np.ndarray,
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
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in km) between centroids and track nodes.
    r_max : np.ndarray of shape (nnodes,)
        Radius (in km) of maximum winds at each track node.
    hol_b : np.ndarray of shape (nnodes,)
        Holland's b parameter at each track node.
    penv : np.ndarray of shape (nnodes,)
        Environmental pressure (in hPa) at each track node.
    pcen : np.ndarray of shape (nnodes,)
        Central pressure (in hPa) at each track node.
    lat : np.ndarray of shape (nnodes,)
        Latitudinal coordinate (in degrees) of each track node.
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
    d_centr, r_max, hol_b, penv, pcen, lat = [
        ar[close_centr] for ar in np.broadcast_arrays(
            d_centr, r_max[:, None], hol_b[:, None], penv[:, None], pcen[:, None], lat[:, None])
    ]

    r_coriolis = 0
    if not cyclostrophic:
        # d_centr is in km, convert to m and apply Coriolis parameter
        r_coriolis = 0.5 * KM_TO_M * d_centr * _coriolis_parameter(lat)

    # the factor 100 is from conversion between mbar and pascal
    r_max_norm = (r_max / d_centr)**hol_b
    sqrt_term = 100 * hol_b / RHO_AIR * r_max_norm * (penv - pcen) \
                * np.exp(-r_max_norm) + r_coriolis**2

    v_ang[close_centr] = np.sqrt(np.fmax(0, sqrt_term)) - r_coriolis
    return v_ang

def _stat_er_2011(
    d_centr: np.ndarray,
    v_max: np.ndarray,
    r_max: np.ndarray,
    lat: np.ndarray,
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
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in km) between centroids and track nodes.
    v_max : np.ndarray of shape (nnodes,)
        Maximum wind speeds (in m/s) of the tropical cyclone at each track node.
    r_max : np.ndarray of shape (nnodes,)
        Radius (in km) of maximum winds at each track node.
    lat : np.ndarray of shape (nnodes,)
        Latitudinal coordinate (in degrees) of each track node.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0) in
        the computation of M_max. Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (m/s) in angular direction.
    """
    # convert to SI units
    r_max = KM_TO_M * r_max
    d_centr = KM_TO_M * d_centr

    # compute the momentum at the maximum
    M_max = r_max * v_max

    if not cyclostrophic:
        # add the influence of the Coriolis force
        M_max += 0.5 * _coriolis_parameter(lat) * r_max**2

    # rescale the momentum using formula (36) in Emanuel and Rotunno 2011 with Ck == Cd
    r_max_norm = (d_centr / r_max[:, None])**2
    M = M_max[:, None] * 2 * r_max_norm / (1 + r_max_norm)

    # extract the velocity from the rescaled momentum through division by r
    v_ang = np.fmax(0, M / (d_centr + 1e-11))

    return v_ang
