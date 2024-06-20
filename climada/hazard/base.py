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

Define Hazard.
"""

__all__ = ['Hazard']

import copy
import datetime as dt
import logging
from typing import Optional,List
import warnings

import numpy as np
from pathos.pools import ProcessPool as Pool
from scipy import sparse

from climada import CONFIG
from climada.hazard.plot import HazardPlot
from climada.hazard.io import HazardIO
from climada.hazard.centroids.centr import Centroids
import climada.util.checker as u_check
import climada.util.constants as u_const
import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt


LOGGER = logging.getLogger(__name__)


class Hazard(HazardIO, HazardPlot):
    """
    Contains events of some hazard type defined at centroids. Loads from
    files with format defined in FILE_EXT.

    Attributes
    ----------
    haz_type : str
        two-letters hazard-type string, e.g., "TC" (tropical cyclone), "RF" (river flood) or "WF"
        (wild fire).
        Note: The acronym is used as reference to the hazard when centroids of multiple hazards
        are assigned to an ``Exposures`` object.
    units : str
        units of the intensity
    centroids : Centroids
        centroids of the events
    event_id : np.array
        id (>0) of each event
    event_name : list(str)
        name of each event (default: event_id)
    date : np.array
        integer date corresponding to the proleptic
        Gregorian ordinal, where January 1 of year 1 has ordinal 1
        (ordinal format of datetime library)
    orig : np.array
        flags indicating historical events (True)
        or probabilistic (False)
    frequency : np.array
        frequency of each event
    frequency_unit : str
        unit of the frequency (default: "1/year")
    intensity : sparse.csr_matrix
        intensity of the events at centroids
    fraction : sparse.csr_matrix
        fraction of affected exposures for each event at each centroid.
        If empty (all 0), it is ignored in the impact computations
        (i.e., is equivalent to fraction is 1 everywhere).
    """
    intensity_thres = 10
    """Intensity threshold per hazard used to filter lower intensities. To be
    set for every hazard type"""

    vars_oblig = {'units',
                  'centroids',
                  'event_id',
                  'frequency',
                  'intensity',
                  'fraction'
                  }
    """Name of the variables needed to compute the impact. Types: scalar, str,
    list, 1dim np.array of size num_events, scipy.sparse matrix of shape
    num_events x num_centroids, Centroids."""

    vars_def = {'date',
                'orig',
                'event_name',
                'frequency_unit'
                }
    """Name of the variables used in impact calculation whose value is
    descriptive and can therefore be set with default values. Types: scalar,
    string, list, 1dim np.array of size num_events.
    """

    vars_opt = set()
    """Name of the variables that aren't need to compute the impact. Types:
    scalar, string, list, 1dim np.array of size num_events."""

    def __init__(self,
                 haz_type: str = "",
                 pool: Optional[Pool] = None,
                 units: str = "",
                 centroids: Optional[Centroids] = None,
                 event_id: Optional[np.ndarray] = None,
                 frequency: Optional[np.ndarray] = None,
                 frequency_unit: str = u_const.DEF_FREQ_UNIT,
                 event_name: Optional[List[str]] = None,
                 date: Optional[np.ndarray] = None,
                 orig: Optional[np.ndarray] = None,
                 intensity: Optional[sparse.csr_matrix] = None,
                 fraction: Optional[sparse.csr_matrix] = None):
        """
        Initialize values.

        Parameters
        ----------
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
        pool : pathos.pool, optional
            Pool that will be used for parallel computation when applicable. Default: None
        units : str, optional
            units of the intensity. Defaults to empty string.
        centroids : Centroids, optional
            centroids of the events. Defaults to empty Centroids object.
        event_id : np.array, optional
            id (>0) of each event. Defaults to empty array.
        event_name : list(str), optional
            name of each event (default: event_id). Defaults to empty list.
        date : np.array, optional
            integer date corresponding to the proleptic
            Gregorian ordinal, where January 1 of year 1 has ordinal 1
            (ordinal format of datetime library). Defaults to empty array.
        orig : np.array, optional
            flags indicating historical events (True)
            or probabilistic (False). Defaults to empty array.
        frequency : np.array, optional
            frequency of each event. Defaults to empty array.
        frequency_unit : str, optional
            unit of the frequency (default: "1/year").
        intensity : sparse.csr_matrix, optional
            intensity of the events at centroids. Defaults to empty matrix.
        fraction : sparse.csr_matrix, optional
            fraction of affected exposures for each event at each centroid. Defaults to
            empty matrix.

        Examples
        --------
        Initialize using keyword arguments:

        >>> haz = Hazard('TC', intensity=sparse.csr_matrix(np.zeros((2, 2))))

        Take hazard values from file:

        >>> haz = Hazard.from_mat(HAZ_DEMO_MAT, 'demo')

        """
        self.haz_type = haz_type
        self.units = units
        self.centroids = centroids if centroids is not None else Centroids(
            lat=np.empty(0), lon=np.empty(0))
        # following values are defined for each event
        self.event_id = event_id if event_id is not None else np.array([], int)
        self.frequency = frequency if frequency is not None else np.array(
            [], float)
        self.frequency_unit = frequency_unit
        self.event_name = event_name if event_name is not None else list()
        self.date = date if date is not None else np.array([], int)
        self.orig = orig if orig is not None else np.array([], bool)
        # following values are defined for each event and centroid
        self.intensity = intensity if intensity is not None else sparse.csr_matrix(
            np.empty((0, 0)))  # events x centroids
        self.fraction = fraction if fraction is not None else sparse.csr_matrix(
            self.intensity.shape)  # events x centroids

        self.pool = pool
        if self.pool:
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)

    @classmethod
    def get_default(cls, attribute):
        """Get the Hazard type default for a given attribute.

        Parameters
        ----------
        attribute : str
            attribute name

        Returns
        ------
        Any
        """
        return {
            'frequency_unit': u_const.DEF_FREQ_UNIT,
        }.get(attribute)

    def check(self):
        """Check dimension of attributes.

        Raises
        ------
        ValueError
        """
        self._check_events()

    def reproject_vector(self, dst_crs):
        """Change current point data to a a given projection

        Parameters
        ----------
        dst_crs : crs
            reproject to given crs
        """
        self.centroids.gdf.to_crs(dst_crs, inplace=True)
        self.check()

    def select(self, event_names=None, event_id=None, date=None, orig=None,
               reg_id=None, extent=None, reset_frequency=False):
        """Select events matching provided criteria

        The frequency of events may need to be recomputed (see `reset_frequency`)!

        Parameters
        ----------
        event_names : list of str, optional
            Names of events.
        event_id : list of int, optional
            Id of events. Default is None.
        date : array-like of length 2 containing str or int, optional
            (initial date, final date) in string ISO format ('2011-01-02') or datetime
            ordinal integer.
        orig : bool, optional
            Select only historical (True) or only synthetic (False) events.
        reg_id : int, optional
            Region identifier of the centroids' region_id attibute.
        extent: tuple(float, float, float, float), optional
            Extent of centroids as (min_lon, max_lon, min_lat, max_lat).
            The default is None.
        reset_frequency : bool, optional
            Change frequency of events proportional to difference between first and last
            year (old and new). Default: False.

        Returns
        -------
        haz : Hazard or None
            If no event matching the specified criteria is found, None is returned.
        """
        # pylint: disable=unidiomatic-typecheck
        if type(self) is Hazard:
            haz = Hazard(self.haz_type)
        else:
            haz = self.__class__()

        #filter events
        sel_ev = np.ones(self.event_id.size, dtype=bool)

        # filter events by date
        if date is not None:
            date_ini, date_end = date
            if isinstance(date_ini, str):
                date_ini = u_dt.str_to_date(date[0])
                date_end = u_dt.str_to_date(date[1])
            sel_ev &= (date_ini <= self.date) & (self.date <= date_end)
            if not np.any(sel_ev):
                LOGGER.info('No hazard in date range %s.', date)
                return None

        # filter events hist/synthetic
        if orig is not None:
            sel_ev &= (self.orig.astype(bool) == orig)
            if not np.any(sel_ev):
                LOGGER.info('No hazard with %s original events.', str(orig))
                return None

        # filter events based on name
        sel_ev = np.argwhere(sel_ev).reshape(-1)
        if event_names is not None:
            filtered_events = [self.event_name[i] for i in sel_ev]
            try:
                new_sel = [filtered_events.index(n) for n in event_names]
            except ValueError as err:
                name = str(err).replace(" is not in list", "")
                LOGGER.info('No hazard with name %s', name)
                return None
            sel_ev = sel_ev[new_sel]

        # filter events based on id
        if event_id is not None:
            # preserves order of event_id
            sel_ev = np.array([
                np.argwhere(self.event_id == n)[0,0]
                for n in event_id
                if n in self.event_id[sel_ev]
                ])

        # filter centroids
        sel_cen = self.centroids.select_mask(reg_id=reg_id, extent=extent)
        if not np.any(sel_cen):
            LOGGER.info('No hazard centroids within extent and region')
            return None

        # Sanitize fraction, because we check non-zero entries later
        self.fraction.eliminate_zeros()

        # Perform attribute selection
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 \
                    and var_val.size > 0:
                setattr(haz, var_name, var_val[sel_ev])
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(haz, var_name, var_val[sel_ev, :][:, sel_cen])
            elif isinstance(var_val, list) and var_val:
                setattr(haz, var_name, [var_val[idx] for idx in sel_ev])
            elif var_name == 'centroids':
                if reg_id is None and extent is None:
                    new_cent = var_val
                else:
                    new_cent = var_val.select(sel_cen=sel_cen)
                setattr(haz, var_name, new_cent)
            else:
                setattr(haz, var_name, var_val)

        # reset frequency if date span has changed (optional):
        if reset_frequency:
            if self.frequency_unit not in ['1/year', 'annual', '1/y', '1/a']:
                LOGGER.warning("Resetting the frequency is based on the calendar year of given"
                    " dates but the frequency unit here is %s. Consider setting the frequency"
                    " manually for the selection or changing the frequency unit to %s.",
                    self.frequency_unit, u_const.DEF_FREQ_UNIT)
            year_span_old = np.abs(dt.datetime.fromordinal(self.date.max()).year -
                                   dt.datetime.fromordinal(self.date.min()).year) + 1
            year_span_new = np.abs(dt.datetime.fromordinal(haz.date.max()).year -
                                   dt.datetime.fromordinal(haz.date.min()).year) + 1
            haz.frequency = haz.frequency * year_span_old / year_span_new

        # Check if new fraction is zero everywhere
        if self._get_fraction() is not None and haz._get_fraction() is None:
            raise RuntimeError(
                "Your selection created a Hazard object where the fraction matrix is "
                "zero everywhere. This hazard will have zero impact everywhere. "
                "We are catching this condition because of an implementation detail: "
                "A fraction matrix without nonzero-valued entries will be completely "
                "ignored. This is surely not what you intended. If you really want to, "
                "you can circumvent this error by setting your original fraction "
                "matrix to zero everywhere, but there probably is no point in doing so."
            )

        haz.sanitize_event_ids()
        return haz

    def select_tight(self, buffer=u_coord.NEAREST_NEIGHBOR_THRESHOLD / u_const.ONE_LAT_KM,
                     val='intensity'):
        """
        Reduce hazard to those centroids spanning a minimal box which
        contains all non-zero intensity or fraction points.

        Parameters
        ----------
        buffer : float, optional
            Buffer of box in the units of the centroids.
            The default is approximately equal to the default threshold
            from the assign_centroids method (works if centroids in
            lat/lon)
        val: string, optional
            Select tight by non-zero 'intensity' or 'fraction'. The
            default is 'intensity'.

        Returns
        -------
        Hazard
            Copy of the Hazard with centroids reduced to minimal box. All other
            hazard properties are carried over without changes.

        See also
        --------
        self.select: Method to select centroids by lat/lon extent
        util.coordinates.match_coordinates: algorithm to match centroids.

        """

        if val == 'intensity':
            cent_nz = (self.intensity != 0).sum(axis=0).nonzero()[1]
        if val == 'fraction':
            cent_nz = (self.fraction != 0).sum(axis=0).nonzero()[1]
        lon_nz = self.centroids.lon[cent_nz]
        lat_nz = self.centroids.lat[cent_nz]
        return self.select(extent=u_coord.toggle_extent_bounds(
            u_coord.latlon_bounds(lat=lat_nz, lon=lon_nz, buffer=buffer)
        ))

    def local_exceedance_inten(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance intensity map for given return periods.

        Parameters
        ----------
        return_periods : np.array
            return periods to consider

        Returns
        -------
        inten_stats: np.array
        """
        # warn if return period is above return period of rarest event:
        for period in return_periods:
            if period > 1 / self.frequency.min():
                LOGGER.warning('Return period %1.1f exceeds max. event return period.', period)
        LOGGER.info('Computing exceedance intenstiy map for return periods: %s',
                    return_periods)
        num_cen = self.intensity.shape[1]
        inten_stats = np.zeros((len(return_periods), num_cen))
        cen_step = CONFIG.max_matrix_size.int() // self.intensity.shape[0]
        if not cen_step:
            raise ValueError('Increase max_matrix_size configuration parameter to >'
                             f' {self.intensity.shape[0]}')
        # separte in chunks
        chk = -1
        for chk in range(int(num_cen / cen_step)):
            self._loc_return_inten(
                np.array(return_periods),
                self.intensity[:, chk * cen_step:(chk + 1) * cen_step].toarray(),
                inten_stats[:, chk * cen_step:(chk + 1) * cen_step])
        self._loc_return_inten(
            np.array(return_periods),
            self.intensity[:, (chk + 1) * cen_step:].toarray(),
            inten_stats[:, (chk + 1) * cen_step:])
        # set values below 0 to zero if minimum of hazard.intensity >= 0:
        if np.min(inten_stats) < 0 <= self.intensity.min():
            LOGGER.warning('Exceedance intenstiy values below 0 are set to 0. \
                   Reason: no negative intensity values were found in hazard.')
            inten_stats[inten_stats < 0] = 0
        return inten_stats

    def sanitize_event_ids(self):
        """Make sure that event ids are unique"""
        if np.unique(self.event_id).size != self.event_id.size:
            LOGGER.debug('Resetting event_id.')
            self.event_id = np.arange(1, self.event_id.size + 1)

    def local_return_period(self, hazard_intensities):
        """Compute local return periods for given hazard intensities.
    
        Parameters
        ----------
        hazard_intensities : np.array
            Hazard intensities to consider.
    
        Returns
        -------
        return_periods : np.array
            Array containing computed local return periods for given hazard intensities.
        """
        # Ensure hazard_intensities is a numpy array
        hazard_intensities = np.array(hazard_intensities)
        
        num_cen = self.intensity.shape[1]
        return_periods = np.zeros((len(hazard_intensities), num_cen))  # Adjusted for 2D structure
        
        # Process each centroid in chunks as in local_exceedance_inten
        cen_step = CONFIG.max_matrix_size.int() // self.intensity.shape[0]
        if not cen_step:
            raise ValueError('Increase max_matrix_size configuration parameter to >'
                             f'{self.intensity.shape[0]}')
        
        chk = -1
        for chk in range(int(num_cen / cen_step)):
            self._loc_return_period(
                hazard_intensities,
                self.intensity[:, chk * cen_step:(chk + 1) * cen_step].toarray(),
                return_periods[:, chk * cen_step:(chk + 1) * cen_step])
        
        if (chk + 1) * cen_step < num_cen:  # Check if there's a remainder
            self._loc_return_period(
                hazard_intensities,
                self.intensity[:, (chk + 1) * cen_step:].toarray(),
                return_periods[:, (chk + 1) * cen_step:])
        
        return return_periods
    
    def get_event_id(self, event_name):
        """Get an event id from its name. Several events might have the same
        name.

        Parameters
        ----------
        event_name: str
            Event name

        Returns
        -------
        list_id: np.array(int)
        """
        list_id = self.event_id[[i_name for i_name, val_name in enumerate(self.event_name)
                                 if val_name == event_name]]
        if list_id.size == 0:
            raise ValueError(f"No event with name: {event_name}")
        return list_id

    def get_event_name(self, event_id):
        """Get the name of an event id.

        Parameters
        ----------
        event_id: int
            id of the event

        Returns
        -------
            str

        Raises
        ------
            ValueError
        """
        try:
            return self.event_name[np.argwhere(
                self.event_id == event_id)[0][0]]
        except IndexError as err:
            raise ValueError(f"No event with id: {event_id}") from err

    def get_event_date(self, event=None):
        """Return list of date strings for given event or for all events,
        if no event provided.

        Parameters
        ----------
        event: str or int, optional
            event name or id.

        Returns
        -------
        l_dates: list(str)
        """
        if event is None:
            l_dates = [u_dt.date_to_str(date) for date in self.date]
        elif isinstance(event, str):
            ev_ids = self.get_event_id(event)
            l_dates = [
                u_dt.date_to_str(self.date[np.argwhere(self.event_id == ev_id)[0][0]])
                for ev_id in ev_ids]
        else:
            ev_idx = np.argwhere(self.event_id == event)[0][0]
            l_dates = [u_dt.date_to_str(self.date[ev_idx])]
        return l_dates

    def calc_year_set(self):
        """From the dates of the original events, get number yearly events.

        Returns
        -------
        orig_yearset: dict
            key are years, values array with event_ids of that year

        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date[self.orig]])
        orig_yearset = {}
        for year in np.unique(orig_year):
            orig_yearset[year] = self.event_id[self.orig][orig_year == year]
        return orig_yearset

    def remove_duplicates(self):
        """Remove duplicate events (events with same name and date)."""
        events = list(zip(self.event_name, self.date))
        set_ev = set(events)
        if len(set_ev) == self.event_id.size:
            return
        unique_pos = sorted([events.index(event) for event in set_ev])
        for var_name, var_val in vars(self).items():
            if isinstance(var_val, sparse.csr_matrix):
                setattr(self, var_name, var_val[unique_pos, :])
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, var_val[unique_pos])
            elif isinstance(var_val, list) and len(var_val) > 0:
                setattr(self, var_name, [var_val[p] for p in unique_pos])

    def set_frequency(self, yearrange=None):
        """Set hazard frequency from yearrange or intensity matrix.

        Parameters
        ----------
        yearrange: tuple or list, optional
            year range to be used to compute frequency
            per event. If yearrange is not given (None), the year range is
            derived from self.date
        """
        if self.frequency_unit not in ['1/year', 'annual', '1/y', '1/a']:
            LOGGER.warning("setting the frequency on a hazard object who's frequency unit"
                "is %s and not %s will most likely lead to unexpected results",
                self.frequency_unit, u_const.DEF_FREQ_UNIT)
        if not yearrange:
            delta_time = dt.datetime.fromordinal(int(np.max(self.date))).year - \
                         dt.datetime.fromordinal(int(np.min(self.date))).year + 1
        else:
            delta_time = max(yearrange) - min(yearrange) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @property
    def size(self):
        """Return number of events."""
        return self.event_id.size

    def _events_set(self):
        """Generate set of tuples with (event_name, event_date)"""
        ev_set = set()
        for ev_name, ev_date in zip(self.event_name, self.date):
            ev_set.add((ev_name, ev_date))
        return ev_set

    def _loc_return_inten(self, return_periods, inten, exc_inten):
        """Compute local exceedence intensity for given return period.

        Parameters
        ----------
        return_periods: np.array
            return periods to consider
        cen_pos: int
            centroid position

        Returns
        -------
            np.array
        """
        # sorted intensity
        sort_pos = np.argsort(inten, axis=0)[::-1, :]
        columns = np.ones(inten.shape, int)
        # pylint: disable=unsubscriptable-object  # pylint/issues/3139
        columns *= np.arange(columns.shape[1])
        inten_sort = inten[sort_pos, columns]
        # cummulative frequency at sorted intensity
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)

        for cen_idx in range(inten.shape[1]):
            exc_inten[:, cen_idx] = self._cen_return_inten(
                inten_sort[:, cen_idx], freq_sort[:, cen_idx],
                self.intensity_thres, return_periods)
            
    def _loc_return_period(self, hazard_intensities, inten, return_periods):
        """Compute local return periods for given hazard intensities for a specific chunk of data.
    
        Parameters
        ----------
        hazard_intensities: np.array
            Given hazard intensities for which to calculate return periods.
        inten: np.array
            The intensity array for a specific chunk of data.
        return_periods: np.array
            Array to store computed return periods for the given hazard intensities.
        """
        # Assuming inten is sorted and calculating cumulative frequency
        sort_pos = np.argsort(inten, axis=0)[::-1, :]
        inten_sort = inten[sort_pos, np.arange(inten.shape[1])]
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)
    
        for cen_idx in range(inten.shape[1]):
            sorted_inten_cen = inten_sort[:, cen_idx]
            cum_freq_cen = freq_sort[:, cen_idx]
    
            for i, intensity in enumerate(hazard_intensities):
                # Find the first occurrence where the intensity is less than the sorted intensities
                exceedance_index = np.searchsorted(sorted_inten_cen[::-1], intensity, side='right')
    
                # Calculate exceedance probability
                if exceedance_index < len(cum_freq_cen):
                    exceedance_probability = cum_freq_cen[-exceedance_index - 1]
                else:
                    exceedance_probability = 0  # Or set a default minimal probability
    
                # Calculate and store return period
                if exceedance_probability > 0:
                    return_periods[i, cen_idx] = 1 / exceedance_probability
                else:
                    return_periods[i, cen_idx] = np.nan  

    def _check_events(self):
        """Check that all attributes but centroids contain consistent data.
        Put default date, event_name and orig if not provided. Check not
        repeated events (i.e. with same date and name)

        Raises
        ------
            ValueError
        """
        num_ev = len(self.event_id)
        num_cen = self.centroids.size
        if np.unique(self.event_id).size != num_ev:
            raise ValueError("There are events with the same identifier.")

        u_check.check_oligatories(self.__dict__, self.vars_oblig, 'Hazard.',
                                  num_ev, num_ev, num_cen)
        u_check.check_optionals(self.__dict__, self.vars_opt, 'Hazard.', num_ev)
        self.event_name = u_check.array_default(num_ev, self.event_name,
                                                'Hazard.event_name', list(self.event_id))
        self.date = u_check.array_default(num_ev, self.date, 'Hazard.date',
                                          np.ones(self.event_id.shape, dtype=int))
        self.orig = u_check.array_default(num_ev, self.orig, 'Hazard.orig',
                                          np.zeros(self.event_id.shape, dtype=bool))
        if len(self._events_set()) != num_ev:
            raise ValueError("There are events with same date and name.")

    @staticmethod
    def _cen_return_inten(inten, freq, inten_th, return_periods):
        """From ordered intensity and cummulative frequency at centroid, get
        exceedance intensity at input return periods.

        Parameters
        ----------
        inten: np.array
            sorted intensity at centroid
        freq: np.array
            cummulative frequency at centroid
        inten_th: float
            intensity threshold
        return_periods: np.array
            return periods

        Returns
        -------
            np.array
        """
        inten_th = np.asarray(inten > inten_th).squeeze()
        inten_cen = inten[inten_th]
        freq_cen = freq[inten_th]
        if not inten_cen.size:
            return np.zeros((return_periods.size,))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=0)
        inten_fit = np.polyval(pol_coef, np.log(1 / return_periods))
        wrong_inten = (return_periods > np.max(1 / freq_cen)) & np.isnan(inten_fit)
        inten_fit[wrong_inten] = 0.

        return inten_fit

    def append(self, *others):
        """Append the events and centroids to this hazard object.

        All of the given hazards must be of the same type and use the same units as self. The
        centroids of all hazards must have the same CRS.

        The following kinds of object attributes are processed:

        - All centroids are combined together using `Centroids.union`.
        - Lists, 1-dimensional arrays (NumPy) and sparse CSR matrices (SciPy) are concatenated.
          Sparse matrices are concatenated along the first (vertical) axis.

        For any other type of attribute: A ValueError is raised if an attribute of that name is
        not defined in all of the non-empty hazards at least. However, there is no check that the
        attribute value is identical among the given hazard objects. The initial attribute value of
        `self` will not be modified.

        Note: Each of the hazard's `centroids` attributes might be modified in place in the sense
        that missing properties are added, but existing ones are not overwritten. In case of raster
        centroids, conversion to point centroids is applied so that raster information (meta) is
        lost. For more information, see `Centroids.union`.

        Parameters
        ----------
        others : one or more climada.hazard.Hazard objects
            Hazard instances to append to self

        Raises
        ------
        TypeError, ValueError

        See Also
        --------
        Hazard.concat : concatenate 2 or more hazards
        Centroids.union : combine centroids
        """
        # pylint: disable=no-member, protected-access
        if len(others) == 0:
            return
        haz_list = [self] + list(others)
        haz_list_nonempty = [haz for haz in haz_list if haz.size > 0]

        for haz in haz_list:
            haz._check_events()

        # check type, unit, and attribute consistency among hazards
        haz_types = {haz.haz_type for haz in haz_list if haz.haz_type != ''}
        if len(haz_types) > 1:
            raise ValueError(f"The given hazards are of different types: {haz_types}. "
                             "The hazards are incompatible and cannot be concatenated.")
        self.haz_type = haz_types.pop()

        haz_classes = {type(haz) for haz in haz_list}
        if len(haz_classes) > 1:
            raise TypeError(f"The given hazards are of different classes: {haz_classes}. "
                            "The hazards are incompatible and cannot be concatenated.")

        freq_units = {haz.frequency_unit for haz in haz_list}
        if len(freq_units) > 1:
            raise ValueError(f"The given hazards have different frequency units: {freq_units}. "
                             "The hazards are incompatible and cannot be concatenated.")
        self.frequency_unit = freq_units.pop()

        units = {haz.units for haz in haz_list if haz.units != ''}
        if len(units) > 1:
            raise ValueError(f"The given hazards use different units: {units}. "
                             "The hazards are incompatible and cannot be concatenated.")
        if len(units) == 0:
            units = {''}
        self.units = units.pop()

        attributes = sorted(set.union(*[set(vars(haz).keys()) for haz in haz_list]))
        for attr_name in attributes:
            if not all(hasattr(haz, attr_name) for haz in haz_list_nonempty):
                raise ValueError(f"Attribute {attr_name} is not shared by all hazards. "
                                 "The hazards are incompatible and cannot be concatenated.")

        # map individual centroids objects to union
        centroids = Centroids.union(*[haz.centroids for haz in haz_list])
        hazcent_in_cent_idx_list = [
            u_coord.match_coordinates(haz.centroids.coord, centroids.coord, threshold=0)
            for haz in haz_list_nonempty
        ]

        # concatenate array and list attributes of non-empty hazards
        for attr_name in attributes:
            attr_val_list = [getattr(haz, attr_name) for haz in haz_list_nonempty]
            if isinstance(attr_val_list[0], sparse.csr_matrix):
                # map sparse matrix onto centroids
                setattr(self, attr_name, sparse.vstack([
                    sparse.csr_matrix(
                        (matrix.data, cent_idx[matrix.indices], matrix.indptr),
                        shape=(matrix.shape[0], centroids.size)
                    )
                    for matrix, cent_idx in zip(attr_val_list, hazcent_in_cent_idx_list)
                ], format='csr'))
            elif isinstance(attr_val_list[0], np.ndarray) and attr_val_list[0].ndim == 1:
                setattr(self, attr_name, np.hstack(attr_val_list))
            elif isinstance(attr_val_list[0], list):
                setattr(self, attr_name, sum(attr_val_list, []))

        self.centroids = centroids
        self.sanitize_event_ids()

    @classmethod
    def concat(cls, haz_list):
        """
        Concatenate events of several hazards of same type.

        This function creates a new hazard of the same class as the first hazard in the given list
        and then applies the `append` method. Please refer to the docs of `Hazard.append` for
        caveats and limitations of the concatenation procedure.

        For centroids, lists, arrays and sparse matrices, the remarks in `Hazard.append`
        apply. All other attributes are copied from the first object in `haz_list`.

        Note that `Hazard.concat` can be used to concatenate hazards of a subclass. The result's
        type will be the subclass. However, calling `concat([])` (with an empty list) is equivalent
        to instantiation without init parameters. So, `Hazard.concat([])` is equivalent to
        `Hazard()`. If `HazardB` is a subclass of `Hazard`, then `HazardB.concat([])` is equivalent
        to `HazardB()` (unless `HazardB` overrides the `concat` method).

        Parameters
        ----------
        haz_list : list of climada.hazard.Hazard objects
            Hazard instances of the same hazard type (subclass).

        Returns
        -------
        haz_concat : instance of climada.hazard.Hazard
            This will be of the same type (subclass) as all the hazards in `haz_list`.

        See Also
        --------
        Hazard.append : append hazards to a hazard in place
        Centroids.union : combine centroids
        """
        if len(haz_list) == 0:
            return cls()
        haz_concat = haz_list[0].__class__(centroids=Centroids(lat=[], lon=[],
                                                               crs=haz_list[0].centroids.crs))
        for attr_name, attr_val in vars(haz_list[0]).items():
            # to save memory, only copy simple attributes like
            # "units" that are not explicitly handled by Hazard.append
            if not (isinstance(attr_val, (list, np.ndarray, sparse.csr_matrix))
                    or attr_name in ["centroids"]):
                setattr(haz_concat, attr_name, copy.deepcopy(attr_val))
        haz_concat.append(*haz_list)
        return haz_concat

    def change_centroids(self, centroids, threshold=u_coord.NEAREST_NEIGHBOR_THRESHOLD):
        """
        Assign (new) centroids to hazard.

        Centoids of the hazard not in centroids are mapped onto the nearest
        point. Fails if a point is further than threshold from the closest
        centroid.

        The centroids must have the same CRS as self.centroids.

        Parameters
        ----------
        haz: Hazard
            Hazard instance
        centroids: Centroids
            Centroids instance on which to map the hazard.
        threshold: int or float
            Threshold (in km) for mapping haz.centroids not in centroids.
            Argument is passed to climada.util.coordinates.match_coordinates.
            Default: 100 (km)

        Returns
        -------
        haz_new_cent: Hazard
            Hazard projected onto centroids

        Raises
        ------
        ValueError


        See Also
        --------
        util.coordinates.match_coordinates: algorithm to match centroids.

        """
        # define empty hazard
        haz_new_cent = copy.deepcopy(self)
        haz_new_cent.centroids = centroids


        new_cent_idx = u_coord.match_coordinates(
            self.centroids.coord, centroids.coord, threshold=threshold
        )
        if -1 in new_cent_idx:
            raise ValueError(
                "At least one hazard centroid is at a larger distance than the given threshold"
                f" {threshold} from the given centroids. Please choose a larger threshold or"
                " enlarge the centroids"
            )

        if np.unique(new_cent_idx).size < new_cent_idx.size:
            raise ValueError(
                "At least two hazard centroids are mapped to the same centroids. Please make sure"
                " that the given centroids cover the same area like the original centroids and are"
                " not of lower resolution."
            )

        # re-assign attributes intensity and fraction
        for attr_name in ["intensity", "fraction"]:
            matrix = getattr(self, attr_name)
            setattr(haz_new_cent, attr_name,
                    sparse.csr_matrix(
                        (matrix.data, new_cent_idx[matrix.indices], matrix.indptr),
                        shape=(matrix.shape[0], centroids.size)
                    ))

        return haz_new_cent

    @property
    def centr_exp_col(self):
        """
        Name of the centroids columns for this hazard in an exposures

        Returns
        -------
        String
            centroids string indicator with hazard type defining column
            in an exposures gdf. E.g. "centr_TC"

        """
        from climada.entity.exposures import INDICATOR_CENTR  # pylint: disable=import-outside-toplevel
        # import outside toplevel is necessary for it not being circular
        return INDICATOR_CENTR + self.haz_type

    def get_mdr(self, cent_idx, impf):
        """
        Return Mean Damage Ratio (mdr) for chosen centroids (cent_idx)
        for given impact function.

        Parameters
        ----------
        cent_idx : array-like
            array of indices of chosen centroids from hazard
        impf : ImpactFunc
            impact function to compute mdr

        Returns
        -------
        sparse.csr_matrix
            sparse matrix (n_events x len(cent_idx)) with mdr values

        See Also
        --------
        get_paa: get the paa ffor the given centroids

        """
        uniq_cent_idx, indices = np.unique(cent_idx, return_inverse=True)
        mdr = self.intensity[:, uniq_cent_idx]
        if impf.calc_mdr(0) == 0:
            mdr.data = impf.calc_mdr(mdr.data)
        else:
            LOGGER.warning("Impact function id=%d has mdr(0) != 0."
                "The mean damage ratio must thus be computed for all values of"
                "hazard intensity including 0 which can be very time consuming.",
            impf.id)
            mdr_array = impf.calc_mdr(mdr.toarray().ravel()).reshape(mdr.shape)
            mdr = sparse.csr_matrix(mdr_array)
        return mdr[:, indices]

    def get_paa(self, cent_idx, impf):
        """
        Return Percentage of Affected Assets (paa) for chosen centroids (cent_idx)
        for given impact function.

        Note that value as intensity = 0 are ignored. This is different from
        get_mdr.

        Parameters
        ----------
        cent_idx : array-like
            array of indices of chosen centroids from hazard
        impf : ImpactFunc
            impact function to compute mdr

        Returns
        -------
        sparse.csr_matrix
            sparse matrix (n_events x len(cent_idx)) with paa values

        See Also
        --------
        get_mdr: get the mean-damage ratio for the given centroids

        """
        uniq_cent_idx, indices = np.unique(cent_idx, return_inverse=True)
        paa = self.intensity[:, uniq_cent_idx]
        paa.data = np.interp(paa.data, impf.intensity, impf.paa)
        return paa[:, indices]

    def _get_fraction(self, cent_idx=None):
        """
        Return fraction for chosen centroids (cent_idx).

        Parameters
        ----------
        cent_idx : array-like
            array of indices of chosen centroids from hazard
            Default is None (full fraction is returned)

        Returns
        -------
        sparse.csr_matrix or None
            sparse matrix (n_events x len(cent_idx)) with fraction values
            None if fraction is empty. (When calculating the impact, an empty fraction is
            equivalent to the identity under multiplication, i.e. a uniform matrix with
            value 1 everywhere.)
        """
        if self.fraction.nnz == 0:
            return None
        if cent_idx is None:
            return self.fraction
        return self.fraction[:, cent_idx]
