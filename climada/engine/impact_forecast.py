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

Define Forecast variant of Impact.
"""

import logging

import numpy as np
import scipy.sparse as sparse

from ..util import log_level
from ..util.checker import size
from ..util.forecast import Forecast
from .impact import Impact

LOGGER = logging.getLogger(__name__)


class ImpactForecast(Forecast, Impact):
    """An impact object with forecast information"""

    def __init__(
        self,
        *,
        lead_time: np.ndarray | None,
        member: np.ndarray | None,
        **impact_kwargs,
    ):
        """Initialize the impact forecast.

        Parameters
        ----------
        lead_time : np.ndarray, optional
            The lead time associated with each event entry, given as timedelta64 type
        member : np.ndarray, optional
            The ensemble member associated with each event entry, given as integers
        impact_kwargs
            Keyword-arguments passed to ~:py:class`climada.engine.impact.Impact`.
        """
        super().__init__(lead_time=lead_time, member=member, **impact_kwargs)
        self._check_sizes()

    @classmethod
    def from_impact(
        cls, impact: Impact, lead_time: np.ndarray | None, member: np.ndarray | None
    ):
        """Create an impact forecast from an impact object and forecast information.

        Parameters
        ----------
        impact : climada.engine.impact.Impact
            The impact object whose data to use in the forecast object
        lead_time : np.ndarray, optional
            The lead time associated with each event entry, given as timedelta64 type
        member : np.ndarray, optional
            The ensemble member associated with each event entry, given as integers
        """
        with log_level("WARNING", "climada.engine.impact"):
            return cls(
                lead_time=lead_time,
                member=member,
                event_id=impact.event_id,
                event_name=impact.event_name,
                date=impact.date,
                frequency=impact.frequency,
                frequency_unit=impact.frequency_unit,
                coord_exp=impact.coord_exp,
                crs=impact.crs,
                eai_exp=impact.eai_exp,
                at_event=impact.at_event,
                tot_value=impact.tot_value,
                aai_agg=impact.aai_agg,
                unit=impact.unit,
                imp_mat=impact.imp_mat,
                haz_type=impact.haz_type,
            )

    @property
    def at_event(self):
        """Get the total impact for each member/lead_time combination."""
        LOGGER.warning(
            "at_event gives the total impact for one specific combination of member and "
            "lead_time."
        )
        return self._at_event

    @at_event.setter
    def at_event(self, value):
        """Set the total impact for each member/lead_time combination."""
        self._at_event = value

    def local_exceedance_impact(
        self,
        return_periods=(25, 50, 100, 250),
        method="interpolate",
        min_impact=0,
        log_frequency=True,
        log_impact=True,
        bin_decimals=None,
    ):
        """Compution of local exceedance impact for given return periods is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.local_exceedance_impact`

        Raises
        ------
        NotImplementedError
        """

        LOGGER.error("local_exceedance_impact is not defined for ImpactForecast")
        raise NotImplementedError(
            "local_exceedance_impact is not defined for ImpactForecast"
        )

    def local_return_period(
        self,
        threshold_impact=(1000.0, 10000.0),
        method="interpolate",
        min_impact=0,
        log_frequency=True,
        log_impact=True,
        bin_decimals=None,
    ):
        """Compution of local return period for given impact thresholds is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.local_return_period`

        Raises
        -------
        NotImplementedError
        """

        LOGGER.error("local_return_period is not defined for ImpactForecast")
        raise NotImplementedError(
            "local_return_period is not defined for ImpactForecast"
        )

    def calc_freq_curve(self, return_per=None):
        """Computation of the impact exceedance frequency curve is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.calc_freq_curve`

        Raises
        ------
        NotImplementedError
        """

        LOGGER.error("calc_freq_curve is not defined for ImpactForecast")
        raise NotImplementedError("calc_freq_curve is not defined for ImpactForecast")

    def _check_sizes(self):
        """Check sizes of forecast data vs. impact data.

        Raises
        ------
        ValueError
            If the sizes of the forecast data do not match the
            :py:attr:`~climada.engine.impact.Impact.event_id`
        """
        num_entries = len(self.event_id)
        size(exp_len=num_entries, var=self.member, var_name="Forecast.member")
        size(exp_len=num_entries, var=self.lead_time, var_name="Forecast.lead_time")

    def _reduce_attrs(self, event_name: str):
        """
        Reduce the attributes of an ImpactForecast to a single value.

        Attributes are modified as follows:
        - lead_time: set to NaT
        - member: set to -1
        - event_id: set to 0
        - event_name: set to the name of the reduction method (default)
        - date: set to 0
        - frequency: set to 1

        Parameters
        ----------
        event_name : str
            The event name given to the reduced data.
        """
        reduced_attrs = {
            "lead_time": np.array([np.timedelta64("NaT")]),
            "member": np.array([-1]),
            "event_id": np.array([0]),
            "event_name": np.array([event_name]),
            "date": np.array([0]),
            "frequency": np.array([1]),
        }

        return reduced_attrs

    def min(self):
        """
        Reduce the impact matrix and at_event of an ImpactForecast to the minimum
        value.

        Parameters
        ----------
        None

        Returns
        -------
        ImpactForecast
            An ImpactForecast object with the min impact matrix and at_event.
        """
        red_imp_mat = self.imp_mat.min(axis=0).tocsr()
        red_at_event = np.array([red_imp_mat.sum()])
        return ImpactForecast(
            frequency_unit=self.frequency_unit,
            coord_exp=self.coord_exp,
            crs=self.crs,
            eai_exp=self.eai_exp,
            at_event=red_at_event,
            tot_value=self.tot_value,
            aai_agg=self.aai_agg,
            unit=self.unit,
            imp_mat=red_imp_mat,
            haz_type=self.haz_type,
            **self._reduce_attrs("min"),
        )

    def max(self):
        """
        Reduce the impact matrix and at_event of an ImpactForecast to the maximum
        value.

        Parameters
        ----------
        None

        Returns
        -------
        ImpactForecast
            An ImpactForecast object with the max impact matrix and at_event.
        """
        red_imp_mat = self.imp_mat.max(axis=0).tocsr()
        red_at_event = np.array([red_imp_mat.sum()])
        return ImpactForecast(
            frequency_unit=self.frequency_unit,
            coord_exp=self.coord_exp,
            crs=self.crs,
            eai_exp=self.eai_exp,
            at_event=red_at_event,
            tot_value=self.tot_value,
            aai_agg=self.aai_agg,
            unit=self.unit,
            imp_mat=red_imp_mat,
            haz_type=self.haz_type,
            **self._reduce_attrs("max"),
        )

    def mean(self):
        """
        Reduce the impact matrix and at_event of an ImpactForecast to the mean value.

        Parameters
        ----------
        None

        Returns
        -------
        ImpactForecast
            An ImpactForecast object with the mean impact matrix and at_event.
        """
        red_imp_mat = sparse.csr_matrix(self.imp_mat.mean(axis=0))
        red_at_event = np.array([red_imp_mat.sum()])
        return ImpactForecast(
            frequency_unit=self.frequency_unit,
            coord_exp=self.coord_exp,
            crs=self.crs,
            eai_exp=self.eai_exp,
            at_event=red_at_event,
            tot_value=self.tot_value,
            aai_agg=self.aai_agg,
            unit=self.unit,
            imp_mat=red_imp_mat,
            haz_type=self.haz_type,
            **self._reduce_attrs("mean"),
        )
