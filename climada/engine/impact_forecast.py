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

from ..util import log_level
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
        # TODO: Maybe assert array lengths?
        super().__init__(lead_time=lead_time, member=member, **impact_kwargs)

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
        LOGGER.warning("at_event for forecasts is not yet implemented.")
        return self._at_event

    @at_event.setter
    def at_event(self, value):
        """Set the total exposure value close to a hazard"""
        LOGGER.warning("at_event for forecasts is not yet implemented.")
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
        implemented for ImpactForecast. See climada.engine.impact.Impact for details.
        Returns
        -------
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
        implemented for ImpactForecast. See climada.engine.impact.Impact for details.
        Returns
        -------
        NotImplementedError
        """

        LOGGER.error("local_return_period is not defined for ImpactForecast")
        raise NotImplementedError(
            "local_return_period is not defined for ImpactForecast"
        )

    def calc_freq_curve(self, return_per=None):
        """Computation of the impact exceedance frequency curve is not
        implemented for ImpactForecast. See climada.engine.impact.Impact for details.
        Returns
        -------
        NotImplementedError
        """

        LOGGER.error("calc_freq_curve is not defined for ImpactForecast")
        raise NotImplementedError("calc_freq_curve is not defined for ImpactForecast")
