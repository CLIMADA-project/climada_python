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
            The lead time associated with each event entry
        member : np.ndarray, optional
            The ensemble member associated with each event entry
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
            The lead time associated with each event entry
        member : np.ndarray, optional
            The ensemble member associated with each event entry
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
