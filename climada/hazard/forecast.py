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

Define Forecast variant of Hazard.
"""

import logging

import numpy as np

from climada.engine.forecast import Forecast
from climada.hazard.hazard import Hazard

LOGGER = logging.getLogger(__name__)


class HazardForecast(Forecast, Hazard):

    def __init__(
        self,
        lead_time: np.ndarray | None = None,
        member: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(lead_time=lead_time, member=member, **kwargs)

        def from_hazard(self, hazard: Hazard):
            return cls(
                haz_type=hazard.haz_type,
                pool=hazard.pool,
                units=hazard.units,
                centroids=hazard.centroids,
                event_id=hazard.event_id,
                frequency=hazard.frequency,
                frequency_unit=hazard.frequency_unit,
                event_name=hazard.event_name,
                date=hazard.date,
                orig=hazard.orig,
                intensity=hazard.intensity,
                fraction=hazard.fraction,
                lead_time=self.lead_time,
                member=self.member,
            )
