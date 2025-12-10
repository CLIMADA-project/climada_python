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

from ..util.checker import size
from ..util.forecast import Forecast
from .base import Hazard

LOGGER = logging.getLogger(__name__)


class HazardForecast(Forecast, Hazard):
    """A hazard object with forecast information"""

    def __init__(
        self,
        lead_time: np.ndarray | None = None,
        member: np.ndarray | None = None,
        **hazard_kwargs,
    ):
        """
        Initialize a HazardForecast object.

        Parameters
        ----------
        lead_time : np.ndarray of np.timedelta64 or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers as integers. Default is empty array.
        **hazard_kwargs
            keyword arguments to pass to :py:class:`~climada.hazard.base.Hazard` See
            py:meth`~climada.hazard.base.Hazard.__init__` for details.
        """
        super().__init__(lead_time=lead_time, member=member, **hazard_kwargs)
        self._check_sizes()

    @classmethod
    def from_hazard(cls, hazard: Hazard, lead_time: np.ndarray, member: np.ndarray):
        """
        Create a HazardForecast object from a Hazard object.

        Parameters
        ----------
        hazard : climada.hazard.base.Hazard
            Hazard object to convert into a HazardForecast.
        lead_time : np.ndarray of np.timedelta64 or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers as integers. Default is empty array.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the same attributes as the input hazard,
            but with lead_time and member attributes set from instance of HazardForecast.
        """
        return cls(
            lead_time=lead_time,
            member=member,
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
        )

    def _check_sizes(self):
        """Check sizes of forecast data vs. hazard data.

        Raises
        ------
        ValueError
            If the sizes of the forecast data do not match the
            :py:attr:`~climada.hazard.base.Hazard.event_id`
        """
        num_entries = len(self.event_id)
        size(exp_len=num_entries, var=self.member, var_name="Forecast.member")
        size(exp_len=num_entries, var=self.lead_time, var_name="Forecast.lead_time")

    @classmethod
    def concat(cls, haz_list: list):
        """Concatenate multiple HazardForecast instances and return a new object"""
        if len(haz_list) == 0:
            return cls()
        hazard = Hazard.concat(haz_list)
        lead_time = np.concatenate(tuple(haz.lead_time for haz in haz_list))
        member = np.concatenate(tuple(haz.member for haz in haz_list))
        return cls.from_hazard(hazard, lead_time=lead_time, member=member)
