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
import scipy.sparse as sparse

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

    def _reduce_attrs(self, event_name: str):
        """
        Reduce the attributes of a HazardForecast to a single value.

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
            The event_name given to the reduced data.
        """
        reduced_attrs = {
            "lead_time": np.array([np.timedelta64("NaT")]),
            "member": np.array([-1]),
            "event_id": np.array([0]),
            "event_name": np.array([event_name]),
            "date": np.array([0]),
            "frequency": np.array([1]),
            "orig": np.array([True]),
        }

        return reduced_attrs

    def min(self):
        """
        Reduce the intensity and fraction of a HazardForecast to the minimum
        value.

        Parameters
        ----------
        None

        Returns
        -------
        HazardForecast
            A HazardForecast object with the min intensity and fraction.
        """
        red_intensity = sparse.csr_matrix(self.intensity.min(axis=0))
        red_fraction = sparse.csr_matrix(self.fraction.min(axis=0))
        reduced_attrs = self._reduce_attrs("min")
        return HazardForecast(
            lead_time=reduced_attrs["lead_time"],
            member=reduced_attrs["member"],
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            event_id=reduced_attrs["event_id"],
            frequency=reduced_attrs["frequency"],
            frequency_unit=self.frequency_unit,
            event_name=reduced_attrs["event_name"],
            date=reduced_attrs["date"],
            orig=reduced_attrs["orig"],
            intensity=red_intensity,
            fraction=red_fraction,
        )

    def max(self):
        """
        Reduce the intensity and fraction of a HazardForecast to the maximum
        value.

        Parameters
        ----------
        None

        Returns
        -------
        HazardForecast
            A HazardForecast object with the min intensity and fraction.
        """
        red_intensity = sparse.csr_matrix(self.intensity.max(axis=0))
        red_fraction = sparse.csr_matrix(self.fraction.max(axis=0))
        reduced_attrs = self._reduce_attrs("max")
        return HazardForecast(
            lead_time=reduced_attrs["lead_time"],
            member=reduced_attrs["member"],
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            event_id=reduced_attrs["event_id"],
            frequency=reduced_attrs["frequency"],
            frequency_unit=self.frequency_unit,
            event_name=reduced_attrs["event_name"],
            date=reduced_attrs["date"],
            orig=reduced_attrs["orig"],
            intensity=red_intensity,
            fraction=red_fraction,
        )

    def mean(self):
        """
        Reduce the intensity and fraction of a HazardForecast to the mean value.

        Parameters
        ----------
        None

        Returns
        -------
        HazardForecast
            A HazardForecast object with the min intensity and fraction.
        """
        red_intensity = sparse.csr_matrix(self.intensity.mean(axis=0))
        red_fraction = sparse.csr_matrix(self.fraction.mean(axis=0))
        reduced_attrs = self._reduce_attrs("mean")
        return HazardForecast(
            lead_time=reduced_attrs["lead_time"],
            member=reduced_attrs["member"],
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            event_id=reduced_attrs["event_id"],
            frequency=reduced_attrs["frequency"],
            frequency_unit=self.frequency_unit,
            event_name=reduced_attrs["event_name"],
            date=reduced_attrs["date"],
            orig=reduced_attrs["orig"],
            intensity=red_intensity,
            fraction=red_fraction,
        )
