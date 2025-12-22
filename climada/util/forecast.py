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

Define Forecast base class.
"""

from typing import Literal

import numpy as np


class Forecast:
    """Mixin class for forecast data.

    Attributes
    ----------
    lead_time : np.ndarray
        Array of forecast lead times, given as timedelta64 objects.
        Represents the lead times of the forecasts.
    member : np.ndarray
        Array of ensemble member identifiers, given as integers.
        Represents different forecast ensemble members.
    """

    def __init__(
        self,
        lead_time: np.ndarray | None = None,
        member: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize Forecast.

        Parameters
        ----------
        lead_time : np.ndarray or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers. Default is empty array.
        """

        self.lead_time = (
            np.asarray(lead_time)
            if lead_time is not None
            else np.array([], dtype="timedelta64[ns]")
        )
        self.member = (
            np.asarray(member) if member is not None else np.array([], dtype="int")
        )
        super().__init__(**kwargs)

    def idx_member(self, member: np.ndarray) -> np.ndarray:
        """Return boolean array where self.member == member using numpy.isin()

        Parameters
        ----------
        member : np.ndarray
            Array of ensemble members (ints) for which to return an indexer

        Returns
        -------
        np.ndarray
            Boolean array where self.member is in member.
        """

        return np.isin(self.member, member)

    def idx_lead_time(self, lead_time: np.ndarray) -> np.ndarray:
        """Return boolean array where self.lead_time == lead_time using numpy.isin()

        Parameters
        ----------
        lead_time : np.ndarray
            Array of lead times (numpy.timedelta64) for which to return an indexer

        Returns
        -------
        np.ndarray
            Boolean array where self.lead_time is in lead_time.
        """

        return np.isin(self.lead_time, lead_time)

    def _unique_or_default(
        self, attr: str, default: np.typing.ArrayLike
    ) -> np.typing.ArrayLike:
        """Return the single unique value of an attribute or the default value"""
        try:
            if len(unique := np.unique(getattr(self, attr))) == 1:
                return unique
        except AttributeError:
            pass
        return default

    def _reduce_iter_dim(
        self, dim: Literal["member", "lead_time"]
    ) -> Literal["lead_time", "member"]:
        """Return the dimension to iterate over when reducing over 'dim'"""
        if dim == "member":
            return "lead_time"
        if dim == "lead_time":
            return "member"
        raise ValueError(f"Cannot reduce over dim '{dim}'")

    def _reduce_attrs(self, event_name: str, **attrs) -> dict[str, np.ndarray | list]:
        """
        Reduce the attributes of a Forecast derived object to a single value.

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
            "lead_time": self._unique_or_default("lead_time", [np.timedelta64("NaT")]),
            "member": self._unique_or_default("member", [-1]),
            "event_id": self._unique_or_default("event_id", [1]),
            "event_name": self._unique_or_default("event_name", [event_name]),
            "date": self._unique_or_default("date", [0]),
            "frequency": [1],
            "orig": self._unique_or_default("orig", [True]),
        } | attrs

        # Filter out attributes that the derived object does not have
        reduced_attrs = {
            key: np.asarray(val)
            for key, val in reduced_attrs.items()
            if key in self.__dict__
        }
        reduced_attrs["event_name"] = reduced_attrs["event_name"].tolist()

        return reduced_attrs


def reduce_unique_selection(
    forecast, values: np.ndarray, select: str, reduce_attr: str, **kwargs
):
    """
    Reduce an attribute of a forecast object by selecting unique values
    and performing an attribute reduction method.

    Parameters
    ----------
    forecast : HazardForecast | ImpactForecast
        Forecast object to reduce.
    values : np.ndarray
        Array of values for which to select and reduce the attribute.
    select : str
        Name of the attribute to select on (e.g. 'lead_time', 'member').
    reduce_attr : str
        Name of the attribute reduction method to call (e.g. 'min', 'mean').

    Returns
    -------
    Forecast
        Forecast object with the attribute reduced by the reduction method
        and selected by the unique values.
    """
    return forecast.concat(
        [
            getattr(forecast.select(**{select: [val]}), reduce_attr)(dim=None, **kwargs)
            for val in np.unique(values)
        ]
    )
