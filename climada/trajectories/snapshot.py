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

This modules implements the Snapshot class.

Snapshot are used to store the a snapshot of Exposure, Hazard, Vulnerability
at a specific date.

"""

import copy
import datetime
import logging

from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)


class Snapshot:
    """
    A snapshot of exposure, hazard, and impact function at a specific date.

    Parameters
    ----------
    exposure : Exposures
    hazard : Hazard
    impfset : ImpactFuncSet
    date : int | datetime.date | str
        The date of the Snapshot, it can be an integer representing a year,
        a datetime object or a string representation of a datetime object
        with format YYYY-MM-DD.

    Attributes
    ----------
    date : datetime
        Date of the snapshot.
    measure: Measure | None
        The possible measure applied to the snapshot.

    Notes
    -----

    The object creates deep copies of the exposure hazard and impact function set.

    To create a snapshot with a measure, create a snapshot `snap` without
    the measure and call `snap.apply_measure(measure)`, which returns a new Snapshot object.
    """

    def __init__(
        self,
        exposure: Exposures,
        hazard: Hazard,
        impfset: ImpactFuncSet,
        date: int | datetime.date | str,
    ) -> None:
        self._exposure = copy.deepcopy(exposure)
        self._hazard = copy.deepcopy(hazard)
        self._impfset = copy.deepcopy(impfset)
        self._measure = None
        self._date = self._convert_to_date(date)

    @property
    def exposure(self) -> Exposures:
        """Exposure data for the snapshot."""
        return self._exposure

    @property
    def hazard(self) -> Hazard:
        """Hazard data for the snapshot."""
        return self._hazard

    @property
    def impfset(self) -> ImpactFuncSet:
        """Impact function set data for the snapshot."""
        return self._impfset

    @property
    def measure(self) -> Measure | None:
        """Impact function set data for the snapshot."""
        return self._measure

    @property
    def date(self) -> datetime.date:
        """Impact function set data for the snapshot."""
        return self._date

    @staticmethod
    def _convert_to_date(date_arg) -> datetime.date:
        """Convert date argument of type int or str to a datetime.date object."""
        if isinstance(date_arg, int):
            # Assume the integer represents a year
            return datetime.date(date_arg, 1, 1)
        elif isinstance(date_arg, str):
            # Try to parse the string as a date
            try:
                return datetime.datetime.strptime(date_arg, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError("String must be in the format 'YYYY-MM-DD'")
        elif isinstance(date_arg, datetime.date):
            # Already a date object
            return date_arg
        else:
            raise TypeError("date_arg must be an int, str, or datetime.date")

    def apply_measure(self, measure: Measure) -> "Snapshot":
        """Create a new snapshot from a measure

        This methods creates a new `Snapshot` object by applying a measure on
        the current one.

        Parameters
        ----------
        measure : Measure
            The measure to be applied to the snapshot.

        Returns
        -------
            The Snapshot with the measure applied.

        """

        LOGGER.debug(f"Applying measure {measure.name} on snapshot {id(self)}")
        snap = Snapshot(
            *measure.apply(self.exposure, self.impfset, self.hazard), self.date
        )
        snap._measure = measure
        return snap
