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

This modules implements the Snapshot and SnapshotsCollection classes.

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

    Attributes
    ----------
    date : datetime
        Date of the snapshot.
    measure: Measure | None
        The possible measure applied to the snapshot.

    Notes
    -----

    The object creates copies of the exposure hazard and impact function set.

    To create a snapshot with a measure use Snapshot.apply_measure(measure).
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
        self.measure = None
        self.date = self._convert_to_date(date)

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

    @staticmethod
    def _convert_to_date(date_arg) -> datetime.date:
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

    def apply_measure(self, measure: Measure):
        LOGGER.debug(f"Applying measure {measure.name} on snapshot {id(self)}")
        exp_new, impfset_new, haz_new = measure.apply(
            self.exposure, self.impfset, self.hazard
        )
        snap = Snapshot(exp_new, haz_new, impfset_new, self.date)
        snap.measure = measure
        return snap
