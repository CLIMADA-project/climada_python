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
import itertools
import logging
from dataclasses import InitVar, dataclass, field
from weakref import WeakValueDictionary

from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)


# TODO: Improve and make it an __eq__ function within Hazard?
def hazard_data_equal(haz1: Hazard, haz2: Hazard) -> bool:
    intensity_eq = (
        haz1.intensity != haz2.intensity
    ).nnz == 0  # type:ignore (__neq__ type hint is bool)
    freq_eq = (haz1.frequency == haz2.frequency).all()
    frac_eq = (
        haz1.fraction != haz2.fraction
    ).nnz == 0  # type:ignore (__neq__ type hint is bool)
    return intensity_eq and freq_eq and frac_eq


class _SnapData:
    """
    A snapshot of exposure, hazard, and impact function.

    Attributes
    ----------
    exposure : Exposures
        Exposure data for the snapshot.
    hazard : Hazard
        Hazard data for the snapshot.
    impfset : ImpactFuncSet
        Impact function set associated with the snapshot.
    """

    # Class-level cache
    def __init__(
        self, exposure: Exposures, hazard: Hazard, impfset: ImpactFuncSet
    ) -> None:
        self.exposure = copy.deepcopy(exposure)
        self.hazard = copy.deepcopy(hazard)
        self.impfset = copy.deepcopy(impfset)

    def __eq__(self, value, /) -> bool:
        if not isinstance(value, _SnapData):
            return False
        if self is value:
            return True
        same_exposure = self.exposure.gdf.equals(value.exposure.gdf)
        same_hazard = hazard_data_equal(self.hazard, value.hazard)
        same_impfset = self.impfset == value.impfset
        return same_exposure and same_hazard and same_impfset


class Snapshot:
    """
    A snapshot of exposure, hazard, and impact function at a specific date.

    Attributes
    ----------
    date : datetime
        Date of the snapshot.

    Notes
    -----

    The object creates copies of the exposure hazard and impact function set.
    """

    def __init__(
        self,
        exposure: Exposures,
        hazard: Hazard,
        impfset: ImpactFuncSet,
        date: int | datetime.date | str,
    ) -> None:
        self._data = _SnapData(exposure, hazard, impfset)
        self.measure = None
        self.date = self._convert_to_date(date)

    @property
    def exposure(self) -> Exposures:
        """Exposure data for the snapshot."""
        return self._data.exposure

    @property
    def hazard(self) -> Hazard:
        """Hazard data for the snapshot."""
        return self._data.hazard

    @property
    def impfset(self) -> ImpactFuncSet:
        """Impact function set data for the snapshot."""
        return self._data.impfset

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


def pairwise(container: list):
    """
    Generate pairs of successive elements from an iterable.

    Parameters
    ----------
    iterable : iterable
        An iterable sequence from which successive pairs of elements are generated.

    Returns
    -------
    zip
        A zip object containing tuples of successive pairs from the input iterable.

    Example
    -------
    >>> list(pairwise([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]
    """
    a, b = itertools.tee(container)
    next(b, None)
    return zip(a, b)
