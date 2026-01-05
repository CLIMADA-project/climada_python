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

Snapshot are used to store a snapshot of Exposure, Hazard and Vulnerability
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

__all__ = ["Snapshot"]


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
        with format "YYYY-MM-DD".
    ref_only : bool, default False
        Should the `Snapshot` contain deep copies of the Exposures, Hazard and Impfset (False)
        or references only (True).

    Attributes
    ----------
    date : datetime
        Date of the snapshot.
    measure: Measure | None
        The possible measure applied to the snapshot.

    Notes
    -----

    The object creates deep copies of the exposure hazard and impact function set.

    Also note that exposure, hazard and impfset are read-only properties.
    Consider snapshot as immutable objects.

    To create a snapshot with a measure, create a snapshot `snap` without
    the measure and call `snap.apply_measure(measure)`, which returns a new Snapshot object
    with the measure applied to its risk dimensions.
    """

    def __init__(
        self,
        *,
        exposure: Exposures,
        hazard: Hazard,
        impfset: ImpactFuncSet,
        measure: Measure | None,
        date: int | datetime.date | str,
        ref_only: bool = False,
    ) -> None:
        self._exposure = exposure if ref_only else copy.deepcopy(exposure)
        self._hazard = hazard if ref_only else copy.deepcopy(hazard)
        self._impfset = impfset if ref_only else copy.deepcopy(impfset)
        self._measure = measure if ref_only else copy.deepcopy(impfset)
        self._date = self._convert_to_date(date)

    @classmethod
    def from_triplet(
        cls,
        *,
        exposure: Exposures,
        hazard: Hazard,
        impfset: ImpactFuncSet,
        date: int | datetime.date | str,
        ref_only: bool = False,
    ) -> "Snapshot":
        """Create a Snapshot from exposure, hazard and impact functions set

        This method is the main point of entry for the creation of Snapshot. It
        creates a new Snapshot object for the given date with copies of the
        hazard, exposure and impact function set given in argument (or
        references if ref_only is True)

        Parameters
        ----------
        exposure : Exposures
        hazard : Hazard
        impfset : ImpactFuncSet
        date : int | datetime.date | str
        ref_only : bool
            If true, uses references to the exposure, hazard and impact
            function objects. Note that modifying the original objects after
            computations using the Snapshot might lead to inconsistencies in
            results.

        Returns
        -------
        Snapshot

        Notes
        -----

        To create a Snapshot with a measure, first create the Snapshot without
        the measure using this method, and use `apply_measure(measure)` afterward.

        """
        return cls(
            exposure=exposure,
            hazard=hazard,
            impfset=impfset,
            measure=None,
            date=date,
            ref_only=ref_only,
        )

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
        """(Adaptation) Measure data for the snapshot."""
        return self._measure

    @property
    def date(self) -> datetime.date:
        """Date of the snapshot."""
        return self._date

    @property
    def impact_calc_data(self) -> dict:
        """Convenience function for ImpactCalc class."""
        return {
            "exposures": self.exposure,
            "hazard": self.hazard,
            "impfset": self.impfset,
        }

    @staticmethod
    def _convert_to_date(date_arg) -> datetime.date:
        """Convert date argument of type int or str to a datetime.date object."""
        if isinstance(date_arg, int):
            # Assume the integer represents a year
            return datetime.date(date_arg, 1, 1)
        if isinstance(date_arg, str):
            # Try to parse the string as a date
            try:
                return datetime.datetime.strptime(date_arg, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError("String must be in the format 'YYYY-MM-DD'") from exc
        if isinstance(date_arg, datetime.date):
            # Already a date object
            return date_arg

        raise TypeError("date_arg must be an int, str, or datetime.date")

    def apply_measure(self, measure: Measure) -> "Snapshot":
        """Create a new snapshot by applying a Measure object.

        This method creates a new `Snapshot` object by applying a measure on
        the current one.

        Parameters
        ----------
        measure : Measure
            The measure to be applied to the snapshot.

        Returns
        -------
            The Snapshot with the measure applied.

        """

        LOGGER.debug("Applying measure %s on snapshot %s", measure.name, id(self))
        exp, impfset, haz = measure.apply(self.exposure, self.impfset, self.hazard)
        snap = Snapshot(
            exposure=exp, hazard=haz, impfset=impfset, date=self.date, measure=measure
        )
        return snap
