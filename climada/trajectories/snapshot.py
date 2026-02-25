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
from typing import cast

import numpy as np
import pandas as pd

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
    date : datetime.date | str | pd.Timestamp
        The date of the Snapshot, it can be an string representing a year,
        a datetime object or a string representation of a datetime object.
    measure : Measure | None, default None.
        Measure associated with the Snapshot. The measure object is *not* applied
        to the other parameters of the object (Exposure, Hazard, Impfset).
        To create a `Snapshot` with a measure use `apply_measure()` instead (see notes).
        The use of anything but None should be reserved for advanced users.
    ref_only : bool, default False
        Should the `Snapshot` contain deep copies of the Exposures, Hazard and Impfset (False)
        or references only (True).

    Attributes
    ----------
    date : datetime
        Date of the snapshot.
    measure: Measure | None
        A possible measure associated with the snapshot.

    Notes
    -----

    Providing a measure to the init assumes that the (Exposure, Hazard, Impfset) triplet
    already corresponds to the triplet once the measure is applied. Measure objects
    contain "the changes to apply". Creating a consistent Snapshot with a measure should
    be done by first creating a Snapshot with the "baseline" (Exposure, Hazard, Impfset) triplet
    and calling `<Snapshot>.apply_measure(<measure>)`, which returns a new Snapshot object
    with the measure applied.

    Instantiating a Snapshot with a measure directly does not garantee the
    consistency between the triplet and the measure, and should be avoided.

    If `ref_only` is True (default) the object creates deep copies of the
    exposure, hazard, and impact function set.

    Also note that exposure, hazard and impfset are read-only properties.
    Consider snapshots as immutable objects.

    """

    def __init__(
        self,
        *,
        exposure: Exposures,
        hazard: Hazard,
        impfset: ImpactFuncSet,
        date: datetime.date | str | pd.Timestamp,
        measure: Measure | None = None,
        ref_only: bool = False,
    ) -> None:
        self._exposure = exposure if ref_only else copy.deepcopy(exposure)
        self._hazard = hazard if ref_only else copy.deepcopy(hazard)
        self._impfset = impfset if ref_only else copy.deepcopy(impfset)
        self._measure = measure if ref_only else copy.deepcopy(measure)
        self._date = self._convert_to_timestamp(date)

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
    def date(self) -> pd.Timestamp:
        """Date of the snapshot."""
        return self._date

    @property
    def impact_calc_kwargs(self) -> dict:
        """Convenience function for ImpactCalc class."""
        return {
            "exposures": self.exposure,
            "hazard": self.hazard,
            "impfset": self.impfset,
        }

    @staticmethod
    def _convert_to_timestamp(
        date_arg: str | datetime.date | pd.Timestamp | np.datetime64,
    ) -> pd.Timestamp:
        """
        Convert date argument of type str, datetime.date,
        np.datetime64, or pandas Timestamp to a pandas Timestamp object.
        """
        if isinstance(date_arg, str):
            try:
                date = pd.Timestamp(date_arg)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    "String must be in a valid date format (e.g., 'YYYY-MM-DD')"
                ) from exc

        elif isinstance(date_arg, (datetime.date, pd.Timestamp, np.datetime64)):
            date = pd.Timestamp(date_arg)

        else:
            raise TypeError(
                f"Unsupported type: {type(date_arg)}. Must be str, date, Timestamp, or datetime64."
            )

        # Final check for NaT (Not-a-Time)
        if date is pd.NaT:
            raise ValueError(
                f"Could not resolve '{date_arg}' to a valid Pandas Timestamp."
            )

        return cast(pd.Timestamp, date)

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
            exposure=exp,
            hazard=haz,
            impfset=impfset,
            date=self.date,
            measure=measure,
            ref_only=True,  # Avoid unecessary copies of new objects
        )
        return snap
