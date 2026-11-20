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

Define useful types for the measure module.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Concatenate

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

HazardChange = Callable[Concatenate[Hazard, ...], Hazard]
ImpfsetChange = Callable[Concatenate[ImpactFuncSet, ...], ImpactFuncSet]
ExposuresChange = Callable[Concatenate[Exposures, ...], Exposures]


@dataclass(frozen=True)
class MeasureKwargs:
    """Keyword arguments for the change functions of a measure.

    Attributes
    ----------
    exposures : dict
        Kwargs passed to the exposures change function.
    impfset : dict
        Kwargs passed to the impact function set change function.
    hazard : dict
        Kwargs passed to the hazard change function.
    """

    exposures: dict[str, Any] = field(default_factory=dict)
    impfset: dict[str, Any] = field(default_factory=dict)
    hazard: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def shared(cls, **kwargs: Any) -> "MeasureKwargs":
        """Pass the same kwargs to the three change functions.

        Functions that do not declare a given kwarg ignore it (see `allow_kwargs`).
        """
        return cls(dict(kwargs), dict(kwargs), dict(kwargs))

    def to_apply_kwargs(self) -> dict[str, dict[str, Any]]:
        """Return the kwargs in the format expected by `Measure.apply`."""
        return {
            "kwargs_exposures": self.exposures,
            "kwargs_impfset": self.impfset,
            "kwargs_hazard": self.hazard,
        }


# Either a dict in the format expected by `Measure.apply`
#   {"kwargs_hazard": {...}, "kwargs_exposures": {...}, "kwargs_impfset": {...}}
# or a function of the snapshot returning such a dict.
KwargsProvider = Callable[[Any], MeasureKwargs]  # e.g. Snapshot -> MeasureKwargs
KwargsSpec = MeasureKwargs | KwargsProvider
