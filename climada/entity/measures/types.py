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
from typing import Concatenate

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

HazardChange = Callable[Concatenate[Hazard, ...], Hazard]
ImpfsetChange = Callable[Concatenate[ImpactFuncSet, ...], ImpactFuncSet]
ExposuresChange = Callable[Concatenate[Exposures, ...], Exposures]
