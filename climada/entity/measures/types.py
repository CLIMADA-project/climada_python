from collections.abc import Callable
from typing import Concatenate

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

HazardChange = Callable[Concatenate[Hazard, ...], Hazard]
ImpfsetChange = Callable[Concatenate[ImpactFuncSet, ...], ImpactFuncSet]
ExposuresChange = Callable[Concatenate[Exposures, ...], Exposures]
