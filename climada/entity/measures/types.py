from collections.abc import Callable

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

MeasureEffect = Callable[
    [Exposures, ImpactFuncSet, Hazard], tuple[Exposures, ImpactFuncSet, Hazard]
]
HazardChange = Callable[[Hazard], Hazard]
ImpfsetChange = Callable[[ImpactFuncSet], ImpactFuncSet]
ExposuresChange = Callable[[Exposures], Exposures]
