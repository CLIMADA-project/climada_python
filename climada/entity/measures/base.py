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

Define Measure class.
"""

__all__ = ["Measure"]

import copy
import logging
from typing import Callable, Dict, Optional, TypeVar, Union

import pandas as pd

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

from .cost_income import CostIncome

LOGGER = logging.getLogger(__name__)

T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)

HazardChange = Callable[[Hazard], Hazard]
ImpfsetChange = Callable[[ImpactFuncSet], ImpactFuncSet]
ExposuresChange = Callable[[Exposures], Exposures]

# TODO: risk transfer?


class Measure:
    """
    Contains a measure to be applied to a set of exposures, impact functions, and hazard.

    Attributes
    ----------
    name : str
        Name of the measure.
    exposures_change : callable
        Function to change exposures.
    impfset_change : callable
        Function to change impact function set.
    hazard_change : callable
        Function to change hazard.
    sub_measures : list of str, optional
        List of measure names that this measure is a combination of.
    cost_income : climada.entity.measures.cost_income.CostIncome
        Cost and income object associated with the measure.
    implementation_duration : pd.DateOffset, optional
        Duration of implementation before the measure is fully functional.
    """

    def __init__(
        self,
        name: str,
        *,
        exposures_change: ExposuresChange = lambda x: x,
        impfset_change: ImpfsetChange = lambda x: x,
        hazard_change: HazardChange = lambda x: x,
        sub_measures: Optional[list[str]] = None,
        cost_income: Optional[CostIncome] = None,
        implementation_duration: Optional[pd.DateOffset] = None,
    ):
        """
        Initialize a new Measure object.

        Parameters
        ----------
        name : str
            Name of the measure.
        exposures_change : callable, optional
            Transformation function for Exposures. Defaults to identity.
        impfset_change : callable, optional
            Transformation function for ImpactFuncSet. Defaults to identity.
        hazard_change : callable, optional
            Transformation function for Hazard. Defaults to identity.
        sub_measures : list of str, optional
            Names of component measures.
        cost_income : CostIncome, optional
            Financial data. If None, an empty CostIncome is initialized.
        implementation_duration : pd.DateOffset, optional
            Time offset for full implementation.
        """

        self.name = name
        self.exposures_change = exposures_change
        self.impfset_change = impfset_change
        self.hazard_change = hazard_change
        self.sub_measures = sub_measures
        self.cost_income = cost_income if cost_income is not None else CostIncome()
        self.implementation_duration = implementation_duration

    def _apply_transformation(
        self, obj: T, func: Callable[[T], T], enforce_copy: bool = True
    ) -> T:
        """Helper to handle the boilerplate of copying and applying changes."""
        obj_to_mod = copy.deepcopy(obj) if enforce_copy else obj
        return func(obj_to_mod)

    def apply_to_exposures(
        self, exposures: Exposures, enforce_copy: bool = True
    ) -> Exposures:
        """Apply measure changes to an Exposures instance."""
        return self._apply_transformation(
            exposures, self.exposures_change, enforce_copy
        )

    def apply_to_impfset(
        self, impfset: ImpactFuncSet, enforce_copy: bool = True
    ) -> ImpactFuncSet:
        """Apply measure changes to an ImpactFuncSet instance."""
        return self._apply_transformation(impfset, self.impfset_change, enforce_copy)

    def apply_to_hazard(self, hazard: Hazard, enforce_copy: bool = True) -> Hazard:
        """Apply measure changes to a Hazard instance."""
        return self._apply_transformation(hazard, self.hazard_change, enforce_copy)

    def apply(
        self,
        exposures: Exposures,
        impfset: ImpactFuncSet,
        hazard: Hazard,
        enforce_copy: bool = True,
    ) -> Dict[str, Union[Exposures, ImpactFuncSet, Hazard]]:
        """
        Apply all measure transformations to the provided entities.

        Returns
        -------
        dict
            Dictionary with keys 'exposure', 'impfset', and 'hazard'.
        """
        return {
            "exposure": self.apply_to_exposures(exposures, enforce_copy),
            "impfset": self.apply_to_impfset(impfset, enforce_copy),
            "hazard": self.apply_to_hazard(hazard, enforce_copy),
        }
