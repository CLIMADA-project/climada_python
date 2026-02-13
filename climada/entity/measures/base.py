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

from __future__ import annotations

__all__ = ["Measure"]

import copy
import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple, TypeVar, Union

import pandas as pd

from climada.entity.measures.helper import build_measure_effects

if TYPE_CHECKING:
    from climada.entity.exposures.base import Exposures
    from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
    from climada.entity.measures.types import (
        ExposuresChange,
        HazardChange,
        ImpfsetChange,
        MeasureEffect,
    )
    from climada.hazard.base import Hazard

    T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)

from .cost_income import CostIncome

LOGGER = logging.getLogger(__name__)


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
        measure_effects: MeasureEffect = lambda exposures, impfset, hazard: (
            exposures,
            impfset,
            hazard,
        ),
        sub_measures: Optional[list[str]] = None,
        cost_income: Optional[CostIncome] = None,
        implementation_duration: Optional[pd.DateOffset] = None,
        color_rgb: Optional[Tuple[float, float, float]] = None,
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
        self.measure_effects = measure_effects
        self.sub_measures = sub_measures
        self.cost_income = cost_income if cost_income is not None else CostIncome()
        self.implementation_duration = implementation_duration
        self.color_rgb = (0, 0, 0) if color_rgb is None else color_rgb

    @classmethod
    def from_changes(
        cls,
        name: str,
        *,
        exposures_change: ExposuresChange,
        impfset_change: ImpfsetChange,
        hazard_change: HazardChange,
        sub_measures: Optional[list[str]] = None,
        cost_income: Optional[CostIncome] = None,
        implementation_duration: Optional[pd.DateOffset] = None,
    ):
        def measure_effects(
            exp: Exposures, impfs: ImpactFuncSet, haz: Hazard
        ) -> tuple[Exposures, ImpactFuncSet, Hazard]:
            return (exposures_change(exp), impfset_change(impfs), hazard_change(haz))

        return cls(
            name,
            measure_effects=measure_effects,
            sub_measures=sub_measures,
            cost_income=cost_income,
            implementation_duration=implementation_duration,
        )

    def apply(
        self,
        exposures: Exposures,
        impfset: ImpactFuncSet,
        hazard: Hazard,
        enforce_copy: bool = True,
    ) -> Tuple[Exposures, ImpactFuncSet, Hazard]:
        """
        Apply all measure transformations to the provided entities.

        Returns
        -------
        dict
            Dictionary with keys 'exposure', 'impfset', and 'hazard'.
        """
        changed_exp = copy.deepcopy(exposures) if enforce_copy else exposures
        changed_impfset = copy.deepcopy(impfset) if enforce_copy else impfset
        changed_haz = copy.deepcopy(hazard) if enforce_copy else hazard

        return self.measure_effects(changed_exp, changed_impfset, changed_haz)

    @classmethod
    def _from_xls_row_args(cls, name: str, haz_type: str, **kwargs):
        # 1. Validation
        has_haz_mod = any(
            kwargs.get(k)
            for k in ["haz_intensity_multiplier", "haz_intensity_add", "new_hazard"]
        )
        if has_haz_mod and kwargs.get("impact_rp_cutoff"):
            raise ValueError(
                "Cannot apply impact return period cutoff AND base hazard changes."
            )

        if kwargs.get("impact_rp_cutoff"):
            LOGGER.warning(
                "Impact return period cutoff provided. You should know about it subtleties."
            )

        # 2. Financials
        cost_inc = CostIncome.from_kwargs(kwargs)

        # 3. Transformation Logic
        effects = build_measure_effects(haz_type, **kwargs)

        return cls(name, measure_effects=effects, cost_income=cost_inc)
