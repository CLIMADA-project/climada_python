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

import logging
from dataclasses import dataclass, field, fields
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.measure_config import (
    ExposuresModifierConfig,
    HazardModifierConfig,
    ImpfsetModifierConfig,
    MeasureConfig,
)
from climada.hazard.base import Hazard

if TYPE_CHECKING:
    from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
    from climada.entity.measures.types import (
        ExposuresChange,
        HazardChange,
        ImpfsetChange,
    )
    from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)

T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)


def identity_function(x: T, **_kwargs: Any) -> T:
    return x


def composite_fun(*funcs: Callable[..., T]) -> Callable[..., T]:
    """
    Composes multiple functions from right to left.
    f(g(h(x)))
    """

    def compose(f: Callable[..., T], g: Callable[..., T]) -> Callable[..., T]:
        def composed(x: T, **kwargs: Any) -> T:
            return f(g(x, **kwargs), **kwargs)

        return composed

    return reduce(compose, funcs, identity_function)


def replace_hazard(new_hazard: Hazard) -> HazardChange:
    """Returns a function that replaces the hazard with given new one."""

    def hazard_change(_: Hazard) -> Hazard:
        return new_hazard

    return hazard_change


def impact_intensity_rp_cutoff_helper(
    cut_off_rp: float,
) -> HazardChange:
    """Helper to generate a function removing events from a hazard for which
    impacts do not exceed the impacts of a given return period.

    This helper returns a function to be applied on a hazard.
    The function returned has to run an impact computation to find out which
    event to remove from the hazard.
    As such it has the following signature:

    ```f(hazard: Hazard,           # The hazard to apply on
         exposures: Exposures,     # The exposure for the impact computation
         impfset: ImpactFuncSet,   # The impfset for the impact computation
         base_hazard: Hazard,      # The hazard for the impact computation
         exposures_region_id: Optional[list[int]] = None, # Region id to filter to
         ) -> Hazard
    ```

    Identifies events exceeding a return period and returns the hazard intensity
    matrix with those event intensities zeroed out.
    """
    from climada.engine.impact_calc import ImpactCalc

    def hazard_change(
        hazard: Hazard,
        base_exposures: Exposures,
        base_impfset: ImpactFuncSet,
        base_hazard: Hazard,
        exposures_region_id: Optional[list[int]] = None,
    ) -> Hazard:
        exp_imp = base_exposures
        if exposures_region_id:
            # Narrowing the type for the LSP via boolean indexing
            in_reg = base_exposures.gdf["region_id"].isin(exposures_region_id)
            exp_imp = Exposures(base_exposures.gdf[in_reg], crs=base_exposures.crs)

        imp = ImpactCalc(exp_imp, base_impfset, base_hazard).impact(save_mat=False)

        # Calculate exceedance frequencies
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        events_below_cutoff = sort_idxs[exceed_freq <= cut_off_rp]

        # Modify sparse data structure
        intensity_modified = base_hazard.intensity.copy()
        for event in events_below_cutoff:
            start, end = (
                intensity_modified.indptr[event],
                intensity_modified.indptr[event + 1],
            )
            intensity_modified.data[start:end] = 0

        hazard.intensity = intensity_modified
        return hazard

    return hazard_change


def helper_hazard(hazard_modifier: HazardModifierConfig) -> HazardChange:
    """Returns a function that scales and shifts hazard intensity."""

    def hazard_change(hazard: Hazard, **_kwargs) -> Hazard:
        changed_hazard = (
            Hazard.from_hdf5(hazard_modifier.new_hazard_path)
            if hazard_modifier.new_hazard_path is not None
            else hazard
        )
        data = cast(np.ndarray, changed_hazard.intensity.data)
        data *= hazard_modifier.haz_int_mult
        data += hazard_modifier.haz_int_add
        data[data < 0] = 0
        changed_hazard.intensity.eliminate_zeros()
        return changed_hazard

    if hazard_modifier.impact_rp_cutoff is not None:
        hazard_change = composite_fun(
            impact_intensity_rp_cutoff_helper(hazard_modifier.impact_rp_cutoff),
            hazard_change,
        )

    return hazard_change


def helper_impfset(impfset_modifier: ImpfsetModifierConfig) -> ImpfsetChange:
    """Returns a function that modifies impact functions (mdd, paa, intensity) by ID."""

    def impfset_change(impfset: ImpactFuncSet, **_kwargs) -> ImpactFuncSet:
        changed_impfset = (
            impfset.from_excel(impfset_modifier.new_impfset_path)
            if impfset_modifier.new_impfset_path is not None
            else impfset
        )
        if impfset_modifier.impf_ids is None or impfset_modifier.impf_ids == "all":
            ids_to_change = impfset.get_ids(haz_type=impfset_modifier.haz_type)
        elif isinstance(impfset_modifier.impf_ids, list):
            ids_to_change = impfset_modifier.impf_ids
        elif isinstance(impfset_modifier.impf_ids, (str, int)):
            ids_to_change = [impfset_modifier.impf_ids]
        else:
            raise ValueError(
                f"Impact function ids to changes are invalid: {impfset_modifier.impf_ids}"
            )

        funcs = changed_impfset.get_func(haz_type=impfset_modifier.haz_type)
        funcs = [funcs] if isinstance(funcs, ImpactFunc) else funcs

        for impf in funcs:
            # Apply Intensity Mod
            if impf.id in ids_to_change:
                mult, add = (
                    impfset_modifier.impf_int_mult,
                    impfset_modifier.impf_int_add,
                )
                impf.intensity = impf.intensity * mult + add

                mult, add = (
                    impfset_modifier.impf_mdd_mult,
                    impfset_modifier.impf_mdd_add,
                )
                impf.mdd = impf.mdd * mult + add

                mult, add = (
                    impfset_modifier.impf_paa_mult,
                    impfset_modifier.impf_paa_add,
                )
                impf.paa = impf.paa * mult + add

        return changed_impfset

    return impfset_change


def change_impfset(new_impfsets: ImpactFuncSet) -> ImpfsetChange:
    """Returns a function that swaps the impact function set with the given one."""

    def impfset_change(_: ImpactFuncSet) -> ImpactFuncSet:
        return new_impfsets

    return impfset_change


def helper_exposure(exposures_modifier: ExposuresModifierConfig) -> ExposuresChange:
    """Returns a function that reassigns impact function IDs and zeros out specific values."""

    def exposures_change(exposures: Exposures, **_kwargs) -> Exposures:
        changed_exposures = (
            exposures
            if exposures_modifier.new_exposures_path is None
            else Exposures.from_hdf5(exposures_modifier.new_exposures_path)
        )
        gdf = cast(pd.DataFrame, changed_exposures.gdf)
        if exposures_modifier.reassign_impf_id is not None:
            for haz_type, mapping in exposures_modifier.reassign_impf_id.items():
                gdf[f"impf_{haz_type}"] = gdf[f"impf_{haz_type}"].replace(mapping)

        if exposures_modifier.set_to_zero is not None:
            gdf.loc[exposures_modifier.set_to_zero, "value"] = 0

        return changed_exposures

    return exposures_change
