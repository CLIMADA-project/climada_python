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

import copy
import logging
from typing import Dict, Optional, cast

import numpy as np
import pandas as pd
from scipy import sparse

from climada.engine.impact_calc import ImpactCalc
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.base import ExposuresChange, HazardChange, ImpfsetChange
from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)


def helper_hazard(
    intensity_multiplier: Optional[float] = None,
    intensity_subtract: Optional[float] = None,
    new_hazard: Optional[Hazard] = None,
) -> HazardChange:
    """Returns a function that scales and shifts hazard intensity."""

    intensity_multiplier = 1 if intensity_multiplier is None else intensity_multiplier
    intensity_subtract = 1 if intensity_subtract is None else intensity_subtract

    def hazard_change(hazard: Hazard) -> Hazard:
        changed_hazard = new_hazard if new_hazard is not None else hazard
        data = cast(np.ndarray, changed_hazard.intensity.data)
        data *= intensity_multiplier
        data -= intensity_subtract
        data[data < 0] = 0
        changed_hazard.intensity.eliminate_zeros()
        return changed_hazard

    return hazard_change


def replace_hazard(new_hazard: Hazard) -> HazardChange:
    """Returns a function that replaces the hazard with given new one."""

    def hazard_change(_: Hazard) -> Hazard:
        return new_hazard

    return hazard_change


def impact_intensity_rp_cutoff_helper(
    cut_off_rp: float,
    exposures: Exposures,
    impfset: ImpactFuncSet,
    base_hazard: Hazard,
    exposures_region_id: Optional[list[int]] = None,
) -> HazardChange:
    """
    Identifies events exceeding a return period and returns the hazard intensity
    matrix with those event intensities zeroed out.
    """

    def hazard_change(hazard: Hazard) -> Hazard:

        exp_imp = exposures
        if exposures_region_id:
            # Narrowing the type for the LSP via boolean indexing
            in_reg = exposures.gdf["region_id"].isin(exposures_region_id)
            exp_imp = Exposures(exposures.gdf[in_reg], crs=exposures.crs)

        imp = ImpactCalc(exp_imp, impfset, base_hazard).impact(save_mat=False)

        # Calculate exceedance frequencies
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        events_above_cutoff = sort_idxs[exceed_freq > cut_off_rp]

        # Modify sparse data structure
        intensity_modified = base_hazard.intensity.copy()
        for event in events_above_cutoff:
            start, end = (
                intensity_modified.indptr[event],
                intensity_modified.indptr[event + 1],
            )
            intensity_modified.data[start:end] = 0

        hazard.intensity = intensity_modified
        return hazard

    return hazard_change


def helper_impfset(
    haz_type: str,
    impf_id: Optional[int | str] = None,
    impf_mdd_modifier: Dict[int | str, tuple[float, float]] | None = None,
    impf_paa_modifier: Dict[int | str, tuple[float, float]] | None = None,
    impf_intensity_modifier: Dict[int | str, tuple[float, float]] | None = None,
    new_impfset: Optional[ImpactFuncSet] = None,
) -> ImpfsetChange:
    """Returns a function that modifies impact functions (mdd, paa, intensity) by ID."""

    def_impf_id = 1 if impf_id is None else impf_id
    impf_mdd_modifier = (
        impf_mdd_modifier
        if impf_mdd_modifier is not None
        else {def_impf_id: (1.0, 0.0)}
    )
    impf_paa_modifier = (
        impf_paa_modifier
        if impf_paa_modifier is not None
        else {def_impf_id: (1.0, 0.0)}
    )
    impf_intensity_modifier = (
        impf_intensity_modifier
        if impf_intensity_modifier is not None
        else {def_impf_id: (1.0, 0.0)}
    )

    def impfset_change(impfset: ImpactFuncSet) -> ImpactFuncSet:
        # impfset_mod = copy.deepcopy(impfset)
        changed_impfset = new_impfset if new_impfset is not None else impfset
        funcs = changed_impfset.get_func(haz_type, impf_id)
        funcs = [funcs] if isinstance(funcs, ImpactFunc) else funcs
        for impf in funcs:
            # Apply Intensity Mod
            if impf.id in impf_intensity_modifier:
                mult, shift = impf_intensity_modifier[impf.id]
                impf.intensity = np.maximum(impf.intensity * mult - shift, 0.0)

            # Apply MDD Mod
            if impf.id in impf_mdd_modifier:
                mult, shift = impf_mdd_modifier[impf.id]
                impf.mdd = np.maximum(impf.mdd * mult + shift, 0.0)

            # Apply PAA Mod
            if impf.id in impf_paa_modifier:
                mult, shift = impf_paa_modifier[impf.id]
                impf.paa = np.maximum(impf.paa * mult + shift, 0.0)

        return changed_impfset

    return impfset_change


def change_impfset(new_impfsets: ImpactFuncSet) -> ImpfsetChange:
    """Returns a function that swaps the impact function set with the given one."""

    def impfset_change(_: ImpactFuncSet) -> ImpactFuncSet:
        return new_impfsets

    return impfset_change


def helper_exposure(
    reassign_impf_id: Optional[Dict[str, Dict[int | str, int | str]]] = None,
    set_to_zero: Optional[list[int]] = None,
    new_exposure: Optional[Exposures] = None,
) -> ExposuresChange:
    """Returns a function that reassigns impact function IDs and zeros out specific values."""
    indices_to_zero = set_to_zero if set_to_zero is not None else []

    def exposures_change(exposures: Exposures) -> Exposures:
        changed_exposures = exposures if new_exposure is None else new_exposure
        gdf = cast(pd.DataFrame, changed_exposures.gdf)
        if reassign_impf_id is not None:
            for haz_type, mapping in reassign_impf_id.items():
                gdf[f"impf_{haz_type}"] = gdf[f"impf_{haz_type}"].replace(mapping)

        if indices_to_zero is not None:
            gdf.loc[indices_to_zero, "value"] = 0

        return changed_exposures

    return exposures_change
