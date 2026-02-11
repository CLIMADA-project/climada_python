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
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.base import ExposuresChange, HazardChange, ImpfsetChange
from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)


def helper_hazard(
    intensity_multiplier: float = 1.0, intensity_subtract: float = 0.0
) -> HazardChange:
    """Returns a function that scales and shifts hazard intensity."""

    def hazard_change(hazard: Hazard) -> Hazard:
        data = cast(np.ndarray, hazard.intensity.data)
        data *= intensity_multiplier
        data -= intensity_subtract
        data[data < 0] = 0
        hazard.intensity.eliminate_zeros()
        return hazard

    return hazard_change


def replace_hazard(new_hazard: Hazard) -> HazardChange:
    """Returns a function that replaces the hazard with given new one."""

    def hazard_change(_: Hazard) -> Hazard:
        return new_hazard

    return hazard_change


def impact_intensity_rp_cutoff(
    cut_off_rp: float,
    exposures: Exposures,
    impfset: ImpactFuncSet,
    hazard: Hazard,
    exposures_region_id: Optional[list[int]] = None,
) -> sparse.csr_matrix:
    """
    Identifies events exceeding a return period and returns the hazard intensity
    matrix with those event intensities zeroed out.
    """

    exp_imp = exposures
    if exposures_region_id:
        # Narrowing the type for the LSP via boolean indexing
        in_reg = exposures.gdf["region_id"].isin(exposures_region_id)
        exp_imp = Exposures(exposures.gdf[in_reg], crs=exposures.crs)

    imp = ImpactCalc(exp_imp, impfset, hazard).impact(save_mat=False)

    # Calculate exceedance frequencies
    sort_idxs = np.argsort(imp.at_event)[::-1]
    exceed_freq = np.cumsum(imp.frequency[sort_idxs])
    events_above_cutoff = sort_idxs[exceed_freq > cut_off_rp]

    # Modify sparse data structure
    intensity_modified = hazard.intensity.copy()
    for event in events_above_cutoff:
        start, end = (
            intensity_modified.indptr[event],
            intensity_modified.indptr[event + 1],
        )
        intensity_modified.data[start:end] = 0

    return intensity_modified


def change_impfset(new_impfsets: ImpactFuncSet) -> ImpfsetChange:
    """Returns a function that swaps the impact function set with the given one."""

    def impfset_change(_: ImpactFuncSet) -> ImpactFuncSet:
        return new_impfsets

    return impfset_change


def helper_impfset(
    haz_type: str,
    impf_mdd_modifier: Dict[int, tuple[float, float]] | None = None,
    impf_paa_modifier: Dict[int, tuple[float, float]] | None = None,
    impf_intensity_modifier: Dict[int, tuple[float, float]] | None = None,
) -> ImpfsetChange:
    """Returns a function that modifies impact functions (mdd, paa, intensity) by ID."""
    impf_mdd_modifier = (
        impf_mdd_modifier if impf_mdd_modifier is not None else {1: (1.0, 0.0)}
    )
    impf_paa_modifier = (
        impf_paa_modifier if impf_paa_modifier is not None else {1: (1.0, 0.0)}
    )
    impf_intensity_modifier = (
        impf_intensity_modifier
        if impf_intensity_modifier is not None
        else {1: (1.0, 0.0)}
    )

    def impfset_change(impfset: ImpactFuncSet) -> ImpactFuncSet:
        # impfset_mod = copy.deepcopy(impfset)
        for impf in impfset.get_func(haz_type):
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

        return impfset

    return impfset_change


def helper_exposure(
    reassign_impf_id: int, haz_type: str, set_to_zero: Optional[list[int]] = None
) -> ExposuresChange:
    """Returns a function that reassigns impact function IDs and zeros out specific values."""
    indices_to_zero = set_to_zero if set_to_zero is not None else []

    def exposures_change(exposures: Exposures) -> Exposures:
        gdf = cast(pd.DataFrame, exposures.gdf)
        gdf[f"impf_{haz_type}"] = reassign_impf_id

        if indices_to_zero:
            gdf.loc[indices_to_zero, "value"] = 0

        return exposures

    return exposures_change
