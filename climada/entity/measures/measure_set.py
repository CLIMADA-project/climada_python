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

Define MeasureSet class.
"""

__all__ = ["MeasureSet"]

import logging
from functools import reduce
from itertools import chain
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, TypeVar

import numpy as np
import pandas as pd

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.entity.measures.cost_income import CostIncome
from climada.hazard.base import Hazard
from climada.util.string_parsers import parse_mapping_string, parse_range

if TYPE_CHECKING:
    from climada.entity.measures.types import MeasureEffect

    T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)


LOGGER = logging.getLogger(__name__)


DEF_VAR_EXCEL = {
    "sheet_name": "measures",
    "col_name": {
        "name": "name",
        "color": "color",
        "cost": "cost",
        "periodic_cost": "periodic cost",
        "periodic_income": "periodic income",
        "income_yearly_growth_rate": "income growth rate (yearly)",
        "cost_yearly_growth_rate": "cost growth rate (yearly)",
        "impf_id": "impact function id",
        "haz_int_a": "hazard intensity impact a",
        "haz_int_b": "hazard intensity impact b",
        "haz_set": "hazard event set",
        "mdd_a": "MDD impact a",
        "mdd_b": "MDD impact b",
        "paa_a": "PAA impact a",
        "paa_b": "PAA impact b",
        "fun_map": "damagefunctions map",
        "exp_set": "assets file",
        "exp_reg": "Region_ID",
        "haz_type": "peril_ID",
        "impact_rp_cutoff": "Impact RP cutoff",
        "assets_to_zero": "assets zeroing",
    },
}
"""Excel variable names"""


class MeasureSet:
    """Contains measures of type Measure.

    Attributes
    ----------
    _data : dict
        Contains Measure objects keyed by their name.
    """

    def __init__(self, measures: Iterable[Measure]):
        """Initialize a new MeasureSet object.

        Parameters
        ----------
        measures : Iterable[Measure].
            The measures to include in the MeasureSet.

        """
        self._data: Dict[str, Measure] = {meas.name: meas for meas in measures}

    def append(self, measure: Measure):
        """
        Append a Measure. Overwrites if a measure with the same name exists.

        Parameters
        ----------
        measure : Measure
            The Measure instance to add.

        Raises
        ------
        TypeError
            If input is not an instance of Measure.

        """
        if not isinstance(measure, Measure):
            raise TypeError(f"Expected Measure, got {type(measure).__name__}")

        self._data[measure.name] = measure

    def measures(self, names: Optional[Iterable[str]] = None) -> Dict[str, Measure]:
        """
        Get a dictionary of measures.

        Parameters
        ----------
        names : Iterable[str], optional
            Filter by these measure names. If None, returns all.

        Returns
        -------
        Dict[str, Measure]
            Dictionary of measure names and objects.

        """
        if names is None:
            return self._data
        return {name: self._data[name] for name in names if name in self._data}

    @property
    def names(self):
        """Get measures names contained for the hazard type provided.
        Return all names for each hazard type if no input hazard type.

        Parameters
        ----------
        haz_type : str, optional
            hazard type from which to obtain the names

        Returns
        -------
        list(Measure.name) (if haz_type provided),
        {Measure.haz_type : list(Measure.name)} (if no haz_type)
        """
        return list(self._data.keys())

    @property
    def size(self) -> int:
        """
        Number of measures in the set.

        Returns
        -------
        int
        """
        return len(self._data)

    def __contains__(self, item: str) -> bool:
        """Check if a measure name exists in the set."""
        return item in self._data

    def compose(self, names: List[str], combo_name: Optional[str] = None) -> Measure:
        """
        Compose multiple measures into a single meta-Measure.

        This method creates a new Measure where the transformation functions
        (hazard, exposures, impfset) are mathematically composed and financial
        values are aggregated.

        Execution Order:
        The composition follows a right-to-left nesting. For a list of
        measures [m1, m2, m3], the resulting transformation is:
        m1(m2(m3(x)))
        This means m3 is applied first, then m2, then m1.

        Financial Aggregation:
        - Costs and incomes are summed across all selected measures.
        - Growth rates and market price years are inherited from the
          first measure in the list.

        Parameters
        ----------
        names : list of str
            Ordered list of measure names to combine.
        combo_name : str, optional
            The name for the resulting Measure. If None, a name is
            generated by joining names with an underscore and
            appending 'composed'.

        Returns
        -------
        Measure
            A new Measure object representing the sequential application
            of the component measures.

        Raises
        ------
        ValueError
            If the provided names do not match any measures in the set,
            or if the MeasureSet is empty.

        """
        meas_list = list(self.measures(names).values())

        if not meas_list:
            raise ValueError("No measures found to compose.")

        def composite_fun(*funcs: MeasureEffect) -> MeasureEffect:

            def compose(f: MeasureEffect, g: MeasureEffect) -> MeasureEffect:
                return lambda exp, impfs, haz: f(*g(exp, impfs, haz))

            return reduce(
                compose,
                funcs,
                lambda exposures, impfset, hazard: (
                    exposures,
                    impfset,
                    hazard,
                ),
            )

        def measure_effects(
            exp: Exposures, impfs: ImpactFuncSet, haz: Hazard
        ) -> tuple[Exposures, ImpactFuncSet, Hazard]:
            return composite_fun(*[meas.measure_effects for meas in meas_list])(
                exp, impfs, haz
            )

        return Measure(
            name=combo_name or "_".join(names) + "composed",
            measure_effects=measure_effects,
            sub_measures=names,
            cost_income=CostIncome.comb_cost_income([m.cost_income for m in meas_list]),
        )

    @staticmethod
    def _combine_hazards(modified_hazards: list[Hazard]) -> Hazard:
        """Finds the maximum effect (minimum intensity/freq) across hazards."""
        intensities = [h.intensity for h in modified_hazards]
        fractions = [h.fraction for h in modified_hazards]
        frequencies = [h.frequency for h in modified_hazards]

        hazard_mod = modified_hazards[0]
        hazard_mod.intensity = reduce(lambda a, b: a.minimum(b), intensities)
        hazard_mod.fraction = reduce(lambda a, b: a.minimum(b), fractions)
        hazard_mod.frequency = np.minimum.reduce(frequencies)
        return hazard_mod

    @staticmethod
    def _combine_impfsets(
        base_set: ImpactFuncSet, modified_sets: list[ImpactFuncSet]
    ) -> ImpactFuncSet:
        """Merges impact functions by taking the safest (minimum) damage parameters."""
        combined = ImpactFuncSet()
        for haz_dict in base_set.get_func().values():
            for impf in haz_dict.values():
                versions = [
                    s.get_func(haz_type=impf.haz_type, fun_id=impf.id)
                    for s in modified_sets
                ]

                combined.append(
                    ImpactFunc(
                        impf.haz_type,
                        impf.id,
                        intensity=np.maximum.reduce([v.intensity for v in versions]),
                        mdd=np.minimum.reduce([v.mdd for v in versions]),
                        paa=np.minimum.reduce([v.paa for v in versions]),
                        intensity_unit=impf.intensity_unit,
                        name=impf.name,
                    )
                )
        return combined

    @staticmethod
    def _combine_exposures(
        base_exp: Exposures, modified_exps: list[Exposures]
    ) -> Exposures:
        """Merges exposure changes, raising ValueError if two measures touch the same cell."""
        new_exps_gdfs = [exp.gdf for exp in modified_exps]
        if not all(
            set(new_gdf.columns) == set(base_exp.gdf.columns)
            for new_gdf in new_exps_gdfs
        ):
            raise ValueError(
                "All change DataFrames must have identical column structure and order."
            )

        # Align all changes into a single MultiIndexed DataFrame
        # This stacks all change-sets on top of each other
        stack = pd.concat(
            new_exps_gdfs,
            keys=range(len(new_exps_gdfs)),
            names=["change_idx", "row_idx"],
        )

        # Create a broadcasted baseline to match the stack's shape
        # We use take() to repeat baseline rows for every change-set
        baseline_repeated = base_exp.gdf.iloc[
            np.tile(np.arange(len(base_exp.gdf)), len(base_exp.gdf))
        ]
        baseline_repeated.index = stack.index  # Align indices for direct comparison

        # Identify changes: Mask is True where a cell differs from baseline
        diff_mask = stack != baseline_repeated

        # Check for Conflicts:
        # Sum the True values across the 'change_idx' level for every (row, col)
        # If any cell has > 1 change, it's a conflict.
        change_counts = diff_mask.groupby(level="row_idx").sum()
        if (change_counts > 1).any().any():
            # Identify exactly where the conflict is for the error message
            conflicting_cells = (
                change_counts[change_counts > 1]
                .dropna(how="all")
                .dropna(axis=1, how="all")
            )
            raise ValueError(
                f"Conflict: Multiple measures change the same cells:\n{conflicting_cells}"
            )

        # Merge:
        # We take the baseline and update it with the sum of differences
        # Only works if the data is numeric. For general objects (like 'if_' IDs):
        result = base_exp.gdf.copy()

        # Efficiently collapse the stack:
        # Since only one change exists per cell (checked in step 5),
        # we can 'first' or 'max' to get the non-baseline value.
        updates = stack.where(diff_mask).groupby(level="row_idx").first()

        exp_mod = Exposures(
            updates.combine_first(result),
            crs=base_exp.crs,
            description=base_exp.description,
            ref_year=base_exp.ref_year,
            value_unit=base_exp.value_unit,
        )
        return exp_mod

    def combine(
        self, names: Optional[list[str]] = None, combo_name: Optional[str] = None
    ) -> Measure:
        names = self.names if names is None else names
        meas_list = list(self.measures(names).values())

        if not meas_list:
            raise ValueError("No measures found to combine.")

        def comb_effect(
            base_exp: Exposures, base_impfs: ImpactFuncSet, base_haz: Hazard
        ):
            # 1. Apply all measures individually
            results = [m.apply(base_exp, base_impfs, base_haz) for m in meas_list]
            mod_exps, mod_impfs, mod_hazs = zip(
                *[(r["exposures"], r["impfset"], r["hazard"]) for r in results]
            )

            # 2. Delegate combination to specialized methods
            return (
                self._combine_exposures(base_exp, mod_exps),
                self._combine_impfsets(base_impfs, mod_impfs),
                self._combine_hazards(mod_hazs),
            )

        return Measure(
            name=combo_name or "_".join(names),
            measure_effects=comb_effect,
            sub_measures=names,
            cost_income=CostIncome.comb_cost_income([m.cost_income for m in meas_list]),
        )

    @classmethod
    def from_excel(cls, file_name: str, var_names: Optional[dict] = None):
        """Read excel file following template and return a MeasureSet."""
        if var_names is None:
            var_names = DEF_VAR_EXCEL

        # 1. Load and clean DataFrame
        df = pd.read_excel(file_name, sheet_name=var_names["sheet_name"])

        # 2. Reverse map the Excel columns to our internal argument names
        # This removes the need for most of those 'try-except' blocks
        inv_map = {v: k for k, v in var_names["col_name"].items()}
        df = df.rename(columns=inv_map)

        meas_set = []

        # 3. Iterate through rows and instantiate
        for _, row in df.iterrows():
            # Handle the special (a, b) tuple logic for modifiers
            # We group them into the format expected by _from_xls_row_args

            haz_type = row["haz_type"]
            impf_id = row.get("impf_id", 1)
            # Prepare modifiers (mapping the Excel a/b columns to our tuple format)
            # Note: If your excel has specific IDs, you'd map them here.
            # For now, we assume ID 1 as per your previous logic.
            mdd_mod = {impf_id: (row.get("mdd_a", 1.0), row.get("mdd_b", 0.0))}
            paa_mod = {impf_id: (row.get("paa_a", 1.0), row.get("paa_b", 0.0))}
            int_mod = {impf_id: (row.get("int_a", 1.0), row.get("int_b", 0.0))}

            zeros_idx = (
                parse_range(assets_to_zero)
                if (assets_to_zero := row.get("assets_to_zero"))
                else None
            )
            reassign_impf_id = (
                parse_mapping_string(fun_map)
                if (fun_map := row.get("fun_map"))
                else None
            )

            new_exposure = None
            if path := row.get("exp_set"):
                if path != "nil":
                    new_exposure = Exposures.from_hdf5(path)

            new_hazard = None
            if path := row.get("haz_set"):
                if path != "nil":
                    new_hazard = Hazard.from_hdf5(path)

            # Build the Measure using the refactored classmethod
            measure = Measure._from_xls_row_args(
                name=row.get("name", "unnamed"),
                # Hazard modifs
                haz_type=haz_type,
                haz_intensity_multiplier=row.get("haz_int_a", 1.0),
                haz_intensity_add=row.get("haz_int_b", 0.0),
                impact_rp_cutoff=row.get("impact_rp_cutoff"),
                new_hazard=new_hazard,
                # Impfset modifs
                impf_mdd_modifier=mdd_mod,
                impf_paa_modifier=paa_mod,
                impf_intensity_modifier=int_mod,
                # Exp modifs
                reassign_impf_id=reassign_impf_id,
                set_to_zero=zeros_idx,
                new_exposure=new_exposure,
                # CostIncome
                init_cost=row.get("cost", 0.0),
                periodic_cost=row.get("periodic_cost", 0.0),
                periodic_income=row.get("periodic_income", 0.0),
                income_yearly_growth_rate=row.get("income_yearly_growth_rate", 0.0),
                cost_yearly_growth_rate=row.get("cost_yearly_growth_rate", 0.0),
            )

            # Handle the color conversion separately if needed
            if "color" in row and isinstance(row["color"], str):
                measure.color_rgb = tuple(row["color"].split(" "))

            meas_set.append(measure)

        return MeasureSet(meas_set)
