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
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, cast

import numpy as np
import pandas as pd

import climada.util.hdf5_handler as u_hdf5
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.entity.measures.cost_income import CostIncome
from climada.entity.measures.helper import composite_fun
from climada.entity.measures.measure_config import ImpfsetModifierConfig, MeasureConfig
from climada.hazard.base import Hazard
from climada.util.string_parsers import parse_mapping_string, parse_range

T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)

LOGGER = logging.getLogger(__name__)

DEF_VAR_MAT = {
    "sup_field_name": "entity",
    "field_name": "measures",
    "var_name": {
        "name": "name",
        "color": "color",
        "cost": "cost",
        "haz_int_a": "hazard_intensity_impact_a",
        "haz_int_b": "hazard_intensity_impact_b",
        "haz_frq": "hazard_high_frequency_cutoff",
        "haz_set": "hazard_event_set",
        "mdd_a": "MDD_impact_a",
        "mdd_b": "MDD_impact_b",
        "paa_a": "PAA_impact_a",
        "paa_b": "PAA_impact_b",
        "fun_map": "damagefunctions_map",
        "exp_set": "assets_file",
        "exp_reg": "Region_ID",
        "risk_att": "risk_transfer_attachement",
        "risk_cov": "risk_transfer_cover",
        "haz": "peril_ID",
    },
}
"""MATLAB variable names"""

DEF_VAR_EXCEL = {
    "sheet_name": "measures",
    "col_name": {
        "name": "name",
        "color": "color_rgb",
        "implementation duration": "implementation_duration",
        "cost": "init_cost",
        "periodic cost": "periodic_cost",
        "periodic income": "periodic_income",
        "income growth rate (yearly)": "income_yearly_growth_rate",
        "cost growth rate (yearly)": "cost_yearly_growth_rate",
        "impact function id": "impf_id",
        "hazard intensity impact a": "haz_int_mult",
        "hazard intensity impact b": "haz_int_add",
        "hazard event set": "haz_set",
        "MDD impact a": "impf_mdd_mult",
        "MDD impact b": "impf_mdd_add",
        "PAA impact a": "impf_paa_mult",
        "PAA impact b": "imfp_paa_add",
        "damagefunctions map": "fun_map",
        "assets file": "exp_set",
        "Region_ID": "exp_reg",
        "peril_ID": "haz_type",
        "Impact RP cutoff": "impact_rp_cutoff",
        "assets zeroing": "assets_to_zero",
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

    def get_measure(self, haz_type=None, name=None):
        """This function is deprecated, use Entity.from_mat instead."""
        LOGGER.warning(
            "The use of MeasureSet.get_measure() is deprecated."
            "Use MeasureSet.measures().values() instead."
        )
        if haz_type is not None:
            LOGGER.warning(
                "Selection per hazard type has been deprecated (as measures are no longer"
                "considered specific to a hazard)"
            )
        return self.measures(names=name).values()

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

        def exposure_changes(exp: Exposures, **kwargs) -> Exposures:
            return composite_fun(*[meas.exposures_changes for meas in meas_list])(
                exp, **kwargs
            )

        def impfset_changes(impfset: ImpactFuncSet, **kwargs) -> ImpactFuncSet:
            return composite_fun(*[meas.impfset_changes for meas in meas_list])(
                impfset, **kwargs
            )

        def hazard_changes(haz: Hazard, **kwargs) -> Hazard:
            return composite_fun(*[meas.hazard_changes for meas in meas_list])(
                haz, **kwargs
            )

        return Measure(
            name=combo_name or "_".join(names) + "composed",
            exposures_changes=exposure_changes,
            impfset_changes=impfset_changes,
            hazard_changes=hazard_changes,
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
        """Combine multiple measures into a single composite Measure object.

        This method creates a new Measure that applies all specified sub-measures
        sequentially to exposures, impact function sets, and hazard data. Each
        sub-measure's transformation is applied individually, then the results are
        aggregated through specialized combination methods that implement "best-of-all"
        principles (taking minimum damage parameters, maximum effects).

        The combined Measure preserves the cost/income information from all sub-measures.

        Parameters
        ----------
        names : list[str], optional
            List of measure names to subset before combining. If None, uses all
            measures. Each name must correspond to an existing measure in the
            collection.
        combo_name : str, optional
            Custom name for the combined measure. If None, defaults to joining the
            sub-measure names with underscores (e.g., "measure1_measure2_measure3").

        Returns
        -------
        Measure
            A new Measure object containing:

            * Combined transformation functions for exposures, impact function sets,
              and hazard data
            * Aggregated cost/income information from all sub-measures
            * Reference to all sub-measure names for traceability

        Raises
        ------
        ValueError
            If no measures are found to combine (empty `names` list or no valid
            measures). Also raised if:

            * Multiple measures attempt to modify the same exposure cells (conflict
              detected during exposure combination)
            * Cost/income objects have mismatched market price years, cost growth
              rates, or income growth rates

        Notes
        -----
        Combination Logic by Entity Type:

        **Hazards**: Maximum effect is taken across all modified hazards, implemented
        as minimum intensity, minimum fraction, and minimum frequency values. This
        represents the most conservative (highest impact) scenario.

        **Impact Functions**: Merged by taking the safest (minimum) damage parameters
        (MDD and PAA) across all modified impact functions, while preserving the
        maximum intensity range. This ensures conservative damage estimation.

        **Exposures**: Changes are merged with strict conflict detection. If multiple
        measures attempt to modify the same exposure cell, a ValueError is raised.
        All change DataFrames must have identical column structure and order.

        **Cost/Income**: Values are summed across all sub-measures. Validation ensures
        all cost/income objects share the same market price year, cost growth rate,
        and income growth rate before aggregation.

        The wrapper functions preserve `**kwargs` support, allowing entity-specific
        configuration to be passed through to individual sub-measure transformations.
        This enables fine-grained control over how each sub-measure behaves during
        the combined operation.
        """
        names = self.names if names is None else names
        meas_list = list(self.measures(names).values())

        if not meas_list:
            raise ValueError("No measures found to combine.")

        def combined_exposure_changes(base_exp: Exposures, **kwargs) -> Exposures:
            # 1. Apply all measures individually
            mod_exps = [
                m.apply_exposures_changes(base_exp, **kwargs) for m in meas_list
            ]

            # 2. Delegate combination to specialized methods
            return self._combine_exposures(base_exp, mod_exps)

        def combined_impfset_changes(base_impfs: ImpactFuncSet, **kwargs):
            # 1. Apply all measures individually
            mod_impfs = [
                m.apply_impfset_changes(base_impfs, **kwargs) for m in meas_list
            ]

            # 2. Delegate combination to specialized methods
            return self._combine_impfsets(base_impfs, mod_impfs)

        def combined_hazard_changes(base_haz: Hazard, **kwargs):
            # 1. Apply all measures individually
            mod_haz = [m.apply_hazard_changes(base_haz, **kwargs) for m in meas_list]

            # 2. Delegate combination to specialized methods
            return self._combine_hazards(mod_haz)

        return Measure(
            name=combo_name or "_".join(names),
            sub_measures=names,
            exposures_changes=combined_exposure_changes,
            impfset_changes=combined_impfset_changes,
            hazard_changes=combined_hazard_changes,
            cost_income=CostIncome.comb_cost_income([m.cost_income for m in meas_list]),
        )

    @classmethod
    def from_excel(cls, file_name: str, var_names: Optional[dict] = None):
        """Read excel file following template and return a MeasureSet."""
        if var_names is None:
            var_names = DEF_VAR_EXCEL

        df = pd.read_excel(file_name, sheet_name=var_names["sheet_name"])
        # inv_map = {v: k for k, v in var_names["col_name"].items()}
        df = df.rename(columns=var_names["col_name"])

        # Extract row processing to reduce locals in main method
        measures = [cls._process_excel_row(row) for _, row in df.iterrows()]
        return MeasureSet(measures)

    @classmethod
    def from_mat(cls, file_name: str, var_names: Optional[dict] = None) -> "MeasureSet":
        """Read MATLAB file generated with previous MATLAB CLIMADA version.

        Parameters
        ----------
        file_name : str
            Absolute path to the MATLAB file.
        var_names : dict, optional
            Name of the variables in the file. Defaults to DEF_VAR_MAT.

        Returns
        -------
        MeasureSet
        """
        if var_names is None:
            var_names = DEF_VAR_MAT

        def _parse_measure(idx: int, data: dict) -> MeasureConfig:
            vn = var_names["var_name"]

            haz_type = u_hdf5.get_str_from_ref(file_name, data[vn["haz"]][idx][0])
            impf_id = 1  # MATLAB format has no explicit impf_id

            # hazard intensity: old files may lack the _a/_b suffix
            try:
                haz_int_a = data[vn["haz_int_a"]][idx][0]
                haz_int_b = data[vn["haz_int_b"]][0][idx]
            except KeyError:
                haz_int_a = data[vn["haz_int_a"][:-2]][idx][0]
                haz_int_b = 0.0

            # optional fields that may be empty strings in legacy files
            haz_set = (
                u_hdf5.get_str_from_ref(file_name, data[vn["haz_set"]][idx][0]) or None
            )
            exp_set = (
                u_hdf5.get_str_from_ref(file_name, data[vn["exp_set"]][idx][0]) or None
            )
            fun_map = (
                u_hdf5.get_str_from_ref(file_name, data[vn["fun_map"]][idx][0]) or None
            )
            color_str = u_hdf5.get_str_from_ref(file_name, data[vn["color"]][idx][0])

            if data[vn["exp_reg"]][idx][0]:
                LOGGER.warning(
                    "Measure '%s' has exp_region_id set, which is no longer supported "
                    "and will be ignored. It will be reimplemented in a future version.",
                    u_hdf5.get_str_from_ref(file_name, data[vn["name"]][idx][0]),
                )
            if data[vn["risk_att"]][idx][0] or data[vn["risk_cov"]][idx][0]:
                LOGGER.warning(
                    "Measure '%s' has risk_transf_attach/cover set, which is no longer "
                    "supported and will be ignored. It will be reimplemented in a future version.",
                    u_hdf5.get_str_from_ref(file_name, data[vn["name"]][idx][0]),
                )

            return MeasureConfig.from_dict(
                dict(
                    name=u_hdf5.get_str_from_ref(file_name, data[vn["name"]][idx][0]),
                    haz_type=haz_type,
                    impf_id=impf_id,
                    impf_mdd_mult=data[vn["mdd_a"]][idx][0],
                    impf_mdd_add=data[vn["mdd_b"]][idx][0],
                    impf_paa_mult=data[vn["paa_a"]][idx][0],
                    impf_paa_add=data[vn["paa_b"]][idx][0],
                    intensity_multiplier=float(haz_int_a),
                    intensity_add=float(haz_int_b),
                    new_hazard_path=haz_set if haz_set != "nil" else None,
                    impact_rp_cutoff=float(data[vn["haz_frq"]][idx][0]) or None,
                    reassign_impf_id=(
                        parse_mapping_string(fun_map)
                        if fun_map and fun_map != "nil"
                        else None
                    ),
                    new_exposures_path=exp_set if exp_set != "nil" else None,
                    init_cost=float(data[vn["cost"]][idx][0]),
                    color_rgb=tuple(np.fromstring(color_str, dtype=float, sep=" ")),
                )
            )

        data = u_hdf5.read(file_name)
        try:
            data = data[var_names["sup_field_name"]]
        except KeyError:
            pass
        try:
            data = data[var_names["field_name"]]
        except KeyError as err:
            raise KeyError("Variable not in MAT file: " + str(err)) from err

        num_measures = len(data[var_names["var_name"]["name"]])
        measures = [
            Measure.from_config(_parse_measure(idx, data))
            for idx in range(num_measures)
        ]
        return cls(measures)

    @classmethod
    def _load_dataset(cls, path: Optional[str], loader_func) -> Optional[Any]:
        """Load dataset from path if valid, otherwise return None."""
        if path and path != "nil":
            return loader_func(path)
        return None

    @classmethod
    def _process_excel_row(cls, row: pd.Series) -> Measure:
        """Process a single Excel row into a Measure object."""
        return Measure.from_config(MeasureConfig.from_row(row))

    def to_dict(self) -> dict:
        """Serialize all serializable measures to a dict. Skips function-only measures."""
        serializable = {
            name: measure._config.to_dict()
            for name, measure in self._data.items()
            if measure.is_serializable
        }
        skipped = [
            name for name, measure in self._data.items() if not measure.is_serializable
        ]
        if skipped:
            LOGGER.warning(
                "The following measures are not serializable and will be skipped: %s",
                skipped,
            )
        return {"measures": list(serializable.values())}

    @classmethod
    def from_dict(cls, d: dict) -> "MeasureSet":
        measures = [
            Measure.from_config(MeasureConfig.from_dict(m)) for m in d["measures"]
        ]
        return cls(measures)

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "MeasureSet":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_json(self, path: str) -> None:
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MeasureSet":
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))
