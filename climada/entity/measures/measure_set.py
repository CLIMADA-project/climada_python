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
from typing import Any, Callable, Dict, Iterable, List, Optional, cast

import numpy as np
from scipy.sparse import csr_matrix

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.entity.measures.cost_income import CostIncome
from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)


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

    def combine(
        self, names: Optional[List[str]] = None, combo_name: Optional[str] = None
    ) -> Measure:
        """
        Combine multiple measures into a single representative Measure.

        The combination is maximalistic, the combined measure has the maximum
        effect of each of its members.

        Parameters
        ----------
        names : List[str], optional
            Names of measures to combine. Defaults to all measures.
        combo_name : str, optional
            Name for the combined measure. Defaults to joined names.

        Returns
        -------
        Measure
            A new combined Measure object.
        """
        names = self.names if names is None else names
        meas_list = list(self.measures(names).values())

        if not meas_list:
            raise ValueError("No measures found to combine.")

        def comb_haz_map(hazard: Hazard) -> Hazard:
            """Apply measures sequentially and reduce hazard intensity/frequency."""
            hazard_mod = meas_list[0].apply_to_hazard(hazard)
            for measure in meas_list[1:]:
                new_haz = measure.apply_to_hazard(hazard)
                hazard_mod.intensity = csr_matrix(
                    np.minimum(
                        new_haz.intensity.toarray(), hazard_mod.intensity.toarray()
                    )
                )
                hazard_mod.fraction = csr_matrix(
                    np.minimum(
                        new_haz.fraction.toarray(), hazard_mod.fraction.toarray()
                    )
                )
                hazard_mod.frequency = np.minimum(
                    new_haz.frequency, hazard_mod.frequency
                )
            return hazard_mod

        def comb_impfset_map(impfset: ImpactFuncSet) -> ImpactFuncSet:
            """Apply measures and reduce impact function parameters."""
            impfset_mod = meas_list[0].apply_to_impfset(impfset)
            for measure in meas_list[1:]:
                new_impfset = measure.apply_to_impfset(impfset)
                for new_impf in new_impfset.get_func():
                    impf_mod = impfset_mod.get_func(new_impf.id)
                    impf_mod = cast(ImpactFunc, impf_mod)
                    impf_mod.paa = np.minimum(new_impf.paa, impf_mod.paa)
                    impf_mod.mdd = np.minimum(new_impf.mdd, impf_mod.mdd)
                    impf_mod.intensity = np.maximum(
                        new_impf.intensity, impf_mod.intensity
                    )
            return impfset_mod

        def comb_exp_map(exposures: Exposures) -> Exposures:
            """Apply measures and update exposure values and impact function IDs."""
            exp_mod = meas_list[0].apply_to_exposures(exposures)
            for measure in meas_list[1:]:
                new_exp = measure.apply_to_exposures(exposures)
                exp_mod.gdf["value"] = np.minimum(
                    new_exp.gdf["value"], exp_mod.gdf["value"]
                )

                # TODO make a choice here
                impf_col = f"impf_{measure.haz_type}"
                if impf_col in new_exp.gdf.columns:
                    changed_ids = new_exp.gdf[impf_col] != exposures.gdf[impf_col]
                    exp_mod.gdf.loc[changed_ids, impf_col] = new_exp.gdf.loc[
                        changed_ids, impf_col
                    ]
            return exp_mod

        def comb_cost_income() -> CostIncome:
            """Sum costs and incomes from all measures."""
            first_ci = meas_list[0].cost_income
            return CostIncome(
                mkt_price_year=first_ci.mkt_price_year.year,
                cost_yearly_growth_rate=first_ci.cost_growth_rate,
                init_cost=sum(m.cost_income.init_cost for m in meas_list),
                periodic_cost=sum(m.cost_income.periodic_cost for m in meas_list),
                periodic_income=sum(m.cost_income.periodic_income for m in meas_list),
                income_yearly_growth_rate=first_ci.income_growth_rate,
            )

        return Measure(
            name=combo_name or "_".join(names),
            exposures_change=comb_exp_map,
            impfset_change=comb_impfset_map,
            hazard_change=comb_haz_map,
            sub_measures=names,
            cost_income=comb_cost_income(),
        )
