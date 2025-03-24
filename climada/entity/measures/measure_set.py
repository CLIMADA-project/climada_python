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
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix

import climada.util.hdf5_handler as u_hdf5
from climada.entity.measures.base import Measure
from climada.entity.measures.cost_income import CostIncome

LOGGER = logging.getLogger(__name__)

# TODOS:
# Implement __contains__ method


class MeasureSet:
    """Contains measures of type Measure.

    Attributes
    ----------
    _data : dict
        Contains Measure objects. This attribute is not suppossed to be accessed directly.
        Use the available methods instead.
    """

    def __init__(self, measures: Iterable[Measure]):
        """Initialize a new MeasureSet object with specified data.

        Parameters
        ----------
        measures : iterable of Measure objects.
            The measures to include in the MeasureSet

        """
        haz_type = np.unique([meas.haz_type for meas in measures])[0]
        self.haz_type = haz_type
        self._data = {meas.name: meas for meas in measures}

    def append(self, measure):
        """Append an Measure. Override if same name and haz_type.

        Parameters
        ----------
        meas : Measure
            Measure instance

        Raises
        ------
        ValueError
        """
        if not isinstance(measure, Measure):
            raise ValueError("Input value is not of type Measure.")
        if measure.haz_type != self.haz_type:
            raise ValueError("Input measures for different hazard type")
        self._data[measure.name] = measure

    def measures(self, names=None):
        """Get measures

        Parameters
        ----------
        name : str, optional
            measure name

        Returns
        -------
        """
        if names is None:
            return self._data
        return {name: meas for name, meas in self._data.items() if name in names}

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
    def size(self):
        """Get number of measures contained with input hazard type and
        /or id. If no input provided, get total number of impact functions.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        name : str, optional
            measure name

        Returns
        -------
        int
        """
        return len(self._data)

    # def combine(self, names=None, start_year=None, end_year=None, combo_name=None):
    def combine(self, names=None, combo_name=None) -> Measure:
        names = self.names if names is None else names
        # if start_year is None:
        #     start_year = np.min([meas.start_year for meas in self.measures(names).values()])
        # if end_year is None:
        #     end_year = np.max([meas.end_year for meas in self.measures(names).values()])
        meas_list = list(self.measures(names).values())

        def comb_haz_map(hazard, year=None):
            hazard_modified = meas_list[0].apply_to_hazard(hazard)
            for measure in meas_list[1:]:
                new_haz = measure.apply_to_hazard(hazard)
                hazard_modified.intensity = csr_matrix(
                    np.minimum(
                        new_haz.intensity.todense(), hazard_modified.intensity.todense()
                    )
                )
                hazard_modified.fraction = csr_matrix(
                    np.minimum(
                        new_haz.fraction.todense(), hazard_modified.fraction.todense()
                    )
                )
                hazard_modified.frequency = np.minimum(
                    new_haz.frequency, hazard_modified.frequency
                )
            return hazard_modified

        def comb_impfset_map(impfset, year=None):
            impfset_modified = meas_list[0].apply_to_impfset(impfset)
            for measure in meas_list[1:]:
                new_impfset = measure.apply_to_impfset(impfset)
                for new_impf in new_impfset.get_func(self.haz_type):
                    impf_modified = impfset_modified.get_func(
                        self.haz_type, new_impf.id
                    )
                    impf_modified.paa = np.minimum(new_impf.paa, impf_modified.paa)
                    impf_modified.mdd = np.minimum(new_impf.mdd, impf_modified.mdd)
                    impf_modified.intensity = np.maximum(
                        new_impf.intensity, impf_modified.intensity
                    )
            return impfset_modified

        def comb_exp_map(exposures, year=None):
            exposures_modified = meas_list[0].apply_to_exposures(exposures)
            for measure in meas_list[1:]:
                new_exposures = measure.apply_to_exposures(exposures)
                exposures_modified.gdf["value"] = np.minimum(
                    new_exposures.gdf["value"], exposures_modified.gdf["value"]
                )
                impf_col = f"impf_{measure.haz_type}"
                changed_impf_ids = np.array(
                    new_exposures.gdf[impf_col] != exposures.gdf[impf_col]
                )
                exposures_modified.gdf[changed_impf_ids] = new_exposures.gdf[
                    changed_impf_ids
                ]
            return exposures_modified

        def comb_cost_income():
            combined_cost_income = CostIncome(
                mkt_price_year=meas_list[0].cost_income.mkt_price_year,
                cost_growth_rate=meas_list[0].cost_income.cost_growth_rate,
                init_cost=sum([meas.cost_income.init_cost for meas in meas_list]),
                annual_cost=sum([meas.cost_income.annual_cost for meas in meas_list]),
                annual_income=sum(
                    [meas.cost_income.annual_income for meas in meas_list]
                ),
                income_growth_rate=meas_list[0].cost_income.income_growth_rate,
            )
            return combined_cost_income

        return Measure(
            name="_".join(names) if combo_name is None else combo_name,
            haz_type=self.haz_type,
            # start_year=start_year,
            # end_year=end_year,
            exposures_change=comb_exp_map,
            impfset_change=comb_impfset_map,
            hazard_change=comb_haz_map,
            combo=names,
            cost_income=comb_cost_income(),
        )
