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
from typing import Callable, Optional

import numpy as np

from climada.engine import ImpactCalc
from climada.entity import ImpactFuncSet
from climada.entity.exposures.base import Exposures
from climada.hazard.base import Hazard

from .cost_income import CostIncome

LOGGER = logging.getLogger(__name__)


##todo: risk transfer, change hazard/exposures/impfset completely
class Measure:
    """Contains a measure to be applied to a set of exposures, impact functions and hazard.

    Attributes
    ----------
    name : str
        Name of the measure
    start_year : int
        Start year of the measure
    end_year : int
        End year of the measure
    haz_type : str
        Type of hazard
    exposures_change : callable
        Function to change exposures
    impfset_change : callable
        Function to change impact function set
    hazard_change : callable
        Function to change hazard
    cost_income : climada.entity.measures.cost_income.CostIncome
        Cost and income object
    """

    def __init__(
        self,
        name: str,
        # start_year: int,
        # end_year: int,
        haz_type: str,
        exposures_change: Callable[[Exposures, int], Exposures] = lambda x, y: x,
        impfset_change: Callable[[ImpactFuncSet, int], ImpactFuncSet] = lambda x, y: x,
        hazard_change: Callable[[Hazard, int], Hazard] = lambda x, y: x,
        combo: list[
            str
        ] = None,  # list of measure names that this measure is a combination of (Probably better to stire the other measures in the measure object)
        cost_income: Optional[CostIncome] = None,
        implenmentation_duration: int = 0,  # duration of implementation in years before the measure is fully implemented (or should this be made later ... )
    ):
        """
        Initialize a new Measure object with specified data.

        Parameters
        ----------
        name : str
            Name of the measure
            End year of the measure
        haz_type : str
            Type of hazard
        exposures_change : callable
            Function to change exposures
        impfset_change : callable
            Function to change impact function set
        hazard_change : callable
            Function to change hazard
        CostIncome : climada.entity.measures.cost_income.CostIncome
            Cost and income object
        implenmentation_duration : int
            Duration of implementation in years before the measure is fully implemented

        """
        self.name = name
        self.exp_map = exposures_change
        self.impfset_map = impfset_change
        self.haz_map = hazard_change
        self.haz_type = haz_type
        self.combo = combo
        self.cost_income = cost_income if cost_income is not None else CostIncome()
        self.implenmentation_duration = implenmentation_duration

    # @property
    # def start_year(self):
    #     return self.years[0]

    # @property
    # def end_year(self):
    #     return self.years[1]

    def apply_to_exposures(self, exposures, year=None):
        """
        Implement measure to exposures.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance

        Returns
        -------
        new_exp : climada.entity.Exposure
            Exposure with implemented measure with all defined parameters
        """
        return self.exp_map(exposures, year)

    def apply_to_impfset(self, impfset, year=None):
        """
        Implement measure to impact function set

        Parameters
        ----------
        impfset : climada.entity.ImpactFuncSet
            impact function set instance

        Returns
        -------
        new_impfset : climada.entity.ImpactFuncSet
            Impact function set with implemented measure with all defined parameters
        """
        return self.impfset_map(impfset, year)

    def apply_to_hazard(self, hazard, year=None):
        """
        Implement measure to hazard.

        Parameters
        ----------
        hazard : climada.hazard.Hazard
            hazard instance

        Returns
        -------
        new_haz : climada.hazard.Hazard
            Hazard with implemented measure with all defined parameters
        """
        return self.haz_map(hazard, year)

    def apply(self, exposures, impfset, hazard, year=None):
        """
        Implement measure with all its defined parameters.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance
        imp_fun_set : climada.entity.ImpactFuncSet
            impact function set instance
        hazard : climada.hazard.Hazard
            hazard instance

        Returns
        -------
        new_exp : climada.entity.Exposure
            Exposure with implemented measure with all defined parameters
        new_ifs : climada.entity.ImpactFuncSet
            Impact function set with implemented measure with all defined parameters
        new_haz : climada.hazard.Hazard
            Hazard with implemented measure with all defined parameters
        """
        # change exposures
        new_exp = self.exp_map(exposures, year)
        # change impact functions
        new_impfs = self.impfset_map(impfset, year)
        # change hazard
        new_haz = self.haz_map(hazard, year)
        return new_exp, new_impfs, new_haz

    def impact(self, exposures, impfset, hazard, year=None, **kwargs):
        meas_exp, meas_impfset, meas_haz = self.apply(
            exposures, impfset, hazard, year=year
        )
        return ImpactCalc(meas_exp, meas_impfset, meas_haz).impact(**kwargs)


def helper_hazard(intensity_multiplier=1, intensity_substract=0):
    def hazard_change(hazard, year=None):
        haz_modified = copy.deepcopy(hazard)
        haz_modified.intensity.data *= intensity_multiplier
        haz_modified.intensity.data -= intensity_substract
        haz_modified.intensity.data[haz_modified.intensity.data < 0] = 0
        haz_modified.intensity.eliminate_zeros()
        return haz_modified

    return hazard_change


def replace_hazard(measure_hazards):
    def hazard_change(hazard, year):
        return measure_hazards[year]

    return hazard_change


def hazard_intensity_rp_cutoff(cut_off_rp, hazard):
    return hazard.local_exceedance_inten(return_periods=(cut_off_rp))


def impact_intensity_rp_cutoff(
    cut_off_rp, exposures, impfset, hazard, exposures_region_id
):
    if exposures_region_id:
        # compute impact only in selected region
        in_reg = np.logical_or.reduce(
            [exposures.gdf.region_id.values == reg for reg in exposures_region_id]
        )
        exp_imp = Exposures(exposures.gdf[in_reg], crs=exposures.crs)
    else:
        exp_imp = exposures
    imp = ImpactCalc(exp_imp, impfset, hazard).impact(save_mat=False)
    sort_idxs = np.argsort(imp.at_event)[::-1]
    exceed_freq = np.cumsum(imp.frequency[sort_idxs])
    events_above_cutoff = sort_idxs[exceed_freq > cut_off_rp]
    intensity_substract = hazard.intensity.data
    for event in events_above_cutoff:
        intensity_substract[
            hazard.intensity.indptr[event] : hazard.intensity.indptr[event + 1]
        ] = 0
    return intensity_substract


def change_impfset(measure_impfsets):
    def impfset_change(impfset, year):
        return measure_impfsets[year]

    return impfset_change


def helper_impfset(
    haz_type,
    impf_mdd_modifier={1: (1, 0)},
    impf_paa_modifier={1: (1, 0)},
    impf_intensity_modifier={1: (1, 0)},
):
    def impfset_change(impfset, year=None):
        impfset_modified = copy.deepcopy(impfset)
        for impf in impfset_modified.get_func(haz_type):
            if impf.id in impf_intensity_modifier.keys():
                impf_inten = impf_intensity_modifier[impf.id]
                impf.intensity = np.maximum(
                    impf.intensity * impf_inten[0] - impf_inten[1], 0.0
                )
            if impf.id in impf_mdd_modifier.keys():
                impf_mdd = impf_mdd_modifier[impf.id]
                impf.mdd = np.maximum(impf.mdd * impf_mdd[0] + impf_mdd[1], 0.0)
            if impf.id in impf_paa_modifier.keys():
                impf_paa = impf_paa_modifier[impf.id]
                impf.paa = np.maximum(impf.paa * impf_paa[0] + impf_paa[1], 0.0)
        return impfset_modified

    return impfset_change


def helper_exposure(reassign_impf_id, haz_type, set_to_zero=None):
    if set_to_zero is None:
        set_to_zero = []

    def exposures_change(exposures, year=None):
        exp_modified = exposures.copy()
        exp_modified.gdf[f"impf_{haz_type}"] = reassign_impf_id
        exp_modified.gdf["value"][set_to_zero] = 0
        return exp_modified

    return exposures_change
