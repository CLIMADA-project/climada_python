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

"""

import copy
from dataclasses import dataclass

import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from scipy.sparse import lil_matrix

from climada import hazard
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard import Hazard
from climada.engine.impact_calc import ImpactCalc


### Utils functions
def pairwise(iterable):
    "s -> (s0, s1), (s1, s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def get_dates(haz: Hazard):
    return [datetime.fromordinal(date) for date in haz.date]


def get_years(haz: Hazard):
    return np.unique(np.array([datetime.fromordinal(date).year for date in haz.date]))


def grow_exp(exp, exp_growth_rate, elapsed):
    exp_grown = copy.deepcopy(exp)
    # Exponential growth
    exp_growth_rate = 0.01
    exp_grown.gdf.value = exp_grown.gdf.value * (1 + exp_growth_rate) ** elapsed
    return exp_grown


def interpolate_sm(mat_start, mat_end, year, year_start, year_end):
    if year < year_start or year > year_end:
        raise ValueError("Year must be within the start and end years")

    # Calculate the ratio of the difference between the target year and the start year
    # to the total number of years between the start and end years
    ratio = (year - year_start) / (year_end - year_start)

    # Convert the input matrices to a format that allows efficient modification of its elements
    mat_start = lil_matrix(mat_start)
    mat_end = lil_matrix(mat_end)

    # Perform the linear interpolation
    mat_interpolated = mat_start + ratio * (mat_end - mat_start)

    return mat_interpolated


# Derive the intermediate probability distributions
def interpolate_years(year_start, year_end):
    # Generate an array of interpolated values between 0 and 1
    values = np.linspace(0, 1, num=year_end - year_start + 1)
    return values


def bayesian_viktypliers(year0, year1):
    prop_H1 = interpolate_years(year0, year1)
    prop_H0 = 1 - prop_H1
    return prop_H0, prop_H1


def snapshot_combinaisons(snapshot0, snapshot1):
    impfset0 = snapshot0.impfset
    impfset1 = snapshot1.impfset
    assert impfset0 is impfset1  # We don't allow for different impfset

    exp_y0 = snapshot0.exposure
    exp_y1 = snapshot1.exposure
    haz_y0 = snapshot0.hazard
    haz_y1 = snapshot1.hazard

    # Case 1 - H2000# Impact 1)  Hazard 2000  and Exposure 2000
    imp_E0H0 = ImpactCalc(exp_y0, impfset0, haz_y0).impact()
    imp_E1H0 = ImpactCalc(
        exp_y1, impfset0, haz_y0
    ).impact()  # Impact 2)  Hazard 2000  and Exposure 2020

    # Case 2 - H2020
    # Impact 1)  Hazard 2000  and Exposure 2000
    imp_E0H1 = ImpactCalc(exp_y0, impfset0, haz_y1).impact()
    imp_E1H1 = ImpactCalc(exp_y1, impfset0, haz_y1).impact()

    return imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1


def interpolate_imp_mat(imp0, imp1, start_year, end_year):
        return [
            interpolate_sm(imp0.imp_mat, imp1.imp_mat, year, start_year, end_year)
            for year in range(start_year, end_year + 1)
        ]

def calc_freq_curve(imp_mat_intrpl, frequency, return_per=None):
    '''
    Calculate the frequency curve

    Parameters:
    imp_mat_intrpl (np.array): The interpolated impact matrix
    frequency (np.array): The frequency of the hazard
    return_per (np.array): The return period

    Returns:
    ifc_return_per (np.array): The impact exceeding frequency
    ifc_impact (np.array): The impact exceeding the return period
    '''

    # Calculate the at_event make the np.array
    at_event = np.sum(imp_mat_intrpl, axis=1).A1

    # Sort descendingly the impacts per events
    sort_idxs = np.argsort(at_event)[::-1]
    # Calculate exceedence frequency
    exceed_freq = np.cumsum(frequency[sort_idxs])
    # Set return period and impact exceeding frequency
    ifc_return_per = 1 / exceed_freq[::-1]
    ifc_impact = at_event[sort_idxs][::-1]

    if return_per is not None:
        interp_imp = np.interp(return_per, ifc_return_per, ifc_impact)
        ifc_return_per = return_per
        ifc_impact = interp_imp

    return ifc_impact


def calc_yearly_eais(imp_mats_0, imp_mats_1, frequency_0, frequency_1):
    yearly_eai_exp_0 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_0) for imp_mat in imp_mats_0
    ]
    yearly_eai_exp_1 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_1) for imp_mat in imp_mats_1
    ]
    return yearly_eai_exp_0, yearly_eai_exp_1

def calc_yearly_rps(imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods):
    rp_0 = [
        calc_freq_curve(imp_mat, frequency_0, return_periods) for imp_mat in imp_mats_0
    ]
    rp_1 = [
        calc_freq_curve(imp_mat, frequency_1, return_periods) for imp_mat in imp_mats_1
    ]
    return rp_0, rp_1

def calc_yearly_aais(yearly_eai_exp_0, yearly_eai_exp_1):
    yearly_aai_0 = [
            ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_0
    ]
    yearly_aai_1 = [
            ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_1
    ]
    return yearly_aai_0, yearly_aai_1


def bayesian_mixer(start_snapshot, end_snapshot, metrics=["eai", "aai", "rp"], return_periods=[100, 500, 1000]):
        # 1. Interpolate in between years
        prop_H0, prop_H1 = bayesian_viktypliers(start_snapshot.year, end_snapshot.year)
        imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1 = snapshot_combinaisons(
            start_snapshot, end_snapshot
        )
        frequency_0 = start_snapshot.hazard.frequency
        frequency_1 = end_snapshot.hazard.frequency

        imp_mats_0 = interpolate_imp_mat(imp_E0H0, imp_E1H0, start_snapshot.year, end_snapshot.year)
        imp_mats_1 = interpolate_imp_mat(imp_E0H1, imp_E1H1, start_snapshot.year, end_snapshot.year)

        yearly_eai_exp_0, yearly_eai_exp_1 = calc_yearly_eais(imp_mats_0, imp_mats_1, frequency_0, frequency_1)

        res = []

        if "eai" in metrics:
            yearly_eai = (
                np.multiply(prop_H0.reshape(-1,1), yearly_eai_exp_0) +
                np.multiply(prop_H1.reshape(-1,1), yearly_eai_exp_1)
            )
            eai_df = pd.DataFrame(index=list(range(start_snapshot.year, end_snapshot.year+1)),
                                  data=yearly_eai)
            eai_df["group"] = pd.NA
            eai_df["metric"] = "eai"
            eai_df.reset_index(inplace=True)
            res.append(eai_df)

        if "aai" in metrics:
            yearly_aai_0, yearly_aai_1 = calc_yearly_aais(yearly_eai_exp_0, yearly_eai_exp_1)
            yearly_aai = prop_H0 * yearly_aai_0 + prop_H1 * yearly_aai_1
            aai_df = pd.DataFrame(index=list(range(start_snapshot.year, end_snapshot.year+1)),
                                  data=yearly_aai)
            aai_df["group"] = pd.NA
            aai_df["metric"] = "aai"
            aai_df.reset_index(inplace=True)
            res.append(aai_df)

        if "rp" in metrics:
            tmp = []
            for rp in return_periods:
                rp_0, rp_1 = calc_yearly_rps(imp_mats_0, imp_mats_1, frequency_0, frequency_1, rp)
                yearly_rp = np.multiply(prop_H0.reshape(-1,1), rp_0) + np.multiply(prop_H1.reshape(-1,1), rp_1)
                tmp_df = pd.DataFrame(index=list(range(start_snapshot.year, end_snapshot.year+1)),
                                      data=yearly_rp)
                tmp_df["group"] = pd.NA
                tmp_df["metric"] = f"rp_{rp}"
                tmp_df.reset_index(inplace=True)
                tmp.append(tmp_df)
            rp_df = pd.concat(tmp)
            res.append(rp_df)

        return pd.concat(res)

@dataclass
class Snapshot:
    exposure: Exposures
    hazard: Hazard
    impfset: ImpactFuncSet
    year: int


class SnapshotsCollection:
    def __init__(self, exposure_set, hazard_set, impfset, snapshot_years):
        self.exposure_set = exposure_set
        self.hazard_set = hazard_set
        self.impfset = impfset
        self.snapshots_years = snapshot_years
        self.data = [
            Snapshot(exposure_set[year], hazard_set[year], impfset, year)
            for year in snapshot_years
        ]

    # Check that at least first and last snap are complete
    # and otherwise it is ok

    @classmethod
    def from_dict(cls, snapshots_dict, impfset):
        snapshot_years = list(snapshots_dict.keys())
        exposure_set = {year: snapshots_dict[year][0] for year in snapshot_years}
        hazard_set = {year: snapshots_dict[year][1] for year in snapshot_years}
        return cls(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )

    @classmethod
    def from_lists(cls, hazard_list, exposure_list, impfset, snapshot_years):
        exposure_set = {year: exposure_list[i] for i, year in enumerate(snapshot_years)}
        hazard_set = {year: hazard_list[i] for i, year in enumerate(snapshot_years)}
        return cls(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )


class CalcImpactsSnapshots:
    def __init__(self, snapshots: SnapshotsCollection):
        self.snapshots = snapshots
        self.group_map_exp_dict = None
        self.yearly_eai_exp_tuples = []

    # An init param could be the region aggregation you want

    def calc_impacts_snapshots(self):
        impacts_list = {}
        for snapshot in self.snapshots.data:
            impacts_list[snapshot.year] = ImpactCalc(
                snapshot.exposure, self.snapshots.impfset, snapshot.hazard
            ).impact()
        return impacts_list

    def calc_all_years(self):
        all_yearly_aai = []
        start = True
        for start_snapshot, end_snapshot in pairwise(self.snapshots.data):
            if start:
                all_yearly_aai.append(self.bayesian_mixer(start_snapshot, end_snapshot))
                start = False
            else:
                all_yearly_aai.append(
                    self.bayesian_mixer(start_snapshot, end_snapshot)[1:]
                )
        return np.concatenate(all_yearly_aai)


#### WIP


class TBRTrajectories:

    @classmethod
    def create_hazard_yearly_set(cls, haz: Hazard):
        haz_set = {}
        years = get_years(haz)
        for year in range(years.min(), years.max(), 1):
            haz_set[year] = haz.select(
                date=[f"{str(year)}-01-01", f"{str(year+1)}-01-01"]
            )

        return haz_set

    @classmethod
    def create_exposure_set(cls, snapshot_years, exp1, exp2=None, growth=None):
        exp_set = {}
        if exp2 is None:
            if growth is None:
                raise ValueError("Need to specify either final exposure or growth.")
            else:
                year_0 = snapshot_years.min()
                exp_set = {
                    year: grow_exp(exp1, year - year_0) for year in snapshot_years
                }
        else:
            exp_set = {
                year: interp(exp1, exp2, year - year_0) for year in snapshot_years
            }
