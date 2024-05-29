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

import numpy as np
from datetime import datetime
from scipy.sparse import lil_matrix

from climada.hazard import Hazard
from climada.engine.impact_calc import ImpactCalc


### Utils functions


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


class SnapshotsCollection:
    def __init__(self, exposure_set, hazard_set, impfset, snapshot_years):
        self.exposure_set = exposure_set
        self.hazard_set = hazard_set
        self.impfset = impfset
        self.snapshots_years = snapshot_years
        self.data_dict = {
            year: [exposure_set[year], hazard_set[year]] for year in snapshot_years
        }

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

    def calc_impacts_list(self):
        impacts_list = {}
        for year, [exp, haz] in self.snapshots.data_dict.items():
            impacts_list[year] = ImpactCalc(exp, self.snapshots.impfset, haz).impact()
        return impacts_list

    # Calculate the eai_exp for each year
    def interpolate_eai_exp(self, imp0, imp1, start_year, end_year, frequency):
        yearly_eai_exp = []
        for year in range(start_year, end_year + 1):
            imp_mat_intrpl = interpolate_sm(
                imp0.imp_mat, imp1.imp_mat, year, start_year, end_year
            )
            # sum across the rows of the sparse matrix
            eai_exp = ImpactCalc.eai_exp_from_mat(imp_mat_intrpl, frequency)
            yearly_eai_exp.append(eai_exp)
        return yearly_eai_exp

    def snapshot_combinaisons(self, year0, year1):
        prop_H1 = interpolate_years(year0, year1)
        prop_H0 = 1 - prop_H1

        exp_y0 = self.snapshots.data_dict[year0][0]
        exp_y1 = self.snapshots.data_dict[year1][0]
        haz_y0 = self.snapshots.data_dict[year0][1]
        haz_y1 = self.snapshots.data_dict[year1][1]

        # Case 1 - H2000# Impact 1)  Hazard 2000  and Exposure 2000
        imp_E0H0 = ImpactCalc(exp_y0, self.snapshots.impfset, haz_y0).impact()
        imp_E1H0 = ImpactCalc(
            exp_y1, self.snapshots.impfset, haz_y0
        ).impact()  # Impact 2)  Hazard 2000  and Exposure 2020

        # Case 2 - H2020
        # Impact 1)  Hazard 2000  and Exposure 2000
        imp_E0H1 = ImpactCalc(exp_y0, self.snapshots.impfset, haz_y1).impact()
        imp_E1H1 = ImpactCalc(exp_y1, self.snapshots.impfset, haz_y1).impact()

        return prop_H0, prop_H1, imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1

    def bayesian_mixer(self):
        # 1. Interpolate in between years
        for i in range(len(self.snapshots.snapshots_years) - 1):
            start_year, end_year = (
                self.snapshots.snapshots_years[i],
                self.snapshots.snapshots_years[i + 1],
            )
            prop_H0, prop_H1, imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1 = (
                self.snapshot_combinaisons(start_year, end_year)
            )
            frequency0 = self.snapshots.data_dict[start_year][1].frequency
            frequency1 = self.snapshots.data_dict[end_year][1].frequency

            yearly_eai_exp_0 = self.interpolate_eai_exp(
                imp_E0H0, imp_E1H0, start_year, end_year, frequency0
            )
            yearly_eai_exp_1 = self.interpolate_eai_exp(
                imp_E0H1, imp_E1H1, start_year, end_year, frequency1
            )
            yearly_aai_0 = [
                ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_0
            ]
            yearly_aai_1 = [
                ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_1
            ]

        # Average Annual Impact across the years
        yearly_aai = prop_H0 * yearly_aai_0 + prop_H1 * yearly_aai_1
        # EAI region_id across the years
        # self.yearly_eai_region_id = np.multiply(prop_H0.reshape(-1,1), yearly_eai_region_id_0) + np.multiply(prop_H1.reshape(-1,1), yearly_eai_region_id_1)
        return yearly_aai


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
