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

# import copy
# from datetime import datetime

# import numpy as np

# from climada.hazard.base import Hazard


# def get_dates(haz: Hazard):
#     """
#     Convert ordinal dates from a Hazard object to datetime objects.

#     Parameters
#     ----------
#     haz : Hazard
#         A Hazard instance with ordinal date values.

#     Returns
#     -------
#     list of datetime
#         List of datetime objects corresponding to the ordinal dates in `haz`.

#     Example
#     -------
#     >>> haz = Hazard(...)
#     >>> get_dates(haz)
#     [datetime(2020, 1, 1), datetime(2020, 1, 2), ...]
#     """
#     return [datetime.fromordinal(date) for date in haz.date]


# def get_years(haz: Hazard):
#     """
#     Extract unique years from ordinal dates in a Hazard object.

#     Parameters
#     ----------
#     haz : Hazard
#         A Hazard instance containing ordinal date values.

#     Returns
#     -------
#     np.ndarray
#         Array of unique years as integers, derived from the ordinal dates in `haz`.

#     Example
#     -------
#     >>> haz = Hazard(...)
#     >>> get_years(haz)
#     array([2020, 2021, ...])
#     """
#     return np.unique(np.array([datetime.fromordinal(date).year for date in haz.date]))


# def grow_exp(exp, exp_growth_rate, elapsed):
#     """
#     Apply exponential growth to the exposure values over a specified period.

#     Parameters
#     ----------
#     exp : Exposures
#         The initial Exposures object with values to be grown.
#     exp_growth_rate : float
#         The annual growth rate to apply (in decimal form, e.g., 0.01 for 1%).
#     elapsed : int
#         Number of years over which to apply the growth.

#     Returns
#     -------
#     Exposures
#         A deep copy of the original Exposures object with grown exposure values.

#     Example
#     -------
#     >>> exp = Exposures(...)
#     >>> grow_exp(exp, 0.01, 5)
#     Exposures object with values grown by 5%.
#     """
#     exp_grown = copy.deepcopy(exp)
#     # Exponential growth
#     exp_growth_rate = 0.01
#     exp_grown.gdf.value = exp_grown.gdf.value * (1 + exp_growth_rate) ** elapsed
#     return exp_grown


# class TBRTrajectories:

#     # Compute impacts for trajectories with present exposure and future exposure and interpolate in between
#     #

#     @classmethod
#     def create_hazard_yearly_set(cls, haz: Hazard):
#         haz_set = {}
#         years = get_years(haz)
#         for year in range(years.min(), years.max(), 1):
#             haz_set[year] = haz.select(
#                 date=[f"{str(year)}-01-01", f"{str(year+1)}-01-01"]
#             )

#         return haz_set

#     @classmethod
#     def create_exposure_set(cls, snapshot_years, exp1, exp2=None, growth=None):
#         exp_set = {}
#         year_0 = snapshot_years.min()
#         if exp2 is None:
#             if growth is None:
#                 raise ValueError("Need to specify either final exposure or growth.")
#             else:
#                 exp_set = {
#                     year: grow_exp(exp1, growth, year - year_0)
#                     for year in snapshot_years
#                 }
#         else:
#             exp_set = {
#                 year: np.interp(exp1, exp2, year - year_0) for year in snapshot_years
#             }
#         return exp_set
