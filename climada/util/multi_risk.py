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
Define functions to handle impact_yearsets
"""
import itertools
import copy
import datetime
import warnings
from itertools import product

import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse
from scipy.sparse import lil_matrix

from climada.engine import Impact, ImpactCalc
from climada.util.constants import DEF_CRS


def calc_impacts(haz_dict, exp_dict, impf_set, mapping_exp_haz=None):
    """
        Calculate impacts based on hazard, exposure, and impact function data.

        Parameters:
        haz_dict (dict): Dictionary of hazard objects with hazard type as the key.
        exp_dict (dict): Dictionary of exposure objects with exposure type as the key.
        impf_set_dict (dict): Dictionary of impact function sets used.
        mapping_exp_haz (list, optional): Mapping of exposure and hazard keys. If None, all combinations will be used.

        Returns:
        dict: Dictionary of impacts calculated for each combination of exposure and hazard.
        """
    if mapping_exp_haz is None:
        exp_keys = list(exp_dict.keys())
        haz_keys = list(haz_dict.keys())

        # Generate all possible combinations of keys
        mapping_exp_haz = list(itertools.product(exp_keys, haz_keys))

    impacts_dict = {keys:
        ImpactCalc(exp_dict[keys[0]], impf_set, haz_dict[keys[1]]).impact()
        for keys in mapping_exp_haz
    }

    return impacts_dict


def find_common_time_definition(dates):
    """
    Find the common time definition among multiple arrays of ordinal dates.

    Parameters:
    dates (list): List of arrays of ordinal dates.

    Returns:
    str: The common time definition among the dates ('year', 'month', 'week', or 'day').
    """
    date_len = [len(date_array) for date_array in dates]
    if np.sum([l>1 for l in date_len]) < 2:
        raise ValueError("All arrays have only a single date. Cannot determine a common time definition.")

    resolutions = []

    for date_array in dates:
        if len(date_array) == 1:
            warnings.warn("Array with a single date encountered. It will not contribute to the common time definition.")
            continue

        date_list = [datetime.date.fromordinal(int(date)) for date in date_array]
        differences = [(date_list[i + 1] - date_list[i]).days for i in range(len(date_list) - 1)]

        if all(diff % 365 == 0 or diff % 366 == 0 for diff in differences):
            res = 'year'
        elif all(diff % 28 == 0 or diff % 29 == 0 or diff % 30 == 0 or diff % 31 == 0 for diff in differences):
            res = 'month'
        elif all(diff % 7 == 0 for diff in differences):
            res = 'week'
        elif all(diff % 1 == 0 for diff in differences):
            res = 'day'
        else:
            res = 'custom'

        resolutions.append(res)

    if resolutions:
        common_definition = min(resolutions, key=['year', 'month', 'week', 'day'].index)
    else:
        common_definition = None

    return common_definition


def find_common_res(coord):
    return


def upscale_dates(impact, by='year'):
    """
    Change the resolution of dates to a lower resolution.

    Parameters:
    impact (dict): Dictionary containing the impact data.
    by (str, optional): The time unit to aggregate the dates. Options are 'year' (default), 'month', or 'week'.

    Returns:
    dict: Dictionary containing the impact data with upscaled dates.
    """
    imp = copy.deepcopy(impact)
    dates = [datetime.datetime.fromordinal(int(date)) for date in imp.date]
    if by == 'year':
        dates = [datetime.date(date.year, 1, 1).toordinal() for date in dates]
    elif by == 'month':
        dates = [datetime.date(date.year, date.month, 1).toordinal() for date in dates]
    elif by == 'week':
        dates = [datetime.date.fromisocalendar(date.year, date.isocalendar()[1], 1).toordinal() for date in dates]

    imp.date = np.array(dates)
    return imp


def aggregate_impact_by_date(impact, how='sum', exp=None):
    """
    Aggregate events to have lower resolution. Maximum impact per year
    at each exposure point is exposure value if exp is not None.

    Parameters
    ----------
    impact : Impact
        Impact with an impact matrix and events with dates per year
    how : How to aggregate impacts, options are 'sum' or 'max'
    exp : Exposure
        Exposure of Impact to cap the impact value at the value of the exposure

    Raises
    ------
    AttributeError
        If impact matrix is empty.

    Returns
    -------
    impact : Impact
        Impact yearset.

    """
    impact_upscaled_dates = copy.deepcopy(impact)
    if how == 'sum':
        mask = [np.ma.make_mask(np.array(impact_upscaled_dates.date) == event).astype(int)
                for event in np.unique(impact_upscaled_dates.date)]
        mask_matrix = sp.sparse.csr_matrix(mask)
        imp_mat = mask_matrix.dot(impact_upscaled_dates.imp_mat)

    elif how == 'max':
        imp_mat = sp.sparse.csr_matrix(sp.sparse.vstack(
            [impact_upscaled_dates.imp_mat[(np.array(impact_upscaled_dates.date) == date).astype(bool)].max(axis=0)
             for date in np.unique(impact_upscaled_dates.date)]))
    else:
        warnings.warn("Unsupported value for 'how' parameter. Only 'sum' and 'max' methods are currently supported.")

    if exp is not None:
        m1 = imp_mat.data
        m2 = np.array(exp.gdf.value[imp_mat.nonzero()[1]])
        imp_mat = sp.sparse.csr_matrix((np.minimum(m1, m2), imp_mat.indices, imp_mat.indptr))

    years = np.unique([datetime.date.fromordinal(int(date)) for date in impact.date])
    frequency = np.ones(imp_mat.shape[0]) / len(np.unique(years))
    at_event, eai_exp, aai_agg = ImpactCalc.risk_metrics(imp_mat, frequency)
    date = np.unique(impact_upscaled_dates.date)
    event_id = np.arange(1, len(at_event) + 1)
    event_name = np.unique(event_id)
    impact_aggr = Impact(
        event_id=event_id,
        event_name=event_name,
        date=date,
        at_event=at_event,
        eai_exp=eai_exp,
        aai_agg=aai_agg,
        coord_exp=impact.coord_exp,
        crs=DEF_CRS,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=5,
        unit="USD",
        frequency_unit="1/year"
    )
    return impact_aggr


def fill_impact_gaps(impact_dict):
    """
    Fill in the gaps in impact of each impact in impact_dict to have the same events and coordinates.

    Parameters
    ----------
    impact_dict : dict
        The impacts to process, where the keys are hazard IDs and the values are objects with attributes.

    Returns
    -------
    dict
        The impacts with gaps filled with 0s.
    """

    # Step 1: Compile a master list of all dates and coordinates from all impacts
    all_dates = sorted(set(date for imp in impact_dict.values() for date in imp.date))
    all_coords = sorted(set(tuple(coord) for imp in impact_dict.values() for coord in imp.coord_exp))

    date_mapping = {date: i for i, date in enumerate(all_dates)}
    coord_mapping = {tuple(coord): i for i, coord in enumerate(all_coords)}

    filled_impacts = {}

    # Step 2: For each impact, create a new sparse matrix with shape (number of dates, number of coordinates)
    for hazard, imp in impact_dict.items():
        if imp.imp_mat.nnz == 0:
            raise ValueError("An element in impact_dict's imp_mat contains only zero values")

        new_mat = lil_matrix((len(all_dates), len(all_coords)))

        # Step 3: Fill the new matrix with the data from the impact
        for date, coord, data in zip(imp.date[imp.imp_mat.nonzero()[0]],
                                     imp.coord_exp[imp.imp_mat.nonzero()[1]], imp.imp_mat.data):

            new_mat[date_mapping[date], coord_mapping[tuple(coord)]] = data

        # Convert the new matrix to a CSR matrix (more efficient for calculations)
        new_mat = new_mat.tocsr()

        # Step 4: Replace the impact's imp_mat, date, and coord_exp with the new ones
        filled_impact = copy.deepcopy(imp)
        filled_impact.imp_mat = new_mat
        filled_impact.date = all_dates
        filled_impact.event_id = np.arange(1, len(all_dates)+1)
        filled_impact.event_name = np.arange(1, len(all_dates) + 1)
        filled_impact.coord_exp = np.array([list(coord) for coord in all_coords])
        years = np.unique([datetime.date.fromordinal(int(date)).year for date in filled_impact.date])
        filled_impact.frequency = np.ones(new_mat.shape[0]) / len(np.unique(years))
        filled_impact.at_event, filled_impact.eai_exp, filled_impact.aai_agg = \
            ImpactCalc.risk_metrics(filled_impact.imp_mat, filled_impact.frequency)
        filled_impacts[hazard] = filled_impact

    return filled_impacts


import itertools

def calculate_combinations_dict(impact_dict, how='sum', by='date', exp=None, combinations=None,combination_type='all'):
    """
    Calculates combinations of impacts from a dictionary.

    Parameters
    ----------
    impact_dict : dict
        Dictionary of impacts with the same coord and dates
    how : str, optional
        How to combine the impacts, options are 'sum', 'max', or 'min', by default 'sum'
    by : str, optional
        Common value used to combine events, can be 'date', 'event_id', or 'event_name', by default 'date'
    exp : object, optional
        If the exposures are given, the impacts are capped at their value, by default None
    combination_type : str, optional
        Type of combinations to calculate, can be 'all' or 'pairs', by default 'all'

    Returns
    -------
    combined_impacts : dict
        Dictionary of combined impacts with hazard keys
    """
    impact_list = list(impact_dict.values())
    hazard_keys = list(impact_dict.keys())

    if combination_type == 'all':
        combinations = []
        for r in range(1, len(impact_list) + 1):
            combinations.extend(list(itertools.combinations(impact_list, r)))
    elif combination_type == 'pairs':
        combinations = list(itertools.combinations(impact_list, 2))
    else:
        raise ValueError("Invalid combination_type. Valid options are 'all' or 'pairs'.")

    combined_impacts = {}
    for combination in combinations:
        combined_impact = combine_impacts(list(combination), how=how, by=by, exp=exp)
        hazard_key = "-".join([str(hazard_keys[impact_list.index(imp)]) for imp in combination])
        combined_impacts[hazard_key] = combined_impact

    return combined_impacts


def combine_impacts(impact_list, how='sum', by='date', exp=None):
    """
    Parameters
    ----------
    impact_list : list or dict of impacts with the same coord and dates
    how : how to combine the impacts, options are 'sum', 'max' or 'min'
    by : array of common value used to combine events, can be date, event_id or event_name
    exp : If the exposures are given, the impacts are caped at their value

    Returns
    -------
    imp : Impact
        Combined impact
    """
    if isinstance(impact_list, dict):
        impact_list = list(impact_list.values())

    imp0 = copy.deepcopy(impact_list[0])
    for imp in impact_list:
        if imp.unit != imp0.unit:
            raise ValueError("The impacts do not have the same units and cannot be combined.")
        # Check if imp_mat contains non-zero values
        if imp.imp_mat.nnz == 0:
            raise ValueError("imp_mat contains only zero values")

        # Check if all imp_mat are of the same shape
        if np.any(imp.date != imp0.date) and by=='date':
            raise ValueError("All impacts must have the same dates to be combined by date. Use the method fill_impact_gaps"
                             "first.")
        if np.any(imp.event_name != imp0.event_name) and by=='event_name':
            raise ValueError("All impacts must have the same event_name to be combined by event_name")

        if np.any(imp.event_id != imp0.event_id) and by == 'event_id':
            raise ValueError("All impacts must have the same event_id to be combined by event_id")

        if np.any(imp.coord_exp != imp0.coord_exp):
            raise ValueError("All impacts must have the same coordinates to be combined. Use the method fill_impact_gaps"
                             "first.")

        if imp.unit != imp0.unit:
            raise ValueError("All impacts must have the same units to be combined. If you want to assess events"
                             "affecting exposures with different unit, you may normalize the impact first.")

    if by == 'event_name' or by == 'date':
        aggr_attr = {'event_name': imp.event_name, 'date':imp.date}
        unique_elements = np.unique(aggr_attr[by])
        # Create a dictionary mapping unique elements to integers
        element_to_int = {element: i + 1 for i, element in enumerate(unique_elements)}

        # Map unique elements to integers using the dictionary
        imp.event_id = np.array([element_to_int[element] for element in aggr_attr[by]])
    elif by != 'event_id':
        raise NotImplementedError("This method is not implemented to combine impacts.")

    if how == 'sum':
        imp_mat_sum = imp0.imp_mat
        for imp in impact_list[1:]:
            imp_mat_sum = imp_mat_sum + imp.imp_mat
        imp_mat = imp_mat_sum

    elif how == 'min':
        imp_mat_min = imp0.imp_mat
        for imp in impact_list[1:]:
            imp_mat_min = imp_mat_min.minimum(imp.imp_mat)
        imp_mat = imp_mat_min

    elif how == 'max':
        imp_mat_max = imp0.imp_mat
        for imp in impact_list[1:]:
            imp_mat_max = imp_mat_max.maximum(imp.imp_mat)
        imp_mat = imp_mat_max
    else:
        raise ValueError(f"'{how}' is not a valid method. The implemented methods are sum, max or min")

    if exp is not None:
        m1 = imp_mat.data
        m2 = exp.gdf.value[imp_mat.nonzero()[1]]
        imp_mat = sp.sparse.csr_matrix((np.minimum(m1, m2), imp_mat.indices, imp_mat.indptr))
        imp_mat.eliminate_zeros()

    years = np.unique([datetime.date.fromordinal(int(date)).year for date in imp0.date])
    frequency = np.ones(imp_mat.shape[0]) / len(np.unique(years))
    at_event, eai_exp, aai_agg = ImpactCalc.risk_metrics(imp_mat, frequency)

    if np.all(imp.event_id == imp0.event_id and imp.event_name == imp0.event_name for imp in impact_list):
        event_id = imp0.event_id
        event_name = imp0.event_name
    else:
        event_id = np.arange(1, len(at_event) + 1)
        event_name = event_id

    impact_aggr = Impact(
        event_id=event_id,
        event_name=event_name,
        date=imp0.date,
        at_event=at_event,
        eai_exp=eai_exp,
        aai_agg=aai_agg,
        coord_exp=imp0.coord_exp,
        crs=DEF_CRS,
        imp_mat=imp_mat,
        frequency=frequency,
        tot_value=5,
        unit=imp0.unit,
        frequency_unit="1/year",
    )
    return impact_aggr

def mask_single_hazard_impact(impact, impact_list):
    """ Mask points in impact matrix of an impact where other impacts provided in impact_list do not cause impacts
    and return a new impact object. This allow to to estimate impacts that happen at the same time at the same exposure.

       Parameters
       ----------
       impact : Impact object
       impact_list : list or dict of impacts
       Returns
       -------
       impact_combi_masked : Impact
       """
    if isinstance(impact_list, dict):
        impact_list = list(impact_list.values())

    shape = impact.imp_mat.shape

    for imp in impact_list:
        # Check if imp_mat contains non-zero values
        if imp.imp_mat.nnz == 0:
            raise ValueError("An element in impact_list's imp_mat contains only zero values, please calculate impacts"
                             "again with save_mat=True")

        # Check if all imp_mat are of the same shape
        if imp.imp_mat.shape != shape:
            raise ValueError("All imp_mat must be of the same shape")

    impact_masked = copy.deepcopy(impact)
    imp_mat_masked = impact_masked.imp_mat
    for imp in impact_list:
        nonzero_mask = np.array(imp.imp_mat[imp_mat_masked.nonzero()] < 1)[0]
        rows = imp_mat_masked.nonzero()[0][nonzero_mask]
        cols = imp_mat_masked.nonzero()[1][nonzero_mask]
        imp_mat_masked[rows, cols] = 0
        imp_mat_masked.eliminate_zeros()
    impact_masked.imp_mat = imp_mat_masked
    impact_masked.at_event, impact_masked.eai_exp, impact_masked.aai_agg = \
        ImpactCalc.risk_metrics(imp_mat_masked, impact.frequency)
    return impact_masked


def corr_btw_hazards(hazard_dict, temporal=True, spatial=False):
    """
    Parameters
    ----------
    hazard_dict : dict of hazards
    temporal : bool, rather to consider the temporal dimension
    spatial : bool, rather to consider the spatial dimension

    Returns
    -------
    corr_df : pd.DataFrame
    """
    all_dates = sorted(set(date for haz in hazard_dict.values() for date in haz.date))
    all_coords = sorted(set(tuple(coord) for haz in hazard_dict.values() for coord in haz.centroids.coord))

    date_mapping = {date: i for i, date in enumerate(all_dates)}
    coord_mapping = {tuple(coord): i for i, coord in enumerate(all_coords)}

    filled_intensities = {}

    for haz_name, hazard in hazard_dict.items():
        if hazard.intensity.nnz == 0:
            raise ValueError("An element in impact_dict's imp_mat contains only zero values")

        new_mat = lil_matrix((len(all_dates), len(all_coords)))

        # Step 3: Fill the new matrix with the data from the impact
        for date, coord, data in zip(hazard.date[hazard.intensity.nonzero()[0]],
                                     hazard.centroids.coord[hazard.intensity.nonzero()[1]], hazard.intensity.data):
            new_mat[date_mapping[date], coord_mapping[tuple(coord)]] = data

        # Convert the new matrix to a CSR matrix (more efficient for calculations)
        filled_intensities[haz_name] = new_mat.tocsr()
    if temporal is True and spatial is False:
        df = pd.DataFrame.from_dict({hazard: filled_intensities[hazard].max(axis=1).toarray().flatten() for hazard in filled_intensities})
    if spatial is True and temporal is False:
        df = pd.DataFrame.from_dict({hazard:  filled_intensities[hazard].max(axis=0).toarray().flatten() for hazard in filled_intensities})
    if spatial is True and temporal is True:
        df = pd.DataFrame.from_dict({hazard: np.array(filled_intensities[hazard].todense().flatten())[0]
                                     for hazard in filled_intensities})
    return df.corr()

def corr_btw_impacts(impact_dict, temporal=True, spatial=False):
    """
    Parameters
    ----------
    impact_dict : dict of Impacts
    temporal : bool, rather to consider the temporal dimension
    spatial : bool, rather to consider the spatial dimension

    Returns
    -------
    corr_df : pd.DataFrame
    """

    if temporal is True and spatial is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].at_event for hazard in impact_dict})
    if spatial is True and temporal is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].eai_exp for hazard in impact_dict})
    if spatial is True and temporal is True:
        df = pd.DataFrame.from_dict({hazard: np.array(impact_dict[hazard].imp_mat.todense().flatten())[0]
                                     for hazard in impact_dict})
    return df.corr()

