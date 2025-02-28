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
import itertools
import logging
from dataclasses import dataclass
from datetime import datetime
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from climada.engine.impact_calc import ImpactCalc
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.entity.measures.measure_set import MeasureSet
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)


### Utils functions
def impact_func_equal(func1: ImpactFunc, func2: ImpactFunc) -> bool:
    """Check equality of two ImpactFunc objects."""
    return (
        func1.haz_type == func2.haz_type
        and func1.id == func2.id
        and func1.name == func2.name
        and func1.intensity_unit == func2.intensity_unit
        and np.array_equal(func1.intensity, func2.intensity)
        and np.array_equal(func1.mdd, func2.mdd)
        and np.array_equal(func1.paa, func2.paa)
    )


def impact_func_set_equal(set1: ImpactFuncSet, set2: ImpactFuncSet) -> bool:
    """Check equality of two ImpactFuncSet objects."""
    if set1._data.keys() != set2._data.keys():
        return False

    for haz_type1, id_map1 in set1._data.items():
        id_map2 = set2._data[haz_type1]
        if id_map1.keys() != id_map2.keys():
            return False
        for fid, func1 in id_map1.items():
            if not impact_func_equal(func1, id_map2[fid]):
                return False

    return True


def hazard_data_equal(haz1: Hazard, haz2: Hazard) -> bool:
    intensity_eq = (haz1.intensity != haz2.intensity).nnz == 0
    freq_eq = (haz1.frequency == haz2.frequency).all()
    frac_eq = (haz1.fraction != haz2.fraction).nnz == 0
    return intensity_eq and freq_eq and frac_eq


def get_dates(haz: Hazard):
    """
    Convert ordinal dates from a Hazard object to datetime objects.

    Parameters
    ----------
    haz : Hazard
        A Hazard instance with ordinal date values.

    Returns
    -------
    list of datetime
        List of datetime objects corresponding to the ordinal dates in `haz`.

    Example
    -------
    >>> haz = Hazard(...)
    >>> get_dates(haz)
    [datetime(2020, 1, 1), datetime(2020, 1, 2), ...]
    """
    return [datetime.fromordinal(date) for date in haz.date]


def get_years(haz: Hazard):
    """
    Extract unique years from ordinal dates in a Hazard object.

    Parameters
    ----------
    haz : Hazard
        A Hazard instance containing ordinal date values.

    Returns
    -------
    np.ndarray
        Array of unique years as integers, derived from the ordinal dates in `haz`.

    Example
    -------
    >>> haz = Hazard(...)
    >>> get_years(haz)
    array([2020, 2021, ...])
    """
    return np.unique(np.array([datetime.fromordinal(date).year for date in haz.date]))


def grow_exp(exp, exp_growth_rate, elapsed):
    """
    Apply exponential growth to the exposure values over a specified period.

    Parameters
    ----------
    exp : Exposures
        The initial Exposures object with values to be grown.
    exp_growth_rate : float
        The annual growth rate to apply (in decimal form, e.g., 0.01 for 1%).
    elapsed : int
        Number of years over which to apply the growth.

    Returns
    -------
    Exposures
        A deep copy of the original Exposures object with grown exposure values.

    Example
    -------
    >>> exp = Exposures(...)
    >>> grow_exp(exp, 0.01, 5)
    Exposures object with values grown by 5%.
    """
    exp_grown = copy.deepcopy(exp)
    # Exponential growth
    exp_growth_rate = 0.01
    exp_grown.gdf.value = exp_grown.gdf.value * (1 + exp_growth_rate) ** elapsed
    return exp_grown


def interpolate_sm(mat_start, mat_end, year, year_start, year_end):
    """
    Perform linear interpolation between two matrices for a specified year.

    Parameters
    ----------
    mat_start : scipy.sparse.lil_matrix
        The starting matrix at `year_start`.
    mat_end : scipy.sparse.lil_matrix
        The ending matrix at `year_end`.
    year : int
        The target year for interpolation.
    year_start : int
        The starting year of the interpolation range.
    year_end : int
        The ending year of the interpolation range.

    Returns
    -------
    scipy.sparse.lil_matrix
        The interpolated matrix for the specified year.

    Raises
    ------
    ValueError
        If `year` is outside the range defined by `year_start` and `year_end`.

    Example
    -------
    >>> interpolate_sm(mat_start, mat_end, 2015, 2010, 2020)
    Interpolated sparse matrix for the year 2015.
    """
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


# Risk transfer functions
def calc_residual_or_risk_transf_imp_mat(
    imp_mat, attachment=None, cover=None, calc_residual=True
):
    """
    Calculate either the residual or the risk transfer impact matrix.

    The impact matrix is adjusted based on the total impact for each event.
    When calculating the residual impact, the result is the total impact minus
    the risk layer. The risk layer is defined as the minimum of the cover and
    the maximum of the difference between the total impact and the attachment.
    If `calc_residual` is False, the function returns the risk layer matrix
    instead of the residual.

    Parameters
    ----------
    imp_mat : scipy.sparse.csr_matrix
        The original impact matrix to be scaled.
    attachment : float, optional
        The attachment point for the risk layer.
    cover : float, optional
        The maximum coverage for the risk layer.
    calc_residual : bool, default=True
        Determines if the function calculates the residual (if True) or the
        risk layer (if False).

    Returns
    -------
    scipy.sparse.csr_matrix
        The adjusted impact matrix, either residual or risk transfer.

    Example
    -------
    >>> calc_residual_or_risk_transf_imp_mat(imp_mat, attachment=100, cover=500, calc_residual=True)
    Residual impact matrix with applied risk layer adjustments.
    """
    if attachment and cover:
        # Make a copy of the impact matrix
        imp_mat = copy.deepcopy(imp_mat)
        # Calculate the total impact per event
        total_at_event = imp_mat.sum(axis=1).A1
        # Risk layer at event
        transfer_at_event = np.minimum(
            np.maximum(total_at_event - attachment, 0), cover
        )
        # Resiudal impact
        residual_at_event = np.maximum(total_at_event - transfer_at_event, 0)

        # Calculate either the residual or transfer impact matrix
        # Choose the denominator to rescale the impact values
        if calc_residual:
            # Rescale the impact values
            numerator = residual_at_event
        else:
            # Rescale the impact values
            numerator = transfer_at_event

        # Rescale the impact values
        rescale_impact_values = np.divide(
            numerator,
            total_at_event,
            out=np.zeros_like(numerator, dtype=float),
            where=total_at_event != 0,
        )

        # The multiplication is broadcasted across the columns for each row
        result_matrix = imp_mat.multiply(rescale_impact_values[:, np.newaxis])

        return result_matrix

    else:

        return imp_mat


# Derive the intermediate probability distributions
def interpolate_years(year_start, year_end):
    """
    Generate an array of interpolated values between 0 and 1 for a range of years.

    Parameters
    ----------
    year_start : int
        The starting year of interpolation.
    year_end : int
        The ending year of interpolation.

    Returns
    -------
    np.ndarray
        Array of interpolated values between 0 and 1 for each year in the range.
    """
    values = np.linspace(0, 1, num=year_end - year_start + 1)
    return values


def bayesian_viktypliers(year0, year1):
    """
    Calculate the Bayesian interpolation proportions for a given year range.

    Parameters
    ----------
    year0 : int
        Starting year.
    year1 : int
        Ending year.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing:
        - prop_H0 : np.ndarray
            Array of proportions for the H0 hypothesis.
        - prop_H1 : np.ndarray
            Array of proportions for the H1 hypothesis.
    """
    prop_H1 = interpolate_years(year0, year1)
    prop_H0 = 1 - prop_H1
    return prop_H0, prop_H1


def snapshot_combinaisons(snapshot0, snapshot1):
    """
    Calculate impact combinations between two snapshots with shared impact function set.

    Parameters
    ----------
    snapshot0 : Snapshot
        The starting snapshot.
    snapshot1 : Snapshot
        The ending snapshot.

    Returns
    -------
    tuple
        Tuple containing the impacts for each scenario:
        - imp_E0H0 : ImpactCalc result for exposure from snapshot0, hazard from snapshot0.
        - imp_E1H0 : ImpactCalc result for exposure from snapshot1, hazard from snapshot0.
        - imp_E0H1 : ImpactCalc result for exposure from snapshot0, hazard from snapshot1.
        - imp_E1H1 : ImpactCalc result for exposure from snapshot1, hazard from snapshot1.
    """
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
    """
    Interpolate between two impact matrices over a specified year range.

    Parameters
    ----------
    imp0 : ImpactCalc
        The impact calculation for the starting year.
    imp1 : ImpactCalc
        The impact calculation for the ending year.
    start_year : int
        The starting year for interpolation.
    end_year : int
        The ending year for interpolation.

    Returns
    -------
    list of np.ndarray
        List of interpolated impact matrices for each year in the specified range.
    """
    LOGGER.debug(f"imp0: {imp0.imp_mat.data[0]}, imp1: {imp1.imp_mat.data[0]}")
    return [
        interpolate_sm(imp0.imp_mat, imp1.imp_mat, year, start_year, end_year)
        for year in range(start_year, end_year + 1)
    ]


def calc_freq_curve(imp_mat_intrpl, frequency, return_per=None):
    """
    Calculate the frequency curve

    Parameters:
    imp_mat_intrpl (np.array): The interpolated impact matrix
    frequency (np.array): The frequency of the hazard
    return_per (np.array): The return period

    Returns:
    ifc_return_per (np.array): The impact exceeding frequency
    ifc_impact (np.array): The impact exceeding the return period
    """

    # Calculate the at_event make the np.array
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
    """
    Calculate yearly expected annual impact (EAI) values for two scenarios.

    Parameters
    ----------
    imp_mats_0 : list of np.ndarray
        List of interpolated impact matrices for scenario 0.
    imp_mats_1 : list of np.ndarray
        List of interpolated impact matrices for scenario 1.
    frequency_0 : np.ndarray
        Frequency values associated with scenario 0.
    frequency_1 : np.ndarray
        Frequency values associated with scenario 1.

    Returns
    -------
    tuple
        Tuple containing:
        - yearly_eai_exp_0 : list of float
            Yearly expected annual impacts for scenario 0.
        - yearly_eai_exp_1 : list of float
            Yearly expected annual impacts for scenario 1.
    """
    yearly_eai_exp_0 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_0) for imp_mat in imp_mats_0
    ]
    yearly_eai_exp_1 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_1) for imp_mat in imp_mats_1
    ]
    return yearly_eai_exp_0, yearly_eai_exp_1


def calc_yearly_rps(imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods):
    """
    Calculate yearly return period impact values for two scenarios.

    Parameters
    ----------
    imp_mats_0 : list of np.ndarray
        List of interpolated impact matrices for scenario 0.
    imp_mats_1 : list of np.ndarray
        List of interpolated impact matrices for scenario 1.
    frequency_0 : np.ndarray
        Frequency values for scenario 0.
    frequency_1 : np.ndarray
        Frequency values for scenario 1.
    return_periods : list of int
        Return periods to calculate impact values for.

    Returns
    -------
    tuple
        Tuple containing:
        - rp_0 : list of np.ndarray
            Yearly return period impact values for scenario 0.
        - rp_1 : list of np.ndarray
            Yearly return period impact values for scenario 1.
    """
    rp_0 = [
        calc_freq_curve(imp_mat, frequency_0, return_periods) for imp_mat in imp_mats_0
    ]
    rp_1 = [
        calc_freq_curve(imp_mat, frequency_1, return_periods) for imp_mat in imp_mats_1
    ]
    return rp_0, rp_1


def calc_yearly_aais(yearly_eai_exp_0, yearly_eai_exp_1):
    """
    Calculate yearly aggregate annual impact (AAI) values for two scenarios.

    Parameters
    ----------
    yearly_eai_exp_0 : list of float
        Yearly expected annual impacts for scenario 0.
    yearly_eai_exp_1 : list of float
        Yearly expected annual impacts for scenario 1.

    Returns
    -------
    tuple
        Tuple containing:
        - yearly_aai_0 : list of float
            Aggregate annual impact values for scenario 0.
        - yearly_aai_1 : list of float
            Aggregate annual impact values for scenario 1.
    """
    yearly_aai_0 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_0
    ]
    yearly_aai_1 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_1
    ]
    return yearly_aai_0, yearly_aai_1


def get_eai_exp(eai_exp, group_map):
    """
    Aggregate expected annual impact (EAI) by groups.

    Parameters
    ----------
    eai_exp : np.ndarray
        Array of EAI values.
    group_map : dict
        Mapping of group names to indices for aggregation.

    Returns
    -------
    dict
        Dictionary of EAI values aggregated by specified groups.
    """
    eai_region_id = {}
    for group_name, exp_indices in group_map.items():
        eai_region_id[group_name] = np.sum(eai_exp[:, exp_indices], axis=1)
    return eai_region_id


def bayesian_mixer_opti(
    risk_period,
    metrics,
    return_periods,
    groups=None,
    all_groups_name=pd.NA,
):
    """
    Perform Bayesian mixing of impacts across snapshots.

    Parameters
    ----------
    start_snapshot : Snapshot
        The starting snapshot.
    end_snapshot : Snapshot
        The ending snapshot.
    metrics : list of str
        Metrics to calculate (e.g., 'eai', 'aai', 'rp').
    return_periods : list of int
        Return periods for calculating impact values.
    groups : dict, optional
        Mapping of group names to indices for aggregating EAI values by group.
    all_groups_name : str, optional
        Name for all-groups aggregation in the output.
    risk_transf_cover : float, optional
        Coverage level for risk transfer calculations.
    risk_transf_attach : float, optional
        Attachment point for risk transfer calculations.
    calc_residual : bool, optional
        Whether to calculate residual impacts after applying risk transfer.

    Returns
    -------
    pd.DataFrame
        DataFrame of calculated impact values by year, group, and metric.
    """
    # 1. Interpolate in between years

    prop_H0, prop_H1 = risk_period._prop_H0, risk_period._prop_H1
    frequency_0 = risk_period.snapshot0.hazard.frequency
    frequency_1 = risk_period.snapshot1.hazard.frequency
    imp_mats_0, imp_mats_1 = risk_period.get_interp()
    yearly_eai_exp_0, yearly_eai_exp_1 = calc_yearly_eais(
        imp_mats_0, imp_mats_1, frequency_0, frequency_1
    )
    year_idx = risk_period.year_idx
    res = []
    if "aai" in metrics:
        yearly_aai_0, yearly_aai_1 = calc_yearly_aais(
            yearly_eai_exp_0, yearly_eai_exp_1
        )
        yearly_aai = prop_H0 * yearly_aai_0 + prop_H1 * yearly_aai_1
        aai_df = pd.DataFrame(index=year_idx, columns=["risk"], data=yearly_aai)
        aai_df["group"] = all_groups_name
        aai_df["metric"] = "aai"
        aai_df.reset_index(inplace=True)
        res.append(aai_df)

    if "rp" in metrics:
        rp_0, rp_1 = calc_yearly_rps(
            imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods
        )
        yearly_rp = np.multiply(prop_H0.reshape(-1, 1), rp_0) + np.multiply(
            prop_H1.reshape(-1, 1), rp_1
        )
        rp_df = pd.DataFrame(
            index=year_idx, columns=return_periods, data=yearly_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = all_groups_name
        rp_df["metric"] = f"rp_" + rp_df["rp"].astype(str)
        res.append(rp_df)

    if groups is not None:
        yearly_eai = np.multiply(
            prop_H0.reshape(-1, 1), yearly_eai_exp_0
        ) + np.multiply(prop_H1.reshape(-1, 1), yearly_eai_exp_1)
        yearly_eai_group = get_eai_exp(yearly_eai, groups)
        eai_group_df = pd.DataFrame(index=year_idx, data=yearly_eai_group).melt(
            value_name="risk", var_name="group", ignore_index=False
        )
        eai_group_df["metric"] = "aai"
        eai_group_df.reset_index(inplace=True)
        res.append(eai_group_df)

    ret = pd.concat(res, axis=0)
    ret["measure"] = risk_period.measure_name
    return ret


def bayesian_mixer(
    start_snapshot,
    end_snapshot,
    metrics,
    return_periods,
    groups=None,
    all_groups_name=pd.NA,
    risk_transf_cover=None,
    risk_transf_attach=None,
    calc_residual=True,
):
    """
    Perform Bayesian mixing of impacts across snapshots.

    Parameters
    ----------
    start_snapshot : Snapshot
        The starting snapshot.
    end_snapshot : Snapshot
        The ending snapshot.
    metrics : list of str
        Metrics to calculate (e.g., 'eai', 'aai', 'rp').
    return_periods : list of int
        Return periods for calculating impact values.
    groups : dict, optional
        Mapping of group names to indices for aggregating EAI values by group.
    all_groups_name : str, optional
        Name for all-groups aggregation in the output.
    risk_transf_cover : float, optional
        Coverage level for risk transfer calculations.
    risk_transf_attach : float, optional
        Attachment point for risk transfer calculations.
    calc_residual : bool, optional
        Whether to calculate residual impacts after applying risk transfer.

    Returns
    -------
    pd.DataFrame
        DataFrame of calculated impact values by year, group, and metric.
    """
    # 1. Interpolate in between years
    prop_H0, prop_H1 = bayesian_viktypliers(start_snapshot.year, end_snapshot.year)
    imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1 = snapshot_combinaisons(
        start_snapshot, end_snapshot
    )
    frequency_0 = start_snapshot.hazard.frequency
    frequency_1 = end_snapshot.hazard.frequency

    # Modeify the impact matrices if risk transfer is provided
    imp_E0H0.imp_mat = calc_residual_or_risk_transf_imp_mat(
        imp_E0H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
    )
    imp_E1H0.imp_mat = calc_residual_or_risk_transf_imp_mat(
        imp_E1H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
    )
    imp_E0H1.imp_mat = calc_residual_or_risk_transf_imp_mat(
        imp_E0H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
    )
    imp_E1H1.imp_mat = calc_residual_or_risk_transf_imp_mat(
        imp_E1H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
    )

    imp_mats_0 = interpolate_imp_mat(
        imp_E0H0, imp_E1H0, start_snapshot.year, end_snapshot.year
    )
    imp_mats_1 = interpolate_imp_mat(
        imp_E0H1, imp_E1H1, start_snapshot.year, end_snapshot.year
    )

    yearly_eai_exp_0, yearly_eai_exp_1 = calc_yearly_eais(
        imp_mats_0, imp_mats_1, frequency_0, frequency_1
    )

    res = []

    year_idx = pd.Index(
        list(range(start_snapshot.year, end_snapshot.year + 1)), name="year"
    )

    if "aai" in metrics:
        yearly_aai_0, yearly_aai_1 = calc_yearly_aais(
            yearly_eai_exp_0, yearly_eai_exp_1
        )
        yearly_aai = prop_H0 * yearly_aai_0 + prop_H1 * yearly_aai_1
        aai_df = pd.DataFrame(index=year_idx, columns=["risk"], data=yearly_aai)
        aai_df["group"] = all_groups_name
        aai_df["metric"] = "aai"
        aai_df.reset_index(inplace=True)
        res.append(aai_df)

    if "rp" in metrics:
        rp_0, rp_1 = calc_yearly_rps(
            imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods
        )
        yearly_rp = np.multiply(prop_H0.reshape(-1, 1), rp_0) + np.multiply(
            prop_H1.reshape(-1, 1), rp_1
        )
        rp_df = pd.DataFrame(
            index=year_idx, columns=return_periods, data=yearly_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = all_groups_name
        rp_df["metric"] = f"rp_" + rp_df["rp"].astype(str)
        res.append(rp_df)

    if groups is not None:
        yearly_eai = np.multiply(
            prop_H0.reshape(-1, 1), yearly_eai_exp_0
        ) + np.multiply(prop_H1.reshape(-1, 1), yearly_eai_exp_1)
        yearly_eai_group = get_eai_exp(yearly_eai, groups)
        eai_group_df = pd.DataFrame(index=year_idx, data=yearly_eai_group).melt(
            value_name="risk", var_name="group", ignore_index=False
        )
        eai_group_df["metric"] = "aai"
        eai_group_df.reset_index(inplace=True)
        res.append(eai_group_df)

    return pd.concat(res, axis=0)


# TODO: Better measure checking (if we change the measure object after, changes are not accounted for!!!)
@dataclass(eq=False, frozen=True)
class Snapshot:
    """
    A snapshot of exposure, hazard, and impact function set for a given year.

    Attributes
    ----------
    exposure : Exposures
        Exposure data for the snapshot.
    hazard : Hazard
        Hazard data for the snapshot.
    impfset : ImpactFuncSet
        Impact function set associated with the snapshot.
    year : int
        Year of the snapshot.
    """

    exposure: Exposures
    hazard: Hazard
    impfset: ImpactFuncSet
    year: int
    measure: None | Measure = None

    # Class-level cache
    _instances = WeakValueDictionary()

    def __new__(cls, exposure, hazard, impfset, year, measure=None):
        """Check if an equal instance exists before creating a new one."""
        for existing_snapshot in cls._instances.values():
            if (
                existing_snapshot.exposure.gdf.equals(exposure.gdf)
                and hazard_data_equal(existing_snapshot.hazard, hazard)
                and impact_func_set_equal(existing_snapshot.impfset, impfset)
                and existing_snapshot.year == year
            ):
                if (
                    existing_snapshot.measure
                    and measure
                    and existing_snapshot.measure.name == measure.name
                ):
                    LOGGER.debug(
                        f"Found existing instance of snapshot for year {year}, measure {measure.name}, with id {id(existing_snapshot)}"
                    )
                    return existing_snapshot  # Return existing instance
                elif existing_snapshot.measure is None and measure is None:
                    LOGGER.debug(
                        f"Found existing instance of snapshot for year {year} (no measure), with id {id(existing_snapshot)}"
                    )
                    return existing_snapshot  # Return existing instance

        # Create new instance if no match is found
        instance = super().__new__(cls)
        return instance

    def __post_init__(self):
        """Store the instance in the cache after initialization."""
        if id(self) not in self._instances:
            LOGGER.debug(f"Created and stored new Snapshot {id(self)}")
            self._instances[id(self)] = self

    def __eq__(self, value, /) -> bool:
        if not isinstance(value, Snapshot):
            return False
        if self is value:
            return True
        same_exposure = self.exposure.gdf.equals(value.exposure.gdf)
        same_hazard = self.hazard == value.hazard
        same_impfset = impact_func_set_equal(self.impfset, value.impfset)
        same_year = self.year == value.year
        same_measure = self.measure == value.measure
        return (
            same_exposure
            and same_hazard
            and same_impfset
            and same_year
            and same_measure
        )

    def apply_measure(self, measure: Measure):
        LOGGER.debug(f"Applying measure {measure.name} on snapshot {id(self)}")
        exp_new, impfset_new, haz_new = measure.apply(
            self.exposure, copy.deepcopy(self.impfset), self.hazard
        )
        return Snapshot(exp_new, haz_new, impfset_new, self.year, measure)


class SnapshotsCollection:
    """
    Collection of snapshots for different years.

    Attributes
    ----------
    exposure_set : dict
        Dictionary of exposure data by year.
    hazard_set : dict
        Dictionary of hazard data by year.
    impfset : ImpactFuncSet
        Impact function set shared across snapshots.
    snapshots_years : list of int
        Years associated with each snapshot in the collection.
    data : list of Snapshot
        List of Snapshot objects in the collection.
    """

    def __init__(self, snaplist):

        self._snapshots = {snap.year: snap for snap in snaplist}
        self._impfset = snaplist[0].impfset

    @classmethod
    def _from_dicts(
        cls,
        exposure_set: dict[int, Exposures],
        hazard_set: dict[int, Hazard],
        impfset: ImpactFuncSet,
        snapshot_years: list[int],
    ):

        # Validate all requested years exist
        missing_exposure = [y for y in snapshot_years if y not in exposure_set]
        missing_hazard = [y for y in snapshot_years if y not in hazard_set]
        if missing_exposure or missing_hazard:
            raise ValueError(
                f"Missing data for years - Exposure: {missing_exposure}, Hazard: {missing_hazard}"
            )

        return cls(
            [
                Snapshot(exposure_set[year], hazard_set[year], impfset, year)
                for year in sorted(snapshot_years)
            ]
        )

    @property
    def data(self):
        return list(self._snapshots.values())

    @property
    def snapshots_years(self):
        return self._snapshots.keys()

    @property
    def exposure_set(self):
        return [snap.exposure for snap in self._snapshots.values()]

    @property
    def hazard_set(self):
        return [snap.hazard for snap in self._snapshots.values()]

    @property
    def impfset(self):
        return self._impfset

    def __len__(self):
        """Return the number of snapshots in the collection."""
        return len(self._snapshots)

    # def __iter__(self):
    #     """Return an iterator over the snapshots in the collection."""
    #     return iter(self.data)

    def __contains__(self, item):
        """Check if a Snapshot or a year exists in the collection."""
        if isinstance(item, int):
            return item in self._snapshots
        if isinstance(item, Snapshot):
            return item in self._snapshots.values()  # Check object identity
        else:
            return False  # Invalid type

    # Check that at least first and last snap are complete
    # and otherwise it is ok

    @classmethod
    def from_dict(cls, snapshots_dict, impfset):
        """
        Create a SnapshotsCollection from a dictionary of snapshots.

        Parameters
        ----------
        snapshots_dict : dict
            Dictionary of snapshots data by year.
        impfset : ImpactFuncSet
            Impact function set shared across snapshots.

        Returns
        -------
        SnapshotsCollection
            A new SnapshotsCollection instance.
        """
        snapshot_years = list(snapshots_dict.keys())
        exposure_set = {year: snapshots_dict[year][0] for year in snapshot_years}
        hazard_set = {year: snapshots_dict[year][1] for year in snapshot_years}
        return cls._from_dicts(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )

    @classmethod
    def from_lists(cls, hazard_list, exposure_list, impfset, snapshot_years):
        """
        Create a SnapshotsCollection from separate lists of hazard and exposure data.

        Parameters
        ----------
        hazard_list : list
            List of hazard data for each year, in the same order as `snapshot_years`.
        exposure_list : list
            List of exposure data for each year, in the same order as `snapshot_years`.
        impfset : ImpactFuncSet
            Impact function set shared across snapshots.
        snapshot_years : list of int
            List of years corresponding to each hazard and exposure data entry.

        Returns
        -------
        SnapshotsCollection
            A new SnapshotsCollection instance.
        """
        exposure_set = {year: exposure_list[i] for i, year in enumerate(snapshot_years)}
        hazard_set = {year: hazard_list[i] for i, year in enumerate(snapshot_years)}
        return cls._from_dicts(
            exposure_set=exposure_set,
            hazard_set=hazard_set,
            impfset=impfset,
            snapshot_years=snapshot_years,
        )

    def add_snapshot(self, snapshot: Snapshot):
        """Adds a snapshot to the collecton"""
        if not isinstance(snapshot, Snapshot):
            raise TypeError("snapshot must be an instance of Snapshot")

        if snapshot in self:  # Identity check
            LOGGER.warning("Snapshot already present.", UserWarning)
            return  # Do nothing if it's the exact same object

        if snapshot.year in self:
            LOGGER.warning(
                "Snapshot already exist for this year. Overwriting.", UserWarning
            )

        # Ensure the impact function set is consistent
        if snapshot.impfset is not self.impfset:
            raise ValueError(
                "Snapshot impact function set does not match existing one."
            )

        self._snapshots[snapshot.year] = snapshot
        self._snapshots = dict(sorted(self._snapshots.items()))

    def pairwise(self):
        """
        Generate pairs of successive elements from an iterable.

        Parameters
        ----------
        iterable : iterable
            An iterable sequence from which successive pairs of elements are generated.

        Returns
        -------
        zip
            A zip object containing tuples of successive pairs from the input iterable.

        Example
        -------
        >>> list(pairwise([1, 2, 3, 4]))
        [(1, 2), (2, 3), (3, 4)]
        """
        a, b = itertools.tee(self._snapshots.values())
        next(b, None)
        return zip(a, b)


class RiskPeriod:

    # TODO: make lazy / delayed interpolation and impacts
    # TODO: make MeasureRiskPeriod child class (with effective start/end)
    # TODO: special case where hazard and exposure don't change (no need to interpolate) ?

    _instances = WeakValueDictionary()

    def __new__(
        cls,
        snapshot0,
        snapshot1,
        measure_name="no_measure",
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ):
        """Ensure only one instance exists per snapshot pair."""
        key = (id(snapshot0), id(snapshot1), measure_name)
        if key in cls._instances:
            LOGGER.debug("Found existing RiskPeriod")
            return cls._instances[key]

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        measure_name="no_measure",
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ):
        if hasattr(self, "_initialized"):
            return  # Avoid re-initialization

        LOGGER.debug(
            f"Initializing new RiskPeriod from {snapshot0.year} to {snapshot1.year}, with snapshot0: {id(snapshot0)}, snapshot1: {id(snapshot1)}"
        )
        self.snapshot0 = snapshot0
        self.snapshot1 = snapshot1
        self.start_year = snapshot0.year
        self.end_year = snapshot1.year
        self.measure_name = measure_name
        self.impfset = snapshot0.impfset
        assert impact_func_set_equal(
            self.impfset, snapshot1.impfset
        )  # Ensure same impfset

        self._prop_H0, self._prop_H1 = bayesian_viktypliers(
            snapshot0.year, snapshot1.year
        )

        self._exp_y0 = snapshot0.exposure
        self._exp_y1 = snapshot1.exposure
        self._haz_y0 = snapshot0.hazard
        self._haz_y1 = snapshot1.hazard

        # Compute impacts once
        LOGGER.debug("Computing snapshots combination impacts")
        imp_E0H0 = self._compute_impact(self._exp_y0, self._haz_y0)
        imp_E1H0 = self._compute_impact(self._exp_y1, self._haz_y0)
        imp_E0H1 = self._compute_impact(self._exp_y0, self._haz_y1)
        imp_E1H1 = self._compute_impact(self._exp_y1, self._haz_y1)

        # Modify the impact matrices if risk transfer is provided
        # TODO: See where this ends up
        imp_E0H0.imp_mat = calc_residual_or_risk_transf_imp_mat(
            imp_E0H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E1H0.imp_mat = calc_residual_or_risk_transf_imp_mat(
            imp_E1H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E0H1.imp_mat = calc_residual_or_risk_transf_imp_mat(
            imp_E0H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E1H1.imp_mat = calc_residual_or_risk_transf_imp_mat(
            imp_E1H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )

        LOGGER.debug("Interpolating impact matrices between E0H0 and E1H0")
        self.imp_mats_0 = interpolate_imp_mat(
            imp_E0H0, imp_E1H0, snapshot0.year, snapshot1.year
        )
        LOGGER.debug("Interpolating impact matrices between E0H1 and E1H1")
        self.imp_mats_1 = interpolate_imp_mat(
            imp_E0H1, imp_E1H1, snapshot0.year, snapshot1.year
        )
        LOGGER.debug("Done")

        self.year_idx = pd.Index(
            list(range(snapshot0.year, snapshot1.year + 1)), name="year"
        )

        self._initialized = True

        # Store the instance in the cache after initialization
        key = (id(self.snapshot0), id(self.snapshot1), self.measure_name)
        if key not in self._instances:
            LOGGER.debug(f"Created and stored new RiskPeriod {key}")
            self._instances[key] = self

    def _compute_impact(self, exposure, hazard):
        """Compute the impact once per unique exposure-hazard pair."""
        return ImpactCalc(exposure, self.impfset, hazard).impact()

    def get_interp(self):
        return self.imp_mats_0, self.imp_mats_1

    def apply_measures(self, measure_set: MeasureSet | None, measure_name_list):
        # Apply measure on snapshot and return risk period instance
        if measure_set is None or measure_name_list == []:
            return self

        combined_measures = measure_set.combine(names=measure_name_list)
        snapshot0 = self.snapshot0.apply_measure(combined_measures)
        snapshot1 = self.snapshot1.apply_measure(combined_measures)
        return RiskPeriod(snapshot0, snapshot1, measure_name=combined_measures.name)


class CalcImpactsSnapshots:
    """
    Calculate impacts for each year in a collection of snapshots.

    Attributes
    ----------
    snapshots : SnapshotsCollection
        Collection of snapshots to calculate impacts for.
    group_map_exp_dict : dict, optional
        Dictionary mapping group names to indices for impact aggregation by group.
    yearly_eai_exp_tuples : list of tuple
        List of tuples containing yearly expected annual impact data.
    """

    def __init__(self, snapshots: SnapshotsCollection, group_map_exp_dict=None):
        self.snapshots = snapshots
        self.group_map_exp_dict = group_map_exp_dict
        self.yearly_eai_exp_tuples = []

    # An init param could be the region aggregation you want

    # def calc_impacts_snapshots(self):
    #     """
    #     Calculate impacts for each snapshot year.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of impacts for each year, keyed by year.
    #     """
    #     impacts_list = {}
    #     for snapshot in self.snapshots.data:
    #         impacts_list[snapshot.year] = ImpactCalc(
    #             snapshot.exposure, self.snapshots.impfset, snapshot.hazard
    #         ).impact()
    #     return impacts_list

    def calc_all_years(
        self,
        metrics=["eai", "aai", "rp"],
        return_periods=[100, 500, 1000],
        compute_groups=False,
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ) -> pd.DataFrame:
        """Calculate impacts for all years in the snapshots collection.

        This method computes specified metrics (e.g., expected annual impact, aggregated annual impact, and return periods) for each year in the snapshot collection. The results are aggregated and returned as a DataFrame.

        Parameters
        ----------
        metrics : list of str, optional
            List of metrics to compute. Options include "eai" (expected annual impact), "aai" (aggregated annual impact), and "rp" (return periods). Default is ["eai", "aai", "rp"].
        return_periods : list of int, optional
            List of return periods (in years) to compute for the "rp" metric. Default is [100, 500, 1000].
        compute_groups : bool, optional
            Whether to compute the metrics for specific groups (e.g., regions). Default is False.
        risk_transf_cover : optional
            Coverage values for risk transfer, used in the calculation of impacts.
        risk_transf_attach : optional
            Attachment points for risk transfer, used in the calculation of impacts.
        calc_residual : bool, optional
            Whether to calculate the residual impacts after risk transfer. Default is True.


        Returns
        -------
        pd.DataFrame
            DataFrame containing the computed metrics, with columns for "group", "year", "metric", and "result".

        """
        results_df = []
        if compute_groups:
            groups = self.group_map_exp_dict
        else:
            groups = None
        for start_snapshot, end_snapshot in self.snapshots.pairwise():
            risk_period = RiskPeriod(
                start_snapshot,
                end_snapshot,
                risk_transf_cover=risk_transf_cover,
                risk_transf_attach=risk_transf_attach,
                calc_residual=calc_residual,
            )
            results_df.append(
                bayesian_mixer_opti(
                    risk_period,
                    metrics,
                    return_periods,
                    groups,
                )
            )
        results_df = pd.concat(results_df, axis=0)

        # duplicate rows arise from overlapping end and start if there's more than two snapshots
        results_df.drop_duplicates(inplace=True)
        return results_df[["group", "year", "metric", "risk"]]


#### WIP

# Implement collections of trajectories.


class TBRTrajectories:

    # Compute impacts for trajectories with present exposure and future exposure and interpolate in between
    #

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
