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

Impact function calibration functionalities:
    Optimization and manual calibration
"""

import datetime as dt
import copy
import itertools
import logging
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize

from climada.engine import ImpactCalc
from climada.entity import ImpactFuncSet, ImpfTropCyclone, impact_funcs
from climada.engine.impact_data import emdat_impact_yearlysum, emdat_impact_event

LOGGER = logging.getLogger(__name__)



def calib_instance(hazard, exposure, impact_func, df_out=pd.DataFrame(),
                   yearly_impact=False, return_cost='False'):

    """calculate one impact instance for the calibration algorithm and write
        to given DataFrame

        Parameters
        ----------
        hazard : Hazard
        exposure : Exposure
        impact_func : ImpactFunc
        df_out : Dataframe, optional
            Output DataFrame with headers of columns defined and optionally with
            first row (index=0) defined with values. If columns "impact",
            "event_id", or "year" are not included, they are created here.
            Data like reported impacts or impact function parameters can be
            given here; values are preserved.
        yearly_impact : boolean, optional
            if set True, impact is returned per year, not per event
        return_cost : str, optional
            if not 'False' but any of 'R2', 'logR2',
            cost is returned instead of df_out

        Returns
        -------
        df_out: DataFrame
            DataFrame with modelled impact written to rows for each year
            or event.
    """
    ifs = ImpactFuncSet([impact_func])
    impacts = ImpactCalc(exposures=exposure, impfset=ifs, hazard=hazard)\
              .impact(assign_centroids=False)
    if yearly_impact:  # impact per year
        iys = impacts.impact_per_year(all_years=True)
        # Loop over whole year range:
        if df_out.empty | df_out.index.shape[0] == 1:
            for cnt_, year in enumerate(np.sort(list((iys.keys())))):
                if cnt_ > 0:
                    df_out.loc[cnt_] = df_out.loc[0]  # copy info from first row
                if year in iys:
                    df_out.loc[cnt_, 'impact_CLIMADA'] = iys[year]
                else:
                    df_out.loc[cnt_, 'impact_CLIMADA'] = 0.0
                df_out.loc[cnt_, 'year'] = year
        else:
            years_in_common = df_out.loc[df_out['year'].isin(np.sort(list((iys.keys())))), 'year']
            for cnt_, year in years_in_common.iteritems():
                df_out.loc[df_out['year'] == year, 'impact_CLIMADA'] = iys[year]


    else:  # impact per event
        if df_out.empty | df_out.index.shape[0] == 1:
            for cnt_, impact in enumerate(impacts.at_event):
                if cnt_ > 0:
                    df_out.loc[cnt_] = df_out.loc[0]  # copy info from first row
                df_out.loc[cnt_, 'impact_CLIMADA'] = impact
                df_out.loc[cnt_, 'event_id'] = int(impacts.event_id[cnt_])
                df_out.loc[cnt_, 'event_name'] = impacts.event_name[cnt_]
                df_out.loc[cnt_, 'year'] = \
                    dt.datetime.fromordinal(impacts.date[cnt_]).year
                df_out.loc[cnt_, 'date'] = impacts.date[cnt_]
        elif df_out.index.shape[0] == impacts.at_event.shape[0]:
            for cnt_, (impact, ind) in enumerate(zip(impacts.at_event, df_out.index)):
                df_out.loc[ind, 'impact_CLIMADA'] = impact
                df_out.loc[ind, 'event_id'] = int(impacts.event_id[cnt_])
                df_out.loc[ind, 'event_name'] = impacts.event_name[cnt_]
                df_out.loc[ind, 'year'] = \
                    dt.datetime.fromordinal(impacts.date[cnt_]).year
                df_out.loc[ind, 'date'] = impacts.date[cnt_]
        else:
            raise ValueError('adding simulated impacts to reported impacts not'
                             ' yet implemented. use yearly_impact=True or run'
                             ' without init_impact_data.')
    if return_cost != 'False':
        df_out = calib_cost_calc(df_out, return_cost)
    return df_out

def init_impf(impf_name_or_instance, param_dict, df_out=pd.DataFrame(index=[0])):
    """create an ImpactFunc based on the parameters in param_dict using the
    method specified in impf_parameterisation_name and document it in df_out.

    Parameters
    ----------
    impf_name_or_instance : str or ImpactFunc
        method of impact function parameterisation e.g. 'emanuel' or an
        instance of ImpactFunc
    param_dict : dict, optional
        dict of parameter_names and values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
        or {'mdd_shift': 1.05, 'mdd_scale': 0.8, 'paa_shift': 1, paa_scale': 1}

    Returns
    -------
    imp_fun : ImpactFunc
        The Impact function based on the parameterisation
    df_out : DataFrame
        Output DataFrame with headers of columns defined and with first row
        (index=0) defined with values. The impact function parameters from
        param_dict are represented here.
    """
    impact_func_final = None
    if isinstance(impf_name_or_instance, str):
        if impf_name_or_instance == 'emanuel':
            impact_func_final = ImpfTropCyclone.from_emanuel_usa(**param_dict)
            impact_func_final.haz_type = 'TC'
            impact_func_final.id = 1
            df_out['impact_function'] = impf_name_or_instance
    elif isinstance(impf_name_or_instance, impact_funcs.ImpactFunc):
        impact_func_final = change_impf(impf_name_or_instance, param_dict)
        df_out['impact_function'] = ('given_' +
                                     impact_func_final.haz_type +
                                     str(impact_func_final.id))
    for key, val in param_dict.items():
        df_out[key] = val
    return impact_func_final, df_out

def change_impf(impf_instance, param_dict):
    """apply a shifting or a scaling defined in param_dict to the impact
    function in impf_istance and return it as a new ImpactFunc object.

    Parameters
    ----------
    impf_instance : ImpactFunc
        an instance of ImpactFunc
    param_dict : dict
        dict of parameter_names and values (interpreted as
        factors, 1 = neutral)
        e.g. {'mdd_shift': 1.05, 'mdd_scale': 0.8,
        'paa_shift': 1, paa_scale': 1}

    Returns
    -------
    ImpactFunc : The Impact function based on the parameterisation
    """
    ImpactFunc_new = copy.deepcopy(impf_instance)
    # create higher resolution impact functions (intensity, mdd ,paa)
    paa_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.paa,
                                    fill_value='extrapolate')
    mdd_func = interpolate.interp1d(ImpactFunc_new.intensity,
                                    ImpactFunc_new.mdd,
                                    fill_value='extrapolate')
    temp_dict = dict()
    temp_dict['paa_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['mdd_intensity_ext'] = np.linspace(ImpactFunc_new.intensity.min(),
                                                 ImpactFunc_new.intensity.max(),
                                                 (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    temp_dict['paa_ext'] = paa_func(temp_dict['paa_intensity_ext'])
    temp_dict['mdd_ext'] = mdd_func(temp_dict['mdd_intensity_ext'])
    # apply changes given in param_dict
    for key, val in param_dict.items():
        field_key, action = key.split('_')
        if action == 'shift':
            shift_absolut = (
                ImpactFunc_new.intensity[np.nonzero(getattr(ImpactFunc_new, field_key))[0][0]]
                * (val - 1))
            temp_dict[field_key + '_intensity_ext'] = \
                temp_dict[field_key + '_intensity_ext'] + shift_absolut
        elif action == 'scale':
            temp_dict[field_key + '_ext'] = \
                    np.clip(temp_dict[field_key + '_ext'] * val,
                            a_min=0,
                            a_max=1)
        else:
            raise AttributeError('keys in param_dict not recognized. Use only:'
                                 'paa_shift, paa_scale, mdd_shift, mdd_scale')

    # map changed, high resolution impact functions back to initial resolution
    ImpactFunc_new.intensity = np.linspace(ImpactFunc_new.intensity.min(),
                                           ImpactFunc_new.intensity.max(),
                                           (ImpactFunc_new.intensity.shape[0] + 1) * 10 + 1)
    paa_func_new = interpolate.interp1d(temp_dict['paa_intensity_ext'],
                                        temp_dict['paa_ext'],
                                        fill_value='extrapolate')
    mdd_func_new = interpolate.interp1d(temp_dict['mdd_intensity_ext'],
                                        temp_dict['mdd_ext'],
                                        fill_value='extrapolate')
    ImpactFunc_new.paa = paa_func_new(ImpactFunc_new.intensity)
    ImpactFunc_new.mdd = mdd_func_new(ImpactFunc_new.intensity)
    return ImpactFunc_new

def init_impact_data(hazard_type,
                     region_ids,
                     year_range,
                     source_file,
                     reference_year,
                     impact_data_source='emdat',
                     yearly_impact=True):
    """creates a dataframe containing the recorded impact data for one hazard
    type and one area (countries, country or local split)

    Parameters
    ----------
    hazard_type : str
        default = 'TC', type of hazard 'WS','FL' etc.
    region_ids : str
        name the region_ids or country names
    year_range : list
        list containting start and end year.
        e.g. [1980, 2017]
    source_file : str
    reference_year : int
        impacts will be scaled to this year
    impact_data_source : str, optional
        default 'emdat', others maybe possible
    yearly_impact : bool, optional
        if set True, impact is returned per year, not per event

    Returns
    -------
    df_out : pd.DataFrame
        Dataframe with recorded impact written to rows for each year
        or event.
    """
    if impact_data_source == 'emdat':
        if yearly_impact:
            em_data = emdat_impact_yearlysum(source_file, countries=region_ids,
                                             hazard=hazard_type,
                                             year_range=year_range,
                                             reference_year=reference_year)
        else:
            raise ValueError('init_impact_data not yet implemented for yearly_impact = False.')
            em_data = emdat_impact_event(source_file)
    else:
        raise ValueError('init_impact_data not yet implemented for other impact_data_sources '
                         'than emdat.')
    return em_data


def calib_cost_calc(df_out, cost_function):
    """calculate the cost function of the modelled impact impact_CLIMADA and
        the reported impact impact_scaled in df_out

    Parameters
    ----------
    df_out : pd.Dataframe
        DataFrame as created in calib_instance
    cost_function : str
        chooses the cost function e.g. 'R2' or 'logR2'

    Returns
    -------
    cost : float
        The results of the cost function when comparing modelled and
        reported impact
    """
    if cost_function == 'R2':
        cost = np.sum((pd.to_numeric(df_out['impact_scaled']) -
                       pd.to_numeric(df_out['impact_CLIMADA']))**2)
    elif cost_function == 'logR2':
        impact1 = pd.to_numeric(df_out['impact_scaled'])
        impact1[impact1 <= 0] = 1
        impact2 = pd.to_numeric(df_out['impact_CLIMADA'])
        impact2[impact2 <= 0] = 1
        cost = np.sum((np.log(impact1) -
                       np.log(impact2))**2)
    else:
        raise ValueError('This cost function is not implemented.')
    return cost


def calib_all(hazard, exposure, impf_name_or_instance, param_full_dict,
              impact_data_source, year_range, yearly_impact=True):
    """portrait the difference between modelled and reported impacts for all
    impact functions described in param_full_dict and impf_name_or_instance

    Parameters
    ----------
    hazard : list or Hazard
    exposure : list or Exposures
        list or instance of exposure of full countries
    impf_name_or_instance: string or ImpactFunc
        the name of a parameterisation or an instance of class
        ImpactFunc e.g. 'emanuel'
    param_full_dict : dict
        a dict containing keys used for
        f_name_or_instance and values which are iterable (lists)
        e.g. {'v_thresh' : [25.7, 20], 'v_half': [70], 'scale': [1, 0.8]}
    impact_data_source : dict or pd.Dataframe
        with name of impact data source and file location or dataframe
    year_range : list
    yearly_impact : bool, optional

    Returns
    -------
    df_result : pd.DataFrame
        df with modelled impact written to rows for each year or event.
    """
    df_result = None  # init return variable

    # prepare hazard and exposure
    region_ids = list(np.unique(exposure.region_id))
    hazard_type = hazard.tag.haz_type
    exposure.assign_centroids(hazard)
    # prepare impact data
    if isinstance(impact_data_source, pd.DataFrame):
        df_impact_data = impact_data_source
    else:
        if list(impact_data_source.keys()) == ['emdat']:
            df_impact_data = init_impact_data(hazard_type, region_ids, year_range,
                                              impact_data_source['emdat'], year_range[-1])
        else:
            raise ValueError('other impact data sources not yet implemented.')
    params_generator = (dict(zip(param_full_dict, x))
                        for x in itertools.product(*param_full_dict.values()))
    for param_dict in params_generator:
        print(param_dict)
        df_out = copy.deepcopy(df_impact_data)
        impact_func_final, df_out = init_impf(impf_name_or_instance, param_dict, df_out)
        df_out = calib_instance(hazard, exposure, impact_func_final, df_out, yearly_impact)
        if df_result is None:
            df_result = copy.deepcopy(df_out)
        else:
            df_result = df_result.append(df_out, input)


    return df_result


def calib_optimize(hazard, exposure, impf_name_or_instance, param_dict,
                   impact_data_source, year_range, yearly_impact=True,
                   cost_fucntion='R2', show_details=False):
    """portrait the difference between modelled and reported impacts for all
    impact functions described in param_full_dict and impf_name_or_instance

    Parameters
    ----------
    hazard: list or Hazard
    exposure: list or Exposures
        list or instance of exposure of full countries
    impf_name_or_instance: string or ImpactFunc
        the name of a parameterisation or an instance of class
        ImpactFunc e.g. 'emanuel'
    param_dict : dict
        a dict containing keys used for
        impf_name_or_instance and one set of values
        e.g. {'v_thresh': 25.7, 'v_half': 70, 'scale': 1}
    impact_data_source : dict or pd. dataframe
        with name of impact data source and file location or dataframe
    year_range : list
    yearly_impact : bool, optional
    cost_function : str, optional
        the argument for function calib_cost_calc, default 'R2'
    show_details : bool, optional
        if True, return a tuple with the parameters AND
        the details of the optimization like success,
        status, number of iterations etc

    Returns
    -------
    param_dict_result : dict or tuple
        the parameters with the best calibration results
        (or a tuple with (1) the parameters and (2) the optimization output)
    """
    param_dict_result = param_dict

    # prepare hazard and exposure
    region_ids = list(np.unique(exposure.region_id))
    hazard_type = hazard.tag.haz_type
    exposure.assign_centroids(hazard)
    # prepare impact data
    if isinstance(impact_data_source, pd.DataFrame):
        df_impact_data = impact_data_source
    else:
        if list(impact_data_source.keys()) == ['emdat']:
            df_impact_data = init_impact_data(hazard_type, region_ids, year_range,
                                              impact_data_source['emdat'], year_range[-1])
        else:
            raise ValueError('other impact data sources not yet implemented.')
    # definie specific function to
    def specific_calib(values):
        param_dict_temp = dict(zip(param_dict.keys(), values))
        print(param_dict_temp)
        return calib_instance(hazard, exposure,
                              init_impf(impf_name_or_instance, param_dict_temp)[0],
                              df_impact_data,
                              yearly_impact=yearly_impact, return_cost=cost_fucntion)
    # define constraints
    if impf_name_or_instance == 'emanuel':
        cons = [{'type': 'ineq', 'fun': lambda x: -x[0] + x[1]},
                {'type': 'ineq', 'fun': lambda x: -x[2] + 0.9999},
                {'type': 'ineq', 'fun': lambda x: x[2]}]
    else:
        cons = [{'type': 'ineq', 'fun': lambda x: -x[0] + 2},
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: -x[1] + 2},
                {'type': 'ineq', 'fun': lambda x: x[1]}]


    values = list(param_dict.values())
    res = minimize(specific_calib, values,
                   # bounds=bounds,
                   # bounds=((0.0, np.inf), (0.0, np.inf), (0.0, 1.0)),
                   constraints=cons,
                   # method='SLSQP',
                   method='trust-constr',
                   options={'xtol': 1e-5, 'disp': True, 'maxiter': 500})

    param_dict_result = dict(zip(param_dict.keys(), res.x))

    if res.success:
        LOGGER.info('Optimization successfully finished.')
    else:
        LOGGER.info('Opimization did not finish successfully. Check you input'
                    ' or consult the detailed returns (with argument'
                    'show_details=True) for further information.')

    if show_details:
        return param_dict_result, res

    return param_dict_result


# if __name__ == "__main__":
#
#
#    ## tryout calib_all
#    hazard = TropCyclone.from_hdf5('C:/Users/ThomasRoosli/tc_NA_hazard.hdf5')
#    exposure = LitPop.from_hdf5('C:/Users/ThomasRoosli/DOM_LitPop.hdf5')
#    impf_name_or_instance = 'emanuel'
#    param_full_dict = {'v_thresh': [25.7, 20], 'v_half': [70], 'scale': [1, 0.8]}
#
#    impact_data_source = {'emdat':('D:/Documents_DATA/EM-DAT/'
#                                   '20181031_disaster_list_all_non-technological/'
#                                   'ThomasRoosli_2018-10-31.csv')}
#    year_range = [2004, 2017]
#    yearly_impact = True
#    df_result = calib_all(hazard,exposure,impf_name_or_instance,param_full_dict,
#                  impact_data_source, year_range, yearly_impact)
#
#
#    ## tryout calib_optimize
#    hazard = TropCyclone.from_hdf5('C:/Users/ThomasRoosli/tc_NA_hazard.hdf5')
#    exposure = LitPop.from_hdf5('C:/Users/ThomasRoosli/DOM_LitPop.hdf5')
#    impf_name_or_instance = 'emanuel'
#    param_dict = {'v_thresh': 25.7, 'v_half': 70, 'scale': 0.6}
#    year_range = [2004, 2017]
#    cost_function = 'R2'
#    show_details = True
#    yearly_impact = True
#    impact_data_source = {'emdat':('D:/Documents_DATA/EM-DAT/'
#                                   '20181031_disaster_list_all_non-technological/'
#                                   'ThomasRoosli_2018-10-31.csv')}
#    param_result,result = calib_optimize(hazard,exposure,impf_name_or_instance,param_dict,
#              impact_data_source, year_range, yearly_impact=yearly_impact,
#              cost_fucntion=cost_function,show_details= show_details)
#
#
