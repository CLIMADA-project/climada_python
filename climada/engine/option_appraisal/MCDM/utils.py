#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:58:15 2024

@author: vwattin
"""

from climada.entity import Entity
from climada.util.api_client import Client
from climada.engine import CostBenefit
from climada.engine.cost_benefit import risk_aai_agg, risk_rp_100, risk_rp_250
from climada.engine.unsequa import InputVar, UncOutput
from IPython.display import clear_output

from climada.engine.unsequa import CalcCostBenefit


import pandas as pd
import copy
import os
import itertools
import scipy as sp
from functools import partial
import numpy as np

#import functions as fcn

# Constants
CURRENT_YEAR = 2018
FUTURE_YEAR = 2040
# Define the risk functions dictionary
RISK_FNCS_DICT = { 'aai': risk_aai_agg,
                  'rp250': risk_rp_250}


import inspect

def filter_dataframe(df, filter_conditions=None, derived_columns=None, base_cols=None):
    """
    This function filters a DataFrame based on provided conditions and calculates derived columns.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.
    - filter_conditions (dict): A dictionary specifying filtering conditions for columns.
    - derived_columns (dict): A dictionary specifying derived columns and their functions.

    Returns:
    - filtered_df (pandas.DataFrame): The filtered DataFrame based on conditions and derived columns.
    - boolean_df (pandas.DataFrame): A boolean DataFrame indicating whether values satisfy conditions.
    """

    # Create a copy of the input DataFrame
    filtered_df = df.copy()
    unfiltered_df = df.copy()
    
    # If conditions or derived columns are not provided, initialize them as empty dictionaries
    if filter_conditions is None:
        filter_conditions = {}
    
    if derived_columns is None:
        derived_columns = {}

    # Calculate and add derived columns to the filtered DataFrame
    if derived_columns:
        for new_col, function in derived_columns.items():
            filtered_df[new_col] = function(df)
            unfiltered_df[new_col] = function(df)

    # Create a boolean DataFrame to track conditions satisfaction
    if base_cols:
        boolean_df = df[base_cols].copy()
    else:
        boolean_df = df.copy()
    
    # Apply filtering conditions and update boolean DataFrame accordingly
    if filter_conditions:
        for col, cond in filter_conditions.items():
            
            if isinstance(cond, list):
                # Filter data based on whether column values are equal to the provided value
                filtered_df = filtered_df[filtered_df[col] == cond['equal']]
                boolean_df[col] = unfiltered_df[col] == cond['equal']
            elif 'equal' in cond:
                # Filter data based on whether column values are equal to the provided value
                filtered_df = filtered_df[filtered_df[col].isin(cond['equal'])]
                boolean_df[col] = unfiltered_df[col].isin(cond['equal'])
            elif 'in' in cond:
                # Filter data based on whether column values are in the provided list
                filtered_df = filtered_df[filtered_df[col].isin(cond['in'])]
                boolean_df[col] = unfiltered_df[col].isin(cond['in'])
            elif 'greater' in cond:
                # Filter data based on whether column values are greater than the provided value
                filtered_df = filtered_df[filtered_df[col] > cond['greater']]
                boolean_df[col] = unfiltered_df[col] > cond['greater']
            elif 'less' in cond:
                # Filter data based on whether column values are less than the provided value
                filtered_df = filtered_df[filtered_df[col] < cond['less']]
                boolean_df[col] = unfiltered_df[col] < cond['less']
            elif 'range' in cond:
                # Filter data based on whether column values are within the provided range
                lower, upper = cond['range']
                filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
                boolean_df[col] = (unfiltered_df[col] >= lower) & (unfiltered_df[col] <= upper)

    # Drop derived columns from the final filtered DataFrame
    if derived_columns:
        filtered_df = filtered_df.drop(derived_columns.keys(), axis=1)
     
    return filtered_df, boolean_df

def generate_unique_sets(items):
    """
    Generate all possible sets of unique items, including an empty set.

    Parameters:
        items (list): List of items to generate sets from.

    Returns:
        list of lists: List of lists representing all unique sets.
    """
    all_sets = [[]]  # Start with an empty set

    for r in range(1, len(items) + 1):
        item_combinations = itertools.combinations(items, r)
        all_sets.extend(item_combinations)

    # Convert sets to lists
    all_sets_as_lists = [list(item_set) for item_set in all_sets]

    return all_sets_as_lists

def expand_dataframe(original_df, values_list, new_column_name):
    # Create a list to hold the expanded rows
    expanded_data = []

    # Iterate through the original DataFrame
    for _, row in original_df.iterrows():
        # For each row in the original DataFrame, create a new row for each value in values_list
        for value in values_list:
            new_row = row.copy()
            new_row[new_column_name] = value
            expanded_data.append(new_row)

    # Create the expanded DataFrame
    expanded_df = pd.DataFrame(expanded_data)

    return expanded_df

def generate_metrics(haz_dict, ent_dict, unc_func_dist_dict={}, groups=[], risk_fncs_dict=RISK_FNCS_DICT, n_samples=1, imp_time_depend=1.2, future_year=FUTURE_YEAR, current_year=CURRENT_YEAR, file_output = None):


    # Initialize a flag variable
    first_iteration = True
    aggr_sets =  generate_unique_sets(groups)

    unc_var_dist = {}
    mean_dict = {}
    unc_vars = []

    # Add years to the entity objects
    for ent_key in ent_dict.keys():
        ent_dict[ent_key]['today'].exposures.ref_year = current_year
        ent_dict[ent_key]['future'].exposures.ref_year = future_year

    # Check if uncertainty variables are given and get them
    if unc_func_dist_dict:
        for unc_var in unc_func_dist_dict.keys():
            if unc_func_dist_dict[unc_var]['func'] and unc_func_dist_dict[unc_var]['distr']:
                for var in unc_func_dist_dict[unc_var]['distr'].keys():
                    # Add the variable to the dictionary
                    unc_vars += [var]
                    unc_var_dist[var] = unc_func_dist_dict[unc_var]['distr'][var]
                    # Check if the variable is continuous or discrete
                    mean_dict[var] = unc_func_dist_dict[unc_var]['distr'][var].mean()
                    # If discrete, you will need all the variables # TODO: Add this later


    # Create container
    # Build decision matrix
    base_col = ['entity', 'hazard', 'haz_type'] + groups 
    #crit_col = ['cost'] + ['tot_' + fcn.__name__ for fcn in risk_fncs] + ['ben_' + fcn.__name__ for fcn in risk_fncs] + ['ben_cost_ratio_' + fcn.__name__ for fcn in risk_fncs]
    #crit_col = ['crit_' + col for col in crit_col]
    metrics_df = pd.DataFrame(columns = base_col) # Decision matrix (contains the criteria values we will use to base a decision)

    for ent_key, haz_key, aggr_set in itertools.product(ent_dict.keys(), haz_dict.keys(), aggr_sets):
        # Clear the output
        clear_output()
        print(ent_key, haz_key, aggr_set)

        # Get the possible group aggregation combinations
        if aggr_set: 
            aggr_combos_df = ent_dict[ent_key]['today'].exposures.gdf[aggr_set].drop_duplicates()
        else:
            aggr_combos_df = pd.DataFrame(['dummy'])

        ## Get the entity objects (only exposure effected) at each aggregation combination
        for index, aggr_combo in aggr_combos_df.iterrows():

            # Get the entities
            ent_today = copy.deepcopy(ent_dict[ent_key]['today'])
            ent_fut = copy.deepcopy(ent_dict[ent_key]['future'])

            # Get the valuue unit

            
            # Filter the exposures to the aggregation combination
            if aggr_set:
                # Get the index of the exposures that match the aggregation combination
                # Enitiy today
                idx_combo = ent_today.exposures.gdf[aggr_set].isin(aggr_combo.values).all(axis=1)
                ent_today.exposures.gdf = ent_today.exposures.gdf[idx_combo]
                # Entity future
                idx_combo = ent_fut.exposures.gdf[aggr_set].isin(aggr_combo.values).all(axis=1)
                ent_fut.exposures.gdf = ent_fut.exposures.gdf[idx_combo]

            # Get the hazards
            # To speed up calculations – reduce the hazard extent to the extent of the exposures
            haz_today = haz_dict[haz_key]['today'].select(extent=(ent_today.exposures.gdf.longitude.min(), ent_today.exposures.gdf.longitude.max(), ent_today.exposures.gdf.latitude.min(), ent_today.exposures.gdf.latitude.max()))
            haz_fut = haz_dict[haz_key]['future'].select(extent=(ent_fut.exposures.gdf.longitude.min(), ent_fut.exposures.gdf.longitude.max(), ent_fut.exposures.gdf.latitude.min(), ent_fut.exposures.gdf.latitude.max()))

            if not haz_today or not haz_fut:
                print(f'No hazard data for {ent_key} {haz_key} {aggr_set}')
                haz_today = copy.deepcopy(haz_dict[haz_key]['today'])
                haz_fut = copy.deepcopy(haz_dict[haz_key]['future'])
                if not haz_today or not haz_fut:
                    print(f'No hazard data for {ent_key} {haz_key} {aggr_set}')
                    continue
                    
            # Get all measure names
            measure_dict = ent_today.measures.get_measure()[haz_today.haz_type]
            measure_names = ['no measure'] + list(measure_dict.keys())

            ## Calculate the criteria values, using ether cost-benefit or Unsequa module, depending if uncertainty var included 
            if not unc_func_dist_dict:
                # Create a temporary dict to later use to populate the data frame
                t_dict_to_df = {'entity': [ent_key for meas in measure_names], 
                            'hazard': [haz_key for meas in measure_names],
                            'haz_type':  [haz_today.haz_type for meas in measure_names], 
                            'measure': measure_names
                            }
                
                # Add aggregation columns
                if aggr_set:
                    t_dict_to_df.update({col_name: [col_value for meas in measure_names] for col_name, col_value in aggr_combo.items()})
                for group in groups:
                    if group not in aggr_set:
                        t_dict_to_df[group] = ['ALL' for meas in measure_names]
                        
                
                # Calc criteria – Costs
                t_dict_to_df['crit_cost'] =  [measure_dict[meas].cost if meas != 'no measure' else 0 for meas in measure_names]
                

                # Calc criteria –  Averted risk and cost benefit
                for name, risk_fcn in risk_fncs_dict.items():
                    costbenefit_disc = CostBenefit()
                    costbenefit_disc.calc(hazard = haz_today, 
                                    entity = ent_today, 
                                    haz_future = haz_fut, 
                                    ent_future = ent_fut,
                                    risk_func=risk_fcn, 
                                    imp_time_depen = imp_time_depend, 
                                    save_imp=True)
                
                    
                    # Save benefit results in tmep dictionary
                    t_dict_to_df['crit_ben_' + name] =  [costbenefit_disc.benefit[meas] if meas != 'no measure' else 0 for meas in measure_names]
                    t_dict_to_df['crit_bcr_' + name] =  [1/costbenefit_disc.cost_ben_ratio[meas] if meas != 'no measure' else 0 for meas in measure_names]
                    t_dict_to_df['crit_npv_' + name] = [costbenefit_disc.tot_climate_risk - costbenefit_disc.benefit[meas] if meas != 'no measure' else costbenefit_disc.tot_climate_risk for meas in measure_names]
                    
                    
                # Populate decision matrix
                metrics_df = pd.concat([metrics_df, pd.DataFrame(t_dict_to_df)], ignore_index=True)

            else:
                
                # Define the input uncertainty variables
                # Recode, generalize, later so that you can define and call other defined uncertainty variables that deterine  
                # the four objects haz_input_var, ent_input_var, haz_fut_input_var, ent_fut_input_var
                # Entity today
                base_dict = {'ent_today_base': ent_today, 'ent_fut_base': ent_fut, 'haz_today_base': haz_today, 'haz_fut_base': haz_fut}
                keys = ['ent_today', 'ent_fut', 'haz_today', 'haz_fut']
                iv_dict = {}

                for key in keys:
                    if unc_func_dist_dict[key]['func'] and unc_func_dist_dict[key]['distr']:
                        iv_dict[key + '_iv'] = InputVar(partial(unc_func_dist_dict[key]['func'], **base_dict), unc_func_dist_dict[key]['distr'])
                    else:
                        iv_dict[key + '_iv'] = locals()[key]


                # Define the uncertainty cost-benefit object
                unc_cb = CalcCostBenefit(haz_input_var=iv_dict['haz_today_iv'], ent_input_var=iv_dict['ent_today_iv'],
                                    haz_fut_input_var=iv_dict['haz_fut_iv'], ent_fut_input_var=iv_dict['ent_fut_iv'])
                
                # Make samples (only first iteration)
                if first_iteration:
                    df_samples = unc_cb.make_sample(N=n_samples, sampling_kwargs={'calc_second_order':False}).get_samples_df()
                    # Add the mean 
                    new_mean_row = {var: [mean_dict[var]] for var in unc_var_dist}
                    # TODO: Add all the varibles for the discrete case
                    # Append the new row to the DataFrame
                    df_samples = pd.concat([df_samples, pd.DataFrame(new_mean_row)], ignore_index=True)
                    nbr_of_samples = len(df_samples)
                    first_iteration = False


                # Create empty data frame to consectively add criteria columns to
                # Later to be used to concatenate with metrics_df
                t_pop_df = expand_dataframe(df_samples, measure_names, 'measure')
                keys = list(t_pop_df.columns)

                # Calc each criteria value for each uncertainty var combo
                for name, risk_fcn in risk_fncs_dict.items():
                    # Calculate criteria values with pool
                    output_cb = unc_cb.uncertainty(UncOutput(df_samples), risk_func=risk_fcn, imp_time_depen = imp_time_depend, future_year = future_year)
                    df_results = output_cb.get_uncertainty(metric_list=['benefit', 'cost_ben_ratio', 'tot_climate_risk'])
                    
                    # Make basic empty data frame containing 
                    t_crit_df = pd.DataFrame(columns= keys + ['crit_ben_' + name] + ['crit_bcr_' + name] + ['crit_npv_' + name])
                    t_meas_crit_df = copy.deepcopy(df_samples)
                    
                    # Get the criteria values for each measure under each uncertainty variable set
                    for meas in measure_names:
                        if  meas == 'no measure':
                            t_meas_crit_df['measure'] = meas
                            t_meas_crit_df['crit_ben_' + name] = 0
                            t_meas_crit_df['crit_bcr_' + name] = 0
                            t_meas_crit_df['crit_npv_' + name] = df_results['tot_climate_risk']
                            t_crit_df = pd.concat([t_crit_df, t_meas_crit_df], ignore_index=True)
                        else:
                            t_meas_crit_df['measure'] = meas
                            t_meas_crit_df['crit_ben_' + name] = df_results[meas + ' Benef']
                            t_meas_crit_df['crit_bcr_' + name] = 1/df_results[meas + ' CostBen']
                            t_meas_crit_df['crit_npv_' + name] = df_results['tot_climate_risk'] - df_results[meas + ' Benef']
                            t_crit_df = pd.concat([t_crit_df, t_meas_crit_df], ignore_index=True)
                            
                    # Right join with t_pop_df
                    t_pop_df = pd.merge(t_pop_df, t_crit_df, on=keys, how='right')

                # Calc criteria – Costs
                t_pop_df['crit_cost'] =  [measure_dict[meas].cost if meas != 'no measure' else 0 for meas in t_pop_df.measure]
                t_pop_df['entity'] = ent_key
                t_pop_df['hazard'] = haz_key
                t_pop_df['haz_type'] = haz_today.haz_type
                
                # Add aggregation columns
                if aggr_set:
                    for group in groups:
                        if group not in aggr_set:
                            t_pop_df[group] = 'ALL' 
                        else:
                            for col_name, col_value in aggr_combo.items():
                                t_pop_df[col_name] = col_value
                else:
                    for group in groups:
                        t_pop_df[group] = 'ALL'
            
                
                # Populate decision matrix
                metrics_df = pd.concat([metrics_df, t_pop_df], ignore_index=True)

    # Make a backup if it fails
    #backup_metrics_df = metrics_df.copy()


    #%%

    cols = [['measure'], ['entity'], ['hazard', 'haz_type'],  unc_vars, groups ]

    for idx, col in enumerate(cols):
        
        # Get unique values from each column
        df_temp = pd.DataFrame(metrics_df[col].drop_duplicates(), columns= col)
        # Add a common column to each DataFrame
        df_temp['_merge'] = 1
        
        # Merge data frames
        if idx == 0:
            df_base = df_temp
        else:
            df_base = df_base.merge(df_temp, on='_merge')
            
    # Drop the common column
    df_base = df_base.drop('_merge', axis=1)

    if unc_vars:
        df_base = df_base.astype(metrics_df[df_base.columns].dtypes)

    # Drop duplicates
    df_base = df_base.drop_duplicates()

    #%%
    # Join criteria values
    metrics_df = pd.merge(df_base, metrics_df, how='left')


    # #%% Update so certain metrics are the same for both provinces, e.g., cost

    #Get non 
    col = 'crit_cost'
    meas_cost_df = metrics_df[['measure', col]].dropna().drop_duplicates().rename(columns= {col: 'clean'})
    # # Merge with metrics_df
    metrics_df = metrics_df.merge(meas_cost_df, how='left')
    metrics_df[col] = metrics_df['clean']
    metrics_df = metrics_df.drop(columns=['clean'])


    #%% Pivot the DataFrame to add exposure to column


    # # Define pivot and index columns for pivot
    base_col = [col for col in metrics_df.columns if 'crit' not in col and 'entity' not in col]
    piv_col = ['entity']
    val_col = [col for col in metrics_df.columns if 'crit' in col]

    metrics_df = metrics_df.pivot(index = base_col,
                            columns = piv_col,
                            values = val_col)

    metrics_df = metrics_df.reset_index()

    # # Flatten the MultiIndex columns and add suffixes
    metrics_df.columns = [f'{"_".join(col)}' if col[1] else f'{col[0]}' for col in metrics_df.columns]

    #%% Drop and rename columns

    #metrics_df = metrics_df.rename(columns={'crit_cost_Assets' :'crit_cost_USD'})
    #metrics_df = metrics_df.drop(columns=['crit_cost_People'])


    #%%% Add additional criteria

    # Average annual values
    #for col in metrics_df.columns:
    #    if 'crit_npv_' in col:
    #        risk_fun = col.replace('crit_npv_', '')
    #        metrics_df['crit_avg_' + risk_fun] = metrics_df[col]/(future_year-current_year)

    # # # Break-even = cost/(benefit/year) (only for assets)
    # # #metrics_df['crit_breakeven'] = metrics_df['crit_cost']/(metrics_df['crit_ben_aai_Assets']/(future_year-CURRENT_YEAR))
        
    #%% Add feasibility and popularity criteria

    # Updated data including "Insurance"
    #data = {
    #    'measure': [
    #        'no measure', 'Retention Reservoirs', 'Swales', 'Waste Management',
    #       'Rehabilitation Drainage', 'Flood Awareness', 'Spillways', 'Rain collection',
    #        'Mobile flood embankments', 'Flood Wall', 'Storage + Sandbags', 'Green Roofs',
    #        'Green Spaces', 'Insurance'  # Added "Insurance"
    #    ],
    #    'crit_approv': [
    #        3, 4, 4, 3, 3, 1, 4, 4, 1, 2, 4, 2, 3, 4  # Approval rating for "Insurance"
    #    ],
    #    'crit_feas': [
    #        0.044420, 0.594257, 0.837318, 0.882075, 0.634717, 0.116425, 0.959374,
    #        0.019499, 0.390462, 0.847239, 0.415067, 0.717244, 0.630990, 0.887256  # Feasibility for "Insurance"
    #    ]
    #}

    # Create DataFrame
    temp_df = pd.DataFrame(metrics_df['measure'].unique(), columns=['measure'])
    temp_df['crit_approv'] = [np.random.randint(1, 5) for _ in range(len(temp_df))]
    temp_df['crit_feas'] = np.random.rand(len(temp_df))
    metrics_df = pd.merge(metrics_df, temp_df, on='measure')

    #%% Drop the criteria prefix
    metrics_df.columns =  [col.replace('crit_','') if 'crit' in col else col for col in metrics_df.columns  ]

    #%% Save the metrics
    if file_output:

        # Path 
        path = os.path.join(os.getcwd(), "Data/Metrics")
        # Make country directory
        try:
            # Create the directory 
            os.mkdir(path)
        except:
            pass

        
        # Save the file as a csv with suffix groups
        file_output_csv = os.path.join(path, file_output + '.csv')
        metrics_df.to_csv(file_output_csv, index=False)

    return metrics_df



def generate_unc_func_dist_dict(func_dict, unc_var_dist_dict):
    # Generate unc_func_dist_dict
    unc_func_dist_dict = {}
    for func_name, func in func_dict.items():
        # Get the argument names of the function
        arg_names = inspect.getfullargspec(func).args
        # Map the argument names to their distributions
        distr = {arg: unc_var_dist_dict[arg] for arg in arg_names if arg in unc_var_dist_dict}
        # Add to unc_func_dist_dict
        unc_func_dist_dict[func_name] = {'func': func, 'distr': distr}
    return unc_func_dist_dict