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

from .impact_trajectories import CalcImpactsSnapshots, SnapshotsCollection
from .impact_metrics import ImpactMetrics
import pandas as pd
import numpy as np
from climada.entity.measures import MeasureSet

# BASE_arm_df_COLUMNS = ['measure', 'group', 'year', 'metric', 'result']


class CalcImpactMetrics:

    def __init__(self, snapshots, group_col_str=None):
        self.snapshots = snapshots
        # Get groups
        self.group_map_exp_dict = create_group_map_exp_dict(snapshots, group_col_str) if group_col_str else None

    # Calculate the annual risk metrics
    def calc_arm_df(self, calc_static_scenario_metrics = True, previous_impact_metrics = None, measure_set = None):
        # Calculate the risk metrics for the non-measure case
        if previous_impact_metrics:
            _df = previous_impact_metrics.arm_df.copy()
        else: 
            _df = calc_annual_risk_metrics(self.snapshots, group_map_exp_dict=self.group_map_exp_dict)

        # Calculate the static risk metrics (used for waterfall levels)
        if previous_impact_metrics and previous_impact_metrics.all_arms_df is not None:
            _all_df = previous_impact_metrics.all_arms_df.copy()
        elif calc_static_scenario_metrics:
            _all_df = calc_annual_risk_metrics_static(self.snapshots, group_map_exp_dict=self.group_map_exp_dict)
        else:
            _all_df = None

        # Calculate the risk metrics for the measure set
        if measure_set:
            # ADHOC!!! This should be done in a more elegant way
            # Remove all the rows in the dataframe that have measure 'All'
            _df = _df[_df['measure'] != 'All']
            # Reduce the measure set if previous_impact_metrics is given
            measure_set = reduce_meas_set(measure_set, previous_impact_metrics.measure_set) if previous_impact_metrics else measure_set
            # Calculate the risk metrics for the measure set
            _temp_df = calc_annual_risk_metrics_measure_set(self.snapshots, measure_set, group_map_exp_dict=self.group_map_exp_dict)
            # Concatenate with arm_df
            _df = pd.concat([_df, _temp_df], ignore_index=True)

        return _df, _all_df

    # Generate an impact metrics object
    def generate_impact_metrics(self, measure_set=None, calc_static_scenario_metrics = True, previous_impact_metrics = None, planner = None, combine_all = True):

        # Generate the measure times DataFrame
        # Check if measure_times_df is given
        if previous_impact_metrics and previous_impact_metrics.measure_times_df is not None:
            measure_times_df = previous_impact_metrics.measure_times_df.copy()
            # Update the measure times DataFrame based on the planner
            if planner:
                measure_times_df = update_measure_times_df(measure_times_df, planner)  
        elif measure_set:
            measure_times_df = generate_meas_times_df(measure_set, self.snapshots, planner) if measure_set else None
        
        # Generate the necessary measure set including the sub_combos
        if measure_set and combine_all:
            measure_set = generate_necessary_combo_measure_set(measure_set, measure_times_df)
        else:
            measure_set = measure_set if measure_set else None
       
        # Get the impact metrics
        _df, _all_df = self.calc_arm_df(calc_static_scenario_metrics, previous_impact_metrics, measure_set)

        # Get the exposure value units
        exp_value_unit = self.snapshots.data[0].exposure.value_unit

        return ImpactMetrics(_df, _all_df, measure_set, measure_times_df, exp_value_unit)



#%% Utility functions

# Reduce the measure set to only include the new measures
def reduce_meas_set(new_meas_set, old_meas_set=None):
    
    # If no old measure set is provided, return the new measure set
    if old_meas_set is None:
        return new_meas_set

    meas_list = []
    # Add to measure set list
    for meas_name, meas in new_meas_set.measures().items():
        if meas_name in old_meas_set.measures().keys() and meas_name != 'All':
            continue
        meas_list.append(meas)
    return MeasureSet(meas_list)


# Update the measure times DataFrame based on the planner
def update_measure_times_df(measure_times_df, planner):
    new_measure_times_df = measure_times_df.copy()

    for meas_name, dates in planner.items():
        # Directly update the start and end year for rows where measure equals meas_name
        new_measure_times_df.loc[new_measure_times_df.measure == meas_name, 'start_year'] = dates[0]
        new_measure_times_df.loc[new_measure_times_df.measure == meas_name, 'end_year'] = dates[1]

    return new_measure_times_df

def create_group_map_exp_dict(snapshots, group_col_str= None):
    '''
    Create a dictionary that maps each group to the indices of the exposures in the gdf.
    '''

    if group_col_str:
        # Get the first snapshot exposures gdf
        gdf = snapshots.data[0].exposure.gdf
        # Get the unique groups
        unique_groups = list(gdf[group_col_str].unique())
        # Create the dictionary
        group_map_exp_dict = {i: list(np.where(gdf[group_col_str] == i)[0]) for i in unique_groups}

    return group_map_exp_dict

def generate_meas_times_df(measure_set, snapshots, planner=None):
    # Create the df
    cols = ['measure', 'start_year', 'end_year']
    meas_times_df = pd.DataFrame(columns=cols)

    # Populate the df
    for _,meas in measure_set.measures().items():

        # Check if the measure is in the planner
        if not planner or meas.name not in planner.keys():
            star_year = min(snapshots.snapshots_years)
            end_year = max(snapshots.snapshots_years)
        else:
            star_year = planner[meas.name][0]
            end_year = planner[meas.name][1]

        # Create a new DataFrame for the current measure
        temp_df = pd.DataFrame({
            'measure': [meas.name],
            'start_year': [star_year],
            'end_year': [end_year]
        })

        # Append to the df
        if meas_times_df.empty:
            meas_times_df = temp_df
        else:
            meas_times_df = pd.concat([meas_times_df, temp_df])

    return meas_times_df


def get_active_measure_combinations(meas_times_df):

    # Get the range of years
    min_year = meas_times_df['start_year'].min()
    max_year = meas_times_df['end_year'].max()
    years = range(min_year, max_year + 1)

    # Initialize a list to store the active measures for each year
    active_measures_by_year = []

    for year in years:
        active_measures = meas_times_df[(meas_times_df['start_year'] <= year) & (meas_times_df['end_year'] >= year)]['measure'].tolist()
        if not active_measures:
            active_measures = ['no_measure']
        active_measures_by_year.append(frozenset(active_measures))  # Use frozenset for unique combinations

    # Get the unique combinations of active measures
    unique_combinations = set(active_measures_by_year)

    # Convert the frozensets to sorted lists for readability and filter out lists with size equal to one
    unique_combinations = [sorted(list(comb)) for comb in unique_combinations if len(comb) > 1]

    return unique_combinations

def include_combos_in_measure_set(measure_set, *other_combos, all_measures=True ):
    """
    Generate possible combinations of measures in a measure set

    Parameters
    ----------
    measure_set : MeasureSet
        The measure set to be combined
    all_measures : bool
        If True, all measures are combined
    other_combos : list
        List of lists of measures to be combined

    Returns
    -------
    dict
        Dictionary of combined measures
    """

    # Make a copy of the measure set
    new_measure_set = copy.deepcopy(measure_set)

    # Combine all measures
    if all_measures:
        meas_all = new_measure_set.combine()
        #meas_all = new_measure_set.combine(combo_name='all')
        #new_measure_set.append(meas_combo)

    # Combine other measures
    for combo in other_combos:
        meas_combo = new_measure_set.combine(names=combo)
        new_measure_set.append(meas_combo)

    # Add the 'all' measure
    if all_measures:
        new_measure_set.append(meas_all)

    return new_measure_set

# make a function that filters out redundant combos. 
# This is consequence that if all measures may be overlapping at some point it generates a combo of this kind which is redundent to the 'all' measures included combo.
def filter_redundant_combos(measure_set):
    # Initialize lists to hold the names of initial measures, individual measures, and combo measures
    init_meas = list(measure_set.measures().keys())
    individual_measures = []
    combo_measures = []

    # Initialize a dictionary to track unique combos
    unique_combos = {}

    for meas_name, meas in measure_set.measures().items():
        if meas.combo:
            combo_tuple = tuple(sorted(meas.combo))
            if combo_tuple not in unique_combos:
                unique_combos[combo_tuple] = meas_name
                # Temporarily add to combo_measures; will decide placement later
                combo_measures.append(meas_name)
            else:
                if meas_name in init_meas:
                    init_meas.remove(meas_name)
        else:
            individual_measures.append(meas_name)

    # Identify the 'All' combo, if it exists
    all_combo = None
    for combo_name in combo_measures:
        if set(measure_set.measures()[combo_name].combo) == set(individual_measures):
            all_combo = combo_name
            break

    if all_combo:
        # Ensure 'All' combo is last
        combo_measures.remove(all_combo)
        combo_measures.append(all_combo)
        # Rename the 'All' combo measure
        measure_set.measures()[all_combo].name = 'All'

    # Merge individual measures and combo measures, excluding redundant combos
    final_meas_names = individual_measures + combo_measures
    meas_list = [measure_set.measures()[meas_name] for meas_name in final_meas_names]

    # Create a new MeasureSet with the unique and properly ordered measures
    unique_measure_set = MeasureSet(measures=meas_list)

    return unique_measure_set

# make a function that generates the updated combo measure set
def generate_necessary_combo_measure_set(measure_set, meas_times_df=None):
    """
    Update a measure set with the unique combinations of active measures
    
    Parameters
    ----------
    measure_set : MeasureSet
        The measure set to be updated
    meas_times_df : pd.DataFrame
        A DataFrame with the measures and the start and end years

    Returns
    -------
    MeasureSet
        The updated measure set
    """

    # Get the unique combinations of active measures
    unique_combinations = get_active_measure_combinations(meas_times_df)
    # Generate the updated measure set
    new_measure_set = include_combos_in_measure_set(measure_set, *unique_combinations, all_measures=True)
    # Filter out redundant combos
    new_measure_set = filter_redundant_combos(new_measure_set)
    
    return new_measure_set

def _update_exposure_data(gdf, other_gdf, unique_exp_columns, update_cols):
    """
    Updates the exposure data by merging it with the first exposure data based on unique columns.
    It updates the specified columns from the first exposure data.

    Parameters:
    - exp: The current exposure data.
    - first_exp: The first exposure data to merge with.
    - unique_exp_columns: Columns used for merging based on unique identifiers.
    - update_cols: Columns to update in the current exposure data.
    """

    # Copy the current exposure data
    gdf = gdf.copy()
    column_order = gdf.columns

    # Remove the update columns from the current exposure data
    gdf = gdf.drop(columns=update_cols, errors='ignore')

    # Merge the first exposure data with the current exposure data based on unique_exp_columns
    merged_gdf = gdf.merge(other_gdf[unique_exp_columns + update_cols],
                               on=unique_exp_columns,
                               how='left')

    # Update the specified columns in the current exposure's DataFrame with those from the merged DataFrame
    for col in update_cols:
        gdf[col] = merged_gdf[col]

    # Reorder the columns
    gdf = gdf[column_order]

    return gdf

# There should ba a function that takes a snapshot and returns the risk metrics in a dataframe
def make_static_snapshot(snapshots, exp_change=True, impfset_change=True, haz_change=True, unique_exp_columns=['latitude', 'longitude']):

    # Change the exposure, hazard and impfset for the affected snapshot years
    hazard_list = []
    exposure_list = []
    snapshot_years = []

    # Get the exposure for the first snapshot year
    first_exp = snapshots.data[0].exposure
    impf_cols = [col for col in  first_exp.gdf.columns if 'impf_' in col]
    value_cols = [col for col in  first_exp.gdf.columns if col in ['value', 'deductible', 'cover']]

    # Iterate over each snapshot
    for idx, snap in enumerate(snapshots.data):

        # Get the current exposure
        exp = copy.deepcopy(snap.exposure)

        # Determine if the exposure values change
        if not exp_change:
            # Update the exposure gdf with the first exposure gdf
            exp.gdf = _update_exposure_data(exp.gdf, first_exp.gdf, unique_exp_columns, value_cols)

        # Determine if the impfset changes
        if not impfset_change:
            # Update the exposure gdf with the first exposure gdf
            exp.gdf = _update_exposure_data(exp.gdf, first_exp.gdf, unique_exp_columns, impf_cols)

        # Determine if the hazard changes
        if not haz_change:
            haz = snapshots.data[0].hazard
        else:
            haz = snap.hazard

        # Append the snapshot year
        hazard_list.append(haz)
        exposure_list.append(exp)
        snapshot_years.append(snap.year)

        # Create a new snapshot collection for the measure case
        static_snapshots = SnapshotsCollection.from_lists( hazard_list, exposure_list, snapshots.impfset, snapshot_years)

    return static_snapshots

# Make measure snapshot
def make_measure_snapshot(snapshots, measure):
    _snapshots = copy.deepcopy(snapshots)
    # Change the exposure, hazard and impfset for the affected snapshot years
    hazard_list = []
    exposure_list = []
    snapshot_years = []

    for snap in _snapshots.data:
        # Apply the measure on all the snapshots
        exp_new, impfset_new, haz_new = measure.apply(snap.exposure, snap.impfset, snap.hazard)

        # Append the new exposure, hazar, impfset and snapshot year
        hazard_list.append(haz_new)
        exposure_list.append(exp_new)
        impfset = impfset_new # The impfset is the same for all the snapshots
        snapshot_years.append(snap.year)

    # Create a new snapshot collection for the measure case
    meas_snapshots = SnapshotsCollection.from_lists( hazard_list, exposure_list, impfset, snapshot_years)

    return meas_snapshots

def calc_annual_risk_metrics(snapshots, measure = None, group_map_exp_dict= None, risk_metrics = ['aai', 'rp', 'eai']):

    # Store the annual risk results in the dataframe where 'year' nan is for path-dependent results and group 'nan' is for aggregated results
    # arm_df = pd.DataFrame(columns=BASE_arm_df_COLUMNS)

    # Check if measure is applied
    if measure:
        snapshots = make_measure_snapshot(snapshots, measure)
        measure_name = measure.name
    else:
        measure_name = 'no_measure'

    # Calculate the risk metrics for the snapshots
    arm_df = CalcImpactsSnapshots(snapshots, group_map_exp_dict).calc_all_years(compute_groups=True)

    # Add the measure name to the dataframe
    arm_df['measure'] = measure_name
    # Reorder as the first column
    arm_df = arm_df[['measure'] + [col for col in arm_df.columns if col != 'measure']]

    # Remove duplicates
    arm_df = arm_df.drop_duplicates(subset=['measure', 'group', 'year', 'metric'], keep='first').reset_index(drop=True)

    return arm_df

def calc_annual_risk_metrics_static(snapshots, group_map_exp_dict=None):

    # Calculate the annual risk metrics for different static snapshots
    first_level_kwargs = {'exp_change': True, 'impfset_change': False, 'haz_change': False} # only exposure changes
    second_level_kwargs = {'exp_change': True, 'impfset_change': True, 'haz_change': False} # exposure and impfset changes

    unique_exp_columns = ['latitude', 'longitude']

    # Initialize an empty DataFrame to concatenate all arm_dfs
    # all_arms_df = pd.DataFrame()
    # Create static snapshots for the first and second waterfall level
    tmp = []
    for idx, level in enumerate([first_level_kwargs, second_level_kwargs]):
        # Create the static snapshots
        static_snapshots = make_static_snapshot(snapshots, **level, unique_exp_columns=unique_exp_columns)
        # Calculate the impacts
        arm_df = calc_annual_risk_metrics(static_snapshots, group_map_exp_dict=group_map_exp_dict)
        # Add the keys from the dictionary as new columns with their respective True/False values
        for key, value in level.items():
            arm_df[key] = value
        # Concatenate the current arm_df to the all_arms_df
        tmp.append(arm_df)

    all_arms_df = pd.concat(tmp, ignore_index=True)

    # Assuming you want to see the concatenated DataFrame
    return all_arms_df


def calc_annual_risk_metrics_measure_set(snapshots, measure_set, group_map_exp_dict= None):

    # _df = pd.DataFrame(columns=BASE_arm_df_COLUMNS)
    tmp = [ calc_annual_risk_metrics(snapshots, measure = meas, group_map_exp_dict= group_map_exp_dict) for _,meas in measure_set.measures().items() ]
    return pd.concat(tmp,ignore_index=True)


