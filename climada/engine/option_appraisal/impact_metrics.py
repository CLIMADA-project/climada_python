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
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from climada.engine.option_appraisal.plot import (
    _calc_waterfall_plot_df,
    _plot_two_years_waterfall,
    _plot_yearly_waterfall,
    plot_CB_summary,
    plot_risk_metrics,
    plot_yearly_averted_cost,
)
from climada.entity.measures import MeasureSet
from climada.util.value_representation import ABBREV, value_to_monetary_unit

from .impact_trajectories import SnapshotsCollection


class ImpactMetrics:

    def __init__(
        self,
        annual_risk_metrics_df: pd.DataFrame,
        measure_set: Optional[MeasureSet] = None,
        measure_times_df: Optional[pd.DataFrame] = None,
        planner: dict[str, tuple[int, int]] | None = None,
        value_unit: str = "USD",
    ):
        self.annual_risk_metrics_df = annual_risk_metrics_df.copy()
        self.measure_set = copy.deepcopy(measure_set)
        self.measure_times_df = measure_times_df.copy()
        self.value_unit = value_unit
        self.planner = planner

    def calc_CB(
        self,
        start_year=None,
        end_year=None,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
    ):
        return calc_CB_df(
            self.annual_risk_metrics_df,
            self.measure_set,
            self.measure_times_df,
            start_year,
            end_year,
            consider_measure_times,
            risk_disc,
            cost_disc,
        )


class ImpactMetrics2:

    def __init__(
        self,
        annual_risk_metrics_df: pd.DataFrame,
        all_annual_risk_metrics_df: Optional[pd.DataFrame] = None,
        measure_set: Optional[MeasureSet] = None,
        measure_times_df: Optional[pd.DataFrame] = None,
        planner: dict[str, tuple[int, int]] | None = None,
        value_unit: str = "USD",
    ):
        self.annual_risk_metrics_df = annual_risk_metrics_df.copy()
        self.all_annual_risk_metrics_df = copy.deepcopy(all_annual_risk_metrics_df)
        self.measure_set = copy.deepcopy(measure_set)
        self.measure_times_df = copy.deepcopy(measure_times_df)
        self.value_unit = value_unit
        self.planner = planner

    @classmethod
    def from_snapshots(
        cls, snapshots: SnapshotsCollection, measure_set: MeasureSet | None = None
    ):
        from .calc_impact_metrics import CalcImpactMetrics

        return CalcImpactMetrics(snapshots).generate_impact_metrics(
            measure_set=measure_set
        )

    def add_measures(self, measure_set: MeasureSet | None = None):
        pass

    ## Waterfall plot functions
    # Plot the waterfall plot
    def waterfall_plot(self, yearly=False, measure=None, metric="aai", group=np.nan):

        # Check if the all_annual_risk_metrics_df is provided
        if self.all_annual_risk_metrics_df is None:
            raise ValueError(
                "The all_annual_risk_metrics_df is required for the waterfall plot. Please reggerate the all_annual_risk_metrics_df using CalcImpactMetrics."
            )

        # Check if yearly or classic waterfall plot
        if yearly:
            # Calculate the averted risk metrics for yearly waterfall plot
            waterfall_df = _calc_waterfall_plot_df(
                self.annual_risk_metrics_df,
                self.all_annual_risk_metrics_df,
                measure=measure,
                metric=metric,
                ref_year=None,
                fut_year=None,
                group=group,
                subtract=True,
            )
            _plot_yearly_waterfall(waterfall_df, self.value_unit)
        else:
            # Calculate the averted risk metrics for the two points waterfall plot
            waterfall_df = _calc_waterfall_plot_df(
                self.annual_risk_metrics_df,
                self.all_annual_risk_metrics_df,
                measure=measure,
                metric=metric,
                ref_year=None,
                fut_year=None,
                group=group,
            )
            _plot_two_years_waterfall(waterfall_df, self.value_unit)

        return

    ## CB analysis
    # Calculate the cost-benefit analysis
    def calc_CB(
        self,
        start_year=None,
        end_year=None,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
    ):
        return calc_CB_df(
            self.annual_risk_metrics_df,
            self.measure_set,
            self.measure_times_df,
            start_year,
            end_year,
            consider_measure_times,
            risk_disc,
            cost_disc,
        )

    # Print the CB summary table
    def print_CB_summary_table(
        self,
        metric="aai",
        start_year=None,
        end_year=None,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
    ):
        # Get the CB analysis
        _, tot_CB_df = self.calc_CB(
            start_year, end_year, consider_measure_times, risk_disc, cost_disc
        )
        # Get the measure_summary_df and total_summary
        measure_summary_df, total_summary_dict, unit = generate_CB_plot_print_data(
            tot_CB_df, self.measure_set, metric, group=np.nan
        )
        # Print the measure summary in tabulate format
        # rename the  the Cost and Benefit headers by adding add ({value_unit} {unit})  to the headers
        measure_summary_df.columns = [
            "Measure",
            f"Cost ({self.value_unit} {unit})",
            f"Benefit ({self.value_unit} {unit})",
            "Benefit/Cost Ratio",
        ]
        print(tabulate(measure_summary_df, headers="keys", tablefmt="fancy_grid"))
        # Print the total summary
        print("\n--------------------  ---------")
        for key, value in total_summary_dict.items():
            print(f"{key:<22} {value:>11.2f}  ({self.value_unit} {unit})")
        print("--------------------  ---------")
        # Create a new line for explanatory text
        label_Benefit = f"{'Discounted ' if risk_disc else ''} Total averted annual risk metric: {metric}"
        label_Cost = f"{'Discounted ' if cost_disc else ''} Total annual cost"
        # Print explanatory text
        print("Explanatory Notes:")
        print(
            f"Benefit: {label_Benefit} – This represents the total benefits in terms of risk reduction."
        )
        print(
            f"Cost: {label_Cost} – This represents the total costs associated with implementing the measures."
        )
        return

    def plot_CB_summary_table(
        self,
        metric="aai",
        start_year=None,
        end_year=None,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
    ):
        # Get the CB analysis
        _, tot_CB_df = self.calc_CB(
            start_year, end_year, consider_measure_times, risk_disc, cost_disc
        )
        # Get the measure_summary_df and total_summary
        measure_summary_df, total_summary_dict, unit = generate_CB_plot_print_data(
            tot_CB_df, self.measure_set, metric, group=np.nan
        )
        # Plot the measure summary
        plot_CB_summary(
            measure_summary_df,
            total_summary_dict,
            measure_colors=None,
            y_label=f"Averted Risk {unit}",
            title=f"Benefit (NPV of total annual {metric}) and Benefit/Cost Ratio by Measure",
        )
        return

    # Plot the yearly averted cost for a specific measure
    def plot_benefit_vs_cost(
        self,
        measure,
        metric="aai",
        start_year=None,
        end_year=None,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
    ):
        ann_CB_df, tot_CB_df = self.calc_CB(
            start_year, end_year, consider_measure_times, risk_disc, cost_disc
        )
        plot_yearly_averted_cost(ann_CB_df, measure, metric, group=None)
        return

    ## Risk analysis
    # Plot the risk metrics for each measure
    def plot_yearly_risk_metrics(
        self,
        metric="aai",
        averted=False,
        consider_measure_times=False,
        risk_disc=None,
        plot_type="line",
        measures=None,
        group=np.nan,
    ):
        # Copy the DataFrame
        plot_df = self.annual_risk_metrics_df.copy()

        # Check if the time should be considered
        if consider_measure_times:
            plot_df = create_meas_mod_annual_risk_metrics_df(
                plot_df,
                self.measure_set,
                self.measure_times_df,
                consider_measure_times=True,
            )

        # Check if the risk metrics should be avereted
        if averted:
            # Get the avereted risk metrics
            plot_df = calc_averted_risk_metrics(plot_df)

        # check if the risk metrics should be discounted
        if risk_disc is not None:
            # Calculate the discounted cash flows
            plot_df, _ = calc_npv_annual_risk_metrics_df(plot_df, disc=risk_disc)
            discounted = True
        else:
            discounted = False

        # Plot the risk metrics
        plot_risk_metrics(
            plot_df,
            metric=metric,
            group=group,
            averted=averted,
            discounted=discounted,
            plot_type=plot_type,
            measures=measures,
            value_unit=self.value_unit,
        )

        return

    # Generate the MCDM dataframe
    def generate_MCDM_dataframe(
        self,
        consider_measure_times=True,
        risk_disc=None,
        cost_disc=None,
        levels=["tot", "avrt"],
        risk_metrics=None,
    ):

        # Calculate the cost-benefit analysis
        _, tot_CB_df = self.calc_CB(
            consider_measure_times=consider_measure_times,
            risk_disc=risk_disc,
            cost_disc=cost_disc,
        )

        # Get he risk metrics
        if risk_metrics is None:
            risk_metrics = self.get_risk_metrics()

        # Create the DecisionMatrix object
        mcdm_df = tot_CB_df.copy()

        # Filter out the risk metrics
        mcdm_df = mcdm_df[mcdm_df["metric"].isin(risk_metrics)]

        # Change the column names
        dict_col_names = {
            "total risk": "tot",
            "averted risk": "avrt",
            "cost (net)": "cost",
        }
        mcdm_df.rename(columns=dict_col_names, inplace=True)

        # Store only the risk metrics
        mcdm_risk_df = mcdm_df[["measure", "group"] + levels]
        # Remove space and _ in the metric column
        mcdm_risk_df["metric"] = mcdm_df["metric"].str.replace(" ", "")
        mcdm_risk_df["metric"] = mcdm_df["metric"].str.replace("_", "")
        # Pivot the table to have the risk_cols + metric + unit as columns
        mcdm_df_piv = mcdm_risk_df.pivot(
            index=["measure", "group"], columns="metric", values=levels
        )
        # # Add the unit to the column names
        mcdm_df_piv.columns = mcdm_df_piv.columns.map("{0[0]}_{0[1]}".format)
        # # Reset the index
        mcdm_df_piv.reset_index(inplace=True)

        # Add the cost by merging the dataframes
        mcmdm_cost_df = mcdm_df[["measure", "group", "cost"]].drop_duplicates()
        mcdm_df_piv = mcdm_df_piv.merge(mcmdm_cost_df, on=["measure", "group"])

        # Add the value unit to all columns except the measure and group
        mcdm_df_piv.columns = [
            f"{col}_{self.value_unit}" if col not in ["measure", "group"] else col
            for col in mcdm_df_piv.columns
        ]

        return mcdm_df_piv

    # Update the planner
    def update_planner(self, planner):

        # Step 1 - Create the new potential measure_times_df
        new_measure_times_df = update_measure_times_df(self.measure_times_df, planner)

        # Step 2 - Check if the 'All' measure combo is in the measure set
        for meas_name, meas in self.measure_set.measures().items():
            if meas_name == "All" and meas.combo:
                all_combos_exist = True

        # Step 3 - If the 'All' measure combo is in the measure set, check if all the necessary subcombos exist
        if all_combos_exist:
            actual_missing_combos = check_if_necessary_subcombos_exist(
                new_measure_times_df, self.measure_set
            )

            # if actual_missing_combos raising an error
            if actual_missing_combos:
                raise ValueError(
                    f"The following combos are missing from the measure set: {actual_missing_combos}. \n Re-run the planner to get the missing combos"
                )

        # Update the measure_times_df
        self.measure_times_df = new_measure_times_df

        return

    # Properties
    def get_annual_risk_df(self):
        # Update to modify be used for the plot funcions
        return self.annual_risk_metrics_df.copy()

    def get_risk_metrics(self):
        return list(self.annual_risk_metrics_df["metric"].unique())

    def get_measures(self):
        return list(self.measure_set.measures().keys())


# %% Update the planner


# Update the measure times DataFrame based on the planner
def update_measure_times_df(measure_times_df, planner):
    new_measure_times_df = measure_times_df.copy()

    for meas_name, dates in planner.items():
        # Directly update the start and end year for rows where measure equals meas_name
        new_measure_times_df.loc[
            new_measure_times_df.measure == meas_name, "start_year"
        ] = dates[0]
        new_measure_times_df.loc[
            new_measure_times_df.measure == meas_name, "end_year"
        ] = dates[1]

    return new_measure_times_df


# This would benefit from a __contains__ method in MeasureSet
# Check if the necessary subcombos exist in the measure set for the 'All' combo
def check_if_necessary_subcombos_exist(measure_times_df, measure_set):

    # Get the needed combos
    needed_combos = get_active_measure_combinations(measure_times_df)

    # Convert needed_combos to a set of frozensets for efficient comparison
    needed_combos_set = {frozenset(combo) for combo in needed_combos}
    found_combos = set()
    missing_combos = set()

    # Check if combo exists in the measure set
    for meas_name, meas in measure_set.measures().items():
        # Check if the measure has a combo
        if meas.combo:
            # Convert the combo to a frozenset
            combo_fset = frozenset(meas.combo)
            # Check if the combo is in the needed_combos_set
            if combo_fset in needed_combos_set:
                found_combos.add(combo_fset)
            else:
                missing_combos.add(combo_fset)

    # Identify missing combos by comparing needed_combos_set with found_combos
    actual_missing_combos = needed_combos_set - found_combos

    return actual_missing_combos


# %%


## Generate active measures for a specific year
def get_meas_times_df(measure_set, incl_combo=False):
    """
    Get a DataFrame with the start and end years of the measures in a MeasureSet

    Parameters
    ----------
    MeasureSet : MeasureSet
        The measure set

    Returns
    -------
    pd.DataFrame
        DataFrame with the start and end years of the measures
    """

    cols = ["measure", "start_year", "end_year"]
    meas_times_df = pd.DataFrame(columns=cols)

    # Populate the df
    for _, meas in measure_set.measures().items():

        # Skip combo measures if not included
        if not incl_combo and meas.combo:
            continue

        # Create a new DataFrame for the current measure
        temp_df = pd.DataFrame(
            {
                "measure": [meas.name],
                "start_year": [meas.start_year],
                "end_year": [meas.end_year],
            }
        )

        # Append to the df
        if meas_times_df.empty:
            meas_times_df = temp_df
        else:
            meas_times_df = pd.concat([meas_times_df, temp_df])

    return meas_times_df


# Get the unique combinations of measures that are mutually active for different time periods
def get_active_measure_combinations(meas_times_df):

    # Get the range of years
    min_year = meas_times_df["start_year"].min()
    max_year = meas_times_df["end_year"].max()
    years = range(min_year, max_year + 1)

    # Initialize a list to store the active measures for each year
    active_measures_by_year = []

    for year in years:
        active_measures = meas_times_df[
            (meas_times_df["start_year"] <= year) & (meas_times_df["end_year"] >= year)
        ]["measure"].tolist()
        if not active_measures:
            active_measures = ["no_measure"]
        active_measures_by_year.append(
            frozenset(active_measures)
        )  # Use frozenset for unique combinations

    # Get the unique combinations of active measures
    unique_combinations = set(active_measures_by_year)

    # Convert the frozensets to sorted lists for readability and filter out lists with size equal to one
    unique_combinations = [
        sorted(list(comb)) for comb in unique_combinations if len(comb) > 1
    ]

    return unique_combinations


def include_combos_in_measure_set(measure_set, *other_combos, all_measures=True):
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
        # meas_all = new_measure_set.combine(combo_name='all')
        # new_measure_set.append(meas_combo)

    # Combine other measures
    for combo in other_combos:
        meas_combo = new_measure_set.combine(names=combo)
        new_measure_set.append(meas_combo)

    # Add the 'all' measure
    if all_measures:
        new_measure_set.append(meas_all)

    return new_measure_set


# make a function that filters out redundant combos
def filter_redundant_combos(measure_set):
    # Initialize a list to hold the names of initial measures
    init_meas = list(measure_set.measures().keys())
    indvidual_measures = []

    # Initialize a dictionary to track unique combos
    unique_combos = {}

    for meas_name, meas in measure_set.measures().items():
        # Check if the measure is a combo
        if meas.combo:
            # Convert the combo to a tuple (or another immutable and hashable structure) for comparison
            combo_tuple = tuple(sorted(meas.combo))
            # If the combo is not already in unique_combos, add it with the current measure name
            if combo_tuple not in unique_combos:
                unique_combos[combo_tuple] = meas_name
            else:
                # If the combo is already tracked, remove the current measure name from init_meas
                if meas_name in init_meas:
                    init_meas.remove(meas_name)
        else:
            indvidual_measures.append(meas_name)

    # Collect measures from the filtered list of measure names
    meas_list = [
        measure_set.measures()[meas_name]
        for meas_name in init_meas
        if meas_name in measure_set.measures()
    ]

    # Create a new MeasureSet with the unique measures
    unique_measure_set = MeasureSet(measures=meas_list)

    # Rename the combo conatining all individual measures as 'All'
    for _, meas in unique_measure_set.measures().items():
        if meas.combo and set(meas.combo) == set(indvidual_measures):
            meas.name = "All"

    return unique_measure_set


# make a function that generates the updated combo measure set
def generate_necessary_combo_measure_set(
    measure_set, combo_consider_measure_times=True
):
    """
    Update a measure set with the unique combinations of active measures

    Parameters
    ----------
    measure_set : MeasureSet
        The measure set to be updated
    combo_consider_measure_times : bool
        If True, the measure times are considered when generating the unique combinations

    Returns
    -------
    MeasureSet
        The updated measure set
    """

    if combo_consider_measure_times:
        # Get the DataFrame of the individul measure times
        meas_times_df = get_meas_times_df(measure_set, incl_combo=False)
        # Get the unique combinations of active measures
        unique_combinations = get_active_measure_combinations(meas_times_df)
        # Generate the updated measure set
        new_measure_set = include_combos_in_measure_set(
            measure_set, *unique_combinations, all_measures=True
        )
    else:
        new_measure_set = include_combos_in_measure_set(measure_set, all_measures=True)

    # Filter out redundant combos
    new_measure_set = filter_redundant_combos(new_measure_set)

    return new_measure_set


## Waterfall plot functions


#
def get_active_measures_for_year(meas_times_df, year):

    # Filter the DataFrame for the specified year
    active_measures = meas_times_df[
        (meas_times_df["start_year"] <= year) & (meas_times_df["end_year"] >= year)
    ]["measure"].tolist()
    if not active_measures:
        active_measures = ["no_measure"]
    return active_measures


# Get the name of the active combo measure
def get_name_of_active_combo(measure_set, active_measures):

    # Get the name of the active combo measure
    # Check if there is only one active measure then return its name
    if len(active_measures) == 1:
        return active_measures[0]

    # Get the DataFrame of the individul measure times
    for _, meas in measure_set.measures().items():
        if meas.combo and set(meas.combo) == set(active_measures):
            return meas.name


# Calculate the averted risk metrics
def calc_averted_risk_metrics(annual_risk_metrics_df):
    # Copy the DataFrame to avoid modifying the original
    averted_annual_risk_metrics_df = pd.DataFrame(
        columns=annual_risk_metrics_df.columns
    )

    # Get the unique groups and metrics
    groups = annual_risk_metrics_df["group"].unique()
    metrics = annual_risk_metrics_df["metric"].unique()

    # Iterate over each combination of group and metric
    for group in groups:
        for metric in metrics:
            # Filter the DataFrame for the current group and metric
            if pd.isna(group):
                sub_df = annual_risk_metrics_df[
                    (annual_risk_metrics_df["group"].isna())
                    & (annual_risk_metrics_df["metric"] == metric)
                ]
            else:
                sub_df = annual_risk_metrics_df[
                    (annual_risk_metrics_df["group"] == group)
                    & (annual_risk_metrics_df["metric"] == metric)
                ]
            # If the sub_df is empty, continue to the next iteration
            if sub_df.empty:
                continue

            # Get the risk metrics for the no measure
            no_meas_df = sub_df[sub_df["measure"] == "no_measure"].sort_values("year")

            # Calculate the averted risk metrics for each measure
            for meas_name in sub_df["measure"].unique():
                # if meas_name == 'no_measure':
                #    continue

                # Get the risk metrics for the measure
                meas_df = sub_df[sub_df["measure"] == meas_name].sort_values("year")

                # Align the DataFrames by year and calculate the averted risk
                sub_averted_risk_df = meas_df.copy()
                sub_averted_risk_df["result"] = (
                    no_meas_df["result"].values - meas_df["result"].values
                )

                # Concatenate the DataFrames
                if averted_annual_risk_metrics_df.empty:
                    averted_annual_risk_metrics_df = sub_averted_risk_df
                else:
                    averted_annual_risk_metrics_df = pd.concat(
                        [averted_annual_risk_metrics_df, sub_averted_risk_df],
                        ignore_index=True,
                    )

    # Drop duplicates and reset the index
    averted_annual_risk_metrics_df = (
        averted_annual_risk_metrics_df.drop_duplicates().reset_index(drop=True)
    )

    return averted_annual_risk_metrics_df


# Calculate the averted risk metrics
def calc_npv_cash_flows(cash_flows, start_year, end_year=None, disc=None):

    # Check if discount rates are provided
    if disc:
        if end_year is None:
            end_year = start_year + len(cash_flows) - 1
        # Get the discount rates
        years = np.array(list(range(start_year, end_year + 1)))
        disc_years = np.intersect1d(years, disc.years)
        disc_rates = disc.rates[np.isin(disc.years, disc_years)]
        years = np.array([year for year in years if year in disc_years])
        # Calculate the discount factors
        discount_factors = []
        for idx, disc_rate in enumerate(disc_rates):
            discount_factors.append(1 / (1 + disc_rate) ** (years[idx] - start_year))
        discount_factors = np.array(discount_factors)
        # Return the discounted cash flows
        npv_cash_flows = cash_flows * discount_factors
    else:
        npv_cash_flows = cash_flows

    # Sum the discounted cash flows
    total_NPV = np.sum(npv_cash_flows)

    return npv_cash_flows, total_NPV


# Calculate the NPV for the annual_risk_metrics_df
def calc_npv_annual_risk_metrics_df(df, disc=None):
    # Copy the DataFrame
    disc_df = df.copy()
    npv_df = pd.DataFrame(columns=df.columns)

    for meas_name in disc_df["measure"].unique():
        for metric in disc_df["metric"].unique():
            for group_in in disc_df["group"].unique():

                # Handle NaN values in the 'group' column
                if pd.isna(group_in):
                    sub_df = disc_df[
                        (disc_df["measure"] == meas_name)
                        & (disc_df["metric"] == metric)
                        & (disc_df["group"].isna())
                    ]
                else:
                    sub_df = disc_df[
                        (disc_df["measure"] == meas_name)
                        & (disc_df["metric"] == metric)
                        & (disc_df["group"] == group_in)
                    ]

                # If the sub_df is empty, continue to the next iteration - Later rasie an error that no data is available
                if sub_df.empty:
                    continue

                cash_flows = sub_df["result"].values
                start_year = sub_df["year"].min()
                end_year = sub_df["year"].max()

                # Calculate the discounted cash flows and the total NPV
                npv_cash_flows, total_NPV = calc_npv_cash_flows(
                    cash_flows, start_year, end_year, disc
                )

                # Update the 'result' column with the discounted cash flows
                disc_df.loc[sub_df.index, "result"] = npv_cash_flows

                # Append the total NPV to the npv_df DataFrame
                npv_row = sub_df.iloc[0].copy()
                npv_row["result"] = total_NPV
                npv_df = pd.concat([npv_df, pd.DataFrame(npv_row).T])
                # Reset the index
                npv_df.reset_index(drop=True, inplace=True)

    # Drop the 'year' column from npv_df
    npv_df = npv_df.drop(columns=["year"])

    # Drop duplicates and reset the index
    disc_df = disc_df.drop_duplicates().reset_index(drop=True)
    npv_df = npv_df.drop_duplicates().reset_index(drop=True)

    return disc_df, npv_df


# %% Functions for the modified annual_risk_metrics_df


# Create the modified annual_risk_metrics_df for the individual measures
def create_indv_meas_mod_annual_risk_metrics_df(
    annual_risk_metrics_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Use start_year and end_year if they are provided and filter the annual_risk_metrics_df
    start_year = annual_risk_metrics_df["year"].min()
    end_year = annual_risk_metrics_df["year"].max()

    # Filter the annual_risk_metrics_df
    time_annual_risk_metrics_df = annual_risk_metrics_df.copy()

    # If the measure times should be considered
    if not consider_measure_times:
        return time_annual_risk_metrics_df

    # Iterate over the measures in the measure set
    # Use start_year and end_year if they are provided and filter the annual_risk_metrics_df
    time_annual_risk_metrics_df = time_annual_risk_metrics_df[
        time_annual_risk_metrics_df["measure"] == "no_measure"
    ].copy()

    # Iterate over the measures in the measure set
    for _, meas in measure_set.measures().items():
        # Skip combo measures
        if meas.combo:
            continue

        # Get the measure name and the start and end years from the measure_times_df
        measure_name = meas.name
        meas_start_year = measure_times_df[measure_times_df["measure"] == measure_name][
            "start_year"
        ].values[0]
        meas_end_year = measure_times_df[measure_times_df["measure"] == measure_name][
            "end_year"
        ].values[0]

        # Iterate over the years
        for year in range(start_year, end_year + 1):
            # If the measure is active this year, use its original results
            if meas_start_year <= year <= meas_end_year:
                mask = (annual_risk_metrics_df["measure"] == measure_name) & (
                    annual_risk_metrics_df["year"] == year
                )
                active_measure_df = annual_risk_metrics_df.loc[mask]
                time_annual_risk_metrics_df = pd.concat(
                    [time_annual_risk_metrics_df, active_measure_df], ignore_index=True
                )
            else:
                # If the measure is not active this year, replace its values with the 'no_measure' values
                for metric in annual_risk_metrics_df["metric"].unique():
                    for group_in in annual_risk_metrics_df["group"].unique():
                        # Create a mask for the 'no_measure' rows for this year, metric, and group
                        mask = (
                            (annual_risk_metrics_df["measure"] == "no_measure")
                            & (annual_risk_metrics_df["metric"] == metric)
                            & (annual_risk_metrics_df["year"] == year)
                            & (
                                (
                                    pd.isna(group_in)
                                    & annual_risk_metrics_df["group"].isna()
                                )
                                | (annual_risk_metrics_df["group"] == group_in)
                            )
                        )

                        # Get the 'no_measure' result for this year, metric, and group
                        result = annual_risk_metrics_df.loc[mask, "result"].values
                        # Skip if there is no result
                        if len(result) == 0:
                            continue

                        # Create a new DataFrame for this row
                        new_row = pd.DataFrame(
                            {
                                "measure": [measure_name],
                                "group": [group_in],
                                "year": [year],
                                "metric": [metric],
                                "result": result,
                            }
                        )

                        # Concatenate the new row to the DataFrame
                        time_annual_risk_metrics_df = pd.concat(
                            [time_annual_risk_metrics_df, new_row], ignore_index=True
                        )

    # Remove duplicates and reset the index
    time_annual_risk_metrics_df = (
        time_annual_risk_metrics_df.drop_duplicates().reset_index(drop=True)
    )

    return time_annual_risk_metrics_df


# Create the modified annual_risk_metrics_df for the combo measures
def create_meas_combo_time_mod_annual_risk_metrics_df(
    annual_risk_metrics_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Calculate the risk for the combo measures
    # Use start_year and end_year if they are provided and filter the annual_risk_metrics_df
    start_year = annual_risk_metrics_df["year"].min()
    end_year = annual_risk_metrics_df["year"].max()

    # Filter the annual_risk_metrics_df
    time_annual_risk_metrics_df = pd.DataFrame(columns=annual_risk_metrics_df.columns)

    # # If the measure times should be considered
    if not consider_measure_times:
        return time_annual_risk_metrics_df

    # Iterate over the measures in the measure set
    for _, meas in measure_set.measures().items():
        # Skip combo measures
        if not meas.combo:
            continue

        # Iterate over the years
        for year in range(start_year, end_year + 1):
            # Get the combos in the measure
            combo_measures = meas.combo
            measure_name = meas.name
            # Get the active measures for the year
            active_measures = get_active_measures_for_year(measure_times_df, year)
            # print(f'For {measure_name} active measures are {active_measures}')
            if len(active_measures) == 1:
                measure_to_use = active_measures[0]
            else:
                # Get the subset of active measures that are in the combo
                active_combo_measures = list(
                    set(active_measures).intersection(combo_measures)
                )
                # get the name of the active combo measure
                measure_to_use = get_name_of_active_combo(
                    measure_set, active_combo_measures
                )
            # print(f'For {measure_name} using {measure_to_use} for year {year}')

            # Filter the DataFrame for the current metric and measure to use
            sub_df = annual_risk_metrics_df[
                (annual_risk_metrics_df["measure"] == measure_to_use)
                & (annual_risk_metrics_df["year"] == year)
            ]

            # Update the measure name to the combo measure name
            sub_df["measure"] = measure_name

            # Append the values to the time_annual_risk_metrics_df
            if time_annual_risk_metrics_df.empty:
                time_annual_risk_metrics_df = sub_df
            else:
                time_annual_risk_metrics_df = pd.concat(
                    [time_annual_risk_metrics_df, sub_df], ignore_index=True
                )

    return time_annual_risk_metrics_df


# Create the modified annual_risk_metrics_df
def create_meas_mod_annual_risk_metrics_df(
    annual_risk_metrics_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Calculate the risk for the individual measures
    mod_annual_risk_metrics_indv_meas_df = create_indv_meas_mod_annual_risk_metrics_df(
        annual_risk_metrics_df, measure_set, measure_times_df, consider_measure_times
    )

    # Calculate the risk for the combo measures
    mod_annual_risk_metrics_combo_meas_df = (
        create_meas_combo_time_mod_annual_risk_metrics_df(
            annual_risk_metrics_df,
            measure_set,
            measure_times_df,
            consider_measure_times,
        )
    )

    # Concatenate the DataFrames
    if mod_annual_risk_metrics_combo_meas_df.empty:
        mod_annual_risk_metrics_df = mod_annual_risk_metrics_indv_meas_df
    else:
        mod_annual_risk_metrics_df = pd.concat(
            [
                mod_annual_risk_metrics_indv_meas_df,
                mod_annual_risk_metrics_combo_meas_df,
            ],
            ignore_index=True,
        )

    # Remove duplicates and reset the index
    mod_annual_risk_metrics_df = (
        mod_annual_risk_metrics_df.drop_duplicates().reset_index(drop=True)
    )

    return mod_annual_risk_metrics_df


# %% Cash flow functions


# Calculate the cash flows for the individual measures
def calc_indv_measure_cash_flows_df(
    measure_set,
    measure_times_df,
    start_year,
    end_year,
    consider_measure_times=True,
    disc=None,
):
    """
    This function calculates the cash flows for a set of measures over a specified time period.

    Parameters:
    measure_set: A set of measures for which to calculate cash flows.
    start_year: The first year of the time period.
    end_year: The last year of the time period.
    disc: The discount rate to apply to future cash flows.

    Returns:
    A DataFrame with the calculated cash flows for each measure.
    """

    # Initialize an empty DataFrame to store the cash flows
    costincome_df = pd.DataFrame(columns=["measure", "year", "cost", "income", "net"])

    # Loop over the measures in the set
    for _, meas in measure_set.measures().items():

        # If the measure is a combination of other measures, skip it
        if meas.combo:
            continue
        else:
            # If we should consider the start and end years of the measure, update the start and end years
            if consider_measure_times:
                measure_name = meas.name
                meas_start_year = measure_times_df[
                    measure_times_df["measure"] == measure_name
                ]["start_year"].values[0]
                meas_end_year = measure_times_df[
                    measure_times_df["measure"] == measure_name
                ]["end_year"].values[0]
            else:
                meas_start_year = start_year
                meas_end_year = end_year

            # Calculate the cash flows for the measure
            temp_df = meas.cost_income.calc_cashflows_df(
                impl_year=meas_start_year,
                start_year=start_year,
                end_year=end_year,
                disc=disc,
            )
            # Set all the cash flows to zero for years outside the measure period
            temp_df.loc[
                (temp_df["year"] < meas_start_year) | (temp_df["year"] > meas_end_year),
                ["cost", "income", "net"],
            ] = 0

            # Add the name of the measure to the DataFrame
            temp_df["measure"] = meas.name

            # If the cash flows DataFrame is empty, set it to the DataFrame for the current measure
            # Otherwise, concatenate the DataFrame for the current measure to the existing DataFrame
            if costincome_df.empty:
                costincome_df = temp_df
            else:
                costincome_df = pd.concat([costincome_df, temp_df])

        # Reset the index of the DataFrame
        costincome_df = costincome_df.reset_index(drop=True)

    # Return the DataFrame with the calculated cash flows
    return costincome_df


# Calculate the cash flows for the combined measures
def calc_combo_measure_cash_flows_df(costincome_df, measure_set):

    # Calculate the cash flows for the combined measures dont reorder the columns
    costincome_combo_df = pd.DataFrame(columns=costincome_df.columns)

    # Loop over the measures in the set
    for _, meas in measure_set.measures().items():

        # If the measure is a combination of other measures, skip it
        if not meas.combo:
            continue
        else:
            # print(meas.name)
            # Get all the measures in the combo measure
            combo_measures = meas.combo
            # Get the sub df of the costincome_d
            sub_df = costincome_df[costincome_df["measure"].isin(combo_measures)]
            # For each year, sum the costs, incomes and net
            sub_df = sub_df.groupby("year").sum().reset_index()
            # Change the measure name to the combo measure name
            sub_df["measure"] = meas.name
            # Concatenate the sub_df to the costincome_combo_df
            if costincome_combo_df.empty:
                costincome_combo_df = sub_df
            else:
                costincome_combo_df = pd.concat([costincome_combo_df, sub_df])

    # Reset the index
    costincome_combo_df.reset_index(drop=True, inplace=True)

    # Concatenate the costincome_df and costincome_combo_df
    if costincome_combo_df.empty:
        return costincome_df
    else:
        costincome_df = pd.concat(
            [costincome_df, costincome_combo_df], ignore_index=True
        )

    return costincome_df


# Calculate the cash flows for a set of measures
def calc_measure_cash_flows_df(
    measure_set,
    measure_times_df,
    start_year,
    end_year,
    consider_measure_times=True,
    disc=None,
):

    # Calculate the individual measure cash flows
    costincome_df = calc_indv_measure_cash_flows_df(
        measure_set,
        measure_times_df,
        start_year,
        end_year,
        consider_measure_times,
        disc,
    )

    # Calculate the combo measure cash flows
    costincome_df = calc_combo_measure_cash_flows_df(costincome_df, measure_set)

    return costincome_df


# %% Cost benefit analysis functions


def calc_CB_df2(
    annual_risk_metrics_df,
    measure_set,
    measure_times_df,
    start_year=None,
    end_year=None,
    consider_measure_times=True,
    risk_disc=None,
    cost_disc=None,
):
    """
    This function calculates the cost-benefit analysis (CB) for a set of measures.

    Parameters:
    annual_risk_metrics_df: A DataFrame with the risk metrics for each measure.
    measure_set: A set of measures for which to calculate the CB.
    start_year: The first year of the analysis (can also be none = the minimum year of annual_risk_metrics_df).
    end_year: The last year of the analysis (can also be none = the maximum year of annual_risk_metrics_df).
    consider_measure_times: A boolean indicating if the measure times should be considered.
    risk_disc: The discount rate to apply to future risk metrics.
    cost_disc: The discount rate to apply to future costs.

    Returns:
    A DataFrame with the calculated cost-benefit analysis.
    """

    # Calculate the averted risk when considering the measure times without discounting
    if start_year is None:
        start_year = annual_risk_metrics_df["year"].min()
    if end_year is None:
        end_year = annual_risk_metrics_df["year"].max()

    # Step 1 - Filter the annual_risk_metrics_df based on the start_year and end_year
    filt_annual_risk_metrics_df = annual_risk_metrics_df[
        (annual_risk_metrics_df["year"] >= start_year)
        & (annual_risk_metrics_df["year"] <= end_year)
    ]

    # Step 2 - Create the modified annual_risk_metrics_df based on the measure times
    filt_annual_risk_metrics_df = create_meas_mod_annual_risk_metrics_df(
        filt_annual_risk_metrics_df,
        measure_set,
        measure_times_df,
        consider_measure_times,
    )

    # Step 3 - Calculate the NPV of the annual_risk_metrics_df to get total risk
    disc_filt_annual_risk_metrics_df, _ = calc_npv_annual_risk_metrics_df(
        filt_annual_risk_metrics_df, disc=risk_disc
    )

    # Get the base CB dataframe
    ann_CB_df = disc_filt_annual_risk_metrics_df[
        ["measure", "year", "group", "metric", "result"]
    ].copy()
    ann_CB_df.columns = ["measure", "year", "group", "metric", "total risk"]

    # Step 4 - Calculate the averted risk metrics
    averted_annual_risk_metrics_df = calc_averted_risk_metrics(
        disc_filt_annual_risk_metrics_df
    )
    # Rename the column 'result' to 'averted risk'
    averted_annual_risk_metrics_df = averted_annual_risk_metrics_df.rename(
        columns={"result": "averted risk"}
    )

    # Merge the averted risk metrics to the CB dataframe
    ann_CB_df = ann_CB_df.merge(
        averted_annual_risk_metrics_df,
        on=["measure", "year", "group", "metric"],
        how="left",
    )

    # Calculate the residual risk
    # ann_CB_df['residual risk'] = ann_CB_df['total risk'] - ann_CB_df['averted risk']

    # Calculate the measure cash flows
    costincome_df = calc_measure_cash_flows_df(
        measure_set,
        measure_times_df,
        start_year,
        end_year,
        consider_measure_times,
        disc=cost_disc,
    )

    # Merge the costincome_df with the ann_CB_df but only keep the 'net' column and rename it to 'cost (net)'
    ann_CB_df = ann_CB_df.merge(
        costincome_df[["measure", "year", "net"]], on=["measure", "year"], how="left"
    )
    ann_CB_df = ann_CB_df.rename(columns={"net": "cost (net)"})
    # For group that is not nan, fill the 'cost (net)' column with nan
    ann_CB_df.loc[~ann_CB_df["group"].isna(), "cost (net)"] = np.nan

    # Step 6 - Aggregate the results
    # Cast the 'group' column to string type and fill NaN values with a placeholder
    ann_CB_df["group"] = ann_CB_df["group"].astype(str).fillna("No Group")

    # Aggregate the results

    # Cast the 'group' column to string type and fill NaN values with a placeholder
    ann_CB_df["group"] = ann_CB_df["group"].astype(str).fillna("No Group")
    # Aggregate the results
    tot_CB_df = ann_CB_df.groupby(["measure", "group", "metric"]).sum().reset_index()
    # Drop the 'year' column
    tot_CB_df = tot_CB_df.drop(columns=["year"])

    # Add the average annual risk
    tot_CB_df["average annual risk"] = tot_CB_df["total risk"] / (
        end_year - start_year + 1
    )

    # Step 7 - Calculate the CB ratio
    # Calculate the CB ratio
    tot_CB_df["B/C ratio"] = tot_CB_df["averted risk"] / tot_CB_df["cost (net)"]

    return ann_CB_df, tot_CB_df


def generate_CB_plot_print_data(tot_CB_df, measure_set, metric="aai", group=np.nan):
    df_filtered = tot_CB_df[tot_CB_df["metric"] == metric]
    group = str(group)  # Cast the 'group' to string type
    df_filtered = df_filtered[df_filtered["group"] == group]

    # Get the total climate risk, annual risk, and residual risk
    total_climate_risk = df_filtered[df_filtered["measure"] == "no_measure"][
        "total risk"
    ].values[0]
    annual_risk = df_filtered[df_filtered["measure"] == "no_measure"][
        "average annual risk"
    ].values[0]
    residual_risk = df_filtered[df_filtered["measure"] != "no_measure"][
        "total risk"
    ].min()

    # Derive the unit for the values
    unit = value_to_monetary_unit(total_climate_risk)[1]
    divisor = {unit: divisor for divisor, unit in ABBREV.items()}[unit]

    # Prepare the measure data
    measure_data = df_filtered[df_filtered["measure"] != "no_measure"]
    measure_summary = []
    for measure in measure_set.measures():
        measure_df = measure_data[measure_data["measure"] == measure]
        total_cost = measure_df["cost (net)"].sum()
        total_benefit = measure_df["averted risk"].sum()
        bc_ratio = -1 * total_benefit / total_cost if total_cost != 0 else np.nan
        measure_summary.append(
            [measure, total_cost / divisor, total_benefit / divisor, bc_ratio]
        )

    # Create the measure summary DataFrame
    measure_summary_df = pd.DataFrame(
        measure_summary, columns=["Measure", f"Cost", f"Benefit", "Benefit/Cost Ratio"]
    )

    # Round the values in the measure summary DataFrame
    measure_summary_df = measure_summary_df.round(2)

    # Prepare and print the total summary with formatted output
    total_summary_dict = {
        "Total climate risk:": total_climate_risk / divisor,
        "Average annual risk:": annual_risk / divisor,
        "Residual risk:": residual_risk / divisor,
    }

    return measure_summary_df, total_summary_dict, unit
