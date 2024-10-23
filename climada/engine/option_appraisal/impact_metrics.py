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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from climada.entity.measures import MeasureSet
from climada.util.value_representation import ABBREV, value_to_monetary_unit


class ImpactMetrics:

    def __init__(
        self,
        arm_df: pd.DataFrame,
        all_arms_df: Optional[pd.DataFrame] = None,
        measure_set: Optional[MeasureSet] = None,
        measure_times_df: Optional[pd.DataFrame] = None,
        value_unit: str = "USD",
    ):
        self.arm_df = arm_df.copy()
        self.all_arms_df = copy.deepcopy(all_arms_df)
        self.measure_set = copy.deepcopy(measure_set)
        self.measure_times_df = copy.deepcopy(measure_times_df)
        self.value_unit = value_unit

    ## Waterfall plot functions
    # Plot the waterfall plot
    def waterfall_plot(self, yearly=False, measure=None, metric="aai", group=np.nan):

        # Check if the all_arms_df is provided
        if self.all_arms_df is None:
            raise ValueError(
                "The all_arms_df is required for the waterfall plot. Please reggerate the all_arms_df using CalcImpactMetrics."
            )

        # Check if yearly or classic waterfall plot
        if yearly:
            # Calculate the averted risk metrics for yearly waterfall plot
            waterfall_df = _calc_waterfall_plot_df(
                self.arm_df,
                self.all_arms_df,
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
                self.arm_df,
                self.all_arms_df,
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
            self.arm_df,
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
        plot_df = self.arm_df.copy()

        # Check if the time should be considered
        if consider_measure_times:
            plot_df = create_meas_mod_arm_df(
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
            plot_df, _ = calc_npv_arm_df(plot_df, disc=risk_disc)
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

            # if actual_missing_combos rasing an error
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
        return self.arm_df.copy()

    def get_risk_metrics(self):
        return list(self.arm_df["metric"].unique())

    def get_measures(self):
        return list(self.measure_set.measures().keys())


# %% Utility functions


def plot_CB_summary(
    measure_summary_df,
    total_summary_dict,
    measure_colors=None,
    y_label="Risk (Bn)",
    title="Benefit and Benefit/Cost Ratio by Measure",
):
    # Calculate the Benefit/Cost Ratio
    measure_summary_df["Benefit/Cost Ratio"] = measure_summary_df["Benefit"] / abs(
        measure_summary_df["Cost"]
    )

    # Sort the DataFrame by 'Benefit' column
    measure_summary_df = measure_summary_df.sort_values(by="Benefit", ascending=True)

    # Retrieve total_climate_risk and residual_risk from the total_summary_dict
    total_climate_risk = total_summary_dict["Total climate risk:"]
    residual_risk = total_summary_dict["Residual risk:"]

    # Calculate the highest benefit
    highest_benefit = measure_summary_df["Benefit"].max()

    # Generate a color palette dynamically if not provided
    if measure_colors is None:
        colors = sns.color_palette("hsv", len(measure_summary_df))
        measure_colors = {
            measure: color
            for measure, color in zip(measure_summary_df["Measure"], colors)
        }

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plotting the Benefit as bars with specific colors
    bars = sns.barplot(
        x="Measure",
        y="Benefit",
        hue="Measure",
        data=measure_summary_df,
        ax=ax1,
        palette=measure_colors,
        dodge=False,
        legend=False,
    )
    ax1.set_ylabel(y_label, fontsize=18, fontweight="bold", color="black")
    ax1.set_xlabel("Measure", fontsize=14)

    # Adding color to the bars
    for bar, measure in zip(bars.patches, measure_summary_df["Measure"]):
        bar.set_color(measure_colors[measure])

    # Placing the benefit values on top of the bars
    for bar in bars.patches:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}",
            ha="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

    # Creating a second y-axis for Benefit/Cost Ratio
    ax2 = ax1.twinx()
    line = sns.lineplot(
        x="Measure",
        y="Benefit/Cost Ratio",
        data=measure_summary_df,
        ax=ax2,
        marker="o",
        color="red",
        linewidth=1,
    )
    ax2.set_ylabel("Benefit/Cost Ratio", color="red", fontsize=10)

    # Setting higher y-limit for the benefit/cost ratio axis
    ax2.set_ylim(0, max(measure_summary_df["Benefit/Cost Ratio"]) * 1.5)

    # Change the color of the right y-axis line, ticks, and labels
    ax2.spines["right"].set_color("red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Change the color of the left y-axis ticks and labels, and make them bigger and bold
    ax1.tick_params(axis="y", labelcolor="black", labelsize=14, width=2)
    ax1.spines["left"].set_linewidth(2)

    # Adding a red horizontal line at Benefit/Cost Ratio = 1 without a label
    ax2.axhline(1, color="red", linestyle="--", linewidth=2)

    # Adding a black horizontal line at Total Climate Risk value
    ax1.axhline(total_climate_risk, color="black", linestyle="--", linewidth=2)
    ax1.text(
        -0.52,
        total_climate_risk * 1.02,
        f"Total climate risk: {total_climate_risk:.2f}",
        color="black",
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Adding a grey arrow to indicate the residual risk, pointing upwards and moved to the right
    arrow_x = len(measure_summary_df) - 0.6  # Adjust x position for arrow
    arrow_y_end = total_climate_risk
    arrow_y_start = highest_benefit

    ax1.annotate(
        "",
        xy=(arrow_x, arrow_y_start),
        xycoords="data",
        xytext=(arrow_x, arrow_y_end),
        textcoords="data",
        arrowprops=dict(arrowstyle="<-", color="grey", lw=2),
    )

    # Positioning the residual risk text box so it touches the arrow on the left side
    bbox_props = dict(
        boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.5
    )
    ax1.text(
        arrow_x - 0.1,
        (arrow_y_end + arrow_y_start) / 2,
        f"Residual risk: {residual_risk:.2f}",
        color="grey",
        va="center",
        ha="right",
        fontsize=12,
        fontweight="bold",
        bbox=bbox_props,
    )

    # Adding title
    plt.title(title, fontsize=16)

    # Show plot
    plt.tight_layout()
    plt.show()


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
def _calc_waterfall_plot_df(
    arm_df,
    all_arms_df,
    measure=None,
    metric="aai",
    ref_year=None,
    fut_year=None,
    subtract=False,
    group=np.nan,
):
    # Check if the reference and future years are provided
    if ref_year is None:
        ref_year = arm_df["year"].min()
    if fut_year is None:
        fut_year = arm_df["year"].max()

    # Initialize the waterfall DataFrame
    waterfall_df = pd.DataFrame(
        columns=[
            "year",
            "metric",
            "ref_risk",
            "exp_change",
            "impfset_change",
            "haz_change",
            "risk",
            "group",
        ]
    )

    # Filter the arm_df
    filt_arm_df = arm_df[
        (arm_df["year"] >= ref_year)
        & (arm_df["year"] <= fut_year)
        & (arm_df["measure"] == "no_measure")
        & (arm_df["metric"] == metric)
        & ((pd.isna(group)) | (arm_df["group"] == group))
    ]

    # Filter the all_arms_df
    filt_all_arms_df = all_arms_df[
        (all_arms_df["year"] >= ref_year)
        & (all_arms_df["year"] <= fut_year)
        & (all_arms_df["measure"] == "no_measure")
        & (all_arms_df["metric"] == metric)
        & ((pd.isna(group)) | (all_arms_df["group"] == group))
    ]

    # Get the reference risk
    ref_risk = filt_arm_df[filt_arm_df["year"] == ref_year]["result"].values[0]

    # Calculate the waterfall columns
    for year in range(ref_year, fut_year + 1):
        curr_value = filt_arm_df[filt_arm_df["year"] == year]["result"].values[0]

        # Filter levels from all_arms_df
        # Change in exposure
        exp_change_level = filt_all_arms_df[
            (filt_all_arms_df["exp_change"] == True)
            & (filt_all_arms_df["impfset_change"] == False)
            & (filt_all_arms_df["haz_change"] == False)
            & (filt_all_arms_df["year"] == year)
        ]["result"].values[0]

        # Change in exposure and impfset
        impfset_change_level = filt_all_arms_df[
            (filt_all_arms_df["exp_change"] == True)
            & (filt_all_arms_df["impfset_change"] == True)
            & (filt_all_arms_df["haz_change"] == False)
            & (filt_all_arms_df["year"] == year)
        ]["result"].values[0]

        # Calculate the changes
        if subtract:
            exp_change = exp_change_level - ref_risk
            impfset_change = impfset_change_level - exp_change_level
            climate_change = curr_value - impfset_change_level
        else:
            exp_change = exp_change_level
            impfset_change = impfset_change_level
            climate_change = curr_value

        # Append the values to the waterfall_df
        temp_df = pd.DataFrame(
            [
                [
                    year,
                    group,
                    metric,
                    ref_risk,
                    exp_change,
                    impfset_change,
                    climate_change,
                    curr_value,
                ]
            ],
            columns=[
                "year",
                "group",
                "metric",
                "ref_risk",
                "exp_change",
                "impfset_change",
                "haz_change",
                "risk",
            ],
        )

        # Check if the waterfall_df is empty
        if waterfall_df.empty:
            waterfall_df = temp_df
        else:
            waterfall_df = pd.concat([waterfall_df, temp_df], ignore_index=True)

    # Add the risk for the measure
    if measure:
        # Get the measure risk
        filt_arm_df = arm_df[
            (arm_df["year"] >= ref_year)
            & (arm_df["year"] <= fut_year)
            & (arm_df["measure"] == measure)
            & (arm_df["metric"] == metric)
            & ((pd.isna(group)) | (arm_df["group"] == group))
        ]

        # Join the risk for the measure and add only the risk column and the measure name
        waterfall_df = waterfall_df.merge(
            filt_arm_df[["year", "measure", "result"]], on="year", how="left"
        )
        waterfall_df = waterfall_df.rename(columns={"result": "measure_risk"})
        # Make an additional column called averted risk
        waterfall_df["averted_risk"] = (
            waterfall_df["risk"] - waterfall_df["measure_risk"]
        )

    return waterfall_df


# Plot the yearly waterfall plot
def _plot_yearly_waterfall(waterfall_df, metric="aai", value_unit="USD"):
    fig, ax = plt.subplots(figsize=(15, 10))

    # Get the reference and future years
    ref_year = waterfall_df["year"].min()

    colors = ["blue", "orange", "green", "red"]
    labels = [f"Risk {ref_year}", "Exp Change", "Impfset Change", "Haz Change"]

    for year in waterfall_df["year"].unique():
        year_df = waterfall_df[waterfall_df["year"] == year]
        ref_risk = year_df["ref_risk"].values[0]
        exp_change = year_df["exp_change"].values[0]
        impfset_change = year_df["impfset_change"].values[0]
        haz_change = year_df["haz_change"].values[0]

        values = [ref_risk, exp_change, impfset_change, haz_change]

        # Calculate cumulative values for positive values only
        cum_values = [0]
        for i in range(1, len(values)):
            if values[i - 1] >= 0:
                cum_values.append(cum_values[i - 1] + values[i - 1])
            else:
                cum_values.append(cum_values[i - 1])

        # Plot cumulative positive values
        for i in range(len(values)):
            if values[i] >= 0:
                ax.bar(
                    year,
                    values[i],
                    bottom=cum_values[i],
                    color=colors[i],
                    label=labels[i] if year == waterfall_df["year"].min() else "",
                )
            else:
                # Placeholder for negative values to adjust cum_values
                cum_values[i + 1] = cum_values[i]

        # Plot negative values separately
        neg_cum_values = [0]
        for i in range(1, len(values)):
            if values[i - 1] < 0:
                neg_cum_values.append(neg_cum_values[i - 1] + values[i - 1])
            else:
                neg_cum_values.append(neg_cum_values[i - 1])

        for i in range(len(values)):
            if values[i] < 0:
                ax.bar(
                    year,
                    values[i],
                    bottom=neg_cum_values[i],
                    color=colors[i],
                    label=labels[i] if year == waterfall_df["year"].min() else "",
                )
                # Adjust cumulative values
                if i < len(values) - 1:
                    neg_cum_values[i + 1] += values[i]

    ax.axhline(
        0, color="black", linewidth=0.8, linestyle="--"
    )  # Add a horizontal grid line at zero

    # Construct y-axis label and title based on parameters
    value_label = f"Risk {metric} ({value_unit})"
    title_label = f"Yearly Waterfall Plot ({metric})"

    ax.set_title(title_label)
    ax.set_xlabel("Year")
    ax.set_ylabel(value_label)

    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plot the waterfall plot for two years
def _plot_two_years_waterfall(waterfall_df, metric="aai", value_unit="USD"):
    # Get the reference and future years
    ref_year = waterfall_df["year"].min()
    fut_year = waterfall_df["year"].max()

    # Filter the DataFrame for the specified years
    ref_data = waterfall_df[waterfall_df["year"] == ref_year].iloc[0]
    fut_data = waterfall_df[waterfall_df["year"] == fut_year].iloc[0]

    # Extract values for the waterfall plot
    ref_risk = ref_data["ref_risk"]
    exp_change = fut_data["exp_change"]
    impfset_change = fut_data["impfset_change"]
    haz_change = fut_data["haz_change"]
    fut_risk = fut_data["risk"]

    # Calculate the intermediate values
    exp_change_diff = exp_change - ref_risk
    impfset_change_diff = impfset_change - exp_change
    haz_change_diff = haz_change - impfset_change

    values = [
        ref_risk,
        exp_change_diff,
        impfset_change_diff,
        haz_change_diff,
        fut_risk - (ref_risk + exp_change_diff + impfset_change_diff + haz_change_diff),
    ]
    labels = [
        f"Risk {ref_year}",
        "Exposure change",
        "Vulnerability change",
        "Climate change",
        f"Risk {fut_year}",
    ]
    colors = ["blue", "orange", "green", "red", "purple"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the initial reference risk
    ax.bar(labels[0], values[0], color=colors[0], edgecolor="black")

    # Plot cumulative changes
    cum_value = values[0]
    for i in range(1, len(values) - 1):
        cum_value += values[i]
        ax.bar(
            labels[i],
            values[i],
            bottom=cum_value - values[i],
            color=colors[i],
            edgecolor="black",
        )

    # Plot the final future risk
    ax.bar(labels[-1], fut_risk, color=colors[-1], edgecolor="black")

    # Adding labels to the bars
    cum_value = values[0]
    for i in range(len(values)):
        if i == 0:
            ax.text(
                labels[i],
                values[i],
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )
        elif i == len(values) - 1:
            ax.text(
                labels[i],
                fut_risk,
                f"{fut_risk:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            cum_value += values[i]
            ax.text(
                labels[i],
                cum_value,
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )

    if "measure_risk" in waterfall_df.columns:
        measure_risk = fut_data["measure_risk"]

        # Add an arrow indicating averted risk
        ax.annotate(
            "",
            xy=(labels[-1], measure_risk),
            xytext=(labels[-1], fut_risk),
            arrowprops=dict(
                facecolor="green", shrink=0.05, width=5, headwidth=20, headlength=10
            ),
        )

        # Place the text in the center of the arrow
        arrow_midpoint = measure_risk * 0.98
        ax.text(
            labels[-1],
            arrow_midpoint,
            "Averted Risk",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )

    # Construct y-axis label and title based on parameters
    value_label = f"{metric} ({value_unit})"
    title_label = f"Risk at {ref_year} and {fut_year} ({metric}) for measure"

    ax.set_title(title_label)
    ax.set_ylabel(value_label)

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout for better fit

    plt.show()


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
def calc_averted_risk_metrics(arm_df):
    # Copy the DataFrame to avoid modifying the original
    averted_arm_df = pd.DataFrame(columns=arm_df.columns)

    # Get the unique groups and metrics
    groups = arm_df["group"].unique()
    metrics = arm_df["metric"].unique()

    # Iterate over each combination of group and metric
    for group in groups:
        for metric in metrics:
            # Filter the DataFrame for the current group and metric
            if pd.isna(group):
                sub_df = arm_df[(arm_df["group"].isna()) & (arm_df["metric"] == metric)]
            else:
                sub_df = arm_df[
                    (arm_df["group"] == group) & (arm_df["metric"] == metric)
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
                if averted_arm_df.empty:
                    averted_arm_df = sub_averted_risk_df
                else:
                    averted_arm_df = pd.concat(
                        [averted_arm_df, sub_averted_risk_df], ignore_index=True
                    )

    # Drop duplicates and reset the index
    averted_arm_df = averted_arm_df.drop_duplicates().reset_index(drop=True)

    return averted_arm_df


# Plot the risk metrics
def plot_risk_metrics(
    arm_df,
    metric="aai",
    group=np.nan,
    averted=False,
    discounted=False,
    plot_type="line",
    measures=None,
    value_unit="USD",
):
    # Filter the DataFrame
    if pd.isna(group):
        filt_arm_df = arm_df[(arm_df["group"].isna()) & (arm_df["metric"] == metric)]
    else:
        filt_arm_df = arm_df[(arm_df["group"] == group) & (arm_df["metric"] == metric)]

    # Filter the measures
    if measures:
        filt_arm_df = filt_arm_df[filt_arm_df["measure"].isin(measures)]

    # Ensure 'year' is treated as a categorical variable for bar plots
    filt_arm_df["year"] = filt_arm_df["year"].astype(str)

    # Create the plot
    plt.figure(figsize=(15, 8))

    if plot_type == "bar":
        sns.barplot(
            data=filt_arm_df,
            x="year",
            y="result",
            hue="measure",
            errorbar=None,
            alpha=0.7,
        )
    else:
        sns.lineplot(data=filt_arm_df, x="year", y="result", hue="measure")

    # Make y-axis from 0 to max aai with margin
    plt.ylim(0, filt_arm_df["result"].max() * 1.1)

    # Tilt the x-axis labels for better readability and display every nth year
    n = max(
        1, len(filt_arm_df["year"].unique()) // 20
    )  # Display every nth year, with a maximum of 20 labels
    plt.xticks(ticks=np.arange(0, len(filt_arm_df["year"].unique()), n), rotation=45)

    # More compressed label construction
    label_prefix = (
        f"{'Discounted ' if discounted else ''}{'Averted ' if averted else ''}"
    )
    plt.ylabel(f"{label_prefix}{metric} ({value_unit})")
    plt.title(
        f"{label_prefix}{metric} per Year"
        + (f" for Group: {group}" if not pd.isna(group) else "")
    )

    # Move the legend to the right side, outside the plot
    plt.legend(
        title="Measure", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
    )

    plt.tight_layout()
    plt.show()


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


# Calculate the NPV for the arm_df
def calc_npv_arm_df(df, disc=None):
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


# %% Functions for the modified arm_df


# Create the modified arm_df for the individual measures
def create_indv_meas_mod_arm_df(
    arm_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Use start_year and end_year if they are provided and filter the arm_df
    start_year = arm_df["year"].min()
    end_year = arm_df["year"].max()

    # Filter the arm_df
    time_arm_df = arm_df.copy()

    # If the measure times should be considered
    if not consider_measure_times:
        return time_arm_df

    # Iterate over the measures in the measure set
    # Use start_year and end_year if they are provided and filter the arm_df
    time_arm_df = time_arm_df[time_arm_df["measure"] == "no_measure"].copy()

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
                mask = (arm_df["measure"] == measure_name) & (arm_df["year"] == year)
                active_measure_df = arm_df.loc[mask]
                time_arm_df = pd.concat(
                    [time_arm_df, active_measure_df], ignore_index=True
                )
            else:
                # If the measure is not active this year, replace its values with the 'no_measure' values
                for metric in arm_df["metric"].unique():
                    for group_in in arm_df["group"].unique():
                        # Create a mask for the 'no_measure' rows for this year, metric, and group
                        mask = (
                            (arm_df["measure"] == "no_measure")
                            & (arm_df["metric"] == metric)
                            & (arm_df["year"] == year)
                            & (
                                (pd.isna(group_in) & arm_df["group"].isna())
                                | (arm_df["group"] == group_in)
                            )
                        )

                        # Get the 'no_measure' result for this year, metric, and group
                        result = arm_df.loc[mask, "result"].values
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
                        time_arm_df = pd.concat(
                            [time_arm_df, new_row], ignore_index=True
                        )

    # Remove duplicates and reset the index
    time_arm_df = time_arm_df.drop_duplicates().reset_index(drop=True)

    return time_arm_df


# Create the modified arm_df for the combo measures
def create_meas_combo_time_mod_arm_df(
    arm_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Calculate the risk for the combo measures
    # Use start_year and end_year if they are provided and filter the arm_df
    start_year = arm_df["year"].min()
    end_year = arm_df["year"].max()

    # Filter the arm_df
    time_arm_df = pd.DataFrame(columns=arm_df.columns)

    # # If the measure times should be considered
    if not consider_measure_times:
        return time_arm_df

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
            sub_df = arm_df[
                (arm_df["measure"] == measure_to_use) & (arm_df["year"] == year)
            ]

            # Update the measure name to the combo measure name
            sub_df["measure"] = measure_name

            # Append the values to the time_arm_df
            if time_arm_df.empty:
                time_arm_df = sub_df
            else:
                time_arm_df = pd.concat([time_arm_df, sub_df], ignore_index=True)

    return time_arm_df


# Create the modified arm_df
def create_meas_mod_arm_df(
    arm_df, measure_set, measure_times_df, consider_measure_times=True
):

    # Calculate the risk for the individual measures
    mod_arm_indv_meas_df = create_indv_meas_mod_arm_df(
        arm_df, measure_set, measure_times_df, consider_measure_times
    )

    # Calculate the risk for the combo measures
    mod_arm_combo_meas_df = create_meas_combo_time_mod_arm_df(
        arm_df, measure_set, measure_times_df, consider_measure_times
    )

    # Concatenate the DataFrames
    if mod_arm_combo_meas_df.empty:
        mod_arm_df = mod_arm_indv_meas_df
    else:
        mod_arm_df = pd.concat(
            [mod_arm_indv_meas_df, mod_arm_combo_meas_df], ignore_index=True
        )

    # Remove duplicates and reset the index
    mod_arm_df = mod_arm_df.drop_duplicates().reset_index(drop=True)

    return mod_arm_df


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


def calc_CB_df(
    arm_df,
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
    arm_df: A DataFrame with the risk metrics for each measure.
    measure_set: A set of measures for which to calculate the CB.
    start_year: The first year of the analysis (can also be none = the minimum year of arm_df).
    end_year: The last year of the analysis (can also be none = the maximum year of arm_df).
    consider_measure_times: A boolean indicating if the measure times should be considered.
    risk_disc: The discount rate to apply to future risk metrics.
    cost_disc: The discount rate to apply to future costs.

    Returns:
    A DataFrame with the calculated cost-benefit analysis.
    """

    # Calculate the averted risk when considering the measure times without discounting
    if start_year is None:
        start_year = arm_df["year"].min()
    if end_year is None:
        end_year = arm_df["year"].max()

    # Step 1 - Filter the arm_df based on the start_year and end_year
    filt_arm_df = arm_df[(arm_df["year"] >= start_year) & (arm_df["year"] <= end_year)]

    # Step 2 - Create the modified arm_df based on the measure times
    filt_arm_df = create_meas_mod_arm_df(
        filt_arm_df, measure_set, measure_times_df, consider_measure_times
    )

    # Step 3 - Calculate the NPV of the arm_df to get total risk
    disc_filt_arm_df, _ = calc_npv_arm_df(filt_arm_df, disc=risk_disc)

    # Get the base CB dataframe
    ann_CB_df = disc_filt_arm_df[
        ["measure", "year", "group", "metric", "result"]
    ].copy()
    ann_CB_df.columns = ["measure", "year", "group", "metric", "total risk"]

    # Step 4 - Calculate the averted risk metrics
    averted_arm_df = calc_averted_risk_metrics(disc_filt_arm_df)
    # Rename the column 'result' to 'averted risk'
    averted_arm_df = averted_arm_df.rename(columns={"result": "averted risk"})

    # Merge the averted risk metrics to the CB dataframe
    ann_CB_df = ann_CB_df.merge(
        averted_arm_df, on=["measure", "year", "group", "metric"], how="left"
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


def plot_yearly_averted_cost(
    ann_CB_df,
    measure,
    metric="aai",
    group=None,
    averted=False,
    discounted=False,
    value_unit="USD",
):
    # Filter the dataframe
    df_filtered = ann_CB_df[
        (ann_CB_df["measure"] == measure) & (ann_CB_df["metric"] == metric)
    ]

    if group is not None:
        df_filtered = df_filtered[df_filtered["group"] == group]

    # Replace None with 0 in 'cost (net)' for plotting
    df_filtered["cost (net)"] = df_filtered["cost (net)"].fillna(0)

    # Separate positive and negative values for stacking correctly
    positive_averted_risks = df_filtered["averted risk"].clip(lower=0)
    negative_averted_risks = df_filtered["averted risk"].clip(upper=0)
    positive_net_costs = df_filtered["cost (net)"].clip(lower=0)
    negative_net_costs = df_filtered["cost (net)"].clip(upper=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    # Construct y-axis label and title based on parameters
    label_prefix = ("Discounted " if discounted else "") + (
        "Averted " if averted else ""
    )
    value_label = f"{label_prefix} {metric} ({value_unit})"
    title_label = (
        f"Yearly {label_prefix} Risk ({metric}) vs Net Cost for measure: {measure}"
    )

    # Plot the averted risk and net cost
    years = df_filtered["year"].unique()

    # Plot positive values
    ax.bar(years, positive_averted_risks, color="green", label=label_prefix + "Risk")
    ax.bar(
        years,
        positive_net_costs,
        bottom=positive_averted_risks,
        color="red",
        label=label_prefix + "Net Cost",
    )

    # Plot negative values
    ax.bar(years, negative_averted_risks, color="green")
    ax.bar(years, negative_net_costs, bottom=negative_averted_risks, color="red")

    # Add labels and title
    ax.set_ylabel(value_label)
    ax.set_title(title_label)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2
    )  # Move legend to top
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")  # Improve x-axis readability

    # Add grid line at y=0
    ax.axhline(0, color="black", linewidth=0.5)

    # Set the y-axis limit as the maximum of the absolute values of the averted risk and net cost
    y_lim = (
        max(
            abs(df_filtered["averted risk"]).max(), abs(df_filtered["cost (net)"]).max()
        )
        * 1.1
    )
    ax.set_ylim(-y_lim, y_lim)

    plt.tight_layout()
    plt.show()
