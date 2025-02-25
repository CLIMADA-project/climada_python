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
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from climada.engine.option_appraisal.calc_impact_metrics import (
    CalcImpactMetrics,
    make_measure_snapshot,
)
from climada.engine.option_appraisal.impact_trajectories import (
    RiskPeriod,
    SnapshotsCollection,
)
from climada.entity.measures.measure_set import MeasureSet

LOGGER = logging.getLogger(__name__)


class OptionAppraiser:

    def __init__(
        self,
        snapshots: SnapshotsCollection,
        measure_set: MeasureSet | None = None,
        combo_all: bool = False,
        planner: dict[str, tuple[int, int]] | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
    ):
        "docstring"
        self._metrics_up_to_date: bool = False
        self.metrics = metrics
        self.return_periods = return_periods
        self.star_year = min(snapshots.snapshots_years)
        self.end_year = max(snapshots.snapshots_years)
        self.measure_set = copy.deepcopy(measure_set)
        self.planner = planner if planner else {}
        self._planning = None
        LOGGER.debug("Computing risk periods")
        self.risk_periods = [
            RiskPeriod(start_snapshot, end_snapshot)
            for start_snapshot, end_snapshot in snapshots.pairwise()
        ]
        self.measure_periods = []
        if self.measure_set:
            LOGGER.debug("Computing risk periods with measures")
            if combo_all:
                self.measure_set.append(self.measure_set.combine())

            if planner:
                self.calc_planning_measures()
            else:
                for name, measure in self.measure_set.measures().items():
                    LOGGER.debug(f"Creating measures snapshots for measure {measure}")
                    meas_snapshots = make_measure_snapshot(snapshots, measure)
                    LOGGER.debug(f"Creating measures riskperiods for measure {measure}")
                    meas_risk_period = [
                        RiskPeriod(start_snapshot, end_snapshot, name)
                        for start_snapshot, end_snapshot in meas_snapshots.pairwise()
                    ]
                    self.measure_periods += meas_risk_period

        self._impact_metrics_calculator = CalcImpactMetrics(
            self.risk_periods + self.measure_periods
        )
        # self.impact_metrics = self._impact_metrics_calculator.generate_impact_metrics(measure_set, planner=self.planner)

    def calc_planning_measures(self):
        if not self.planner:
            raise ValueError("No planner set.")

        self._planning = _get_unique_measure_periods(_planner_to_planning(self.planner))
        # For each planned period, find correponding risk periods and create the periods with measure from planning
        for measure_name_list, start_year, end_year in self._planning:
            # Not sure this works as intended (pbly could be simplified anyway)
            periods = self.get_risk_periods(start_year, end_year)
            meas_periods = [
                period.apply_measures(self.measure_set, measure_name_list)
                for period in periods
            ]
            self.measure_periods += meas_periods

    def get_risk_periods(self, start_year: int, end_year: int):
        return [
            period
            for period in self.risk_periods
            if (start_year >= period.start_year or end_year <= period.end_year)
        ]

    @property
    def annual_risk_metrics(self):
        if self._metrics_up_to_date:
            return self._annual_risk_metrics
        else:
            self.update_risk_metrics()
            return self._annual_risk_metrics

    def update_risk_metrics(
        self, compute_groups=False, risk_transf_cover=None, risk_transf_attach=None
    ):
        self._annual_risk_metrics = (
            self._impact_metrics_calculator.calc_risk_periods_metric(
                metrics=self.metrics,
                return_periods=self.return_periods,
                compute_groups=compute_groups,
                risk_transf_cover=risk_transf_cover,
                risk_transf_attach=risk_transf_attach,
            )
        )
        self._metrics_up_to_date = True

    def update_measure_set(
        self, measure_set: MeasureSet, planner: dict[str, tuple[int, int]] | None = None
    ):

        # update self.measure_set

        # recompute impact metrics
        pass

    def set_planner(self, planner: dict[str, tuple[int, int]]):
        # update self.planner

        # recompute impact metrics
        pass

    def measure_risk_metrics(self, measure_name):
        year_start, year_end = self.planner.get(
            measure_name, (self.star_year, self.end_year)
        )
        ret = self.annual_risk_metrics.copy()
        ret = ret.set_index(["measure", "year"]).sort_index()
        ret = ret.loc[pd.IndexSlice[measure_name, year_start:year_end],].reset_index()
        return ret

    @property
    def planning_risk_metrics(self):
        if not self._planning:
            raise ValueError("No planner set. Use `.set_planner()`.")

        df = self.annual_risk_metrics.copy()
        df = df.set_index(["measure", "year"]).sort_index()

        # Initiate mask a false everywhere
        mask = pd.Series(False, index=df.index)

        # for each measure set mask to true at corresponding year (bitwise or)
        for measure_combo, start, end in self._planning:
            mask |= (
                (df.index.get_level_values("measure") == "_".join(measure_combo))
                & (df.index.get_level_values("year") >= start)
                & (df.index.get_level_values("year") <= end)
            )

        # for the no_measure case, set mask to true at corresponding year,
        # only for those years that have no active measures
        no_measure_mask = mask.groupby("year").sum() == 0
        mask.loc[
            pd.IndexSlice["no_measure"], no_measure_mask[no_measure_mask].index
        ] = True

        return df[mask].reset_index()

    def calc_planning_cash_flows_df(self, disc=None):
        if not self._planning:
            raise ValueError("No planner set. Use `.set_planner()`.")

        res = []
        for measure, (start, end) in self.planner.items():
            df = self.measure_set.measures()[measure].cost_income.calc_cashflows_df(
                impl_year=start, start_year=start, end_year=end, disc=disc
            )
            df["measure"] = measure
            res.append(df)

        df = pd.concat(res)
        df = df.groupby("year", as_index=False).agg(
            {
                col: "sum" if is_numeric_dtype(df[col]) else lambda x: "_".join(x)
                for col in df.columns
                if col != "year"
            }
        )
        return df

    def calc_CB_df(
        self,
        start_year=None,
        end_year=None,
        consider_planning=True,
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
        if consider_planning:
            risk_metrics_df = self.planning_risk_metrics
        else:
            risk_metrics_df = self.annual_risk_metrics
        if start_year is None:
            start_year = self.star_year
        if end_year is None:
            end_year = self.end_year

        # Step 1 - Filter the annual_risk_metrics_df based on the start_year and end_year
        risk_metrics_df = risk_metrics_df[
            (risk_metrics_df["year"] >= start_year)
            & (risk_metrics_df["year"] <= end_year)
        ]

        # Step 3 - Calculate the NPV of the annual_risk_metrics_df to get total risk
        risk_metrics_df, _ = calc_npv_annual_risk_metrics_df(
            risk_metrics_df, disc=risk_disc
        )

        # Get the base CB dataframe
        ann_CB_df = risk_metrics_df[
            ["measure", "year", "group", "metric", "result"]
        ].copy()
        ann_CB_df.columns = ["measure", "year", "group", "metric", "total risk"]

        # Step 4 - Calculate the averted risk metrics
        averted_annual_risk_metrics_df = calc_averted_risk_metrics(risk_metrics_df)
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
        costincome_df = self.calc_planning_cash_flows_df(disc=cost_disc)

        # Merge the costincome_df with the ann_CB_df but only keep the 'net' column and rename it to 'cost (net)'
        ann_CB_df = ann_CB_df.merge(
            costincome_df[["measure", "year", "net"]],
            on=["measure", "year"],
            how="left",
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
        tot_CB_df = (
            ann_CB_df.groupby(["measure", "group", "metric"]).sum().reset_index()
        )
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


def _planner_to_planning(planner: dict[str, tuple[int, int]]) -> dict[int, list[str]]:
    """Transform a dictionary of measures with year spans into a dictionary
    where each key is a year, and the value is a list of active measures.

    Parameters
    ----------
    measures : dict[str, tuple[int, int]]
        Dictionary where keys are measure names and values are (start_year, end_year).

    Returns
    -------
    dict[int, list[str]]
        Dictionary where each key is a year, and the value is a list of active measures.
    """
    year_to_measures = defaultdict(list)

    for measure, (start, end) in planner.items():
        for year in range(start, end + 1):  # Include both start and end year
            year_to_measures[year].append(measure)

    return dict(year_to_measures)


def _get_unique_measure_periods(
    year_to_measures: dict[int, list[str]]
) -> list[tuple[list[str], int, int]]:
    """Extract unique measure lists with their corresponding min and max year.

    Parameters
    ----------
    year_to_measures : dict[int, list[str]]
        Dictionary where keys are years and values are lists of active measures.

    Returns
    -------
    list[tuple[list[str], int, int]]
        A list of tuples containing (unique measure list, min year, max year).
    """
    sorted_years = sorted(year_to_measures.keys())  # Ensure chronological order
    unique_periods = []

    current_measures = None
    start_year = None

    for year in sorted_years:
        measures = sorted(
            year_to_measures[year]
        )  # Sorting ensures consistency in comparison

        if measures != current_measures:  # New unique set detected
            if current_measures is not None:  # Save the previous period
                unique_periods.append((current_measures, start_year, year - 1))

            # Start a new period
            current_measures = measures
            start_year = year

    # Add the last recorded period
    if current_measures is not None:
        unique_periods.append((current_measures, start_year, sorted_years[-1]))

    return unique_periods


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

                # If the sub_df is empty, continue to the next iteration - Later raise an error that no data is available
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
