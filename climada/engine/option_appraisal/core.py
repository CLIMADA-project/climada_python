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
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet

LOGGER = logging.getLogger(__name__)


class YearlyRiskAppraiser:

    _grouper = ["group", "metric"]

    def __init__(
        self,
        snapshots: SnapshotsCollection,
        risk_disc: DiscRates | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
    ):
        "docstring"
        self._metrics_up_to_date: bool = False
        self.metrics = metrics
        self.return_periods = return_periods
        self.start_year = min(snapshots.snapshots_years)
        self.end_year = max(snapshots.snapshots_years)
        self.risk_disc = risk_disc
        LOGGER.debug("Computing risk periods")
        self.risk_periods = self._calc_risk_periods(snapshots)
        self._impact_metrics_calculator = CalcImpactMetrics(self.risk_periods)
        # Here we can do some change in the future to include groups and risk transfer
        self._update_risk_metrics(
            compute_groups=False, risk_transf_cover=None, risk_transf_attach=None
        )

    def _calc_risk_periods(self, snapshots):
        return [
            RiskPeriod(start_snapshot, end_snapshot)
            for start_snapshot, end_snapshot in snapshots.pairwise()
        ]

    def _update_risk_metrics(
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

    @staticmethod
    def _get_risk_periods(risk_periods, start_year: int, end_year: int):
        return [
            period
            for period in risk_periods
            if (start_year >= period.start_year or end_year <= period.end_year)
        ]

    def _calc_annual_risk_metrics(self, npv=True):
        def npv_transform(group):
            start_year = group.index.get_level_values("year").min()
            end_year = group.index.get_level_values("year").max()
            return calc_npv_cash_flows(
                group.values, start_year, end_year, self.risk_disc
            )

        if self._metrics_up_to_date:
            df = self._annual_risk_metrics
        else:
            self._update_risk_metrics()
            df = self._annual_risk_metrics

        if npv:
            df = df.set_index("year")
            df["risk"] = df.groupby(
                self._grouper,
                dropna=False,
                as_index=False,
                group_keys=False,
            )["risk"].transform(npv_transform)
            df = df.reset_index()

        return df

    @classmethod
    def _calc_periods_risk(cls, df: pd.DataFrame, colname="risk"):
        def identify_continuous_periods(group):
            # Calculate the difference between consecutive years
            group["year_diff"] = group["year"].diff()
            # Identify breaks in continuity
            group["period_id"] = (group["year_diff"] != 1).cumsum()
            return group

        df_sorted = df.sort_values(by=cls._grouper + ["year"])
        # Apply the function to identify continuous periods
        df_periods = df_sorted.groupby(
            cls._grouper, dropna=False, group_keys=False
        ).apply(identify_continuous_periods)

        # Group by the identified periods and calculate start and end years
        df_periods = (
            df_periods.groupby(cls._grouper + ["period_id"], dropna=False)
            .agg(
                start_year=pd.NamedAgg(column="year", aggfunc="min"),
                end_year=pd.NamedAgg(column="year", aggfunc="max"),
                total=pd.NamedAgg(column=colname, aggfunc="sum"),
            )
            .reset_index()
        )

        df_periods["period"] = (
            df_periods["start_year"].astype(str)
            + "-"
            + df_periods["end_year"].astype(str)
        )
        df_periods = df_periods.rename(columns={"total": f"{colname}"})
        return df_periods.drop(["period_id", "start_year", "end_year"], axis=1)

    def calc_risk_metrics(self, total=False, npv=True):
        df = self._calc_annual_risk_metrics(npv=npv)
        if total:
            return self._calc_periods_risk(df)
        else:
            return df


class MeasuresAppraiser(YearlyRiskAppraiser):
    # TODO: To reflect on:
    # - Do we want "_planned", "_npv", "_total", "_single_measure" as parameter attributes instead of arguments?
    # - Do we keep "combo_all" ?
    _grouper = ["measure", "group", "metric"]

    def __init__(
        self,
        snapshots: SnapshotsCollection,
        measure_set: MeasureSet,
        cost_disc: DiscRates | None = None,
        risk_disc: DiscRates | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
    ):
        "docstring"
        self.cost_disc = cost_disc
        self.measure_set = copy.deepcopy(measure_set)
        super().__init__(snapshots, risk_disc, metrics, return_periods)

    def _calc_risk_periods(self, snapshots):
        risk_periods = super()._calc_risk_periods(snapshots)
        risk_periods += self._calc_measure_periods(risk_periods)
        return risk_periods

    def _calc_measure_periods(self, risk_periods):
        res = []
        for _, measure in self.measure_set.measures().items():
            LOGGER.debug(f"Creating measures risk_period for measure {measure.name}")
            meas_p = [
                risk_period.apply_measure(measure) for risk_period in risk_periods
            ]
            res += meas_p
        return res

    def _single_measure_risk_metrics(self, measure_name):
        """Not currently used"""
        ret = self._calc_annual_risk_metrics().copy()
        ret = ret.set_index(["measure", "year"]).sort_index()
        ret = ret.loc[pd.IndexSlice[measure_name, :],].reset_index()
        return ret

    def _calc_per_measure_annual_averted_risk(self, npv=True):
        # We want super() because we want all years not just planning
        no_measure = super()._calc_annual_risk_metrics(npv=npv)
        no_measure = no_measure[no_measure["measure"] == "no_measure"].copy()

        def subtract_no_measure(group):
            # Merge with no_measure to get the corresponding "no_measure" value
            merged = group.merge(
                no_measure, on=["group", "metric", "year"], suffixes=("", "_no_measure")
            )
            # Subtract the "no_measure" risk from the current risk
            merged["risk"] = merged["risk_no_measure"] - merged["risk"]
            return merged[group.columns]

        averted = self._calc_annual_risk_metrics(npv=npv).copy()
        averted = averted.groupby(
            ["group", "metric", "year"], group_keys=False, dropna=False
        ).apply(subtract_no_measure)
        averted = averted.rename(columns={"risk": "averted risk"})
        return averted

    def calc_averted_risk(self, total=False, npv=True):
        df = self._calc_per_measure_annual_averted_risk(npv=npv)
        if total:
            return self._calc_periods_risk(df, colname="averted risk")
        else:
            return df

    def calc_cash_flow(self, total=False):
        df = self._calc_per_measure_annual_cash_flows()
        if total:
            return self._calc_periods_cashflow(df)
        else:
            return df

    @staticmethod
    def _calc_periods_cashflow(df: pd.DataFrame):
        def identify_continuous_periods(group):
            # Calculate the difference between consecutive years
            group["year_diff"] = group["year"].diff()
            # Identify breaks in continuity
            group["period_id"] = (group["year_diff"] != 1).cumsum()
            return group

        # Apply the function to identify continuous periods
        df_periods = df.groupby(
            ["measure", "year"], dropna=False, group_keys=False
        ).apply(identify_continuous_periods)

        # Group by the identified periods and calculate start and end years
        df_periods = (
            df_periods.groupby(["measure", "period_id"], dropna=False)
            .agg(
                start_year=pd.NamedAgg(column="year", aggfunc="min"),
                end_year=pd.NamedAgg(column="year", aggfunc="max"),
                net=pd.NamedAgg(column="net", aggfunc="sum"),
                cost=pd.NamedAgg(column="cost", aggfunc="sum"),
                income=pd.NamedAgg(column="income", aggfunc="sum"),
            )
            .reset_index()
        )

        df_periods["period"] = (
            df_periods["start_year"].astype(str)
            + "-"
            + df_periods["end_year"].astype(str)
        )
        return df_periods.drop(["period_id", "start_year", "end_year"], axis=1)

    def _calc_per_measure_annual_cash_flows(self):
        res = []
        for meas_name, measure in self.measure_set.measures().items():
            df = measure.cost_income.calc_cashflows(
                impl_year=self.start_year,
                start_year=self.start_year,
                end_year=self.end_year,
                disc=self.cost_disc,
            )
            df["measure"] = meas_name
            res.append(df)
        df = pd.concat(res)
        return df

    def calc_CB(
        self,
        net_present_value=True,
        yearly=False,
    ):
        """
        This function calculates the cost-benefit analysis (CB) for a set of measures.

        Parameters:
        annual_risk_metrics: A DataFrame with the risk metrics for each measure.
        measure_set: A set of measures for which to calculate the CB.
        start_year: The first year of the analysis (can also be none = the minimum year of annual_risk_metrics).
        end_year: The last year of the analysis (can also be none = the maximum year of annual_risk_metrics).
        consider_measure_times: A boolean indicating if the measure times should be considered.
        risk_disc: The discount rate to apply to future risk metrics.
        cost_disc: The discount rate to apply to future costs.

        Returns:
        A DataFrame with the calculated cost-benefit analysis.
        """

        # Calculate the averted risk when considering the measure times without discounting
        # For group that is not nan, fill the 'cost (net)' column with nan
        # find why?
        # ann_CB.loc[~ann_CB["group"].isna(), "cost (net)"] = np.nan

        # Step 6 - Aggregate the results
        # Cast the 'group' column to string type and fill NaN values with a placeholder
        # ann_CB["group"] = ann_CB["group"].astype(str).fillna("No Group")

        # Aggregate the results

        # Cast the 'group' column to string type and fill NaN values with a placeholder
        # ann_CB["group"] = ann_CB["group"].astype(str).fillna("No Group")
        # Aggregate the results
        time_grouper = "year" if yearly else "period"

        res = self.calc_risk_metrics(
            total=(not yearly),
            npv=net_present_value,
        )
        res = res.merge(
            self.calc_averted_risk(
                total=(not yearly),
                npv=net_present_value,
            ),
            on=["measure", time_grouper, "group", "metric"],
            how="left",
        )
        res["residual risk"] = res["risk"] - res["averted risk"]

        res = res.merge(
            self.calc_cash_flow(
                total=(not yearly),
            )[["measure", time_grouper, "net"]],
            on=["measure", time_grouper],
            how="left",
        )

        res["net"] *= -1
        res["net"] = res["net"].fillna(0.0)
        res = res.rename(columns={"net": "cost (net)", "risk": "base risk"})
        res["group"] = res["group"].fillna("No Group")
        if not yearly:
            period_extracted = res["period"].str.extract(r"(\d{4})-(\d{4})").astype(int)
            start_year = period_extracted[0]
            end_year = period_extracted[1]
            years_span = end_year - start_year + 1
            res["average annual base risk"] = res["base risk"] / years_span
            res["average annual residual risk"] = res["residual risk"] / years_span
            res["average annual averted risk"] = res["averted risk"] / years_span
            res["average annual cost"] = res["cost (net)"] / years_span

        res["B/C ratio"] = (res["averted risk"] / res["cost (net)"]).fillna(0.0)
        first_cols = [
            "measure",
            time_grouper,
            "group",
            "metric",
        ]
        res = res.sort_values(time_grouper)[
            first_cols + [col for col in res.columns if col not in first_cols]
        ]
        return res


class PlannedMeasuresAppraiser(MeasuresAppraiser):

    def __init__(
        self,
        snapshots: SnapshotsCollection,
        measure_set: MeasureSet,
        planner: dict[str, tuple[int, int]],
        cost_disc: DiscRates | None = None,
        risk_disc: DiscRates | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
    ):
        self.planner = planner
        self._planning = _get_unique_measure_periods(_planner_to_planning(self.planner))
        super().__init__(
            snapshots,
            measure_set,
            cost_disc,
            risk_disc,
            metrics,
            return_periods,
        )

    def _single_measure_risk_metrics(self, measure_name):
        """Not currently used"""
        year_start, year_end = self.planner.get(
            measure_name, (self.start_year, self.end_year)
        )
        ret = self._calc_annual_risk_metrics().copy()
        ret = ret.set_index(["measure", "year"]).sort_index()
        ret = ret.loc[pd.IndexSlice[measure_name, year_start:year_end],].reset_index()
        return ret

    def _calc_measure_periods(self, risk_periods):
        # For each planned period, find correponding risk periods and create the periods with measure from planning
        res = []
        for measure_name_list, start_year, end_year in self._planning:
            # Not sure this works as intended (pbly could be simplified anyway)
            measure = self.measure_set.combine(names=measure_name_list)
            periods = self._get_risk_periods(risk_periods, start_year, end_year)
            meas_periods = [period.apply_measure(measure) for period in periods]
            res += meas_periods
        return res

    def _calc_annual_risk_metrics(self, npv=True):
        df = super()._calc_annual_risk_metrics(npv)
        df = df.set_index(["measure", "year"]).sort_index()
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

        columns_to_front = ["measure", "group", "year", "metric"]
        return (
            df[mask]
            .reset_index()
            .sort_values("year")[
                columns_to_front
                + [
                    col
                    for col in df.columns
                    if col not in columns_to_front + ["risk", "rp"]
                ]
                + ["risk"]
            ]
        )

    def _calc_per_measure_annual_cash_flows(self, disc=None):
        res = []
        for measure, (start, end) in self.planner.items():
            df = self.measure_set.measures()[measure].cost_income.calc_cashflows(
                impl_year=start, start_year=start, end_year=end, disc=disc
            )
            df["measure"] = measure
            res.append(df)

        df = pd.concat(res)
        df = df.groupby("year", as_index=False).agg(
            {
                col: ("sum" if is_numeric_dtype(df[col]) else lambda x: "_".join(x))
                for col in df.columns
                if col != "year"
            }
        )
        return df


class AdaptationPlansAppraiser:
    def __init__(
        self,
        snapshots: SnapshotsCollection,
        measure_set: MeasureSet,
        plans: list[dict[str, tuple[int, int]]],
        use_net_present_value: bool = True,
        cost_disc: DiscRates | None = None,
        risk_disc: DiscRates | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
    ):
        self._use_npv = use_net_present_value
        self.plans = [
            PlannedMeasuresAppraiser(
                snapshots=snapshots,
                measure_set=measure_set,
                planner=plan,
                cost_disc=cost_disc,
                risk_disc=risk_disc,
                metrics=metrics,
                return_periods=return_periods,
            )
            for plan in plans
        ]

    def calc_CB(
        self,
    ):
        res = []
        for plan in self.plans:
            planner = plan.planner
            df = plan.calc_CB(net_present_value=self._use_npv, yearly=True)
            df = df.drop("year", axis=1)
            df = df.groupby(["group", "metric"], as_index=False).sum(numeric_only=True)
            df["plan"] = format_periods_dict(planner)
            res.append(df)

        return (
            pd.concat(res)
            .set_index(["plan", "group", "metric"])
            .reset_index()
            .sort_values(["metric", "plan"])
        )


def format_periods_dict(periods_dict):
    formatted_string = ""
    for measure, (start_year, end_year) in periods_dict.items():
        formatted_string += f"{measure}: {start_year} - {end_year} ; "
    return formatted_string.strip()


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


def calc_npv_annual_risk_metrics(df, disc=None):
    # Copy the DataFrame
    disc = df.copy()
    npv = pd.DataFrame(columns=df.columns)

    for meas_name in disc["measure"].unique():
        for metric in disc["metric"].unique():
            for group_in in disc["group"].unique():

                # Handle NaN values in the 'group' column
                if pd.isna(group_in):
                    sub = disc[
                        (disc["measure"] == meas_name)
                        & (disc["metric"] == metric)
                        & (disc["group"].isna())
                    ]
                else:
                    sub = disc[
                        (disc["measure"] == meas_name)
                        & (disc["metric"] == metric)
                        & (disc["group"] == group_in)
                    ]

                # If the sub is empty, continue to the next iteration - Later raise an error that no data is available
                if sub.empty:
                    continue

                cash_flows = sub["risk"].values
                start_year = sub["year"].min()
                end_year = sub["year"].max()

                # Calculate the discounted cash flows and the total NPV
                npv_cash_flows, total_NPV = calc_npv_cash_flows(
                    cash_flows, start_year, end_year, disc
                )

                # Update the 'risk' column with the discounted cash flows
                disc.loc[sub.index, "risk"] = npv_cash_flows

                # Append the total NPV to the npv DataFrame
                npv_row = sub.iloc[0].copy()
                npv_row["risk"] = total_NPV
                npv = pd.concat([npv, pd.DataFrame(npv_row).T])
                # Reset the index
                npv.reset_index(drop=True, inplace=True)

    # Drop the 'year' column from npv
    npv = npv.drop(columns=["year"])

    # Drop duplicates and reset the index
    disc = disc.drop_duplicates().reset_index(drop=True)
    npv = npv.drop_duplicates().reset_index(drop=True)

    return disc, npv


def calc_npv_cash_flows(cash_flows, start_year, end_year=None, disc=None):
    # If no discount rates are provided, return the cash flows as is
    if not disc:
        return cash_flows

    # Determine the end year if not provided
    end_year = end_year or (start_year + len(cash_flows) - 1)

    # Generate an array of years
    years = np.arange(start_year, end_year + 1)

    # Find the intersection of years and discount years
    disc_years = np.intersect1d(years, disc.years)
    disc_rates = disc.rates[np.isin(disc.years, disc_years)]

    # Calculate the discount factors
    discount_factors = (1 / (1 + disc_rates)) ** (disc_years - start_year)

    # Apply the discount factors to the cash flows
    npv_cash_flows = cash_flows * discount_factors

    return npv_cash_flows


def calc_averted_risk_metrics(annual_risk_metrics):
    # Copy the DataFrame to avoid modifying the original
    averted_annual_risk_metrics = pd.DataFrame(columns=annual_risk_metrics.columns)

    # Get the unique groups and metrics
    groups = annual_risk_metrics["group"].unique()
    metrics = annual_risk_metrics["metric"].unique()

    # Iterate over each combination of group and metric
    for group in groups:
        for metric in metrics:
            # Filter the DataFrame for the current group and metric
            if pd.isna(group):
                sub = annual_risk_metrics[
                    (annual_risk_metrics["group"].isna())
                    & (annual_risk_metrics["metric"] == metric)
                ]
            else:
                sub = annual_risk_metrics[
                    (annual_risk_metrics["group"] == group)
                    & (annual_risk_metrics["metric"] == metric)
                ]
            # If the sub is empty, continue to the next iteration
            if sub.empty:
                continue

            # Get the risk metrics for the no measure
            no_meas = sub[sub["measure"] == "no_measure"].sort_values("year")

            # Calculate the averted risk metrics for each measure
            for meas_name in sub["measure"].unique():
                # if meas_name == 'no_measure':
                #    continue

                # Get the risk metrics for the measure
                meas = sub[sub["measure"] == meas_name].sort_values("year")

                # Align the DataFrames by year and calculate the averted risk
                sub_averted_risk = meas.copy()
                sub_averted_risk["risk"] = no_meas["risk"].values - meas["risk"].values

                # Concatenate the DataFrames
                if averted_annual_risk_metrics.empty:
                    averted_annual_risk_metrics = sub_averted_risk
                else:
                    averted_annual_risk_metrics = pd.concat(
                        [averted_annual_risk_metrics, sub_averted_risk],
                        ignore_index=True,
                    )

    # Drop duplicates and reset the index
    averted_annual_risk_metrics = (
        averted_annual_risk_metrics.drop_duplicates().reset_index(drop=True)
    )

    return averted_annual_risk_metrics


def calc_measure_cash_flows(
    measure_set,
    measure_times,
    start_year,
    end_year,
    consider_measure_times=True,
    disc=None,
):

    # Calculate the individual measure cash flows
    costincome = calc_indv_measure_cash_flows(
        measure_set,
        measure_times,
        start_year,
        end_year,
        consider_measure_times,
        disc,
    )

    # Calculate the combo measure cash flows
    costincome = calc_combo_measure_cash_flows(costincome, measure_set)

    return costincome


def calc_indv_measure_cash_flows(
    measure_set,
    measure_times,
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
    costincome = pd.DataFrame(columns=["measure", "year", "cost", "income", "net"])

    # Loop over the measures in the set
    for _, meas in measure_set.measures().items():

        # If the measure is a combination of other measures, skip it
        if meas.combo:
            continue
        else:
            # If we should consider the start and end years of the measure, update the start and end years
            if consider_measure_times:
                measure_name = meas.name
                meas_start_year = measure_times[
                    measure_times["measure"] == measure_name
                ]["start_year"].values[0]
                meas_end_year = measure_times[measure_times["measure"] == measure_name][
                    "end_year"
                ].values[0]
            else:
                meas_start_year = start_year
                meas_end_year = end_year

            # Calculate the cash flows for the measure
            temp = meas.cost_income.calc_cashflows(
                impl_year=meas_start_year,
                start_year=start_year,
                end_year=end_year,
                disc=disc,
            )
            # Set all the cash flows to zero for years outside the measure period
            temp.loc[
                (temp["year"] < meas_start_year) | (temp["year"] > meas_end_year),
                ["cost", "income", "net"],
            ] = 0

            # Add the name of the measure to the DataFrame
            temp["measure"] = meas.name

            # If the cash flows DataFrame is empty, set it to the DataFrame for the current measure
            # Otherwise, concatenate the DataFrame for the current measure to the existing DataFrame
            if costincome.empty:
                costincome = temp
            else:
                costincome = pd.concat([costincome, temp])

        # Reset the index of the DataFrame
        costincome = costincome.reset_index(drop=True)

    # Return the DataFrame with the calculated cash flows
    return costincome


def calc_combo_measure_cash_flows(costincome, measure_set):

    # Calculate the cash flows for the combined measures dont reorder the columns
    costincome_combo = pd.DataFrame(columns=costincome.columns)

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
            sub = costincome[costincome["measure"].isin(combo_measures)]
            # For each year, sum the costs, incomes and net
            sub = sub.groupby("year").sum().reset_index()
            # Change the measure name to the combo measure name
            sub["measure"] = meas.name
            # Concatenate the sub to the costincome_combo
            if costincome_combo.empty:
                costincome_combo = sub
            else:
                costincome_combo = pd.concat([costincome_combo, sub])

    # Reset the index
    costincome_combo.reset_index(drop=True, inplace=True)

    # Concatenate the costincome and costincome_combo
    if costincome_combo.empty:
        return costincome
    else:
        costincome = pd.concat([costincome, costincome_combo], ignore_index=True)

    return costincome
