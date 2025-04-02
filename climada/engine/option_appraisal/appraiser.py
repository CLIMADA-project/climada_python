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
import datetime
import logging
from collections import defaultdict
from typing import Union

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.frame import ValueKeyFunc

from climada.engine.option_appraisal.plot import plot_CB_summary, plot_dately
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.trajectories.risk_trajectory import RiskTrajectory, calc_npv_cash_flows
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


class MeasuresAppraiser(RiskTrajectory):
    # TODO: To reflect on:
    # - Do we want "_planned", "_npv", "_total", "_single_measure" as parameter attributes instead of arguments?
    # - Do we keep "combo_all" ?
    _grouper = ["measure", "group", "metric"]

    def __init__(
        self,
        snapshots: list[Snapshot],
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
        ret = ret.set_index(["measure", "date"]).sort_index()
        ret = ret.loc[pd.IndexSlice[measure_name, :],].reset_index()
        return ret

    def _calc_per_measure_annual_averted_risk(self, npv=True):
        # We want super() because we want all dates not just planning
        no_measure = super()._calc_annual_risk_metrics(npv=npv)
        no_measure = no_measure[no_measure["measure"] == "no_measure"].copy()

        def subtract_no_measure(group):
            # Merge with no_measure to get the corresponding "no_measure" value
            merged = group.merge(
                no_measure, on=["group", "metric", "date"], suffixes=("", "_no_measure")
            )
            # Subtract the "no_measure" risk from the current risk
            merged["risk"] = merged["risk_no_measure"] - merged["risk"]
            return merged[group.columns]

        averted = self._calc_annual_risk_metrics(npv=npv).copy()
        averted = averted.groupby(
            ["group", "metric", "date"], group_keys=False, dropna=False
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
    def _calc_periods_cashflow(df: pd.DataFrame, time_unit="date"):
        def identify_continuous_periods(group, time_unit):
            # Calculate the difference between consecutive dates
            if time_unit == "date":
                group["date_diff"] = group["date"].dt.year.diff()
            if time_unit == "month":
                group["date_diff"] = group["date"].dt.month.diff()
            if time_unit == "day":
                group["date_diff"] = group["date"].dt.day.diff()
            if time_unit == "hour":
                group["date_diff"] = group["date"].dt.hour.diff()
            # Identify breaks in continuity
            group["period_id"] = (group["date_diff"] != 1).cumsum()
            return group

        # Apply the function to identify continuous periods
        df_periods = df.groupby(
            ["measure", "date"], dropna=False, group_keys=False
        ).apply(identify_continuous_periods, time_unit)

        # Group by the identified periods and calculate start and end dates
        df_periods = (
            df_periods.groupby(["measure", "period_id"], dropna=False)
            .agg(
                start_date=pd.NamedAgg(column="date", aggfunc="min"),
                end_date=pd.NamedAgg(column="date", aggfunc="max"),
                net=pd.NamedAgg(column="net", aggfunc="sum"),
                cost=pd.NamedAgg(column="cost", aggfunc="sum"),
                income=pd.NamedAgg(column="income", aggfunc="sum"),
            )
            .reset_index()
        )

        df_periods["period"] = (
            df_periods["start_date"].astype(str)
            + "-"
            + df_periods["end_date"].astype(str)
        )
        return df_periods.drop(["period_id", "start_date", "end_date"], axis=1)

    def _calc_per_measure_annual_cash_flows(self):
        res = []
        for meas_name, measure in self.measure_set.measures().items():
            df = measure.cost_income.calc_cashflows(
                impl_year=self.start_date,
                start_year=self.start_date,
                end_year=self.end_date,
                disc=self.cost_disc,
            )
            df["measure"] = meas_name
            res.append(df)
        df = pd.concat(res)
        return df

    def calc_CB(
        self,
        net_present_value=True,
        dately=False,
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
        time_grouper = "date" if dately else "period"

        res = self._calc_risk_metrics(
            total=(not dately),
            npv=net_present_value,
        )
        res = res.merge(
            self.calc_averted_risk(
                total=(not dately),
                npv=net_present_value,
            ),
            on=["measure", time_grouper, "group", "metric"],
            how="left",
        )

        res = res.merge(
            self.calc_cash_flow(
                total=(not dately),
            )[["measure", time_grouper, "net"]],
            on=["measure", time_grouper],
            how="left",
        )

        res["net"] *= -1
        res["net"] = res["net"].fillna(0.0)
        res = res.rename(columns={"net": "cost (net)", "risk": "residual risk"})
        res["group"] = res["group"].fillna("No Group")
        res["base risk"] = res["residual risk"] + res["averted risk"]

        if not dately:
            period_extracted = res["period"].str.extract(r"(\d{4})-(\d{4})").astype(int)
            start_date = period_extracted[0]
            end_date = period_extracted[1]
            dates_span = end_date - start_date + 1
            res["average annual base risk"] = res["base risk"] / dates_span
            res["average annual residual risk"] = res["residual risk"] / dates_span
            res["average annual averted risk"] = res["averted risk"] / dates_span
            res["average annual cost"] = res["cost (net)"] / dates_span

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

    def plot_CB_summary(
        self,
        metric="aai",
        measure_colors=None,
        y_label="Risk",
        title="Benefit and Benefit/Cost Ratio by Measure",
    ):
        plot_CB_summary(
            self.calc_CB(),
            metric=metric,
            measure_colors=measure_colors,
            y_label=y_label,
            title=title,
        )

    def plot_dately(
        self,
        to_plot="residual risk",
        metric="aai",
        y_label=None,
        title=None,
        with_measure=True,
        measure_colors=None,
    ):
        plot_dately(
            self.calc_CB(dately=True).sort_values("residual risk"),
            to_plot=to_plot,
            with_measure=with_measure,
            metric=metric,
            y_label=y_label,
            title=title,
            measure_colors=measure_colors,
        )

    def plot_waterfall(self, ax=None, start_date=None, end_date=None):
        df = self.calc_CB(dately=True)
        averted = df.loc[
            (df["date"] == 2080)
            & (df["metric"] == "aai")
            & (df["measure"] != "no_measure")
        ]
        ax = super().plot_waterfall(ax=ax, start_date=start_date, end_date=end_date)
        ax.bar("Averted risk", ax.patches[-1].get_height(), width=1, visible=False)
        # ax.text(
        #     x=ax.get_xticks()[-1] - ax.patches[-1].get_width() / 2 + 0.02,
        #     y=ax.patches[-1].get_height() * 0.96,
        #     ha="left",
        #     s="Averted risk",
        #     size=12,
        # )
        averted = averted.sort_values("averted risk")
        for i, meas in enumerate(averted["measure"].unique()):
            measure_risk = averted.loc[
                (averted["measure"] == meas), "averted risk"
            ].values[0]
            x_arrow = (
                ax.get_xticks()[-1]
                - ax.patches[-1].get_width() / 2
                + 0.1
                + (ax.patches[-1].get_width() / averted["measure"].nunique()) * i
            )
            top_arrow = ax.patches[-1].get_height()
            bottom_arrow = top_arrow - measure_risk
            ax.annotate(
                "",
                xy=(x_arrow, bottom_arrow),
                xytext=(x_arrow, top_arrow),
                arrowprops=dict(
                    facecolor="tab:green", width=12, headwidth=20, headlength=10
                ),
            )
            ax.text(
                x=x_arrow,
                y=top_arrow - (top_arrow - bottom_arrow) / 2,
                va="center",
                ha="center",
                s=meas,
                rotation=-90,
                size="x-small",
            )

        return ax


class _PlannedMeasuresAppraiser(MeasuresAppraiser):

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
        date_start, date_end = self.planner.get(
            measure_name, (self.start_date, self.end_date)
        )
        ret = self._calc_annual_risk_metrics().copy()
        ret = ret.set_index(["measure", "date"]).sort_index()
        ret = ret.loc[pd.IndexSlice[measure_name, date_start:date_end],].reset_index()
        return ret

    def _calc_measure_periods(self, risk_periods):
        # For each planned period, find correponding risk periods and create the periods with measure from planning
        res = []
        for measure_name_list, start_date, end_date in self._planning:
            # Not sure this works as intended (pbly could be simplified anyway)
            measure = self.measure_set.combine(names=measure_name_list)
            periods = self._get_risk_periods(risk_periods, start_date, end_date)
            LOGGER.debug(f"Creating measures risk_period for measure {measure.name}")
            meas_periods = [period.apply_measure(measure) for period in periods]
            res += meas_periods
        return res

    def _calc_annual_risk_metrics(self, npv=True):
        df = super()._calc_annual_risk_metrics(npv)
        df = df.set_index(["measure", "date"]).sort_index()
        mask = pd.Series(False, index=df.index)
        # for each measure set mask to true at corresponding date (bitwise or)
        for measure_combo, start, end in self._planning:
            mask |= (
                (df.index.get_level_values("measure") == "_".join(measure_combo))
                & (df.index.get_level_values("date") >= start)
                & (df.index.get_level_values("date") <= end)
            )

        # for the no_measure case, set mask to true at corresponding date,
        # only for those dates that have no active measures
        no_measure_mask = mask.groupby("date").sum() == 0
        mask.loc[
            pd.IndexSlice["no_measure"], no_measure_mask[no_measure_mask].index
        ] = True

        return df[mask].reset_index().sort_values("date")

    def _calc_per_measure_annual_cash_flows(self, disc=None):
        res = []
        for measure, (start, end) in self.planner.items():
            df = self.measure_set.measures()[measure].cost_income.calc_cashflows(
                impl_date=start, start_date=start, end_date=end, disc=disc
            )
            df["measure"] = measure
            res.append(df)

        df = pd.concat(res)
        df = df.groupby("date", as_index=False).agg(
            {
                col: ("sum" if is_numeric_dtype(df[col]) else lambda x: "_".join(x))
                for col in df.columns
                if col != "date"
            }
        )
        return df

    def plot_CB_summary(
        self,
        metric="aai",
        measure_colors=None,
        y_label="Risk",
        title="Benefit and Benefit/Cost Ratio by Measure",
    ):
        raise NotImplementedError("Not Implemented for that class")
        df = self.calc_CB(dately=True)
        df_plan = df.groupby(["group", "metric"], as_index=False).agg(
            start_date=pd.NamedAgg(column="date", aggfunc="min"),
            end_date=pd.NamedAgg(column="date", aggfunc="max"),
            base_risk=pd.NamedAgg(column="base risk", aggfunc="sum"),
            residual_risk=pd.NamedAgg(column="residual risk", aggfunc="sum"),
            averted_risk=pd.NamedAgg(column="averted risk", aggfunc="sum"),
            cost_net=pd.NamedAgg(column="cost (net)", aggfunc="sum"),
        )
        df_plan["measure"] = "Whole risk period"
        df.columns = df.columns.str.replace("_", " ")
        df["B/C ratio"] = (df["averted risk"] / df["cost net"]).fillna(0.0)
        plot_CB_summary(
            df,
            metric=metric,
            measure_colors=measure_colors,
            y_label=y_label,
            title=title,
        )

    def plot_dately(
        self,
        to_plot="residual risk",
        metric="aai",
        y_label=None,
        title=None,
        with_measure=True,
        measure_colors=None,
    ):
        plot_dately(
            self.calc_CB(dately=True).sort_values("residual risk"),
            to_plot=to_plot,
            with_measure=with_measure,
            metric=metric,
            y_label=y_label,
            title=title,
            measure_colors=measure_colors,
        )

    def plot_waterfall(self, ax=None, start_date=None, end_date=None):
        df = self.calc_CB(dately=True)
        averted = df.loc[
            (df["date"] == 2080)
            & (df["metric"] == "aai")
            & (df["measure"] != "no_measure")
        ]
        ax = super().plot_waterfall(ax=ax, start_date=start_date, end_date=end_date)
        ax.bar("Averted risk", ax.patches[-1].get_height(), width=1, visible=False)
        # ax.text(
        #     x=ax.get_xticks()[-1] - ax.patches[-1].get_width() / 2 + 0.02,
        #     y=ax.patches[-1].get_height() * 0.96,
        #     ha="left",
        #     s="Averted risk",
        #     size=12,
        # )
        averted = averted.sort_values("averted risk")
        for i, meas in enumerate(averted["measure"].unique()):
            measure_risk = averted.loc[
                (averted["measure"] == meas), "averted risk"
            ].values[0]
            x_arrow = (
                ax.get_xticks()[-1]
                - ax.patches[-1].get_width() / 2
                + 0.1
                + (ax.patches[-1].get_width() / averted["measure"].nunique()) * i
            )
            top_arrow = ax.patches[-1].get_height()
            bottom_arrow = top_arrow - measure_risk
            ax.annotate(
                "",
                xy=(x_arrow, bottom_arrow),
                xytext=(x_arrow, top_arrow),
                arrowprops=dict(
                    facecolor="tab:green", width=12, headwidth=20, headlength=10
                ),
            )
            ax.text(
                x=x_arrow,
                y=top_arrow - (top_arrow - bottom_arrow) / 2,
                va="center",
                ha="center",
                s=meas,
                rotation=-90,
                size="x-small",
            )

        return ax


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
            _PlannedMeasuresAppraiser(
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
            df = plan.calc_CB(net_present_value=self._use_npv, dately=True)
            df = df.drop("date", axis=1)
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
    for measure, (start_date, end_date) in periods_dict.items():
        formatted_string += f"{measure}: {start_date} - {end_date} ; "
    return formatted_string.strip()


def _planner_to_planning(planner: dict[str, tuple[int, int]]) -> dict[int, list[str]]:
    """Transform a dictionary of measures with date spans into a dictionary
    where each key is a date, and the value is a list of active measures.

    Parameters
    ----------
    measures : dict[str, tuple[int, int]]
        Dictionary where keys are measure names and values are (start_date, end_date).

    Returns
    -------
    dict[int, list[str]]
        Dictionary where each key is a date, and the value is a list of active measures.
    """
    date_to_measures = defaultdict(list)

    for measure, (start, end) in planner.items():
        for date in range(start, end + 1):  # Include both start and end date
            date_to_measures[date].append(measure)

    return dict(date_to_measures)


def _get_unique_measure_periods(
    date_to_measures: dict[Union[int, datetime.date], list[str]]
) -> list[tuple[list[str], datetime.date, datetime.date]]:
    """Extract unique measure lists with their corresponding min and max date.

    Parameters
    ----------
    date_to_measures : dict[Union[int, date], list[str]]
        Dictionary where keys are dates (as int or datetime.date) and values are lists of active measures.

    Returns
    -------
    list[tuple[list[str], date, date]]
        A list of tuples containing (unique measure list, min date, max date).
    """
    # Convert all keys to datetime.date if they are integers
    if isinstance(list(date_to_measures.keys())[0], int):
        LOGGER.info("Found integer keys from planner dict, assuming they are years.")
        converted_dates = {
            datetime.date(d, 1, 1): measures for d, measures in date_to_measures.items()
        }
    elif isinstance(list(date_to_measures.keys())[0], datetime.date):
        converted_dates = {d: measures for d, measures in date_to_measures.items()}
    elif isinstance(list(date_to_measures.keys())[0], str):
        converted_dates = {
            datetime.date.fromisoformat(d): measures
            for d, measures in date_to_measures.items()
        }
    else:
        raise ValueError(
            "Planner dict must have years (int) or datetime.date or str date in iso format as keys."
        )

    sorted_dates = sorted(converted_dates.keys())  # Ensure chronological order
    unique_periods = []

    current_measures = None
    start_date = None

    for d in sorted_dates:
        measures = sorted(
            converted_dates[d]
        )  # Sorting ensures consistency in comparison

        if measures != current_measures:  # New unique set detected
            if current_measures is not None:  # Save the previous period

                ## HERE WE NEED A FREQ OR SMTHG
                unique_periods.append(
                    (current_measures, start_date, d - datetime.timedelta(days=1))
                )

            # Start a new period
            current_measures = measures
            start_date = d

    # Add the last recorded period
    if current_measures is not None:
        unique_periods.append((current_measures, start_date, sorted_dates[-1]))

    return unique_periods


def _get_unique_measure_periods(
    date_to_measures: dict[int, list[str]]
) -> list[tuple[list[str], int, int]]:
    """Extract unique measure lists with their corresponding min and max date.

    Parameters
    ----------
    date_to_measures : dict[int, list[str]]
        Dictionary where keys are dates and values are lists of active measures.

    Returns
    -------
    list[tuple[list[str], int, int]]
        A list of tuples containing (unique measure list, min date, max date).
    """
    sorted_dates = sorted(date_to_measures.keys())  # Ensure chronological order
    unique_periods = []

    current_measures = None
    start_date = None

    for date in sorted_dates:
        measures = sorted(
            date_to_measures[date]
        )  # Sorting ensures consistency in comparison

        if measures != current_measures:  # New unique set detected
            if current_measures is not None:  # Save the previous period
                unique_periods.append((current_measures, start_date, date - 1))

            # Start a new period
            current_measures = measures
            start_date = date

    # Add the last recorded period
    if current_measures is not None:
        unique_periods.append((current_measures, start_date, sorted_dates[-1]))

    return unique_periods
