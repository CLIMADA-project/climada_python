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
import warnings
from typing import Container, Iterable

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.trajectories.impact_calc_strat import ImpactComputationStrategy
from climada.trajectories.interpolation import InterpolationStrategy
from climada.trajectories.risk_trajectory import RiskTrajectory
from climada.trajectories.riskperiod import CalcRiskPeriod
from climada.trajectories.snapshot import Snapshot

# from pandas.core.frame import ValueKeyFunc
tqdm.pandas()

LOGGER = logging.getLogger(__name__)


class AdaptationTrajectoryAppraiser(RiskTrajectory):
    _risk_vars = [
        "reference risk",
        "averted risk",
        "risk",
    ]

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        time_resolution: str = "YS",
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        cost_disc: DiscRates | None = None,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        self.cost_disc = cost_disc
        self.measure_set = copy.deepcopy(measure_set)
        super().__init__(
            snapshots_list,
            time_resolution=time_resolution,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
            interpolation_strategy=interpolation_strategy,
            impact_computation_strategy=impact_computation_strategy,
        )

    def _calc_risk_periods(self, snapshots: list[Snapshot]):
        LOGGER.debug(f"{self.__class__.__name__}: Calc risk periods")
        risk_periods = super()._calc_risk_periods(snapshots)
        risk_periods += self._calc_measure_periods(risk_periods)
        return risk_periods

    def _calc_measure_periods(self, risk_periods: list[CalcRiskPeriod]):
        LOGGER.debug(f"{self.__class__.__name__}: Calc risk periods with measures")
        res = []
        for _, measure in self.measure_set.measures().items():
            LOGGER.debug(f"Creating measures risk_period for measure {measure.name}")
            meas_p = [
                risk_period.apply_measure(measure) for risk_period in risk_periods
            ]
            res += meas_p
        return res

    def _generic_metrics(
        self,
        npv=True,
        metric_name=None,
        metric_meth=None,
        measures: list[str] | None = None,
        **kwargs,
    ):
        LOGGER.debug(f"Computing base metric: {metric_name}.")
        base_metrics = super()._generic_metrics(npv, metric_name, metric_meth, **kwargs)
        if base_metrics is not None:
            LOGGER.debug(f"Computing averted risk for: {metric_name}.")
            base_metrics = self._calc_averted(base_metrics)
            no_measures = base_metrics[base_metrics["measure"] == "no_measure"].copy()
            no_measures["reference risk"] = no_measures["risk"]
            no_measures["averted risk"] = 0.0
            no_measures["measure net cost"] = 0.0
            LOGGER.debug(f"Computing cash flow for: {metric_name}.")
            cash_flow_metrics = self._calc_per_measure_annual_cash_flows(npv)
            LOGGER.debug(f"Merging with base metric: {metric_name}.")
            base_metrics = base_metrics.merge(
                cash_flow_metrics[["date", "measure", "measure net cost"]],
                on=["measure", "date"],
            )
            LOGGER.debug(f"Merging with no measure: {metric_name}.")
            base_metrics = pd.concat([no_measures, base_metrics])

            if measures is not None:
                base_metrics = base_metrics.loc[
                    base_metrics["measure"].isin(measures)
                ].reset_index()

        return base_metrics

    @staticmethod
    def _calc_averted(base_metrics: pd.DataFrame) -> pd.DataFrame:
        def subtract_no_measure(group, no_measure, merger):
            # Merge with no_measure to get the corresponding "no_measure" value
            merged = group.merge(no_measure, on=merger, suffixes=("", "_no_measure"))
            # Subtract the "no_measure" risk from the current risk
            merged["reference risk"] = merged["risk_no_measure"]
            merged["averted risk"] = merged["risk_no_measure"] - merged["risk"]
            return merged[list(group.columns) + ["reference risk", "averted risk"]]

        no_measures_metrics = base_metrics[
            base_metrics["measure"] == "no_measure"
        ].copy()
        merger = ["group", "metric", "date"]
        if "coord_id" in base_metrics.columns:
            merger.append("coord_id")

        return base_metrics.groupby(
            ["group", "metric", "date"], group_keys=False, dropna=False, observed=False
        ).progress_apply(
            subtract_no_measure, no_measure=no_measures_metrics, merger=merger
        )

    @classmethod
    def _date_to_period_agg(
        cls,
        df: pd.DataFrame,
        grouper: list[str] | None = None,
        time_unit="year",
        colname: str | list[str] | None = None,
    ) -> pd.DataFrame | pd.Series:
        colname = cls._risk_vars if colname is None else colname
        if grouper is None:
            grouper = cls._grouper
        return super()._date_to_period_agg(df, grouper, time_unit, colname)

    def per_date_CB(
        self,
        metrics: list[str] = ["aai", "return_periods", "aai_per_group"],
        include_no_measure=False,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_date_risk_metrics(metrics, **kwargs)
        if not include_no_measure:
            metrics_df = metrics_df[metrics_df["measure"] != "no_measure"]

        metrics_df.rename(columns={"risk": "residual risk"}, inplace=True)
        metrics_df["cumulated measure cost"] = metrics_df.groupby(
            ["group", "measure", "metric"], observed=True
        )["measure net cost"].cumsum()
        metrics_df["cumulated measure benefit"] = metrics_df.groupby(
            ["group", "measure", "metric"], observed=True
        )["averted risk"].cumsum()
        metrics_df["cost/benefit ratio"] = (
            metrics_df["cumulated measure cost"]
            / metrics_df["cumulated measure benefit"]
        )
        return metrics_df

    def per_period_CB(
        self,
        metrics: list[str] = ["aai", "return_periods", "aai_per_group"],
        npv: bool = True,
        include_no_measure=False,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_period_risk_metrics(metrics, **kwargs)
        cost_df = self._calc_per_measure_annual_cash_flows(npv)
        cost_df = self._date_to_period_agg(
            cost_df, grouper=["measure"], colname="measure net cost"
        )
        metrics_df = metrics_df.merge(cost_df, on=["period", "measure"], how="outer")
        metrics_df["measure net cost"] = metrics_df["measure net cost"].fillna(0.0)
        if not include_no_measure:
            metrics_df = metrics_df[metrics_df["measure"] != "no_measure"]

        return metrics_df

    def _calc_per_measure_annual_cash_flows(self, npv: bool):
        res = []
        for meas_name, measure in self.measure_set.measures().items():
            need_agg = False
            if measure.cost_income.freq != self._time_resolution:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self._time_resolution}). "
                        f"Cash flows will be aggregated to {measure.cost_income.freq} "
                        "but this **may** lead to inconsistencies."
                    ),
                    stacklevel=2,
                )

            df = measure.cost_income.calc_cashflows(
                impl_date=self.start_date,
                start_date=self.start_date,
                end_date=self.end_date,
                disc=self.cost_disc if npv else None,
            )
            if need_agg:
                df = df.groupby(df["date"].dt.year, as_index=False).agg(
                    {"net": "sum", "cost": "sum", "income": "sum", "date": "first"}
                )
            df["measure"] = meas_name
            res.append(df)
        df = pd.concat(res)
        df["net"] *= -1
        df = df.rename(columns={"net": "measure net cost"})
        return df

    def _calc_waterfall_CB_plot_data(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        npv: bool = True,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_components = self.risk_components_metrics(npv)
        risk_components = risk_components.loc[
            (risk_components["date"].dt.date >= start_date)
            & (risk_components["date"].dt.date <= end_date)
            & (risk_components["measure"] != "no_measure")
        ]
        risk_components = risk_components.set_index(["date", "measure", "metric"])[
            ["risk", "reference risk", "averted risk", "measure net cost"]
        ].unstack()
        return risk_components

    def plot_per_date_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        """Plot a waterfall chart of risk components over a specified date range.

        This method generates a stacked bar chart to visualize the
        risk components between specified start and end dates, for each date in between.
        If no dates are provided, it defaults to the start and end dates of the risk trajectory.
        See the notes on how risk is attributed to each components.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The matplotlib axes on which to plot. If None, a new figure and axes are created.
        start_date : datetime, optional
            The start date for the waterfall plot. If None, defaults to the start date of the risk trajectory.
        end_date : datetime, optional
            The end date for the waterfall plot. If None, defaults to the end date of the risk trajectory.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the plotted waterfall chart.

        Notes
        -----
        The "risk components" are plotted such that the increase in risk due to the hazard component
        really denotes the difference between the risk associated with both future exposure and hazard
        compared to the risk associated with future exposure and present hazard.
        """
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        df = self._calc_waterfall_CB_plot_data(start_date=start_date, end_date=end_date)
        df = df.swaplevel()
        metrics = [
            "base risk",
            "exposure contribution",
            "hazard contribution",
            "vulnerability contribution",
            "interaction contribution",
        ]
        colors = {
            "base risk": "tab:blue",
            "exposure contribution": "tab:orange",
            "hazard contribution": "tab:green",
            "vulnerability contribution": "tab:red",
            "interaction contribution": "tab:purple",
        }
        hatch_style = "///"

        # Unique measures
        measures = df.index.get_level_values(0).unique()

        _, axs = plt.subplots(
            1 + len(measures),
            1,
            figsize=(14, 5 * len(measures)),
            sharex=False,
            sharey=False,
        )
        self.plot_per_date_waterfall(ax=axs[0])

        for i, measure in enumerate(measures):
            ax = axs[i + 1]
            d = df.loc[measure]

            # Pivot for stacked bars
            averted = d.loc[:, "averted risk"].sum(axis=1)
            risk = d.loc[:, "risk"].sum(axis=1)
            ax.stackplot(
                d.index,
                [risk, averted],
                labels=["Residual risk", "Averted"],
                colors=["purple", "pink"],
                hatch=["", "/"],
            )
            # Labels and ticks
            ax.set_title(f"Measure: {measure}")
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            # # Custom legend to add hatch explanation
            # handles = [mpatches.Patch(facecolor=colors[m], label=m) for m in metrics]
            # handles.append(
            #     mpatches.Patch(
            #         facecolor="white",
            #         edgecolor="tab:olive",
            #         hatch=hatch_style,
            #         label="averted with measure",
            #     )
            # )
            # handles.append(mpatches.Patch(facecolor="tab:cyan", label="residual risk"))
            ax.legend()

        return axs

    def plot_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        measures: list[str] | None = None,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_component = self._calc_waterfall_CB_plot_data(
            start_date=start_date, end_date=end_date
        )
        meas = (
            list(
                np.setdiff1d(
                    risk_component.index.get_level_values(1).unique(), ["no_measure"]
                )
            )
            if measures is None
            else measures
        )
        num_cols = 3 if 3 < len(meas) else len(meas)
        num_rows = len(meas) // num_cols
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 8, num_rows * 5)
        )

        labels = [
            "Risk",
            "Averted Risk",
            "Residual risk",
            # "Measure cost",
            # "Cost benefit",
        ]
        # measure_costs = risk_component.loc[:,("measure net cost","base risk")].unstack().sum()
        average_risk = (
            risk_component.groupby(level=1)
            .mean()
            .stack()
            .groupby(level=0)
            .agg(
                {
                    "averted risk": "sum",
                    "measure net cost": "first",
                    "reference risk": "sum",
                    "risk": "sum",
                }
            )
        )
        # risk_component = risk_component.loc[str(end_date)]

        for i, measure in enumerate(meas):
            m_average_risk = average_risk.loc[measure]
            values = [
                m_average_risk["reference risk"],
                m_average_risk["averted risk"],
                m_average_risk["reference risk"] - m_average_risk["averted risk"],
                # m_average_risk["measure net cost"],
                # m_average_risk["averted risk"] - m_average_risk["measure net cost"],
            ]
            bottoms = [
                0.0,
                m_average_risk["reference risk"] - m_average_risk["averted risk"],
                0.0,
                # m_average_risk["reference risk"] - m_average_risk["averted risk"],
                # m_average_risk["reference risk"]
                # - m_average_risk["averted risk"]
                # + m_average_risk["measure net cost"],
            ]
            axs[i].bar(
                labels,
                values,
                bottom=bottoms,
                edgecolor="black",
                color=["tab:blue", "tab:olive", "tab:cyan", "tab:brown", "tab:pink"],
            )
            for j in range(len(values)):
                axs[i].text(
                    labels[j],
                    max(values[j] + bottoms[j], bottoms[j]),
                    f"{values[j]:.0e}",
                    ha="center",
                    va="bottom",
                    color="black",
                )

            axs[i].spines["left"].set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].spines["right"].set_visible(False)
            axs[i].set_yticks([])
            axs[i].set_title(f"{measure}")
            axs[i].annotate(
                "",
                xy=(
                    1,
                    (m_average_risk["reference risk"] - m_average_risk["averted risk"]),
                ),
                xycoords="data",
                xytext=(1, m_average_risk["reference risk"]),
                textcoords="data",
                arrowprops=dict(color="red", lw=2, shrink=0.1, width=12),
            )

            # axs[i].annotate(
            #     "",
            #     xy=(
            #         3,
            #         m_average_risk["measure net cost"]
            #         + (
            #             m_average_risk["reference risk"]
            #             - m_average_risk["averted risk"]
            #         ),
            #     ),
            #     xycoords="data",
            #     xytext=(
            #         3,
            #         (m_average_risk["reference risk"] - m_average_risk["averted risk"]),
            #     ),
            #     textcoords="data",
            #     arrowprops=dict(color="red", lw=2, shrink=0.1, width=12),
            # )

            # Construct y-axis label and title based on parameters
            value_label = "USD (Average annual value)"
            axs[i].set_ylabel(value_label)
            axs[i].tick_params(
                axis="x",
                labelrotation=0,
            )

        title_label = f"Measures cost benefit (Averaged values over {start_date} to {end_date} period)"
        fig.suptitle(title_label)

        return axs


class PlannedAdaptationAppraiser(AdaptationTrajectoryAppraiser):
    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        planner: (
            dict[str, tuple[int, int]] | dict[str, tuple[datetime.date, datetime.date]]
        ),
        interval_freq: str = "AS-JAN",
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        cost_disc: DiscRates | None = None,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        if all(
            isinstance(value, tuple)
            and all(isinstance(element, int) for element in value)
            for value in planner.values()
        ):
            planner = {
                k: (datetime.date(v1, 1, 1), datetime.date(v2, 1, 1))  # type: ignore
                for k, (v1, v2) in planner.items()
            }
        self.planner: dict[str, tuple[datetime.date, datetime.date]] = planner
        self._planning = _get_unique_measure_periods(self.planner)
        super().__init__(
            snapshots_list,
            measure_set=measure_set,
            time_resolution=interval_freq,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
            cost_disc=cost_disc,
            interpolation_strategy=interpolation_strategy,
            impact_computation_strategy=impact_computation_strategy,
        )

    def _calc_measure_periods(self, risk_periods):
        # For each planned period, find correponding risk periods and create the periods with measure from planning
        LOGGER.debug(
            f"{self.__class__.__name__}: Calc risk periods with planned measures"
        )
        res = []
        for (start_date, end_date), measure_name_list in self._planning.items():
            # Not sure this works as intended (pbly could be simplified anyway)
            if len(measure_name_list) > 1:
                measure = self.measure_set.combine(names=measure_name_list)
            elif len(measure_name_list) == 1:
                measure = self.measure_set._data[measure_name_list[0]]
            else:
                measure = None

            periods = self._get_risk_periods(risk_periods, start_date, end_date)
            if measure:
                LOGGER.debug(
                    f"Creating measures risk_period for measure {measure.name}"
                )
                meas_periods = [period.apply_measure(measure) for period in periods]
                res += meas_periods
        return res

    def _generic_metrics(
        self,
        npv=True,
        metric_name=None,
        metric_meth=None,
        measures: list[str] | None = None,
        **kwargs,
    ):
        LOGGER.info(f"Computing base metric: {metric_name}.")
        base_metrics = super()._generic_metrics(
            npv,
            metric_name,
            metric_meth,
            measures,
            **kwargs,
        )
        LOGGER.info(f"Computing planning metric: {metric_name}.")
        base_metrics = base_metrics.set_index(["measure", "date"]).sort_index()
        mask = pd.Series(False, index=base_metrics.index)
        for (start, end), measure_name_list in self._planning.items():
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            mask |= (
                (
                    base_metrics.index.get_level_values("measure")
                    == "_".join(measure_name_list)
                )
                & (base_metrics.index.get_level_values("date") >= start)
                & (base_metrics.index.get_level_values("date") <= end)
            )

        no_measure_mask = mask.groupby("date").sum() == 0
        mask.loc[
            pd.IndexSlice["no_measure"], no_measure_mask[no_measure_mask].index
        ] = True

        return base_metrics[mask].reset_index().sort_values("date")

    def _calc_per_measure_annual_cash_flows(self):
        res = []
        for meas_name, (start, end) in self.planner.items():
            need_agg = False
            measure = self.measure_set.measures()[meas_name]
            if measure.cost_income.freq != self._time_resolution:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self._time_resolution}). "
                        f"Cash flows will be aggregated to {measure.cost_income.freq} "
                        "but this **may** lead to inconsistencies."
                    ),
                    stacklevel=2,
                )

            df = measure.cost_income.calc_cashflows(
                impl_date=start,
                start_date=start,
                end_date=end,
                disc=self.cost_disc,
            )
            if need_agg:
                df = df.groupby(df["date"].dt.year, as_index=False).agg(
                    {"net": "sum", "cost": "sum", "income": "sum", "date": "first"}
                )
            df["measure"] = meas_name
            res.append(df)
        df = pd.concat(res)
        df = df.groupby("date", as_index=False).agg(
            {
                col: ("sum" if is_numeric_dtype(df[col]) else lambda x: "_".join(x))
                for col in df.columns
                if col != "date"
            }
        )
        df["net"] *= -1
        df = df.rename(columns={"net": "measure net cost"})
        return df

    def _calc_waterfall_CB_plot_data(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        npv: bool = True,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_components = self.risk_components_metrics(npv)
        risk_components = risk_components.loc[
            (risk_components["date"].dt.date >= start_date)
            & (risk_components["date"].dt.date <= end_date)
        ]
        risk_components = risk_components.set_index(["date", "measure", "metric"])[
            ["risk", "reference risk", "averted risk", "measure net cost"]
        ].unstack()
        return risk_components

    def plot_per_date_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        # Unique measures
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        df = self._calc_waterfall_CB_plot_data(start_date=start_date, end_date=end_date)
        df = df.swaplevel()
        metrics = [
            "base risk",
            "exposure contribution",
            "hazard contribution",
            "vulnerability contribution",
            "interaction contribution",
        ]
        colors = {
            "base risk": "tab:blue",
            "exposure contribution": "tab:orange",
            "hazard contribution": "tab:green",
            "vulnerability contribution": "tab:red",
            "interaction contribution": "tab:purple",
        }
        hatch_style = "///"

        measures = df.index.get_level_values(0).unique().drop("no_measure")
        reference_risk = df["reference risk"].droplevel(0)
        _, axs = plt.subplots(
            3, 1, figsize=(14, 5 * len(measures)), sharex=True, sharey=False
        )
        axs[0].stackplot(
            reference_risk.index,
            [reference_risk[col] for col in reference_risk.columns],
            labels=reference_risk.columns,
        )
        axs[0].legend()
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax = axs[1]
        ax.sharey(axs[0])
        d = df.copy().droplevel("measure")

        # Pivot for stacked bars
        averted = d.loc[:, "averted risk"].sum(axis=1)
        risk = d.loc[:, "risk"].sum(axis=1)
        ax.stackplot(
            d.index,
            [risk, averted],
            labels=["Residual risk", "Averted"],
            colors=["purple", "pink"],
            hatch=["", "/"],
        )
        # Labels and ticks
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.legend()

        y = 0
        planner_t = {
            label: (pd.Timestamp(v1), pd.Timestamp(v2))
            for label, (v1, v2) in self.planner.items()
        }

        for label_text, (start, end) in planner_t.items():
            axs[2].barh(
                y,
                (end - start).days,
                left=start,
                height=0.7,
                color="skyblue",
                edgecolor="none",
            )

            axs[2].text(
                start,
                y,
                "  " + label_text,
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
            y += 1
        axs[2].xaxis.set_major_locator(locator)
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axs[2].set_xlim(
            min(df.index.get_level_values(1)), max(df.index.get_level_values(1))
        )
        axs[0].xaxis.set_major_locator(locator)
        axs[0].xaxis.set_major_formatter(formatter)
        # axs[2].set_yticks([])
        # axs[2].spines["left"].set_visible(False)
        # axs[2].spines["top"].set_visible(False)
        # axs[2].spines["right"].set_visible(False)
        # axs[2].spines["bottom"].set_visible(False)
        # box = axs[2].get_position()
        # box.y0 = box.y0 + 0.03
        # box.y1 = box.y1 + 0.03
        # axs[2].set_position(box)
        # axs[0].set_xticks([])
        # axs[0].set_xlabel("")
        # axs[1].set_xticks([])
        # axs[1].set_xlabel("")
        return axs

    def plot_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        measures: list[str] | None = None,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_component = self._calc_waterfall_CB_plot_data(
            start_date=start_date, end_date=end_date
        )
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = [
            "Risk",
            "Averted Risk",
            "Residual risk",
            "Measure cost",
            "Cost benefit",
        ]
        # measure_costs = risk_component.loc[:,("measure net cost","base risk")].unstack().sum()
        average_risk = (
            risk_component.mean()
            .unstack()
            .T.agg(
                {
                    "averted risk": "sum",
                    "measure net cost": "mean",
                    "reference risk": "sum",
                    "risk": "sum",
                }
            )
        )
        # risk_component = risk_component.loc[str(end_date)]

        m_average_risk = average_risk.copy()
        values = [
            m_average_risk["reference risk"],
            m_average_risk["averted risk"],
            m_average_risk["reference risk"] - m_average_risk["averted risk"],
            m_average_risk["measure net cost"],
            m_average_risk["averted risk"] - m_average_risk["measure net cost"],
        ]
        bottoms = [
            0.0,
            m_average_risk["reference risk"] - m_average_risk["averted risk"],
            0.0,
            m_average_risk["reference risk"] - m_average_risk["averted risk"],
            m_average_risk["reference risk"]
            - m_average_risk["averted risk"]
            + m_average_risk["measure net cost"],
        ]
        ax.bar(
            labels,
            values,
            bottom=bottoms,
            edgecolor="black",
            color=["tab:blue", "tab:olive", "tab:cyan", "tab:brown", "tab:pink"],
        )
        for j in range(len(values)):
            ax.text(
                labels[j],
                max(values[j] + bottoms[j], bottoms[j]),
                f"{values[j]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )

        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
        ax.annotate(
            "",
            xy=(
                1,
                (m_average_risk["reference risk"] - m_average_risk["averted risk"]),
            ),
            xycoords="data",
            xytext=(1, m_average_risk["reference risk"]),
            textcoords="data",
            arrowprops=dict(color="red", lw=2, shrink=0.1, width=12),
        )

        ax.annotate(
            "",
            xy=(
                3,
                m_average_risk["measure net cost"]
                + (m_average_risk["reference risk"] - m_average_risk["averted risk"]),
            ),
            xycoords="data",
            xytext=(
                3,
                (m_average_risk["reference risk"] - m_average_risk["averted risk"]),
            ),
            textcoords="data",
            arrowprops=dict(color="red", lw=2, shrink=0.1, width=12),
        )

        # Construct y-axis label and title based on parameters
        value_label = "USD (Average annual value)"
        ax.set_ylabel(value_label)
        ax.tick_params(
            axis="x",
            labelrotation=0,
        )

        title_label = f"Planning cost benefit (Averaged values over {start_date} to {end_date} period)"
        ax.set_title(title_label, pad=20)

        return ax


def format_periods_dict(periods_dict):
    formatted_string = ""
    for measure, (start_date, end_date) in periods_dict.items():
        formatted_string += f"{measure}: {start_date} - {end_date} ; "
    return formatted_string.strip()


def _get_unique_measure_periods(
    planner: dict[str, tuple[datetime.date, datetime.date]],
) -> dict[tuple[datetime.date, datetime.date], list[str]]:
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
    boundaries = sorted(
        {pt for _, (start, end) in planner.items() for pt in (start, end)}
    )
    subintervals = [
        (boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
    ]

    return {
        (s, e): [
            key for key, (start, end) in planner.items() if start <= s and e <= end
        ]
        for s, e in subintervals
    }
