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

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.trajectories.impact_calc_strat import ImpactComputationStrategy
from climada.trajectories.interpolation import InterpolationStrategy
from climada.trajectories.risk_trajectory import RiskTrajectory
from climada.trajectories.riskperiod import CalcRiskPeriod
from climada.trajectories.snapshot import Snapshot

# from pandas.core.frame import ValueKeyFunc


LOGGER = logging.getLogger(__name__)


class MeasuresAppraiser(RiskTrajectory):
    # TODO: To reflect on:
    # - Do we want "_planned", "_npv", "_total", "_single_measure" as parameter attributes instead of arguments?
    # - Do we keep "combo_all" ?
    _grouper = ["measure", "group", "metric"]
    _risk_vars = [
        "reference risk",
        "averted risk",
        "risk",
        "measure net cost",
        "measure cost benefit",
    ]

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        interval_freq: str = "AS-JAN",
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        cost_disc: DiscRates | None = None,
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual: bool = True,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        self.cost_disc = cost_disc
        self.measure_set = copy.deepcopy(measure_set)
        super().__init__(
            snapshots_list,
            interval_freq=interval_freq,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
            risk_transf_cover=risk_transf_cover,
            risk_transf_attach=risk_transf_attach,
            calc_residual=calc_residual,
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
        base_metrics = super()._generic_metrics(npv, metric_name, metric_meth, **kwargs)
        base_metrics = self._calc_averted(base_metrics)
        no_measures = base_metrics[base_metrics["measure"] == "no_measure"].copy()
        no_measures["reference risk"] = no_measures["risk"]
        no_measures["averted risk"] = 0.0
        no_measures["measure net cost"] = 0.0
        no_measures["measure cost benefit"] = 0.0
        cash_flow_metrics = self._calc_per_measure_annual_cash_flows()
        base_metrics = base_metrics.merge(
            cash_flow_metrics[["date", "measure", "measure net cost"]],
            on=["measure", "date"],
        )
        base_metrics = pd.concat([no_measures, base_metrics])

        averted_risk = base_metrics["averted risk"]
        cash_flow_metrics = base_metrics["measure net cost"]
        base_metrics["measure cost benefit"] = averted_risk - cash_flow_metrics

        if measures is not None:
            base_metrics = base_metrics.loc[
                base_metrics["measure"].isin(measures)
            ].reset_index()

        return base_metrics

    @staticmethod
    def _calc_averted(base_metrics: pd.DataFrame) -> pd.DataFrame:
        def subtract_no_measure(group, no_measure):
            # Merge with no_measure to get the corresponding "no_measure" value
            merged = group.merge(
                no_measure, on=["group", "metric", "date"], suffixes=("", "_no_measure")
            )
            # Subtract the "no_measure" risk from the current risk
            merged["reference risk"] = merged["risk_no_measure"]
            merged["averted risk"] = merged["risk_no_measure"] - merged["risk"]
            return merged[list(group.columns) + ["reference risk", "averted risk"]]

        no_measures_metrics = base_metrics[
            base_metrics["measure"] == "no_measure"
        ].copy()

        return base_metrics.groupby(
            ["group", "metric", "date"], group_keys=False, dropna=False
        ).apply(subtract_no_measure, no_measures_metrics)

    @classmethod
    def _per_period_risk(
        cls, df: pd.DataFrame, time_unit="year", colname=None
    ) -> pd.DataFrame | pd.Series:
        colname = cls._risk_vars if colname is None else colname
        return super()._per_period_risk(df, time_unit, colname)

    #                 net=pd.NamedAgg(column="net", aggfunc="sum"),
    #                 cost=pd.NamedAgg(column="cost", aggfunc="sum"),
    #                 income=pd.NamedAgg(column="income", aggfunc="sum"),

    def per_date_CB(self, metrics=None, **kwargs) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_date_risk_metrics(metrics, **kwargs)
        metrics_df["cumulated measure cost"] = metrics_df["measure net cost"].cumsum()
        metrics_df["cumulated measure benefit"] = metrics_df["averted risk"].cumsum()
        metrics_df["cost/benefit ratio"] = (
            metrics_df["cumulated measure cost"]
            / metrics_df["cumulated measure benefit"]
        )
        return metrics_df

    def per_period_CB(
        self, metrics: list[str] = ["aai", "return_periods", "aai_per_group"], **kwargs
    ) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_period_risk_metrics(metrics=metrics, **kwargs)
        return metrics_df

    def _calc_per_measure_annual_cash_flows(self):
        res = []
        for meas_name, measure in self.measure_set.measures().items():
            need_agg = False
            if measure.cost_income.freq != self._interval_freq:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self._interval_freq}). "
                        f"Cash flows will be aggregated to {measure.cost_income.freq} "
                        "but this **may** lead to inconsistencies."
                    ),
                    stacklevel=2,
                )

            df = measure.cost_income.calc_cashflows(
                impl_date=self.start_date,
                start_date=self.start_date,
                end_date=self.end_date,
                disc=self.cost_disc,
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

    def _calc_cb(self, base_metrics: pd.DataFrame):
        averted_risk = base_metrics["averted risk"]
        cash_flow_metrics = base_metrics["measure net cost"]
        base_metrics["measure cost benefit"] = averted_risk - cash_flow_metrics
        return base_metrics

    def _calc_waterfall_plot_data(
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
        # risk_components = risk_components.set_index(["date", "measure", "metric"])[
        #    ["risk","reference risk","averted risk", "measure net cost"]
        # ].unstack()
        return risk_components

    def plot_per_date_waterfall(
        self,
        ax=None,
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
        df = self._calc_waterfall_plot_data(start_date=start_date, end_date=end_date)
        metrics = ["base risk", "delta from exposure", "delta from hazard"]
        colors = {
            "base risk": "skyblue",
            "delta from exposure": "orange",
            "delta from hazard": "lightgreen",
        }
        hatch_style = "///"

        # Filter relevant rows
        filtered = df[df["metric"].isin(metrics)]

        # Unique measures
        measures = filtered["measure"].unique()

        _, axs = plt.subplots(
            len(measures), 1, figsize=(14, 5 * len(measures)), sharex=True
        )

        if len(measures) == 1:
            axs = [axs]

        for i, measure in enumerate(measures):
            ax = axs[i]
            d = filtered[filtered["measure"] == measure]

            # Pivot for stacked bars
            reference = d.pivot(
                index="date", columns="metric", values="reference risk"
            ).fillna(0)
            averted = (
                d.pivot(index="date", columns="metric", values="averted risk")
                .fillna(0)
                .sum(axis=1)
            )
            risk = (
                d.pivot(index="date", columns="metric", values="risk")
                .fillna(0)
                .sum(axis=1)
            )

            x = np.arange(len(reference.index))
            width = 0.35  # width of each bar
            spacing = width + 0.05  # space between stacked bar and residual risk bar

            bottom = np.zeros(len(x))

            # Stacked + hatched
            for metric in metrics:
                h = reference[metric].values
                # Main reference segment
                ax.bar(
                    x,
                    h,
                    bottom=bottom,
                    width=width,
                    color=colors[metric],
                    label=metric,
                    edgecolor=colors[metric],
                    linewidth=0.5,
                )
                bottom += h
                # Hatched averted overlay

            ax.bar(
                x + spacing,
                risk,
                width=width,
                color="grey",
                edgecolor="grey",
                label="residual risk",
                linewidth=0.5,
            )
            ax.bar(
                x + spacing,
                averted,
                bottom=risk,
                width=width,
                color="none",
                edgecolor="grey",
                hatch=hatch_style,
                linewidth=0.5,
            )

            # Labels and ticks
            ax.set_title(f"Measure: {measure}")
            ax.set_xticks(x + spacing / 2)
            ax.set_xticklabels(reference.index, rotation=45)

            # Custom legend to add hatch explanation
            handles = [mpatches.Patch(facecolor=colors[m], label=m) for m in metrics]
            handles.append(
                mpatches.Patch(
                    facecolor="white",
                    edgecolor="grey",
                    hatch=hatch_style,
                    label="averted with measure",
                )
            )
            handles.append(mpatches.Patch(facecolor="gray", label="residual risk"))
            ax.legend(handles=handles, loc="upper left")

        return axs

    def plot(
        self,
        measures: list[str] | None = None,
        per_date: bool = False,
        variables: list[str] | None = None,
        groups: list[int] | list[str] | None = None,
        **kwargs,
    ):
        # TODO: redo with matplotlib to have some stacked and bottom
        variables = self._risk_vars if variables is None else variables
        groups = ["All"] if groups is None else groups
        to_plot, time_g = (
            (self.per_date_risk_metrics(**kwargs), "date")
            if per_date
            else (
                self.per_period_risk_metrics(**kwargs),
                "period",
            )
        )
        measures = (
            np.setdiff1d(to_plot["measure"].unique(), ["no_measure"])
            if measures is None
            else measures
        )

        to_plot["averted risk"] *= -1

        to_plot = to_plot.melt(id_vars=[time_g, "group", "measure", "metric"])
        to_plot = to_plot.loc[
            (to_plot["metric"] == "aai")
            & (to_plot["variable"].isin(variables))
            & (to_plot["group"].isin(groups))
            & (to_plot["measure"].isin(measures))
        ]
        g = sns.catplot(
            to_plot.loc[to_plot["group"] == "All"],
            col="measure",
            col_wrap=3 if 3 < len(measures) else len(measures),
            hue="variable",
            hue_order=variables,
            x=time_g,
            y="value",
            sharey=False,
            kind="bar",
        )
        if per_date:
            g.tick_params(rotation=90)
        return g


class _PlannedMeasuresAppraiser(MeasuresAppraiser):
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
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual: bool = True,
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
            interval_freq=interval_freq,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
            cost_disc=cost_disc,
            risk_transf_cover=risk_transf_cover,
            risk_transf_attach=risk_transf_attach,
            calc_residual=calc_residual,
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
        base_metrics = super()._generic_metrics(
            npv,
            metric_name,
            metric_meth,
            measures,
            **kwargs,
        )
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
            if measure.cost_income.freq != self._interval_freq:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self._interval_freq}). "
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

    def plot(
        self,
        measures: list[str] | None = None,
        per_date: bool = False,
        variables: list[str] | None = None,
        groups: list[int] | list[str] | None = None,
        **kwargs,
    ):
        variables = self._risk_vars if variables is None else variables
        groups = ["All"] if groups is None else groups
        to_plot, time_g = (
            (self.per_date_risk_metrics(**kwargs), "date")
            if per_date
            else (
                self.per_period_risk_metrics(**kwargs),
                "period",
            )
        )
        measures = to_plot["measure"].unique() if measures is None else measures

        to_plot["averted risk"] *= -1

        to_plot = to_plot.melt(id_vars=[time_g, "group", "measure", "metric"])
        to_plot = to_plot.loc[
            (to_plot["metric"] == "aai")
            & (to_plot["variable"].isin(variables))
            & (to_plot["group"].isin(groups))
            & (to_plot["measure"].isin(measures))
        ]

        if per_date:
            planner_t = {
                label: (pd.Timestamp(v1), pd.Timestamp(v2))
                for label, (v1, v2) in self.planner.items()
            }
            _, axs = plt.subplots(
                2, figsize=(10, 8), gridspec_kw={"height_ratios": [12, 1]}
            )
            sns.barplot(
                to_plot.loc[to_plot["group"] == "All"],
                hue="variable",
                hue_order=variables,
                x="date",
                y="value",
                ax=axs[0],
            )
            axs[0].tick_params(rotation=45)
            y = 0
            for label_text, (start, end) in planner_t.items():
                axs[1].barh(
                    y,
                    (end - start).days,
                    left=start,
                    height=0.6,
                    color="skyblue",
                    edgecolor="k",
                )
                axs[1].text(
                    start,
                    y,
                    "  " + label_text,
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="black",
                )
                y += 1

            axs[1].xaxis.set_major_locator(mdates.YearLocator(5))
            axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            axs[1].set_xlim(min(to_plot["date"]), max(to_plot["date"]))
            axs[1].set_yticks([])
            axs[0].set_xticks([])
            axs[0].set_xlabel("")
            return axs
        else:
            g = sns.barplot(
                to_plot.loc[to_plot["group"] == "All"],
                hue="variable",
                hue_order=variables,
                x=time_g,
                y="value",
            )
            g.tick_params(axis="x", which="both", rotation=45)
            return g

    # def plot_CB_summary(
    #     self,
    #     metric="aai",
    #     measure_colors=None,
    #     y_label="Risk",
    #     title="Benefit and Benefit/Cost Ratio by Measure",
    # ):
    #     raise NotImplementedError("Not Implemented for that class")
    #     df = self.calc_CB(dately=True)
    #     df_plan = df.groupby(["group", "metric"], as_index=False).agg(
    #         start_date=pd.NamedAgg(column="date", aggfunc="min"),
    #         end_date=pd.NamedAgg(column="date", aggfunc="max"),
    #         base_risk=pd.NamedAgg(column="base risk", aggfunc="sum"),
    #         residual_risk=pd.NamedAgg(column="residual risk", aggfunc="sum"),
    #         averted_risk=pd.NamedAgg(column="averted risk", aggfunc="sum"),
    #         cost_net=pd.NamedAgg(column="cost (net)", aggfunc="sum"),
    #     )
    #     df_plan["measure"] = "Whole risk period"
    #     df.columns = df.columns.str.replace("_", " ")
    #     df["B/C ratio"] = (df["averted risk"] / df["cost net"]).fillna(0.0)
    #     plot_CB_summary(
    #         df,
    #         metric=metric,
    #         measure_colors=measure_colors,
    #         y_label=y_label,
    #         title=title,
    #     )

    # def plot_dately(
    #     self,
    #     to_plot="residual risk",
    #     metric="aai",
    #     y_label=None,
    #     title=None,
    #     with_measure=True,
    #     measure_colors=None,
    # ):
    #     plot_dately(
    #         self.calc_CB(dately=True).sort_values("residual risk"),
    #         to_plot=to_plot,
    #         with_measure=with_measure,
    #         metric=metric,
    #         y_label=y_label,
    #         title=title,
    #         measure_colors=measure_colors,
    #     )

    # def plot_waterfall(self, ax=None, start_date=None, end_date=None):
    #     df = self.calc_CB(dately=True)
    #     averted = df.loc[
    #         (df["date"] == 2080)
    #         & (df["metric"] == "aai")
    #         & (df["measure"] != "no_measure")
    #     ]
    #     ax = super().plot_waterfall(ax=ax, start_date=start_date, end_date=end_date)
    #     ax.bar("Averted risk", ax.patches[-1].get_height(), width=1, visible=False)
    #     # ax.text(
    #     #     x=ax.get_xticks()[-1] - ax.patches[-1].get_width() / 2 + 0.02,
    #     #     y=ax.patches[-1].get_height() * 0.96,
    #     #     ha="left",
    #     #     s="Averted risk",
    #     #     size=12,
    #     # )
    #     averted = averted.sort_values("averted risk")
    #     for i, meas in enumerate(averted["measure"].unique()):
    #         measure_risk = averted.loc[
    #             (averted["measure"] == meas), "averted risk"
    #         ].values[0]
    #         x_arrow = (
    #             ax.get_xticks()[-1]
    #             - ax.patches[-1].get_width() / 2
    #             + 0.1
    #             + (ax.patches[-1].get_width() / averted["measure"].nunique()) * i
    #         )
    #         top_arrow = ax.patches[-1].get_height()
    #         bottom_arrow = top_arrow - measure_risk
    #         ax.annotate(
    #             "",
    #             xy=(x_arrow, bottom_arrow),
    #             xytext=(x_arrow, top_arrow),
    #             arrowprops=dict(
    #                 facecolor="tab:green", width=12, headwidth=20, headlength=10
    #             ),
    #         )
    #         ax.text(
    #             x=x_arrow,
    #             y=top_arrow - (top_arrow - bottom_arrow) / 2,
    #             va="center",
    #             ha="center",
    #             s=meas,
    #             rotation=-90,
    #             size="x-small",
    #         )

    #     return ax


# class AdaptationPlansAppraiser:
#     def __init__(
#         self,
#         snapshots: SnapshotsCollection,
#         measure_set: MeasureSet,
#         plans: list[dict[str, tuple[int, int]]],
#         use_net_present_value: bool = True,
#         cost_disc: DiscRates | None = None,
#         risk_disc: DiscRates | None = None,
#         metrics: list[str] = ["aai", "eai", "rp"],
#         return_periods: list[int] = [100, 500, 1000],
#     ):
#         self._use_npv = use_net_present_value
#         self.plans = [
#             _PlannedMeasuresAppraiser(
#                 snapshots=snapshots,
#                 measure_set=measure_set,
#                 planner=plan,
#                 cost_disc=cost_disc,
#                 risk_disc=risk_disc,
#                 metrics=metrics,
#                 return_periods=return_periods,
#             )
#             for plan in plans
#         ]

#     def calc_CB(
#         self,
#     ):
#         res = []
#         for plan in self.plans:
#             planner = plan.planner
#             df = plan.calc_CB(net_present_value=self._use_npv, dately=True)
#             df = df.drop("date", axis=1)
#             df = df.groupby(["group", "metric"], as_index=False).sum(numeric_only=True)
#             df["plan"] = format_periods_dict(planner)
#             res.append(df)

#         return (
#             pd.concat(res)
#             .set_index(["plan", "group", "metric"])
#             .reset_index()
#             .sort_values(["metric", "plan"])
#         )


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
