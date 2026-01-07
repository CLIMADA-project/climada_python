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
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

from climada.engine.option_appraisal.constants import (
    AVERTED_RISK_NAME,
    MEASURE_NET_COST_NAME,
    REFERENCE_RISK_NAME,
    RESIDUAL_RISK_NAME,
)
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    CONTRIBUTIONS_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_TIME_RESOLUTION,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    PERIOD_COL_NAME,
    RETURN_PERIOD_METRIC_NAME,
    RISK_COL_NAME,
)
from climada.trajectories.impact_calc_strat import ImpactComputationStrategy
from climada.trajectories.interpolated_trajectory import InterpolatedRiskTrajectory
from climada.trajectories.interpolation import ImpactInterpolationStrategy
from climada.trajectories.riskperiod import CalcRiskMetricsPeriod, CalcRiskMetricsPoints
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.static_trajectory import StaticRiskTrajectory
from climada.trajectories.trajectory import DEFAULT_DF_COLUMN_PRIORITY, DEFAULT_RP
from climada.util import log_level
from climada.util.config import CONFIG
from climada.util.dataframe_handling import reorder_dataframe_columns

tqdm.pandas()

LOGGER = logging.getLogger(__name__)


class StaticAppraiser(StaticRiskTrajectory):
    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        return_periods: Iterable[int] = DEFAULT_RP,
        all_groups_name: str = DEFAULT_ALLGROUP_NAME,
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        """Initialize a new `StaticAppraiser`.

        Parameters
        ----------
        snapshots_list : list[Snapshot]
            The list of `Snapshot` object to compute risk from.
        measure_set: MeasureSet
            The set of adaptation measures to appraise.
        return_periods: list[int], optional
            The return periods to use when computing the `return_periods_metric`.
            Defaults to `DEFAULT_RP` ([20, 50, 100]).
        all_groups_name: str, optional
            The string that should be used to define "all exposure points" subgroup.
            Defaults to `DEFAULT_ALLGROUP_NAME` ("All").
        risk_disc_rates: DiscRates, optional
            The discount rate to apply to future risk. Defaults to None.
        cost_disc_rates: DiscRates, optional
            The discount rate to apply to future costs (of adaptation measures).
            Defaults to None.
        impact_computation_strategy: ImpactComputationStrategy, optional
            The method used to calculate the impact from the (Haz,Exp,Vul)
            of the two snapshots. Defaults to :class:`ImpactCalcComputation`.

        """

        self._cost_disc_rates = cost_disc_rates
        self.measure_set = copy.deepcopy(measure_set)
        super().__init__(
            snapshots_list,
            return_periods=return_periods,
            all_groups_name=all_groups_name,
            risk_disc_rates=risk_disc_rates,
            impact_computation_strategy=impact_computation_strategy,
        )
        self._risk_metrics_calculators = self._add_adaptation_metrics_calculators(
            self._risk_metrics_calculators, measure_set
        )

    @staticmethod
    def _add_adaptation_metrics_calculators(
        risk_metrics_calculators, measure_set: MeasureSet
    ) -> list[CalcRiskMetricsPoints]:
        """Adds the risk metric calculators for the different adaptation options."""
        calculators = [risk_metrics_calculators] + [
            risk_metrics_calculators.apply_measure(meas)
            for _, meas in measure_set.measures().items()
        ]
        return calculators

    @property
    def cost_disc_rates(self) -> DiscRates | None:
        """The discount rate applied to compute net present values of costs.
        None means no discount rate.

        Notes
        -----

        Changing its value resets the metrics.
        """
        return self._cost_disc_rates

    @cost_disc_rates.setter
    def cost_disc_rates(self, value, /):
        if not isinstance(value, DiscRates):
            raise ValueError("Risk discount needs to be a `DiscRates` object.")

        self._reset_metrics()
        self._cost_disc_rates = value

    def _generic_metrics(
        self,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generic method to compute metrics based on the provided metric name and method.

        This method calls the appropriate method from each calculators (corresponding to
        each adaptation) to return the results for the given metric,
        in a tidy formatted dataframe.

        It first checks whether the requested metric is a valid one.
        Then looks for a possible cached value and otherwised asks the
        calculators (`self._risk_metric_calculators`) to run the computation.
        The results are then regrouped in a nice and tidy DataFrame.
        If a `risk_disc_rates` was set, values are converted to net present values.
        Results are then cached within `self._<metric_name>_metrics` and returned.

        Parameters
        ----------
        metric_name : str, optional
            The name of the metric to return results for.
        metric_meth : str, optional
            The name of the specific method of the calculator to call.

        Returns
        -------
        pd.DataFrame
            A tidy formatted dataframe of the risk metric computed for the
            different snapshots.

        Raises
        ------
        NotImplementedError
            If the requested metric is not part of `POSSIBLE_METRICS`.
        ValueError
            If either of the arguments are not provided.

        """

        if metric_name is None or metric_meth is None:
            raise ValueError("Both metric_name and metric_meth must be provided.")

        if metric_name not in self.POSSIBLE_METRICS:
            raise NotImplementedError(
                f"{metric_name} not implemented ({self.POSSIBLE_METRICS})."
            )

        # Construct the attribute name for storing the metric results
        attr_name = f"_{metric_name}_metrics"

        if getattr(self, attr_name) is not None:
            LOGGER.debug("Returning cached %s", attr_name)
            return getattr(self, attr_name)

        LOGGER.debug("Computing %s", attr_name)
        with log_level(level="WARNING", name_prefix="climada"):
            tmp = [
                getattr(calc_period, metric_meth)(**kwargs)
                for calc_period in self._risk_metrics_calculators
            ]

        try:
            tmp = pd.concat(tmp)
        except ValueError as exc:
            if str(exc) == "All objects passed were None":
                return pd.DataFrame()
            raise exc

        if len(tmp) == 0:
            return pd.DataFrame()

        tmp = self._metric_post_treatment(tmp, metric_name)

        if CONFIG.trajectory_caching.bool():
            LOGGER.debug("All computing done, caching value.")
            setattr(self, attr_name, tmp)
            return getattr(self, attr_name)

        return tmp

    def _metric_post_treatment(
        self, metric_df: pd.DataFrame, metric_name: str
    ) -> pd.DataFrame:
        # Notably for per_group_aai being None:
        metric_df = self._handle_group_categories(metric_df)
        if self._risk_disc_rates:
            LOGGER.debug("Found risk discount rate. Computing NPV.")
            metric_df = self.npv_transform(metric_df, self._risk_disc_rates)

        LOGGER.debug(f"Computing averted risk for: {metric_name}.")
        metric_df = self._calc_averted(metric_df)
        metric_df = reorder_dataframe_columns(metric_df, DEFAULT_DF_COLUMN_PRIORITY)
        return metric_df

    def _handle_group_categories(self, metric_df: pd.DataFrame) -> pd.DataFrame:
        if self._all_groups_name not in metric_df[GROUP_COL_NAME].cat.categories:
            metric_df[GROUP_COL_NAME] = metric_df[GROUP_COL_NAME].cat.add_categories(
                [self._all_groups_name]
            )
            metric_df[GROUP_COL_NAME] = metric_df[GROUP_COL_NAME].fillna(
                self._all_groups_name
            )

        return metric_df

    @staticmethod
    def _calc_averted(base_metrics: pd.DataFrame) -> pd.DataFrame:
        def subtract_no_measure(group, no_measure, merger):
            # Merge with no_measure to get the corresponding NO_MEASURE_VALUE value
            merged = group.merge(
                no_measure, on=merger, suffixes=("", "_" + NO_MEASURE_VALUE)
            )
            # Subtract the NO_MEASURE_VALUE risk from the current risk
            merged[REFERENCE_RISK_NAME] = merged[RISK_COL_NAME + "_" + NO_MEASURE_VALUE]
            merged[AVERTED_RISK_NAME] = (
                merged[RISK_COL_NAME + "_" + NO_MEASURE_VALUE] - merged[RISK_COL_NAME]
            )
            return merged[
                list(group.columns) + [REFERENCE_RISK_NAME, AVERTED_RISK_NAME]
            ]

        no_measures_metrics = base_metrics[
            base_metrics[MEASURE_COL_NAME] == NO_MEASURE_VALUE
        ].copy()
        merger = [GROUP_COL_NAME, METRIC_COL_NAME, DATE_COL_NAME]
        if COORD_ID_COL_NAME in base_metrics.columns:
            merger.append(COORD_ID_COL_NAME)

        return base_metrics.groupby(
            [GROUP_COL_NAME, METRIC_COL_NAME, DATE_COL_NAME],
            group_keys=False,
            dropna=False,
            observed=False,
        ).apply(subtract_no_measure, no_measure=no_measures_metrics, merger=merger)


class InterpolatedAppraiser(InterpolatedRiskTrajectory):
    _risk_vars = [
        REFERENCE_RISK_NAME,
        AVERTED_RISK_NAME,
        RISK_COL_NAME,
    ]

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        return_periods: Iterable[int] = DEFAULT_RP,
        time_resolution: str = DEFAULT_TIME_RESOLUTION,
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
        interpolation_strategy: ImpactInterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        self._cost_disc_rates = cost_disc_rates
        self.measure_set = copy.deepcopy(measure_set)
        super().__init__(
            snapshots_list,
            time_resolution=time_resolution,
            return_periods=return_periods,
            risk_disc_rates=risk_disc_rates,
            interpolation_strategy=interpolation_strategy,
            impact_computation_strategy=impact_computation_strategy,
        )
        self._risk_metrics_calculators += self._add_adaptation_metrics_calculators(
            self._risk_metrics_calculators, measure_set
        )

    @staticmethod
    def _add_adaptation_metrics_calculators(
        risk_metrics_calculators, measure_set: MeasureSet
    ) -> list[CalcRiskMetricsPeriod]:
        adapt_calc = []
        for _, measure in measure_set.measures().items():
            LOGGER.debug(f"Creating measures risk_period for measure {measure.name}")
            meas_p = [
                rmcalc.apply_measure(measure) for rmcalc in risk_metrics_calculators
            ]
            adapt_calc += meas_p
        return adapt_calc

    @property
    def cost_disc_rates(self) -> DiscRates | None:
        """The discount rate applied to compute net present values of costs.
        None means no discount rate.

        Notes
        -----

        Changing its value resets the metrics.
        """
        return self._cost_disc_rates

    @cost_disc_rates.setter
    def cost_disc_rates(self, value, /):
        if not isinstance(value, DiscRates):
            raise ValueError("Risk discount needs to be a `DiscRates` object.")

        self._reset_metrics()
        self._cost_disc_rates = value

    def _generic_metrics(
        self,
        metric_name=None,
        metric_meth=None,
        measures: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        LOGGER.debug(f"Computing base metric: {metric_name}.")
        base_metrics = super()._generic_metrics(metric_name, metric_meth, **kwargs)
        if base_metrics is not None:
            LOGGER.debug(f"Computing averted risk for: {metric_name}.")
            base_metrics = self._calc_averted(base_metrics)
            no_measures = base_metrics[
                base_metrics[MEASURE_COL_NAME] == NO_MEASURE_VALUE
            ].copy()
            no_measures[REFERENCE_RISK_NAME] = no_measures[RISK_COL_NAME]
            no_measures[AVERTED_RISK_NAME] = 0.0
            no_measures[MEASURE_NET_COST_NAME] = 0.0
            LOGGER.debug(f"Computing cash flow for: {metric_name}.")
            cash_flow_metrics = self.annual_cash_flows()
            LOGGER.debug(f"Merging with base metric: {metric_name}.")
            base_metrics = base_metrics.merge(
                cash_flow_metrics[
                    [DATE_COL_NAME, MEASURE_COL_NAME, MEASURE_NET_COST_NAME]
                ],
                on=[MEASURE_COL_NAME, DATE_COL_NAME],
            )
            LOGGER.debug(f"Merging with no measure: {metric_name}.")
            base_metrics = pd.concat([no_measures, base_metrics])

            if measures is not None:
                base_metrics = base_metrics.loc[
                    base_metrics[MEASURE_COL_NAME].isin(measures)
                ].reset_index()

        return base_metrics

    @staticmethod
    def _calc_averted(base_metrics: pd.DataFrame) -> pd.DataFrame:
        def subtract_no_measure(group, no_measure, merger):
            # Merge with no_measure to get the corresponding NO_MEASURE_VALUE value
            merged = group.merge(
                no_measure, on=merger, suffixes=("", "_" + NO_MEASURE_VALUE)
            )
            # Subtract the NO_MEASURE_VALUE risk from the current risk
            merged[REFERENCE_RISK_NAME] = merged[RISK_COL_NAME + "_" + NO_MEASURE_VALUE]
            merged[AVERTED_RISK_NAME] = (
                merged[RISK_COL_NAME + "_" + NO_MEASURE_VALUE] - merged[RISK_COL_NAME]
            )
            return merged[
                list(group.columns) + [REFERENCE_RISK_NAME, AVERTED_RISK_NAME]
            ]

        no_measures_metrics = base_metrics[
            base_metrics[MEASURE_COL_NAME] == NO_MEASURE_VALUE
        ].copy()
        merger = [GROUP_COL_NAME, METRIC_COL_NAME, DATE_COL_NAME]
        if COORD_ID_COL_NAME in base_metrics.columns:
            merger.append(COORD_ID_COL_NAME)

        return base_metrics.groupby(
            [GROUP_COL_NAME, METRIC_COL_NAME, DATE_COL_NAME],
            group_keys=False,
            dropna=False,
            observed=False,
        ).apply(subtract_no_measure, no_measure=no_measures_metrics, merger=merger)

    @classmethod
    def _date_to_period_agg(
        cls,
        metric_df: pd.DataFrame,
        grouper: list[str] | None = None,
        time_unit="year",
        colname: str | list[str] | None = None,
    ) -> pd.DataFrame:
        colname = cls._risk_vars if colname is None else colname
        if grouper is None:
            grouper = cls._grouper
        return super()._date_to_period_agg(metric_df, grouper, time_unit, colname)

    def per_date_CB(
        self,
        metrics: list[str] = [
            AAI_METRIC_NAME,
            RETURN_PERIOD_METRIC_NAME,
            AAI_PER_GROUP_METRIC_NAME,
        ],
        include_no_measure=False,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_date_risk_metrics(metrics, **kwargs)
        if not include_no_measure:
            metrics_df = metrics_df[metrics_df[MEASURE_COL_NAME] != NO_MEASURE_VALUE]

        metrics_df.rename(columns={RISK_COL_NAME: RESIDUAL_RISK_NAME}, inplace=True)
        metrics_df["cumulated measure cost"] = metrics_df.groupby(
            [GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME], observed=True
        )[MEASURE_NET_COST_NAME].cumsum()
        metrics_df["cumulated measure benefit"] = metrics_df.groupby(
            [GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME], observed=True
        )[AVERTED_RISK_NAME].cumsum()
        metrics_df["cost/benefit ratio"] = (
            metrics_df["cumulated measure cost"]
            / metrics_df["cumulated measure benefit"]
        )
        return metrics_df

    def per_period_CB(
        self,
        metrics: list[str] = [
            AAI_METRIC_NAME,
            RETURN_PERIOD_METRIC_NAME,
            AAI_PER_GROUP_METRIC_NAME,
        ],
        npv: bool = True,
        include_no_measure=False,
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        metrics_df = self.per_period_risk_metrics(metrics, **kwargs)
        cost_df = self.annual_cash_flows()
        cost_df = self._date_to_period_agg(
            cost_df, grouper=[MEASURE_COL_NAME], colname=MEASURE_NET_COST_NAME
        )
        metrics_df = metrics_df.merge(
            cost_df, on=[PERIOD_COL_NAME, MEASURE_COL_NAME], how="outer"
        )
        metrics_df[MEASURE_NET_COST_NAME] = metrics_df[MEASURE_NET_COST_NAME].fillna(
            0.0
        )
        if not include_no_measure:
            metrics_df = metrics_df[metrics_df[MEASURE_COL_NAME] != NO_MEASURE_VALUE]

        return metrics_df

    def annual_cash_flows(self):
        res = []
        for meas_name, measure in self.measure_set.measures().items():
            need_agg = False
            if measure.cost_income.freq != self.time_resolution:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self.time_resolution}). "
                        f"Cash flows will be aggregated to {measure.cost_income.freq} "
                        "but this **may** lead to inconsistencies."
                    ),
                    stacklevel=2,
                )

            df = measure.cost_income.calc_cashflows(
                impl_date=self.start_date,
                start_date=self.start_date,
                end_date=self.end_date,
                disc=self.cost_disc_rates,
            )
            if need_agg:
                df = df.groupby(df[DATE_COL_NAME].dt.year, as_index=False).agg(
                    {
                        "net": "sum",
                        "cost": "sum",
                        "income": "sum",
                        DATE_COL_NAME: "first",
                    }
                )
            df[MEASURE_COL_NAME] = meas_name
            res.append(df)
        df = pd.concat(res)
        df["net"] *= -1
        df = df.rename(columns={"net": MEASURE_NET_COST_NAME})
        return df

    def _calc_waterfall_CB_plot_data(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_contributions = self.risk_contributions_metrics()
        risk_contributions = risk_contributions.loc[
            (risk_contributions[DATE_COL_NAME] >= str(start_date))
            & (risk_contributions[DATE_COL_NAME] <= str(end_date))
            & (risk_contributions[MEASURE_COL_NAME] != NO_MEASURE_VALUE)
        ]
        risk_contributions = risk_contributions.set_index(
            [DATE_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME]
        )[
            [
                RISK_COL_NAME,
                REFERENCE_RISK_NAME,
                AVERTED_RISK_NAME,
                MEASURE_NET_COST_NAME,
            ]
        ].unstack()
        return risk_contributions

    def plot_per_date_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        """Plot a waterfall chart of risk contributions over a specified date range.

        This method generates a stacked bar chart to visualize the
        risk contributions between specified start and end dates, for each date in between.
        If no dates are provided, it defaults to the start and end dates of the risk trajectory.
        See the notes on how risk is attributed to each contributions.

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
        The "risk contributions" are plotted such that the increase in risk due to the hazard contribution
        really denotes the difference between the risk associated with both future exposure and hazard
        compared to the risk associated with future exposure and present hazard.
        """
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        df = self._calc_waterfall_CB_plot_data(start_date=start_date, end_date=end_date)
        df = df.swaplevel()
        # Unique measures
        measures = df.index.get_level_values(0).unique()

        value_label = "USD"

        _, axs = plt.subplots(
            1 + len(measures),
            1,
            figsize=(14, 5 * len(measures)),
            sharex=False,
            sharey=False,
        )
        self.plot_time_waterfall(ax=axs[0], start_date=start_date, end_date=end_date)

        for i, measure in enumerate(measures):
            ax = axs[i + 1]
            d = df.loc[measure]

            # Pivot for stacked bars
            averted = d.loc[:, AVERTED_RISK_NAME].sum(axis=1)
            risk = d.loc[:, RISK_COL_NAME].sum(axis=1)
            ax.stackplot(
                d.index.to_timestamp(),
                [risk, averted],
                labels=[RESIDUAL_RISK_NAME, "Averted"],
                colors=["purple", "pink"],
                hatch=["", "/"],
            )
            # Labels and ticks
            ax.set_title(f"Measure: {measure}")
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(ticker.EngFormatter())
            ax.set_ylabel(value_label)
            ax.legend()

        return axs

    def plot_waterfall_CB(
        self,
        risk_reference_date: datetime.date | None = None,
        measure_effect_date: datetime.date | None = None,
        measures: list[str] | None = None,
    ):
        risk_reference_period = pd.Period(
            self.start_date if measure_effect_date is None else risk_reference_date,
            self.time_resolution,
        )
        measure_effect_period = pd.Period(
            self.end_date if measure_effect_date is None else measure_effect_date,
            self.time_resolution,
        )
        risk_contribution = self.risk_contributions_metrics()
        risk_contribution = risk_contribution.set_index(DATE_COL_NAME).loc[
            [risk_reference_period, measure_effect_period]
        ]
        meas = (
            np.setdiff1d(risk_contribution.measure.unique(), [NO_MEASURE_VALUE])
            if measures is None
            else measures
        )
        num_cols = 2 if 2 < len(meas) else len(meas)
        num_rows = len(meas) // num_cols
        risk_contribution = risk_contribution.loc[
            risk_contribution[MEASURE_COL_NAME].isin(meas)
        ]
        risk_contribution.set_index(
            [MEASURE_COL_NAME, METRIC_COL_NAME], inplace=True, append=True
        )
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_cols * 8, num_rows * 5)
        )

        labels = [
            f"Base Risk in {risk_reference_date}",
            "Exposure contribution",
            "Hazard contribution",
            "Vulnerability contribution",
            "Interaction contribution",
            f"Total risk in {measure_effect_date}",
        ]
        reference_risk = risk_contribution.loc[
            (str(risk_reference_period), meas[0], "base risk"), REFERENCE_RISK_NAME
        ]
        base_risk_when_measure_effect = risk_contribution.loc[
            (str(measure_effect_period), meas[0]), REFERENCE_RISK_NAME
        ].sum()

        for i, measure in enumerate(meas):
            exposure_contribution = risk_contribution.loc[
                (str(measure_effect_period), measure, "exposure contribution"),
                REFERENCE_RISK_NAME,
            ]
            hazard_contribution = risk_contribution.loc[
                (str(measure_effect_period), measure, "hazard contribution"),
                REFERENCE_RISK_NAME,
            ]
            vulnerability_contribution = risk_contribution.loc[
                (str(measure_effect_period), measure, "vulnerability contribution"),
                REFERENCE_RISK_NAME,
            ]
            interaction_contribution = risk_contribution.loc[
                (str(measure_effect_period), measure, "interaction contribution"),
                REFERENCE_RISK_NAME,
            ]
            averted_risk = risk_contribution.loc[
                (str(measure_effect_period), measure), AVERTED_RISK_NAME
            ].sum()
            values = [
                reference_risk,
                exposure_contribution,
                hazard_contribution,
                vulnerability_contribution,
                interaction_contribution,
                base_risk_when_measure_effect,
            ]
            bottoms = [
                0.0,
                reference_risk,
                reference_risk + exposure_contribution,
                reference_risk + exposure_contribution + hazard_contribution,
                reference_risk
                + exposure_contribution
                + hazard_contribution
                + vulnerability_contribution,
                0.0,
            ]
            axs[i].bar(
                labels,
                values,
                bottom=bottoms,
                edgecolor="black",
                color=[
                    "tab:blue",
                    "tab:olive",
                    "tab:cyan",
                    "tab:brown",
                    "tab:pink",
                    "tab:blue",
                ],
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

            arrow = mpatches.FancyArrowPatch(
                (5, base_risk_when_measure_effect),
                (5, base_risk_when_measure_effect - averted_risk),
                mutation_scale=50,
                color="red",
            )
            axs[i].text(
                x=5,
                y=(2 * base_risk_when_measure_effect - averted_risk) / 2,
                s="Averted",
                rotation=90,
                ha="center",
                va="center",
            )
            axs[i].add_patch(arrow)
            # Construct y-axis label and title based on parameters
            value_label = "USD (Average annual value)"
            axs[i].set_ylabel(value_label)
            axs[i].tick_params(
                axis="x",
                labelrotation=90,
            )
        plt.tight_layout()
        return axs


class PlannedAdaptationAppraiser(InterpolatedAppraiser):
    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        planner: (
            dict[str, tuple[int, int]] | dict[str, tuple[datetime.date, datetime.date]]
        ),
        interval_freq: str = "YS",
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
        interpolation_strategy: ImpactInterpolationStrategy | None = None,
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
            risk_disc_rates=risk_disc_rates,
            cost_disc_rates=cost_disc_rates,
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
                self.measure_set.append(measure)
            elif len(measure_name_list) == 1:
                measure = self.measure_set._data[measure_name_list[0]]
            else:
                measure = None

            LOGGER.debug(f"Fetching risk_periods within {start_date} and {end_date}")
            periods = self._get_risk_periods(
                risk_periods, start_date, end_date, strict=False
            )
            if measure:
                LOGGER.debug(
                    f"Creating measures risk_period for measure {measure.name} on {periods}"
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
        base_metrics = base_metrics.set_index(
            [MEASURE_COL_NAME, DATE_COL_NAME]
        ).sort_index()
        mask = pd.Series(False, index=base_metrics.index)
        for (start, end), measure_name_list in self._planning.items():
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            mask |= (
                (
                    base_metrics.index.get_level_values(MEASURE_COL_NAME)
                    == "_".join(measure_name_list)
                )
                & (base_metrics.index.get_level_values(DATE_COL_NAME) >= start)
                & (base_metrics.index.get_level_values(DATE_COL_NAME) <= end)
            )

        no_measure_mask = mask.groupby(DATE_COL_NAME).sum() == 0
        mask.loc[
            pd.IndexSlice[NO_MEASURE_VALUE], no_measure_mask[no_measure_mask].index
        ] = True

        return base_metrics[mask].reset_index().sort_values(DATE_COL_NAME)

    def _calc_per_measure_annual_cash_flows(self, npv: bool):
        res = []
        for meas_name, (start, end) in self.planner.items():
            need_agg = False
            measure = self.measure_set.measures()[meas_name]
            if measure.cost_income.freq != self.time_resolution:
                need_agg = True
                warnings.warn(
                    (
                        f"{meas_name} has a different CostIncome interval frequency ({measure.cost_income.freq}) "
                        f"than the MeasureAppraiser ({self.time_resolution}). "
                        f"Cash flows will be aggregated to {measure.cost_income.freq} "
                        "but this **may** lead to inconsistencies."
                    ),
                    stacklevel=2,
                )

            df = measure.cost_income.calc_cashflows(
                impl_date=start,
                start_date=start,
                end_date=end,
                disc=self.cost_disc_rates if npv else None,
            )
            if need_agg:
                df = df.groupby(df[DATE_COL_NAME].dt.year, as_index=False).agg(
                    {
                        "net": "sum",
                        "cost": "sum",
                        "income": "sum",
                        DATE_COL_NAME: "first",
                    }
                )
            df[MEASURE_COL_NAME] = meas_name
            res.append(df)
        df = pd.concat(res)
        df = df.groupby(DATE_COL_NAME, as_index=False).agg(
            {
                col: ("sum" if is_numeric_dtype(df[col]) else lambda x: "_".join(x))
                for col in df.columns
                if col != DATE_COL_NAME
            }
        )
        df["net"] *= -1
        df = df.rename(columns={"net": MEASURE_NET_COST_NAME})
        return df

    def plot_per_date_waterfall_CB(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        # Unique measures
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        df = self._calc_waterfall_CB_plot_data(
            start_date=start_date, end_date=end_date, include_no_measure=True
        )
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

        measures = (
            df.index.get_level_values(0)
            .unique()
            .drop(NO_MEASURE_VALUE, errors="ignore")
        )
        reference_risk = df[REFERENCE_RISK_NAME].droplevel(0)
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
        d = df.copy().droplevel(MEASURE_COL_NAME)

        # Pivot for stacked bars
        averted = d.loc[:, AVERTED_RISK_NAME].sum(axis=1)
        risk = d.loc[:, RISK_COL_NAME].sum(axis=1)
        ax.stackplot(
            d.index,
            [risk, averted],
            labels=[RESIDUAL_RISK_NAME, "Averted"],
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
        risk_contribution = self._calc_waterfall_CB_plot_data(
            start_date=start_date, end_date=end_date
        )
        fig, ax = plt.subplots(figsize=(8, 5))

        labels = [
            RISK_COL_NAME,
            AVERTED_RISK_NAME,
            RESIDUAL_RISK_NAME,
            "Measure cost",
            "Cost benefit",
        ]
        # measure_costs = risk_contribution.loc[:,(MEASURE_NET_COST_NAME,"base risk")].unstack().sum()
        average_risk = (
            risk_contribution.mean()
            .unstack()
            .T.agg(
                {
                    AVERTED_RISK_NAME: "sum",
                    MEASURE_NET_COST_NAME: "mean",
                    REFERENCE_RISK_NAME: "sum",
                    RISK_COL_NAME: "sum",
                }
            )
        )
        # risk_contribution = risk_contribution.loc[str(end_date)]

        m_average_risk = average_risk.copy()
        values = [
            m_average_risk[REFERENCE_RISK_NAME],
            m_average_risk[AVERTED_RISK_NAME],
            m_average_risk[REFERENCE_RISK_NAME] - m_average_risk[AVERTED_RISK_NAME],
            m_average_risk[MEASURE_NET_COST_NAME],
            m_average_risk[AVERTED_RISK_NAME] - m_average_risk[MEASURE_NET_COST_NAME],
        ]
        bottoms = [
            0.0,
            m_average_risk[REFERENCE_RISK_NAME] - m_average_risk[AVERTED_RISK_NAME],
            0.0,
            m_average_risk[REFERENCE_RISK_NAME] - m_average_risk[AVERTED_RISK_NAME],
            m_average_risk[REFERENCE_RISK_NAME]
            - m_average_risk[AVERTED_RISK_NAME]
            + m_average_risk[MEASURE_NET_COST_NAME],
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
                (
                    m_average_risk[REFERENCE_RISK_NAME]
                    - m_average_risk[AVERTED_RISK_NAME]
                ),
            ),
            xycoords="data",
            xytext=(1, m_average_risk[REFERENCE_RISK_NAME]),
            textcoords="data",
            arrowprops=dict(color="red", lw=2, shrink=0.1, width=12),
        )

        ax.annotate(
            "",
            xy=(
                3,
                m_average_risk[MEASURE_NET_COST_NAME]
                + (
                    m_average_risk[REFERENCE_RISK_NAME]
                    - m_average_risk[AVERTED_RISK_NAME]
                ),
            ),
            xycoords="data",
            xytext=(
                3,
                (
                    m_average_risk[REFERENCE_RISK_NAME]
                    - m_average_risk[AVERTED_RISK_NAME]
                ),
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
