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

This file implements interpolated risk trajectory objects, to allow a better evaluation
of risk in between points in time (snapshots).

"""

import datetime
import itertools
import logging
from typing import cast

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.interpolation import (
    AllLinearStrategy,
    InterpolationStrategyBase,
)
from climada.trajectories.riskperiod import CalcRiskMetricsPeriod
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import (
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_RP,
    RiskTrajectory,
)
from climada.util import log_level

LOGGER = logging.getLogger(__name__)

DEFAULT_TIME_RESOLUTION = "Y"

__all__ = ["InterpolatedRiskTrajectory"]


class InterpolatedRiskTrajectory(RiskTrajectory):
    """This class implements interpolated risk trajectories, objects that
    regroup impacts computations for multiple dates, and interpolate risk
    metrics in between.

    This class computes risk metrics over a series of snapshots,
    optionally applying risk discounting. It interpolate risk
    between each pair of snapshots and provides dataframes of risk metric on a
    given time resolution.

    """

    _grouper = ["measure", "metric"]
    """Results dataframe grouper"""

    POSSIBLE_METRICS = [
        "eai",
        "aai",
        "return_periods",
        "risk_contributions",
        "aai_per_group",
    ]
    """Class variable listing the risk metrics that can be computed.

    Currently:

    - eai, expected impact (per exposure point within a period of 1/frequency unit of the hazard object)
    - aai, average annual impact (aggregated eai over the whole exposure)
    - aai_per_group, average annual impact per exposure subgroup (defined from the exposure geodataframe)
    - return_periods, estimated impacts aggregated over the whole exposure for different return periods
    - risk_contributions, estimated contribution part of, respectively exposure, hazard, vulnerability and their interaction to the change in risk over the considered period
    """

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        return_periods: list[int] = DEFAULT_RP,
        time_resolution: str = DEFAULT_TIME_RESOLUTION,
        all_groups_name: str = DEFAULT_ALLGROUP_NAME,
        risk_disc_rates: DiscRates | None = None,
        interpolation_strategy: InterpolationStrategyBase | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        """Initialize a new `StaticRiskTrajectory`.

        Parameters
        ----------
        snapshot_list : list[Snapshot]
            The list of `Snapshot` object to compute risk from.
        return_periods: list[int], optional
            The return periods to use when computing the `return_periods_metric`.
            Defaults to `DEFAULT_RP` ([20, 50, 100]).
        time_resolution: str, optional
            The time resolution to use for interpolation.
            It must be a valid pandas string used to define periods,
            e.g., "Y" for years, "M" for months, "3M" for trimester, etc.
            Defaults to `DEFAULT_TIME_RESOLUTION` ("Y").
        all_groups_name: str, optional
            The string to use to define all exposure points subgroup.
            Defaults to `DEFAULT_ALLGROUP_NAME` ("All").
        risk_disc_rates: DiscRates, optional
            The discount rate to apply to future risk. Defaults to None.
        interpolation_strategy: InterpolationStrategyBase, optional
            The interpolation strategy to use when interpolating.
            Defaults to :class:`AllLinearStrategy`
        impact_computation_strategy: ImpactComputationStrategy, optional
            The method used to calculate the impact from the (Haz,Exp,Vul)
            of the two snapshots. Defaults to :class:`ImpactCalcComputation`.

        """
        super().__init__(
            snapshots_list,
            return_periods=return_periods,
            all_groups_name=all_groups_name,
            risk_disc_rates=risk_disc_rates,
        )
        self._risk_metrics_up_to_date: bool = False
        self.start_date = min([snapshot.date for snapshot in snapshots_list])
        self.end_date = max([snapshot.date for snapshot in snapshots_list])
        self._risk_metrics_calculators = self._reset_risk_metrics_calculators(
            self._snapshots,
            time_resolution,
            interpolation_strategy or AllLinearStrategy(),
            impact_computation_strategy or ImpactCalcComputation(),
        )

    @property
    def interpolation_strategy(self) -> InterpolationStrategyBase:
        """The approach used to interpolate impact matrices in between the two snapshots."""
        return self._risk_metrics_calculators[0].interpolation_strategy

    @interpolation_strategy.setter
    def interpolation_strategy(self, value, /):
        if not isinstance(value, InterpolationStrategyBase):
            raise ValueError("Not an interpolation strategy")

        self._reset_metrics()
        for rmcalc in self._risk_metrics_calculators:
            rmcalc.interpolation_strategy = value

    @property
    def impact_computation_strategy(self) -> ImpactComputationStrategy:
        """The method used to calculate the impact from the (Haz,Exp,Vul) triplets."""
        return self._risk_metrics_calculators[0].impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._reset_metrics()
        for rmcalc in self._risk_metrics_calculators:
            rmcalc.impact_computation_strategy = value

    @property
    def time_resolution(self) -> str:
        """The time resolution to use when interpolating.

        It must be a valid pandas string used to define periods,
        e.g., "Y" for years, "M" for months, "3M" for trimester, etc.

        See `here <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases>`_

        Notes
        -----

        Changing its value resets the corresponding metric.
        """
        return self._risk_metrics_calculators[0].time_resolution

    @time_resolution.setter
    def time_resolution(self, value, /):
        if not isinstance(value, str):
            raise ValueError(
                'time_resolution should be a valid pandas Period frequency string (e.g., `"Y"`, `"M"`, `"D"`).'
            )
        self._reset_metrics()
        for rmcalc in self._risk_metrics_calculators:
            rmcalc.time_resolution = value

    @staticmethod
    def _reset_risk_metrics_calculators(
        snapshots: list[Snapshot],
        time_resolution,
        interpolation_strategy,
        impact_computation_strategy,
    ) -> list[CalcRiskMetricsPeriod]:
        """Initialize or reset the internal risk metrics calculators.

        Notes
        -----

        This methods sorts the snapshots per date.
        """

        def pairwise(container: list):
            """
            Generate pairs of successive elements from an iterable.

            Parameters
            ----------
            iterable : iterable
                An iterable sequence from which successive pairs of elements are generated.

            Returns
            -------
            zip
                A zip object containing tuples of successive pairs from the input iterable.

            Example
            -------
            >>> list(pairwise([1, 2, 3, 4]))
            [(1, 2), (2, 3), (3, 4)]
            """
            a, b = itertools.tee(container)
            next(b, None)
            return zip(a, b)

        return [
            CalcRiskMetricsPeriod(
                start_snapshot,
                end_snapshot,
                time_resolution=time_resolution,
                interpolation_strategy=interpolation_strategy,
                impact_computation_strategy=impact_computation_strategy,
            )
            for start_snapshot, end_snapshot in pairwise(
                sorted(snapshots, key=lambda snap: snap.date)
            )
        ]

    def _generic_metrics(
        self,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generic method to compute metrics based on the provided metric name and method.

        This method calls the appropriate method from the calculator to return
        the results for the given metric, in a tidy formatted dataframe.

        It first checks whether the requested metric is a valid one.
        Then looks for a possible cached value and otherwised asks the
        calculators (`self._risk_metric_calculators`) to run the computations.
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
            LOGGER.debug(f"Returning cached {attr_name}")
            return getattr(self, attr_name)

        LOGGER.debug(f"Computing {attr_name}")
        with log_level(level="WARNING", name_prefix="climada"):
            tmp = [
                getattr(calc_period, metric_meth)(**kwargs)
                for calc_period in self._risk_metrics_calculators
            ]

        # Notably for per_group_aai being None:
        try:
            tmp = pd.concat(tmp)
            if len(tmp) == 0:
                return pd.DataFrame()
        except ValueError as e:
            if str(e) == "All objects passed were None":
                return pd.DataFrame()
            else:
                raise e

        else:
            tmp = tmp.set_index(["date", "group", "measure", "metric"])
            if "coord_id" in tmp.columns:
                tmp = tmp.set_index(["coord_id"], append=True)

            # When more than 2 snapshots, there are duplicated rows, we need to remove them.
            tmp = tmp[~tmp.index.duplicated(keep="first")]
            tmp = tmp.reset_index()
            if self._all_groups_name not in tmp["group"].cat.categories:
                tmp["group"] = tmp["group"].cat.add_categories([self._all_groups_name])
                tmp["group"] = tmp["group"].fillna(self._all_groups_name)
            columns_to_front = ["group", "date", "measure", "metric"]
            tmp = tmp[
                columns_to_front
                + [
                    col
                    for col in tmp.columns
                    if col not in columns_to_front + ["group", "risk", "rp"]
                ]
                + ["risk"]
            ]

            if metric_name == "risk_contributions" and len(self._snapshots) > 2:
                # If there is more than one Snapshot, we need to update the
                # contributions from previous periods for continuity
                # and to set the base risk from the first period
                # This is not elegant, but we need the concatenated metrics from each period,
                # so we can't do it in the calculators, and we need
                # to do it before caching in the private attribute
                tmp = self._risk_contributions_post_treatment(tmp)

            if self._risk_disc_rates:
                LOGGER.debug("Found risk discount rate. Computing NPV.")
                tmp = self.npv_transform(tmp, self._risk_disc_rates)

            LOGGER.debug("All computing done, caching value.")
            setattr(self, attr_name, tmp)
            return getattr(self, attr_name)

    def _compute_period_metrics(
        self, metric_name: str, metric_meth: str, **kwargs
    ) -> pd.DataFrame:
        """Helper method to compute total metrics per period (i.e. whole ranges between pairs of consecutive snapshots)."""
        df = self._generic_metrics(
            metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )
        return self._date_to_period_agg(df, grouper=self._grouper)

    def eai_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the estimated annual impacts at each exposure point for each date.

        This method computes and return a `DataFrame` with eai metric
        (for each exposure point) for each date.

        Notes
        -----

        This computation may become quite expensive for big areas with high resolution.

        """
        df = self._compute_metrics(
            metric_name="eai", metric_meth="calc_eai_gdf", **kwargs
        )
        return df

    def aai_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each date.

        This method computes and return a `DataFrame` with aai metric for each date.

        """

        return self._compute_metrics(
            metric_name="aai", metric_meth="calc_aai_metric", **kwargs
        )

    def return_periods_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the estimated impacts for different return periods.

        Return periods to estimate impacts for are defined by `self.return_periods`.

        """

        return self._compute_metrics(
            metric_name="return_periods",
            metric_meth="calc_return_periods_metric",
            return_periods=self.return_periods,
            **kwargs,
        )

    def aai_per_group_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each exposure group ID.

        This method computes and return a `DataFrame` with aai metric for each
        of the exposure group defined by a group id, for each date.

        """

        return self._compute_metrics(
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
            **kwargs,
        )

    def risk_contributions_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the "contributions" of change in future risk (Exposure and Hazard)

        This method returns the contributions of the change in risk at each date:

           - The 'base risk', i.e., the risk without change in hazard or exposure, compared to trajectory's earliest date.
           - The 'exposure contribution', i.e., the additional risks due to change in exposure (only)
           - The 'hazard contribution', i.e., the additional risks due to change in hazard (only)
           - The 'vulnerability contribution', i.e., the additional risks due to change in vulnerability (only)
           - The 'interaction contribution', i.e., the additional risks due to the interaction term


        """

        return self._compute_metrics(
            metric_name="risk_contributions",
            metric_meth="calc_risk_contributions_metric",
            **kwargs,
        )

    def _risk_contributions_post_treatment(self, df) -> pd.DataFrame:
        """Post treat the risk contributions metrics.

        When more than two snapshots are provided, the total risk of the previous pair
        (period) becomes the base risk for the subsequent one.
        This method straightens this by resetting the base risk to the risk from
        the first snapshot of the list and correcting the different contributions
        by cumulating the contributions from the previous periods.

        """

        df.set_index(["group", "date", "measure", "metric"], inplace=True)
        start_dates = [snap.date for snap in self._snapshots[:-1]]
        end_dates = [snap.date for snap in self._snapshots[1:]]
        periods_dates = list(zip(start_dates, end_dates))
        df.loc[pd.IndexSlice[:, :, :, "base risk"]] = df.loc[
            pd.IndexSlice[
                :,
                pd.to_datetime(self.start_date).to_period(self.time_resolution),
                :,
                "base risk",
            ]  # type: ignore
        ].values
        for p2 in periods_dates[1:]:
            for metric in [
                "exposure contribution",
                "hazard contribution",
                "vulnerability contribution",
                "interaction contribution",
            ]:
                mask_last_previous = (
                    df.index.get_level_values(1)
                    == pd.to_datetime(p2[0]).to_period(self.time_resolution)
                ) & (df.index.get_level_values(3) == metric)
                mask_to_update = (
                    (
                        df.index.get_level_values(1)
                        > pd.to_datetime(p2[0]).to_period(self.time_resolution)
                    )
                    & (
                        df.index.get_level_values(1)
                        <= pd.to_datetime(p2[1]).to_period(self.time_resolution)
                    )
                    & (df.index.get_level_values(3) == metric)
                )

                df.loc[mask_to_update, "risk"] += df.loc[
                    mask_last_previous, "risk"
                ].iloc[0]

        return df.reset_index()

    def per_date_risk_metrics(
        self,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame:
        """Returns a DataFrame of risk metrics for each dates

        This methods collects (and if needed computes) the `metrics`
        (Defaulting to "aai", "return_periods" and "aai_per_group").

        Parameters
        ----------
        metrics : list[str], optional
            The list of metrics to return (defaults to
            ["aai","return_periods","aai_per_group"])
        return_periods : list[int], optional
            The return periods to consider for the return periods metric
            (default to the value of the `.default_rp` attribute)

        Returns
        -------
        pd.DataFrame | pd.Series
            A tidy DataFrame with metrics value for all possible dates.

        """

        metrics = (
            ["aai", "return_periods", "aai_per_group"] if metrics is None else metrics
        )
        return pd.concat([getattr(self, f"{metric}_metrics")() for metric in metrics])

    @staticmethod
    def _get_risk_periods(
        risk_periods: list[CalcRiskMetricsPeriod],
        start_date: datetime.date,
        end_date: datetime.date,
        strict: bool = True,
    ):
        """Returns risk periods from the given list that are within `start_date` and `end_date`.

        Either using a strict inclusion (period is stricly within start and end) or extending
        to overlap inclusion, i.e., start or end is within the period.

        Parameters
        ----------
        risk_periods : list[CalcRiskPeriod]
            The list of risk periods to look through
        start_date : datetime.date
        end_date : datetime.date
        strict: bool, default True
            If true, only returns periods stricly within start and end dates. Else,
            additionaly returns periods that have an overlap within start and end.
        """
        if strict:
            return [
                period
                for period in risk_periods
                if (
                    start_date <= period.snapshot_start.date
                    and end_date >= period.snapshot_end.date
                )
            ]
        else:
            return [
                period
                for period in risk_periods
                if not (
                    start_date >= period.snapshot_end.date
                    or end_date <= period.snapshot_start.date
                )
            ]

    @staticmethod
    def _identify_continuous_periods(group, time_unit):
        """Calculate the difference between consecutive dates."""

        if time_unit == "year":
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

    @classmethod
    def _date_to_period_agg(
        cls,
        df: pd.DataFrame,
        grouper: list[str],
        time_unit: str = "year",
        colname: str | list[str] = "risk",
    ) -> pd.DataFrame:
        """Group per date risk metric to periods."""

        def conditional_agg(group):
            try:
                if "rp" in group.name[2]:
                    return group.mean()
                else:
                    return group.sum()
            except IndexError:
                return group.sum()

        df_sorted = df.sort_values(by=grouper + ["date"])

        if "group" in df.columns and "group" not in grouper:
            grouper = ["group"] + grouper

        # Apply the function to identify continuous periods
        df_periods = df_sorted.groupby(
            grouper, dropna=False, group_keys=False, observed=True
        )[df_sorted.columns].apply(cls._identify_continuous_periods, time_unit)

        if isinstance(colname, str):
            colname = [colname]
        agg_dict = {
            "start_date": pd.NamedAgg(column="date", aggfunc="min"),
            "end_date": pd.NamedAgg(column="date", aggfunc="max"),
        }
        df_periods_dates = (
            df_periods.groupby(grouper + ["period_id"], dropna=False, observed=True)
            .agg(func=None, **agg_dict)  # type: ignore
            .reset_index()
        )

        df_periods_dates["period"] = (
            df_periods_dates["start_date"].astype(str)
            + " to "
            + df_periods_dates["end_date"].astype(str)
        )
        df_periods = (
            df_periods.groupby(grouper + ["period_id"], dropna=False, observed=True)[
                colname
            ]
            .apply(conditional_agg)
            .reset_index()
        )
        df_periods = pd.merge(
            df_periods_dates[grouper + ["period", "period_id"]],
            df_periods,
            on=grouper + ["period_id"],
        )
        df_periods = df_periods.drop(["period_id"], axis=1)
        return df_periods[
            ["period"] + [col for col in df_periods.columns if col != "period"]
        ]

    def per_period_risk_metrics(
        self, metrics: list[str] = ["aai", "return_periods", "aai_per_group"], **kwargs
    ) -> pd.DataFrame:
        """Return a tidy dataframe of the risk metrics with the total for each different period (pair of snapshots)."""

        df = self.per_date_risk_metrics(metrics=metrics, **kwargs)
        return self._date_to_period_agg(df, grouper=self._grouper, **kwargs)

    def _calc_waterfall_plot_data(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ):
        """Compute the required data for the waterfall plot between `start_date` and `end_date`."""
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_contributions = self.risk_contributions_metrics()
        risk_contributions = risk_contributions.loc[
            (risk_contributions["date"] >= str(start_date))
            & (risk_contributions["date"] <= str(end_date))
            & (risk_contributions["measure"] == "no_measure")
        ]
        risk_contributions = risk_contributions.set_index(["date", "metric"])[
            "risk"
        ].unstack()
        return risk_contributions

    def plot_time_waterfall(
        self,
        ax=None,
        figsize=(12, 6),
    ):
        """Plot a waterfall chart of risk contributions over a specified date range.

        This method generates a stacked bar chart to visualize the
        risk contributions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The matplotlib axes on which to plot. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the plotted waterfall chart.

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # get parent figure from the axis

        risk_contribution = self._calc_waterfall_plot_data(
            start_date=self.start_date, end_date=self.end_date
        )
        risk_contribution = risk_contribution[
            [
                "base risk",
                "exposure contribution",
                "hazard contribution",
                "vulnerability contribution",
                "interaction contribution",
            ]
        ]
        risk_contribution["base risk"] = risk_contribution.iloc[0]["base risk"]
        # risk_contribution.plot(x="date", ax=ax, kind="bar", stacked=True)
        ax.stackplot(
            risk_contribution.index.to_timestamp(),  # type: ignore
            [risk_contribution[col] for col in risk_contribution.columns],
            labels=risk_contribution.columns,
        )
        ax.legend()
        # bottom = [0] * len(risk_contribution)
        # for col in risk_contribution.columns:
        #     bottom =  [b + v for b, v in zip(bottom, risk_contribution[col])]
        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = (
            f"Risk between {self.start_date} and {self.end_date} (Average impact)"
        )

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        ax.set_ylim(0.0, 1.1 * ax.get_ylim()[1])
        return fig, ax

    def plot_waterfall(
        self,
        ax=None,
    ):
        """Plot a waterfall chart of risk contributions between two dates.

        This method generates a waterfall plot to visualize the changes in risk contributions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The matplotlib axes on which to plot. If None, a new figure and axes are created.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes with the plotted waterfall chart.

        """
        start_date_p = pd.to_datetime(self.start_date).to_period(self.time_resolution)
        end_date_p = pd.to_datetime(self.end_date).to_period(self.time_resolution)
        risk_contribution = self._calc_waterfall_plot_data(
            start_date=self.start_date, end_date=self.end_date
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        risk_contribution = risk_contribution.loc[
            (risk_contribution.index == str(self.end_date))
        ].squeeze()
        risk_contribution = cast(pd.Series, risk_contribution)

        labels = [
            f"Risk {start_date_p}",
            f"Exposure contribution {end_date_p}",
            f"Hazard contribution {end_date_p}",
            f"Vulnerability contribution {end_date_p}",
            f"Interaction contribution {end_date_p}",
            f"Total Risk {end_date_p}",
        ]
        values = [
            risk_contribution["base risk"],
            risk_contribution["exposure contribution"],
            risk_contribution["hazard contribution"],
            risk_contribution["vulnerability contribution"],
            risk_contribution["interaction contribution"],
            risk_contribution.sum(),
        ]
        bottoms = [
            0.0,
            risk_contribution["base risk"],
            risk_contribution["base risk"] + risk_contribution["exposure contribution"],
            risk_contribution["base risk"]
            + risk_contribution["exposure contribution"]
            + risk_contribution["hazard contribution"],
            risk_contribution["base risk"]
            + risk_contribution["exposure contribution"]
            + risk_contribution["hazard contribution"]
            + risk_contribution["vulnerability contribution"],
            0.0,
        ]

        ax.bar(
            labels,
            values,
            bottom=bottoms,
            edgecolor="black",
            color=[
                "tab:cyan",
                "tab:orange",
                "tab:green",
                "tab:red",
                "tab:purple",
                "tab:blue",
            ],
        )
        for i in range(len(values)):
            ax.text(
                labels[i],  # type: ignore
                values[i] + bottoms[i],
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )

        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = f"Evolution of the contributions of risk between {start_date_p} and {end_date_p} (Average impact)"
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        ax.set_ylim(0.0, 1.1 * ax.get_ylim()[1])
        ax.tick_params(
            axis="x",
            labelrotation=90,
        )

        return ax
