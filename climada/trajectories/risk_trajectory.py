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

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from pandas.tseries.frequencies import to_offset

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.interpolation import InterpolationStrategyBase
from climada.trajectories.riskperiod import (
    AllLinearStrategy,
    CalcRiskMetricsPeriod,
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import RiskTrajectory
from climada.util import log_level

LOGGER = logging.getLogger(__name__)


class InterpolatedRiskTrajectory(RiskTrajectory):
    """Calculates risk trajectories over a series of snapshots.

    This class computes risk metrics over a series of snapshots,
    optionally applying risk discounting.

    """

    _grouper = ["measure", "metric"]
    """Results dataframe grouper"""

    POSSIBLE_METRICS = [
        "eai",
        "aai",
        "return_periods",
        "risk_components",
        "aai_per_group",
    ]
    DEFAULT_RP = [50, 100, 500]

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        time_resolution: str = "Y",
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        interpolation_strategy: InterpolationStrategyBase | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        super().__init__(
            snapshots_list,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
            impact_computation_strategy=impact_computation_strategy,
        )
        self._risk_period_up_to_date: bool = False
        self.start_date = min([snapshot.date for snapshot in snapshots_list])
        self.end_date = max([snapshot.date for snapshot in snapshots_list])
        self._time_resolution = time_resolution
        self._interpolation_strategy = interpolation_strategy or AllLinearStrategy()
        self._risk_periods_calculators = None

    @property
    def _risk_periods(self) -> list[CalcRiskMetricsPeriod]:
        """The risk periods computing objects."""
        if self._risk_periods_calculators is None or not self._risk_period_up_to_date:
            self._risk_periods_calculators = self._calc_risk_periods(self._snapshots)
            self._risk_period_up_to_date = True

        return self._risk_periods_calculators

    def _calc_risk_periods(
        self, snapshots: list[Snapshot]
    ) -> list[CalcRiskMetricsPeriod]:
        """Creates the `CalcRiskPeriod` objects corresponding to a given list of snapshots."""

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

        LOGGER.debug(f"{self.__class__.__name__}: Calc risk periods")
        # impfset = self._merge_impfset(snapshots)
        return [
            CalcRiskMetricsPeriod(
                start_snapshot,
                end_snapshot,
                time_resolution=self._time_resolution,
                interpolation_strategy=self._interpolation_strategy,
                impact_computation_strategy=self._impact_computation_strategy,
            )
            for start_snapshot, end_snapshot in pairwise(
                sorted(snapshots, key=lambda snap: snap.date)
            )
        ]

    def _generic_metrics(
        self,
        npv: bool = True,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Generic method to compute metrics based on the provided metric name and method."""
        if metric_name is None or metric_meth is None:
            raise ValueError("Both metric_name and metric_meth must be provided.")

        if metric_name not in self.POSSIBLE_METRICS:
            raise NotImplementedError(
                f"{metric_name} not implemented ({self.POSSIBLE_METRICS})."
            )

        # Construct the attribute name for storing the metric results
        attr_name = f"_{metric_name}_metrics"

        tmp = []
        for calc_period in self._risk_periods:
            # Call the specified method on the calc_period object
            with log_level(level="WARNING", name_prefix="climada"):
                tmp.append(getattr(calc_period, metric_meth)(**kwargs))

        # Notably for per_group_aai being None:
        try:
            tmp = pd.concat(tmp)
            if len(tmp) == 0:
                return None
        except ValueError as e:
            if str(e) == "All objects passed were None":
                return None
            else:
                raise e

        else:
            tmp = tmp.set_index(["date", "group", "measure", "metric"])
            if "coord_id" in tmp.columns:
                tmp = tmp.set_index(["coord_id"], append=True)

            # When more than 2 snapshots, there are duplicated rows, we need to remove them.
            tmp = tmp[~tmp.index.duplicated(keep="first")]
            tmp = tmp.reset_index()
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
            setattr(self, attr_name, tmp)

            if npv and self._risk_disc:
                return self.npv_transform(getattr(self, attr_name), self._risk_disc)

            return getattr(self, attr_name)

    def _compute_period_metrics(
        self, metric_name: str, metric_meth: str, npv: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Helper method to compute total metrics per period (i.e. whole ranges between pairs of consecutive snapshots)."""
        df = self._generic_metrics(
            npv=npv, metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )
        return self._date_to_period_agg(df, grouper=self._grouper)

    def _compute_metrics(
        self, metric_name: str, metric_meth: str, npv: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Helper method to compute metrics.

        Notes
        -----

        This method exists for the sake of the children option appraisal classes, for which
        `_generic_metrics` can have an additional keyword argument and call and extend on its
        parent method, while this method can stay the same.
        """
        df = self._generic_metrics(
            npv=npv, metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )
        return df

    def eai_metrics(self, npv: bool = True, **kwargs) -> pd.DataFrame:
        """Return the estimated annual impacts at each exposure point for each date.

        This method computes and return a `DataFrame` with eai metric
        (for each exposure point) for each date.

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.

        Notes
        -----

        This computation may become quite expensive for big areas with high resolution.

        """
        df = self._compute_metrics(
            npv=npv, metric_name="eai", metric_meth="calc_eai_gdf", **kwargs
        )
        return df

    def aai_metrics(self, npv: bool = True, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each date.

        This method computes and return a `DataFrame` with aai metric for each date.

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.
        """

        return self._compute_metrics(
            npv=npv, metric_name="aai", metric_meth="calc_aai_metric", **kwargs
        )

    def return_periods_metrics(
        self, return_periods, npv: bool = True, **kwargs
    ) -> pd.DataFrame:
        return self._compute_metrics(
            npv=npv,
            metric_name="return_periods",
            metric_meth="calc_return_periods_metric",
            return_periods=return_periods,
            **kwargs,
        )

    def aai_per_group_metrics(self, npv: bool = True, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each exposure group ID.

        This method computes and return a `DataFrame` with aai metric for each
        of the exposure group defined by a group id, for each date.

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.
        """

        return self._compute_metrics(
            npv=npv,
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
            **kwargs,
        )

    def risk_components_metrics(self, npv: bool = True, **kwargs) -> pd.DataFrame:
        """Return the "components" of change in future risk (Exposure and Hazard)

        This method returns the components of the change in risk at each date:

           - The 'base risk', i.e., the risk without change in hazard or exposure, compared to trajectory's earliest date.
           - The 'exposure contribution', i.e., the additional risks due to change in exposure (only)
           - The 'hazard contribution', i.e., the additional risks due to change in hazard (only)
           - The 'vulnerability contribution', i.e., the additional risks due to change in vulnerability (only)
           - The 'interaction contribution', i.e., the additional risks due to the interaction term

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.

        """

        tmp = self._compute_metrics(
            npv=npv,
            metric_name="risk_components",
            metric_meth="calc_risk_components_metric",
            **kwargs,
        )

        # If there is more than one Snapshot, we need to update the
        # components from previous periods for for continuity
        # and to set the base risk from the first period
        if len(self._snapshots) > 2:
            tmp.set_index(["group", "date", "measure", "metric"], inplace=True)
            start_dates = [snap.date for snap in self._snapshots[:-1]]
            end_dates = [
                snap.date - to_offset(self._time_resolution)
                for snap in self._snapshots[1:]
            ]
            periods_dates = list(zip(start_dates, end_dates))
            tmp.loc[pd.IndexSlice[:, :, :, "base risk"]] = tmp.loc[
                pd.IndexSlice[:, str(self.start_date), :, "base risk"]
            ].values
            for p2 in periods_dates[1:]:
                for metric in [
                    "exposure contribution",
                    "hazard contribution",
                    "vulnerability contribution",
                    "interaction contribution",
                ]:
                    mask_last_previous = (
                        tmp.index.get_level_values(1).date == p2[0]
                    ) & (tmp.index.get_level_values(3) == metric)
                    mask_to_update = (
                        (tmp.index.get_level_values(1).date > p2[0])
                        & (tmp.index.get_level_values(1).date <= p2[1])
                        & (tmp.index.get_level_values(3) == metric)
                    )

                    tmp.loc[mask_to_update, "risk"] += tmp.loc[
                        mask_last_previous, "risk"
                    ].iloc[0]

        tmp.reset_index(inplace=True)
        return tmp.drop("index", axis=1, errors="ignore")

    def per_date_risk_metrics(
        self,
        metrics: list[str] | None = None,
        return_periods: list[int] | None = None,
        npv: bool = True,
    ) -> pd.DataFrame | pd.Series:
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
        npv : bool
            Whether to apply the (risk) discount rate if it was defined
            when instantiating the trajectory. Defaults to `True`.

        Returns
        -------
        pd.DataFrame | pd.Series
            A tidy DataFrame with metrics value for all possible dates.

        """

        metrics_df = []
        metrics = (
            ["aai", "return_periods", "aai_per_group"] if metrics is None else metrics
        )
        return_periods = return_periods if return_periods else self.default_rp
        if "aai" in metrics:
            metrics_df.append(self.aai_metrics(npv))
        if "return_periods" in metrics:
            metrics_df.append(self.return_periods_metrics(return_periods, npv))
        if "aai_per_group" in metrics:
            metrics_df.append(self.aai_per_group_metrics(npv))

        return pd.concat(metrics_df)

    @staticmethod
    def _get_risk_periods(
        risk_periods: list[CalcRiskMetricsPeriod],
        start_date: datetime.date,
        end_date: datetime.date,
        strict: bool = True,
    ):
        """Returns risk periods from the given list that are within `start_date` and `end_date`.

        Parameters
        ----------
        risk_periods : list[CalcRiskPeriod]
            The list of risk periods to look through
        start_date : datetime.date
        end_date : datetime.date
        strict: bool, default True
            If true, only returns periods stricly within start and end dates. Else,
            returns periods that have an overlap within start and end.
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
    def identify_continuous_periods(group, time_unit):
        # Calculate the difference between consecutive dates
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
    ) -> pd.DataFrame | pd.Series:
        """Groups per date risk metric to periods."""

        ## I'm thinking this does not work with RPs... As you can't just sum impacts
        ## Not sure what to do with it. -> Fixed I take the avg RP impact of the period
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
        ).apply(cls.identify_continuous_periods, time_unit)

        if isinstance(colname, str):
            colname = [colname]

        agg_dict = {
            "start_date": pd.NamedAgg(column="date", aggfunc="min"),
            "end_date": pd.NamedAgg(column="date", aggfunc="max"),
        }

        df_periods_dates = (
            df_periods.groupby(grouper + ["period_id"], dropna=False, observed=True)
            .agg(**agg_dict)
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
            df_periods_dates[grouper + ["period"]], df_periods, on=grouper
        )
        df_periods = df_periods.drop(["period_id"], axis=1)
        return df_periods[
            ["period"] + [col for col in df_periods.columns if col != "period"]
        ]

    def per_period_risk_metrics(
        self, metrics: list[str] = ["aai", "return_periods", "aai_per_group"], **kwargs
    ) -> pd.DataFrame | pd.Series:
        """Returns a tidy dataframe of the risk metrics with the total for each different period."""
        df = self.per_date_risk_metrics(metrics=metrics, **kwargs)
        return self._date_to_period_agg(df, grouper=self._grouper, **kwargs)

    def _calc_waterfall_plot_data(
        self,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        npv: bool = True,
    ):
        """Compute the required data for the waterfall plot between `start_date` and `end_date`."""
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_components = self.risk_components_metrics(npv)
        risk_components = risk_components.loc[
            (risk_components["date"] >= str(start_date))
            & (risk_components["date"] <= str(end_date))
        ]
        risk_components = risk_components.set_index(["date", "metric"])[
            "risk"
        ].unstack()
        return risk_components

    def plot_per_date_waterfall(
        self,
        ax=None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        figsize=(12, 6),
        npv=True,
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

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # get parent figure from the axis
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_component = self._calc_waterfall_plot_data(
            start_date=start_date, end_date=end_date, npv=npv
        )
        risk_component = risk_component[
            [
                "base risk",
                "exposure contribution",
                "hazard contribution",
                "vulnerability contribution",
                "interaction contribution",
            ]
        ]
        risk_component["base risk"] = risk_component.iloc[0]["base risk"]
        # risk_component.plot(x="date", ax=ax, kind="bar", stacked=True)
        ax.stackplot(
            risk_component.index.to_timestamp(),
            [risk_component[col] for col in risk_component.columns],
            labels=risk_component.columns,
        )
        ax.legend()
        # bottom = [0] * len(risk_component)
        # for col in risk_component.columns:
        #     bottom =  [b + v for b, v in zip(bottom, risk_component[col])]
        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = f"Risk between {start_date} and {end_date} (Average impact)"

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        ax.set_ylim(0.0, 1.1 * ax.get_ylim()[1])
        return fig, ax

    def plot_waterfall(
        self,
        ax=None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
        npv=True,
    ):
        """Plot a waterfall chart of risk components between two dates.

        This method generates a waterfall plot to visualize the changes in risk components
        between a specified start and end date. If no dates are provided, it defaults to
        the start and end dates of the risk trajectory.

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

        """
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        start_date_p = pd.to_datetime(start_date).to_period(self._time_resolution)
        end_date_p = pd.to_datetime(end_date).to_period(self._time_resolution)
        risk_component = self._calc_waterfall_plot_data(
            start_date=start_date, end_date=end_date, npv=npv
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        risk_component = risk_component.loc[
            (risk_component.index == str(end_date))
        ].squeeze()

        labels = [
            f"Risk {start_date_p}",
            f"Exposure contribution {end_date_p}",
            f"Hazard contribution {end_date_p}",
            f"Vulnerability contribution {end_date_p}",
            f"Interaction contribution {end_date_p}",
            f"Total Risk {end_date_p}",
        ]
        values = [
            risk_component["base risk"],
            risk_component["exposure contribution"],
            risk_component["hazard contribution"],
            risk_component["vulnerability contribution"],
            risk_component["interaction contribution"],
            risk_component.sum(),
        ]
        bottoms = [
            0.0,
            risk_component["base risk"],
            risk_component["base risk"] + risk_component["exposure contribution"],
            risk_component["base risk"]
            + risk_component["exposure contribution"]
            + risk_component["hazard contribution"],
            risk_component["base risk"]
            + risk_component["exposure contribution"]
            + risk_component["hazard contribution"]
            + risk_component["vulnerability contribution"],
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
                labels[i],
                values[i] + bottoms[i],
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )

        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = f"Evolution of the components of risk between {start_date_p} and {end_date_p} (Average impact)"
        ax.yaxis.set_major_formatter(ticker.EngFormatter())
        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        ax.set_ylim(0.0, 1.1 * ax.get_ylim()[1])
        ax.tick_params(
            axis="x",
            labelrotation=90,
        )

        return ax
