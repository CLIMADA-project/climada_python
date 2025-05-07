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

import datetime
import itertools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.riskperiod import (
    CalcRiskPeriod,
    ImpactCalcComputation,
    ImpactComputationStrategy,
    InterpolationStrategy,
    LinearInterpolation,
)
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)

POSSIBLE_METRICS = ["eai", "aai", "return_periods", "risk_components", "aai_per_group"]
DEFAULT_RP = [50, 100, 500]


class RiskTrajectory:
    """Calculates risk trajectories over a series of snapshots.

    This class computes risk metrics over a series of snapshots,
    optionally applying risk discounting and risk transfer adjustments.

    Attributes
    ----------
    start_date : datetime
        The start date of the risk trajectory.
    end_date : datetime
        The end date of the risk trajectory.
    risk_disc : DiscRates | None
        The discount rates for risk, default is None.
    risk_transf_cover : optional
        The risk transfer coverage, default is None.
    risk_transf_attach : optional
        The risk transfer attachment, default is None.
    risk_periods : list
        The computed RiskPeriod objects from the snapshots.
    """

    _grouper = ["measure", "metric"]
    """Grouping class attribute"""

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        interval_freq: str = "YS",
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual: bool = True,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        self._reset_metrics()
        self._risk_period_up_to_date: bool = False
        self._snapshots = snapshots_list
        self._all_groups_name = all_groups_name
        self._default_rp = DEFAULT_RP
        self.start_date = min([snapshot.date for snapshot in snapshots_list])
        self.end_date = max([snapshot.date for snapshot in snapshots_list])
        self._interval_freq = interval_freq
        self.risk_disc = risk_disc
        self._risk_transf_cover = risk_transf_cover
        self._risk_transf_attach = risk_transf_attach
        self._calc_residual = calc_residual
        self._interpolation_strategy = interpolation_strategy or LinearInterpolation()
        self._impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        LOGGER.debug("Computing risk periods")
        self._risk_periods_calculators = self._calc_risk_periods(snapshots_list)

    def _reset_metrics(self):
        for metric in POSSIBLE_METRICS:
            setattr(self, "_" + metric + "_metrics", None)

        self._all_risk_metrics = None

    @property
    def default_rp(self):
        return self._default_rp

    @default_rp.setter
    def default_rp(self, value):
        if not isinstance(value, list):
            ValueError("Return periods need to be a list of int.")
        if any(not isinstance(i, int) for i in value):
            ValueError("Return periods need to be a list of int.")
        self._return_periods_metrics = None
        self._all_risk_metrics = None
        self._default_rp = value

    @property
    def risk_transf_cover(self):
        """The risk transfer coverage."""
        return self._risk_transf_cover

    @risk_transf_cover.setter
    def risk_transf_cover(self, value):
        self._risk_transf_cover = value
        self._risk_period_up_to_date = False
        self._reset_metrics

    @property
    def risk_transf_attach(self):
        """The risk transfer attachment."""
        return self._risk_transf_attach

    @risk_transf_attach.setter
    def risk_transf_attach(self, value):
        self._risk_transf_attach = value
        self._risk_period_up_to_date = False
        self._reset_metrics

    @property
    def risk_periods(self) -> list:
        """The computed risk periods from the snapshots."""
        if not self._risk_period_up_to_date:
            self._risk_periods_calculators = self._calc_risk_periods(self._snapshots)
            self._risk_period_up_to_date = True

        return self._risk_periods_calculators

    def _calc_risk_periods(self, snapshots: list[Snapshot]) -> list[CalcRiskPeriod]:
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

        # impfset = self._merge_impfset(snapshots)
        return [
            CalcRiskPeriod(
                start_snapshot,
                end_snapshot,
                interval_freq=self._interval_freq,
                interpolation_strategy=self._interpolation_strategy,
                impact_computation_strategy=self._impact_computation_strategy,
                risk_transf_cover=self.risk_transf_cover,
                risk_transf_attach=self.risk_transf_attach,
                calc_residual=self._calc_residual,
            )
            for start_snapshot, end_snapshot in pairwise(snapshots)
        ]

    @classmethod
    def npv_transform(cls, df: pd.DataFrame, risk_disc) -> pd.DataFrame:
        def _npv_group(group, disc):
            start_date = group.index.get_level_values("date").min()
            end_date = group.index.get_level_values("date").max()
            return calc_npv_cash_flows(group, start_date, end_date, disc)

        df = df.set_index("date")
        grouper = cls._grouper
        if "group" in df.columns:
            grouper = ["group"] + grouper

        df["risk"] = df.groupby(
            grouper,
            dropna=False,
            as_index=False,
            group_keys=False,
        )["risk"].transform(_npv_group, risk_disc)
        df = df.reset_index()
        return df

    def _generic_metrics(
        self,
        npv: bool = True,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ):
        """Generic method to compute metrics based on the provided metric name and method."""
        if metric_name is None or metric_meth is None:
            raise ValueError("Both metric_name and metric_meth must be provided.")

        if metric_name not in POSSIBLE_METRICS:
            raise NotImplementedError(
                f"{metric_name} not implemented ({POSSIBLE_METRICS})."
            )
        # Construct the attribute name for storing the metric results
        attr_name = f"_{metric_name}_metrics"

        if getattr(self, attr_name, None) is None:
            tmp = []
            for calc_period in self.risk_periods:
                # Call the specified method on the calc_period object
                tmp.append(getattr(calc_period, metric_meth)(**kwargs))

            tmp = pd.concat(tmp)
            tmp.drop_duplicates(inplace=True)
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

        if npv:
            return self.npv_transform(getattr(self, attr_name), self.risk_disc)

        return getattr(self, attr_name)

    def _compute_period_metrics(
        self, metric_name: str, metric_meth: str, npv: bool = True, **kwargs
    ):
        """Helper method to compute total metrics per period."""
        df = self._generic_metrics(
            npv=npv, metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )
        return self._per_period_risk(df)

    def _compute_metrics(
        self, metric_name: str, metric_meth: str, npv: bool = True, **kwargs
    ):
        """Helper method to compute metrics."""
        df = self._generic_metrics(
            npv=npv, metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )
        return df

    def eai_metrics(self, npv: bool = True):
        return self._compute_metrics(
            npv=npv,
            metric_name="eai",
            metric_meth="calc_eai_gdf",
        )

    def aai_metrics(self, npv: bool = True):
        return self._compute_metrics(
            npv=npv,
            metric_name="aai",
            metric_meth="calc_aai_metric",
        )

    def return_periods_metrics(self, return_periods, npv: bool = True):
        return self._compute_metrics(
            npv=npv,
            metric_name="return_periods",
            metric_meth="calc_return_periods_metric",
            return_periods=return_periods,
        )

    def aai_per_group_metrics(self, npv: bool = True):
        return self._compute_metrics(
            npv=npv,
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
        )

    def risk_components_metrics(self, npv: bool = True):
        return self._compute_metrics(
            npv=npv,
            metric_name="risk_components",
            metric_meth="calc_risk_components_metric",
        )

    def per_date_risk_metrics(
        self,
        metrics: list[str] | None = None,
        return_periods: list[int] | None = None,
        npv: bool = True,
    ) -> pd.DataFrame | pd.Series:
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
        risk_periods: list[CalcRiskPeriod],
        start_date: datetime.date,
        end_date: datetime.date,
    ):
        return [
            period
            for period in risk_periods
            if (
                start_date >= period.snapshot0.date or end_date <= period.snapshot1.date
            )
        ]

    @classmethod
    def _per_period_risk(
        cls,
        df: pd.DataFrame,
        time_unit: str = "year",
        colname: str | list[str] = "risk",
    ) -> pd.DataFrame | pd.Series:
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

        grouper = cls._grouper
        if "group" in df.columns and "group" not in grouper:
            grouper = ["group"] + grouper

        df_sorted = df.sort_values(by=cls._grouper + ["date"])
        # Apply the function to identify continuous periods
        df_periods = df_sorted.groupby(grouper, dropna=False, group_keys=False).apply(
            identify_continuous_periods, time_unit
        )

        if isinstance(colname, str):
            colname = [colname]

        agg_dict = {
            "start_date": pd.NamedAgg(column="date", aggfunc="min"),
            "end_date": pd.NamedAgg(column="date", aggfunc="max"),
        }
        for col in colname:
            agg_dict[col] = pd.NamedAgg(column=col, aggfunc="sum")
        # Group by the identified periods and calculate start and end dates
        df_periods = (
            df_periods.groupby(grouper + ["period_id"], dropna=False)
            .agg(**agg_dict)
            .reset_index()
        )

        df_periods["period"] = (
            df_periods["start_date"].astype(str)
            + " to "
            + df_periods["end_date"].astype(str)
        )
        df_periods = df_periods.drop(["period_id", "start_date", "end_date"], axis=1)
        return df_periods[
            ["period"] + [col for col in df_periods.columns if col != "period"]
        ]

    def per_period_risk_metrics(
        self, metrics: list[str] = ["aai", "return_periods", "aai_per_group"], **kwargs
    ) -> pd.DataFrame | pd.Series:
        """Returns a tidy dataframe of the risk metrics with the total for each different period."""
        df = self.per_date_risk_metrics(metrics=metrics, **kwargs)
        return self._per_period_risk(df, **kwargs)

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
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_component = self._calc_waterfall_plot_data(
            start_date=start_date, end_date=end_date
        )
        risk_component.plot(ax=ax, kind="bar", stacked=True)
        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = (
            f"Risk between {start_date} and {end_date} (Annual Average impact)"
        )

        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        return ax

    def plot_waterfall(
        self,
        ax=None,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
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

        Notes
        -----
        The "risk components" are plotted such that the increase in risk due to the hazard component
        really denotes the difference between the risk associated with both future exposure and hazard
        compared to the risk associated with future exposure and present hazard.
        """
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        risk_component = self._calc_waterfall_plot_data(
            start_date=start_date, end_date=end_date
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        risk_component = risk_component.loc[
            (risk_component.index.date == end_date)
        ].squeeze()

        labels = [
            f"Risk {start_date}",
            f"Exposure {end_date}",
            f"Hazard {end_date}¹",
            f"Total Risk {end_date}",
        ]
        values = [
            risk_component["base risk"],
            risk_component["delta from exposure"],
            risk_component["delta from hazard"],
            risk_component["base risk"]
            + risk_component["delta from exposure"]
            + risk_component["delta from hazard"],
        ]
        bottoms = [
            0.0,
            risk_component["base risk"],
            risk_component["base risk"] + risk_component["delta from exposure"],
            0.0,
        ]

        ax.bar(
            labels,
            values,
            bottom=bottoms,
            edgecolor="black",
            color=["tab:blue", "tab:orange", "tab:green", "tab:red"],
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
        title_label = f"Risk at {start_date} and {end_date} (Annual Average impact)"

        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        # ax.tick_params(axis='x', labelrotation=90,)
        ax.annotate(
            """¹: The increase in risk due to hazard denotes the difference in risk with future exposure
and hazard compared to risk with future exposure and present hazard.""",
            xy=(0.0, -0.15),
            xycoords="axes fraction",
            ha="left",
            va="center",
            fontsize=8,
        )

        return ax


def calc_npv_cash_flows(
    cash_flows: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date | None = None,
    disc: DiscRates | None = None,
):
    # If no discount rates are provided, return the cash flows as is
    if not disc:
        return cash_flows

    if not isinstance(cash_flows.index, pd.DatetimeIndex):
        raise ValueError("cash_flows must be a pandas Series with a datetime index")

    # Determine the end date if not provided
    if end_date is None:
        end_date = cash_flows.index[-1]

    df = cash_flows.to_frame(name="cash_flow")
    df["year"] = df.index.year

    # Merge with the discount rates based on the year
    tmp = df.merge(
        pd.DataFrame({"year": disc.years, "rate": disc.rates}), on="year", how="left"
    )
    tmp.index = df.index
    df = tmp.copy()
    df["discount_factor"] = (1 / (1 + df["rate"])) ** (
        (df.index - start_date).days // 365
    )

    # Apply the discount factors to the cash flows
    df["npv_cash_flow"] = df["cash_flow"] * df["discount_factor"]

    return df["npv_cash_flow"]
