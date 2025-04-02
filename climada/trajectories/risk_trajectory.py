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

from climada.engine.impact_calc import ImpactCalc
from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.riskperiod import RiskPeriod
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


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
        risk_disc: DiscRates | None = None,
        metrics: list[str] = ["aai", "eai", "rp"],
        return_periods: list[int] = [100, 500, 1000],
        compute_groups=False,
        risk_transf_cover=None,
        risk_transf_attach=None,
    ):
        "docstring"
        self._metrics_up_to_date: bool = False
        self.metrics = metrics
        self._return_periods = return_periods
        self.start_date = min([snapshot.date for snapshot in snapshots_list])
        self.end_date = max([snapshot.date for snapshot in snapshots_list])
        self.risk_disc = risk_disc
        self._risk_transf_cover = risk_transf_cover
        self._risk_transf_attach = risk_transf_attach
        LOGGER.debug("Computing risk periods")
        self._risk_periods = self._calc_risk_periods(snapshots_list)
        self._update_risk_metrics(compute_groups=compute_groups)

    @property
    def return_periods(self) -> list[int]:
        """The return periods considered in the risk trajectory."""
        return self._return_periods

    @return_periods.setter
    def return_periods(self, value: list[int]):
        if not isinstance(value, list):
            raise ValueError("Not a list")
        if any(not isinstance(i, int) for i in value):
            raise ValueError("List elements are not int")
        self._return_periods = value
        self._metrics_up_to_date = False

    @property
    def risk_transf_cover(self):
        """The risk transfer coverage."""
        return self._risk_transf_cover

    @risk_transf_cover.setter
    def risk_transf_cover(self, value):
        self._risk_transf_cover = value
        self._metrics_up_to_date = False

    @property
    def risk_transf_attach(self):
        """The risk transfer attachment."""
        return self._risk_transf_attach

    @risk_transf_attach.setter
    def risk_transf_attach(self, value):
        self._risk_transf_attach = value
        self._metrics_up_to_date = False

    @property
    def risk_periods(self) -> list[RiskPeriod]:
        """The computed risk periods from the snapshots."""
        return self._risk_periods

    def _calc_risk_periods(self, snapshots):
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
            RiskPeriod(start_snapshot, end_snapshot)
            for start_snapshot, end_snapshot in pairwise(snapshots)
        ]

    def _update_risk_metrics(self, compute_groups=False):
        results_df = []
        for period in self._risk_periods:
            results_df.append(
                impact_mixer(
                    period,
                    self.metrics,
                    self.return_periods,
                    compute_groups,
                    all_groups_name="All",
                )
            )
        results_df = pd.concat(results_df, axis=0)

        # duplicate rows may arise from overlapping end and start if there's more than two snapshots
        results_df.drop_duplicates(inplace=True)

        # reorder the columns (but make sure not to remove possibly important ones in the future)
        columns_to_front = ["date", "measure", "metric"]
        if compute_groups:
            columns_to_front = ["group"] + columns_to_front
        self._annual_risk_metrics = results_df[
            columns_to_front
            + [
                col
                for col in results_df.columns
                if col not in columns_to_front + ["group", "risk", "rp"]
            ]
            + ["risk"]
        ]
        self._metrics_up_to_date = True

    @staticmethod
    def _get_risk_periods(
        risk_periods, start_date: datetime.date, end_date: datetime.date
    ):
        return [
            period
            for period in risk_periods
            if (start_date >= period.start_date or end_date <= period.end_date)
        ]

    def _calc_per_date_risk_metrics(self, npv=True):
        def npv_transform(group):
            start_date = group.index.get_level_values("date").min()
            end_date = group.index.get_level_values("date").max()
            return calc_npv_cash_flows(group, start_date, end_date, self.risk_disc)

        if self._metrics_up_to_date:
            df = self._annual_risk_metrics
        else:
            self._update_risk_metrics()
            df = self._annual_risk_metrics

        if npv:
            df = df.set_index("date")
            grouper = self._grouper
            if "group" in df.columns:
                grouper = ["group"] + grouper

            df["risk"] = df.groupby(
                grouper,
                dropna=False,
                as_index=False,
                group_keys=False,
            )["risk"].transform(npv_transform)
            df = df.reset_index()

        return df

    @classmethod
    def _calc_periods_risk(cls, df: pd.DataFrame, time_unit="year", colname="risk"):
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
        if "group" in df.columns:
            grouper = ["group"] + grouper

        df_sorted = df.sort_values(by=cls._grouper + ["date"])
        # Apply the function to identify continuous periods
        df_periods = df_sorted.groupby(grouper, dropna=False, group_keys=False).apply(
            identify_continuous_periods, time_unit
        )

        # Group by the identified periods and calculate start and end dates
        df_periods = (
            df_periods.groupby(grouper + ["period_id"], dropna=False)
            .agg(
                start_date=pd.NamedAgg(column="date", aggfunc="min"),
                end_date=pd.NamedAgg(column="date", aggfunc="max"),
                total=pd.NamedAgg(column=colname, aggfunc="sum"),
            )
            .reset_index()
        )

        df_periods["period"] = (
            df_periods["start_date"].astype(str)
            + " to "
            + df_periods["end_date"].astype(str)
        )
        df_periods = df_periods.rename(columns={"total": f"{colname}"})
        df_periods = df_periods.drop(["period_id", "start_date", "end_date"], axis=1)
        return df_periods[
            ["period"] + [col for col in df_periods.columns if col != "period"]
        ]

    @property
    def per_date_risk_metrics(self) -> pd.DataFrame | pd.Series:
        """Returns a tidy dataframe of the risk metrics for all dates."""
        return self._calc_risk_metrics(total=False, npv=True)

    @property
    def total_risk_metrics(self):
        """Returns a tidy dataframe of the risk metrics with the total for each different period."""
        return self._calc_risk_metrics(total=True, npv=True)

    def _calc_risk_metrics(self, total=False, npv=True):
        df = self._calc_per_date_risk_metrics(npv=npv)
        if total:
            return self._calc_periods_risk(df)

        return df

    def _calc_waterfall_plot_data(self, start_date=None, end_date=None):
        start_date = self.start_date if start_date is None else start_date
        end_date = self.end_date if end_date is None else end_date
        considered_risk_periods = self._get_risk_periods(
            self._risk_periods, start_date=start_date, end_date=end_date
        )

        risk_component = {
            str(period.start_date)
            + "-"
            + str(period.end_date): self._calc_risk_component(period)
            for period in considered_risk_periods
        }
        risk_component = pd.concat(
            risk_component.values(), keys=risk_component.keys(), names=["Period"]
        ).reset_index()
        risk_component = risk_component.loc[
            (risk_component["date"].dt.date >= start_date)
            & (risk_component["date"].dt.date <= end_date)
        ]
        risk_component["Base risk"] = risk_component["Base risk"].min()
        risk_component[["Change in Exposure", "Change in Hazard (with Exposure)"]] = (
            risk_component[["Change in Exposure", "Change in Hazard (with Exposure)"]]
            .replace(0, None)
            .ffill()
            .fillna(0.0)
        )
        return risk_component

    def _calc_risk_component(self, period: RiskPeriod):
        imp_mats_H0 = period._imp_mats_0
        imp_mats_H1 = period._imp_mats_1
        freq_H0 = period.snapshot0.hazard.frequency
        freq_H1 = period.snapshot1.hazard.frequency
        per_date_eai_H0, per_date_eai_H1 = calc_per_date_eais(
            imp_mats_H0, imp_mats_H1, freq_H0, freq_H1
        )
        per_date_aai_H0, per_date_aai_H1 = calc_per_date_aais(
            per_date_eai_H0, per_date_eai_H1
        )
        prop_H1 = np.linspace(0, 1, num=len(period.date_idx))
        prop_H0 = 1 - prop_H1
        per_date_aai = prop_H0 * per_date_aai_H0 + prop_H1 * per_date_aai_H1

        risk_dev_0 = per_date_aai_H0 - per_date_aai[0]
        risk_cc_0 = per_date_aai - (risk_dev_0 + per_date_aai[0])
        df = pd.DataFrame(
            {
                "Base risk": per_date_aai - (risk_dev_0 + risk_cc_0),
                "Change in Exposure": risk_dev_0,
                "Change in Hazard (with Exposure)": risk_cc_0,
            },
            index=period.date_idx,
        )
        return df.round(1)

    def plot_per_date_waterfall(self, ax=None, start_date=None, end_date=None):
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
        risk_component.plot(ax=ax, kind="bar", x="date", stacked=True)
        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = (
            f"Risk between {start_date} and {end_date} (Annual Average impact)"
        )

        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        return ax

    def plot_waterfall(self, ax=None, start_date=None, end_date=None):
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
            (risk_component["date"].dt.date == end_date)
        ].squeeze()

        labels = [
            f"Risk {start_date}",
            f"Exposure {end_date}",
            f"Hazard {end_date}¹",
            f"Total Risk {end_date}",
        ]
        values = [
            risk_component["Base risk"],
            risk_component["Change in Exposure"],
            risk_component["Change in Hazard (with Exposure)"],
            risk_component["Base risk"]
            + risk_component["Change in Exposure"]
            + risk_component["Change in Hazard (with Exposure)"],
        ]
        bottoms = [
            0.0,
            risk_component["Base risk"],
            risk_component["Base risk"] + risk_component["Change in Exposure"],
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


def calc_npv_cash_flows(cash_flows, start_date, end_date=None, disc=None):
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


def calc_per_date_eais(imp_mats_0, imp_mats_1, frequency_0, frequency_1):
    """
    Calculate per_date expected annual impact (EAI) values for two scenarios.

    Parameters
    ----------
    imp_mats_0 : list of np.ndarray
        List of interpolated impact matrices for scenario 0.
    imp_mats_1 : list of np.ndarray
        List of interpolated impact matrices for scenario 1.
    frequency_0 : np.ndarray
        Frequency values associated with scenario 0.
    frequency_1 : np.ndarray
        Frequency values associated with scenario 1.

    Returns
    -------
    tuple
        Tuple containing:
        - per_date_eai_exp_0 : list of float
            Per_Date expected annual impacts for scenario 0.
        - per_date_eai_exp_1 : list of float
            Per_Date expected annual impacts for scenario 1.
    """
    per_date_eai_exp_0 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_0) for imp_mat in imp_mats_0
    ]
    per_date_eai_exp_1 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_1) for imp_mat in imp_mats_1
    ]
    return per_date_eai_exp_0, per_date_eai_exp_1


def calc_per_date_aais(per_date_eai_exp_0, per_date_eai_exp_1):
    """
    Calculate per_date aggregate annual impact (AAI) values for two scenarios.

    Parameters
    ----------
    per_date_eai_exp_0 : list of float
        Per_Date expected annual impacts for scenario 0.
    per_date_eai_exp_1 : list of float
        Per_Date expected annual impacts for scenario 1.

    Returns
    -------
    tuple
        Tuple containing:
        - per_date_aai_0 : list of float
            Aggregate annual impact values for scenario 0.
        - per_date_aai_1 : list of float
            Aggregate annual impact values for scenario 1.
    """
    per_date_aai_0 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in per_date_eai_exp_0
    ]
    per_date_aai_1 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in per_date_eai_exp_1
    ]
    return per_date_aai_0, per_date_aai_1


def calc_freq_curve(imp_mat_intrpl, frequency, return_per=None):
    """
    Calculate the frequency curve

    Parameters:
    imp_mat_intrpl (np.array): The interpolated impact matrix
    frequency (np.array): The frequency of the hazard
    return_per (np.array): The return period

    Returns:
    ifc_return_per (np.array): The impact exceeding frequency
    ifc_impact (np.array): The impact exceeding the return period
    """

    # Calculate the at_event make the np.array
    at_event = np.sum(imp_mat_intrpl, axis=1).A1

    # Sort descendingly the impacts per events
    sort_idxs = np.argsort(at_event)[::-1]
    # Calculate exceedence frequency
    exceed_freq = np.cumsum(frequency[sort_idxs])
    # Set return period and impact exceeding frequency
    ifc_return_per = 1 / exceed_freq[::-1]
    ifc_impact = at_event[sort_idxs][::-1]

    if return_per is not None:
        interp_imp = np.interp(return_per, ifc_return_per, ifc_impact)
        ifc_return_per = return_per
        ifc_impact = interp_imp

    return ifc_impact


def calc_per_date_rps(imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods):
    """
    Calculate per_date return period impact values for two scenarios.

    Parameters
    ----------
    imp_mats_0 : list of np.ndarray
        List of interpolated impact matrices for scenario 0.
    imp_mats_1 : list of np.ndarray
        List of interpolated impact matrices for scenario 1.
    frequency_0 : np.ndarray
        Frequency values for scenario 0.
    frequency_1 : np.ndarray
        Frequency values for scenario 1.
    return_periods : list of int
        Return periods to calculate impact values for.

    Returns
    -------
    tuple
        Tuple containing:
        - rp_0 : list of np.ndarray
            Per_Date return period impact values for scenario 0.
        - rp_1 : list of np.ndarray
            Per_Date return period impact values for scenario 1.
    """
    rp_0 = [
        calc_freq_curve(imp_mat, frequency_0, return_periods) for imp_mat in imp_mats_0
    ]
    rp_1 = [
        calc_freq_curve(imp_mat, frequency_1, return_periods) for imp_mat in imp_mats_1
    ]
    return rp_0, rp_1


def get_eai_exp(eai_exp, group_map):
    """
    Aggregate expected annual impact (EAI) by groups.

    Parameters
    ----------
    eai_exp : np.ndarray
        Array of EAI values.
    group_map : dict
        Mapping of group names to indices for aggregation.

    Returns
    -------
    dict
        Dictionary of EAI values aggregated by specified groups.
    """
    eai_region_id = {}
    for group_name, exp_indices in group_map.items():
        eai_region_id[group_name] = np.sum(eai_exp[:, exp_indices], axis=1)
    return eai_region_id


def impact_mixer(
    risk_period,
    metrics,
    return_periods,
    compute_groups=False,
    all_groups_name: str | None = None,
):
    """
    Perform Bayesian mixing of impacts across snapshots.

    Parameters
    ----------
    start_snapshot : Snapshot
        The starting snapshot.
    end_snapshot : Snapshot
        The ending snapshot.
    metrics : list of str
        Metrics to calculate (e.g., 'eai', 'aai', 'rp').
    return_periods : list of int
        Return periods for calculating impact values.
    groups : dict, optional
        Mapping of group names to indices for aggregating EAI values by group.
    all_groups_name : str, optional
        Name for all-groups aggregation in the output.
    risk_transf_cover : float, optional
        Coverage level for risk transfer calculations.
    risk_transf_attach : float, optional
        Attachment point for risk transfer calculations.
    calc_residual : bool, optional
        Whether to calculate residual impacts after applying risk transfer.

    Returns
    -------
    pd.DataFrame
        DataFrame of calculated impact values by date, group, and metric.
    """

    # Posterity comment: This was called bayesian_mixing in its initial version,
    # although there is nothing really bayesian here, (but it did sound cool!)

    all_groups_n = pd.NA if all_groups_name is None else all_groups_name

    prop_H0, prop_H1 = risk_period._prop_H0, risk_period._prop_H1
    frequency_0 = risk_period.snapshot0.hazard.frequency
    frequency_1 = risk_period.snapshot1.hazard.frequency
    imp_mats_0, imp_mats_1 = risk_period.get_interp()
    per_date_eai_exp_0, per_date_eai_exp_1 = calc_per_date_eais(
        imp_mats_0, imp_mats_1, frequency_0, frequency_1
    )
    date_idx = risk_period.date_idx
    res = []
    if "aai" in metrics:
        per_date_aai_0, per_date_aai_1 = calc_per_date_aais(
            per_date_eai_exp_0, per_date_eai_exp_1
        )
        per_date_aai = prop_H0 * per_date_aai_0 + prop_H1 * per_date_aai_1
        aai_df = pd.DataFrame(index=date_idx, columns=["risk"], data=per_date_aai)
        aai_df["group"] = all_groups_n
        aai_df["metric"] = "aai"
        aai_df.reset_index(inplace=True)
        res.append(aai_df)

    if "rp" in metrics:
        rp_0, rp_1 = calc_per_date_rps(
            imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods
        )
        per_date_rp = np.multiply(prop_H0.reshape(-1, 1), rp_0) + np.multiply(
            prop_H1.reshape(-1, 1), rp_1
        )
        rp_df = pd.DataFrame(
            index=date_idx, columns=return_periods, data=per_date_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = all_groups_n
        rp_df["metric"] = "rp_" + rp_df["rp"].astype(str)
        res.append(rp_df)

    if compute_groups:
        per_date_eai = np.multiply(
            prop_H0.reshape(-1, 1), per_date_eai_exp_0
        ) + np.multiply(prop_H1.reshape(-1, 1), per_date_eai_exp_1)
        eai_group_df = pd.DataFrame(
            data=per_date_eai.T,
            index=risk_period.snapshot1.exposure.gdf["group_id"],
            columns=risk_period.date_idx,
        )
        eai_group_df = eai_group_df.groupby(eai_group_df.index).sum()
        eai_group_df = eai_group_df.melt(
            ignore_index=False, value_name="risk"
        ).reset_index(names="group")
        eai_group_df["metric"] = "aai"
        res.append(eai_group_df)

    ret = pd.concat(res, axis=0)
    ret["measure"] = risk_period.measure_name
    return ret
