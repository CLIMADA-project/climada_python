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

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climada.engine.impact_calc import ImpactCalc
from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.riskperiod import RiskPeriod, bayesian_viktypliers
from climada.trajectories.snapshot import SnapshotsCollection

LOGGER = logging.getLogger(__name__)


class YearlyRiskTrajectory:

    _grouper = ["measure", "metric"]

    def __init__(
        self,
        snapshots: SnapshotsCollection,
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
        self.return_periods = return_periods
        self.start_year = min(snapshots.snapshots_years)
        self.end_year = max(snapshots.snapshots_years)
        self.risk_disc = risk_disc
        self.risk_transf_cover = risk_transf_cover
        self.risk_transf_attach = risk_transf_attach
        LOGGER.debug("Computing risk periods")
        self.risk_periods = self._calc_risk_periods(snapshots)
        self._update_risk_metrics(compute_groups=compute_groups)

    def _calc_risk_periods(self, snapshots):
        return [
            RiskPeriod(start_snapshot, end_snapshot)
            for start_snapshot, end_snapshot in snapshots.pairwise()
        ]

    def _update_risk_metrics(self, compute_groups=False):
        results_df = []
        for period in self.risk_periods:
            results_df.append(
                bayesian_mixer_opti(
                    period,
                    self.metrics,
                    self.return_periods,
                    compute_groups,
                    all_groups_name="All",
                )
            )
        results_df = pd.concat(results_df, axis=0)

        # duplicate rows arise from overlapping end and start if there's more than two snapshots
        results_df.drop_duplicates(inplace=True)

        # reorder the columns (but make sure not to remove possibly important ones in the future)
        columns_to_front = ["year", "measure", "metric"]
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
    def _calc_periods_risk(cls, df: pd.DataFrame, colname="risk"):
        def identify_continuous_periods(group):
            # Calculate the difference between consecutive years
            group["year_diff"] = group["year"].diff()
            # Identify breaks in continuity
            group["period_id"] = (group["year_diff"] != 1).cumsum()
            return group

        grouper = cls._grouper
        if "group" in df.columns:
            grouper = ["group"] + grouper

        df_sorted = df.sort_values(by=cls._grouper + ["year"])
        # Apply the function to identify continuous periods
        df_periods = df_sorted.groupby(grouper, dropna=False, group_keys=False).apply(
            identify_continuous_periods
        )

        # Group by the identified periods and calculate start and end years
        df_periods = (
            df_periods.groupby(grouper + ["period_id"], dropna=False)
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
        df_periods = df_periods.drop(["period_id", "start_year", "end_year"], axis=1)
        return df_periods[
            ["period"] + [col for col in df_periods.columns if col != "period"]
        ]

    @property
    def yearly_risk_metrics(self):
        return self._calc_risk_metrics(total=False, npv=True)

    @property
    def total_risk_metrics(self):
        return self._calc_risk_metrics(total=True, npv=True)

    def _calc_risk_metrics(self, total=False, npv=True):
        df = self._calc_annual_risk_metrics(npv=npv)
        if total:
            return self._calc_periods_risk(df)

        return df

    def _calc_waterfall_plot_data(self, start_year=None, end_year=None):
        start_year = self.start_year if start_year is None else start_year
        end_year = self.end_year if end_year is None else end_year
        considered_risk_periods = self._get_risk_periods(
            self.risk_periods, start_year=start_year, end_year=end_year
        )

        risk_component = {
            str(period.start_year)
            + "-"
            + str(period.end_year): self._calc_risk_component(period)
            for period in considered_risk_periods
        }
        risk_component = pd.concat(
            risk_component.values(), keys=risk_component.keys(), names=["Period"]
        ).reset_index()
        risk_component = risk_component.loc[
            (risk_component["Year"] >= start_year)
            & (risk_component["Year"] <= end_year)
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
        imp_mats_H0 = period.imp_mats_0
        imp_mats_H1 = period.imp_mats_1
        freq_H0 = period.snapshot0.hazard.frequency
        freq_H1 = period.snapshot1.hazard.frequency
        yearly_eai_H0, yearly_eai_H1 = calc_yearly_eais(
            imp_mats_H0, imp_mats_H1, freq_H0, freq_H1
        )
        yearly_aai_H0, yearly_aai_H1 = calc_yearly_aais(yearly_eai_H0, yearly_eai_H1)
        prop_H0, prop_H1 = bayesian_viktypliers(period.start_year, period.end_year)
        yearly_aai = prop_H0 * yearly_aai_H0 + prop_H1 * yearly_aai_H1

        risk_dev_0 = yearly_aai_H0 - yearly_aai[0]
        risk_cc_0 = yearly_aai - (risk_dev_0 + yearly_aai[0])
        df = pd.DataFrame(
            {
                "Base risk": yearly_aai - (risk_dev_0 + risk_cc_0),
                "Change in Exposure": risk_dev_0,
                "Change in Hazard (with Exposure)": risk_cc_0,
            },
            index=pd.Index(
                [year for year in range(period.start_year, period.end_year + 1)],
                name="Year",
            ),
        )
        return df.round(1)

    def plot_yearly_waterfall(self, ax=None, start_year=None, end_year=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 6))
        start_year = self.start_year if start_year is None else start_year
        end_year = self.end_year if end_year is None else end_year
        risk_component = self._calc_waterfall_plot_data(
            start_year=start_year, end_year=end_year
        )
        risk_component.plot(ax=ax, kind="bar", x="Year", stacked=True)
        # Construct y-axis label and title based on parameters
        value_label = "USD"
        title_label = (
            f"Risk between {start_year} and {end_year} (Annual Average impact)"
        )

        ax.set_title(title_label)
        ax.set_ylabel(value_label)
        return ax

    def plot_waterfall(self, ax=None, start_year=None, end_year=None):
        start_year = self.start_year if start_year is None else start_year
        end_year = self.end_year if end_year is None else end_year
        risk_component = self._calc_waterfall_plot_data(
            start_year=start_year, end_year=end_year
        )
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        risk_component = risk_component.loc[
            (risk_component["Year"] == end_year)
        ].squeeze()

        labels = [
            f"Risk {start_year}",
            f"Exposure {end_year}",
            f"Hazard {end_year}¹",
            f"Total Risk {end_year}",
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
        title_label = f"Risk at {start_year} and {end_year} (Annual Average impact)"

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


def calc_yearly_eais(imp_mats_0, imp_mats_1, frequency_0, frequency_1):
    """
    Calculate yearly expected annual impact (EAI) values for two scenarios.

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
        - yearly_eai_exp_0 : list of float
            Yearly expected annual impacts for scenario 0.
        - yearly_eai_exp_1 : list of float
            Yearly expected annual impacts for scenario 1.
    """
    yearly_eai_exp_0 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_0) for imp_mat in imp_mats_0
    ]
    yearly_eai_exp_1 = [
        ImpactCalc.eai_exp_from_mat(imp_mat, frequency_1) for imp_mat in imp_mats_1
    ]
    return yearly_eai_exp_0, yearly_eai_exp_1


def calc_yearly_aais(yearly_eai_exp_0, yearly_eai_exp_1):
    """
    Calculate yearly aggregate annual impact (AAI) values for two scenarios.

    Parameters
    ----------
    yearly_eai_exp_0 : list of float
        Yearly expected annual impacts for scenario 0.
    yearly_eai_exp_1 : list of float
        Yearly expected annual impacts for scenario 1.

    Returns
    -------
    tuple
        Tuple containing:
        - yearly_aai_0 : list of float
            Aggregate annual impact values for scenario 0.
        - yearly_aai_1 : list of float
            Aggregate annual impact values for scenario 1.
    """
    yearly_aai_0 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_0
    ]
    yearly_aai_1 = [
        ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in yearly_eai_exp_1
    ]
    return yearly_aai_0, yearly_aai_1


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


def calc_yearly_rps(imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods):
    """
    Calculate yearly return period impact values for two scenarios.

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
            Yearly return period impact values for scenario 0.
        - rp_1 : list of np.ndarray
            Yearly return period impact values for scenario 1.
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


def bayesian_mixer_opti(
    risk_period,
    metrics,
    return_periods,
    compute_groups=False,
    all_groups_name=pd.NA,
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
        DataFrame of calculated impact values by year, group, and metric.
    """
    # 1. Interpolate in between years

    prop_H0, prop_H1 = risk_period._prop_H0, risk_period._prop_H1
    frequency_0 = risk_period.snapshot0.hazard.frequency
    frequency_1 = risk_period.snapshot1.hazard.frequency
    imp_mats_0, imp_mats_1 = risk_period.get_interp()
    yearly_eai_exp_0, yearly_eai_exp_1 = calc_yearly_eais(
        imp_mats_0, imp_mats_1, frequency_0, frequency_1
    )
    year_idx = risk_period.year_idx
    res = []
    if "aai" in metrics:
        yearly_aai_0, yearly_aai_1 = calc_yearly_aais(
            yearly_eai_exp_0, yearly_eai_exp_1
        )
        yearly_aai = prop_H0 * yearly_aai_0 + prop_H1 * yearly_aai_1
        aai_df = pd.DataFrame(index=year_idx, columns=["risk"], data=yearly_aai)
        aai_df["group"] = all_groups_name
        aai_df["metric"] = "aai"
        aai_df.reset_index(inplace=True)
        res.append(aai_df)

    if "rp" in metrics:
        rp_0, rp_1 = calc_yearly_rps(
            imp_mats_0, imp_mats_1, frequency_0, frequency_1, return_periods
        )
        yearly_rp = np.multiply(prop_H0.reshape(-1, 1), rp_0) + np.multiply(
            prop_H1.reshape(-1, 1), rp_1
        )
        rp_df = pd.DataFrame(
            index=year_idx, columns=return_periods, data=yearly_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = all_groups_name
        rp_df["metric"] = "rp_" + rp_df["rp"].astype(str)
        res.append(rp_df)

    if compute_groups:
        yearly_eai = np.multiply(
            prop_H0.reshape(-1, 1), yearly_eai_exp_0
        ) + np.multiply(prop_H1.reshape(-1, 1), yearly_eai_exp_1)
        eai_group_df = pd.DataFrame(
            data=yearly_eai.T,
            index=risk_period.snapshot1.exposure.gdf["group_id"],
            columns=risk_period.year_idx,
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
