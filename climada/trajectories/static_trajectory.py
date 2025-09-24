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

This file implements \"static\" risk trajectory objects, for an easier evaluation
of risk at multiple points in time (snapshots).

"""

import logging

import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.impact_calc_strat import ImpactCalcComputation
from climada.trajectories.riskperiod import (
    CalcRiskMetricsPoints,
    ImpactComputationStrategy,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import RiskTrajectory
from climada.util import log_level

LOGGER = logging.getLogger(__name__)


class StaticRiskTrajectory(RiskTrajectory):
    """Calculates risk trajectories over a series of snapshots.

    This class computes risk metrics over a series of snapshots,
    optionally applying risk discounting.

    """

    POSSIBLE_METRICS = [
        "eai",
        "aai",
        "return_periods",
        "aai_per_group",
    ]

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        super().__init__(
            snapshots_list,
            all_groups_name=all_groups_name,
            risk_disc=risk_disc,
        )
        self._risk_metrics_calculators = CalcRiskMetricsPoints(
            self._snapshots,
            impact_computation_strategy=impact_computation_strategy
            or ImpactCalcComputation(),
        )

    @property
    def impact_computation_strategy(self) -> ImpactComputationStrategy:
        """The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots."""
        return self._risk_metrics_calculators.impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._reset_metrics()
        self._risk_metrics_calculators.impact_computation_strategy = value

    def _generic_metrics(
        self,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
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
        with log_level(level="WARNING", name_prefix="climada"):
            tmp.append(getattr(self._risk_metrics_calculators, metric_meth)(**kwargs))

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

            if self._risk_disc:
                return self.npv_transform(getattr(self, attr_name), self._risk_disc)

            return getattr(self, attr_name)

    def eai_metrics(self, **kwargs) -> pd.DataFrame:
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
            metric_name="eai", metric_meth="calc_eai_gdf", **kwargs
        )
        return df

    def aai_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each date.

        This method computes and return a `DataFrame` with aai metric for each date.

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.
        """

        return self._compute_metrics(
            metric_name="aai", metric_meth="calc_aai_metric", **kwargs
        )

    def return_periods_metrics(self, **kwargs) -> pd.DataFrame:
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

        Parameters
        ----------
        npv : bool
            Whether to apply the (risk) discount rate if it is defined.
            Defaults to `True`.
        """

        return self._compute_metrics(
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
            **kwargs,
        )

    def per_date_risk_metrics(
        self,
        metrics: list[str] | None = None,
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
        if "aai" in metrics:
            metrics_df.append(self.aai_metrics())
        if "return_periods" in metrics:
            metrics_df.append(self.return_periods_metrics())
        if "aai_per_group" in metrics:
            metrics_df.append(self.aai_per_group_metrics())

        return pd.concat(metrics_df)
