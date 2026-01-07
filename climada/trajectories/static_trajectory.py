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
from typing import Iterable

import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.calc_risk_metrics import CalcRiskMetricsPoints
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    RETURN_PERIOD_METRIC_NAME,
)
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import (
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_DF_COLUMN_PRIORITY,
    DEFAULT_RP,
    RiskTrajectory,
)
from climada.util import log_level
from climada.util.dataframe_handling import reorder_dataframe_columns

LOGGER = logging.getLogger(__name__)

__all__ = ["StaticRiskTrajectory"]


class StaticRiskTrajectory(RiskTrajectory):
    """This class implements static risk trajectories, objects that
    regroup impacts computations for multiple dates.

    This class computes risk metrics over a series of snapshots,
    optionally applying risk discounting. It does not interpolate risk
    between the snapshot and only provides results for each snapshot.

    """

    POSSIBLE_METRICS = [
        EAI_METRIC_NAME,
        AAI_METRIC_NAME,
        RETURN_PERIOD_METRIC_NAME,
        AAI_PER_GROUP_METRIC_NAME,
    ]
    """Class variable listing the risk metrics that can be computed.

    Currently:

    - eai, expected impact (per exposure point within a period of 1/frequency
    unit of the hazard object)
    - aai, average annual impact (aggregated eai over the whole exposure)
    - aai_per_group, average annual impact per exposure subgroup (defined from
    the exposure geodataframe)
    - return_periods, estimated impacts aggregated over the whole exposure for
    different return periods

    """

    _DEFAULT_ALL_METRICS = [
        AAI_METRIC_NAME,
        RETURN_PERIOD_METRIC_NAME,
        AAI_PER_GROUP_METRIC_NAME,
    ]

    def __init__(
        self,
        snapshots_list: Iterable[Snapshot],
        *,
        return_periods: Iterable[int] = DEFAULT_RP,
        all_groups_name: str = DEFAULT_ALLGROUP_NAME,
        risk_disc_rates: DiscRates | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        """Initialize a new `StaticRiskTrajectory`.

        Parameters
        ----------
        snapshots_list : list[Snapshot]
            The list of `Snapshot` object to compute risk from.
        return_periods: list[int], optional
            The return periods to use when computing the `return_periods_metric`.
            Defaults to `DEFAULT_RP` ([20, 50, 100]).
        all_groups_name: str, optional
            The string that should be used to define "all exposure points" subgroup.
            Defaults to `DEFAULT_ALLGROUP_NAME` ("All").
        risk_disc_rates: DiscRates, optional
            The discount rate to apply to future risk. Defaults to None.
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
        self._risk_metrics_calculators = CalcRiskMetricsPoints(
            self._snapshots,
            impact_computation_strategy=impact_computation_strategy
            or ImpactCalcComputation(),
        )

    @property
    def impact_computation_strategy(self) -> ImpactComputationStrategy:
        """The approach or strategy used to calculate the impact from the snapshots."""
        return self._risk_metrics_calculators.impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._reset_metrics()
        self._risk_metrics_calculators.impact_computation_strategy = value

    def _generic_metrics(
        self,
        /,
        metric_name: str | None = None,
        metric_meth: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generic method to compute metrics based on the provided metric name and method.

        This method calls the appropriate method from the calculator to return
        the results for the given metric, in a tidy formatted dataframe.

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

        with log_level(level="WARNING", name_prefix="climada"):
            tmp = getattr(self._risk_metrics_calculators, metric_meth)(**kwargs)
            if tmp is None:
                return tmp

        tmp = tmp.set_index(
            [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME]
        )
        if COORD_ID_COL_NAME in tmp.columns:
            tmp = tmp.set_index([COORD_ID_COL_NAME], append=True)

        # When more than 2 snapshots, there might be duplicated rows, we need to remove them.
        # Should not be the case in static trajectory, but in any case we really don't want
        # duplicated rows, which would mess up some dataframe manipulation down the road.
        tmp = tmp[~tmp.index.duplicated(keep="first")]
        tmp = tmp.reset_index()
        if self._all_groups_name not in tmp[GROUP_COL_NAME].cat.categories:
            tmp[GROUP_COL_NAME] = tmp[GROUP_COL_NAME].cat.add_categories(
                [self._all_groups_name]
            )
            tmp[GROUP_COL_NAME] = tmp[GROUP_COL_NAME].fillna(self._all_groups_name)

        if self._risk_disc_rates:
            tmp = self.npv_transform(tmp, self._risk_disc_rates)

        tmp = reorder_dataframe_columns(tmp, DEFAULT_DF_COLUMN_PRIORITY)

        setattr(self, attr_name, tmp)
        return getattr(self, attr_name)

    def eai_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the estimated annual impacts at each exposure point for each date.

        This method computes and return a `DataFrame` with eai metric
        (for each exposure point) for each date.

        Notes
        -----

        This computation may become quite expensive for big areas with high resolution.

        """
        metric_df = self._compute_metrics(
            metric_name=EAI_METRIC_NAME, metric_meth="calc_eai_gdf", **kwargs
        )
        return metric_df

    def aai_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the average annual impacts for each date.

        This method computes and return a `DataFrame` with aai metric for each date.

        """

        return self._compute_metrics(
            metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric", **kwargs
        )

    def return_periods_metrics(self, **kwargs) -> pd.DataFrame:
        """Return the estimated impacts for different return periods.

        Return periods to estimate impacts for are defined by `self.return_periods`.

        """
        return self._compute_metrics(
            metric_name=RETURN_PERIOD_METRIC_NAME,
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
            metric_name=AAI_PER_GROUP_METRIC_NAME,
            metric_meth="calc_aai_per_group_metric",
            **kwargs,
        )

    def per_date_risk_metrics(
        self,
        metrics: list[str] | None = None,
    ) -> pd.DataFrame | pd.Series:
        """Returns a DataFrame of risk metrics for each dates.

        This methods collects (and if needed computes) the `metrics`
        (Defaulting to AAI_METRIC_NAME, RETURN_PERIOD_METRIC_NAME and AAI_PER_GROUP_METRIC_NAME).

        Parameters
        ----------
        metrics : list[str], optional
            The list of metrics to return (defaults to
            [AAI_METRIC_NAME,RETURN_PERIOD_METRIC_NAME,AAI_PER_GROUP_METRIC_NAME])

        Returns
        -------
        pd.DataFrame | pd.Series
            A tidy DataFrame with metric values for all possible dates.

        """

        metrics = (
            [AAI_METRIC_NAME, RETURN_PERIOD_METRIC_NAME, AAI_PER_GROUP_METRIC_NAME]
            if metrics is None
            else metrics
        )
        return pd.concat(
            [getattr(self, f"{metric}_metrics")() for metric in metrics],
            ignore_index=True,
        )
