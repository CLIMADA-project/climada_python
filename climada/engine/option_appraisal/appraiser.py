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
import logging
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from climada.engine.option_appraisal.constants import (
    AVERTED_RISK_NAME,
    MEASURE_IMPL_COST_NAME,
    REFERENCE_RISK_NAME,
)
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.trajectories.calc_risk_metrics import CalcRiskMetricsPoints
from climada.trajectories.constants import (
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RISK_COL_NAME,
)
from climada.trajectories.impact_calc_strat import ImpactComputationStrategy
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
        snapshots_list: Iterable[Snapshot],
        *,
        measure_set: MeasureSet,
        return_periods: Iterable[int] = DEFAULT_RP,
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
        if value is not None and not isinstance(value, DiscRates):
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
        def meas_impl_cost(measure_name: str) -> float:
            if measure_name == NO_MEASURE_VALUE:
                return 0.0

            return self.measure_set.measures()[measure_name].cost_income.init_cost

        metric_df = self._handle_group_categories(metric_df)
        if self._risk_disc_rates:
            LOGGER.debug("Found risk discount rate. Computing NPV.")
            metric_df = self.npv_transform(metric_df, self._risk_disc_rates)

        LOGGER.debug("Computing averted risk for: %s.", metric_name)
        metric_df = self._calc_averted(metric_df)
        metric_df[MEASURE_IMPL_COST_NAME] = metric_df[MEASURE_COL_NAME].map(
            meas_impl_cost
        )
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
