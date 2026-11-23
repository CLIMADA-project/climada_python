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
from collections.abc import Iterable

import pandas as pd
from tqdm import tqdm

from climada.engine.option_appraisal.constants import (
    AVERTED_RISK_NAME,
    MEASURE_IMPL_COST_NAME,
    REFERENCE_RISK_NAME,
)
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures import MeasureSet
from climada.trajectories.calc_risk_metrics import resolve_kwargs
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
        snapshots_list: list[Snapshot],
        *,
        measure_set: MeasureSet,
        return_periods: Iterable[int] = DEFAULT_RP,
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
        measure_kwargs=None,
        enforce_baseline_copy=True,
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
        self._measure_kwargs = dict(measure_kwargs or {})
        self._enforce_copy = enforce_baseline_copy
        unknown = set(self._measure_kwargs) - set(measure_set.names)
        if unknown:
            raise ValueError(f"Kwargs given for unknown measures: {sorted(unknown)}")

        self._cost_disc_rates = cost_disc_rates
        self.measure_set = copy.deepcopy(measure_set)

        snapshots_adapt = []
        for meas_name, meas in self.measure_set.items():
            for snap in snapshots_list:
                snapshots_adapt.append(
                    snap.apply_measure(
                        meas,
                        enforce_copy=self._enforce_copy,
                        **resolve_kwargs(self._measure_kwargs.get(meas_name), snap),
                    )
                )
        super().__init__(
            snapshots_list + snapshots_adapt,
            return_periods=return_periods,
            risk_disc_rates=risk_disc_rates,
            impact_computation_strategy=impact_computation_strategy,
        )

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

    def _metric_post_treatment(self, metric_df: pd.DataFrame) -> pd.DataFrame:
        def meas_impl_cost(measure_name: str) -> float:
            if measure_name == NO_MEASURE_VALUE:
                return 0.0

            return self.measure_set[measure_name].cost_income.init_cost

        metric_df = self._avoid_duplicates(metric_df)
        metric_df = self._handle_group_categories(metric_df)
        if self._risk_disc_rates:
            LOGGER.debug("Found risk discount rate. Computing NPV.")
            metric_df = self.npv_transform(metric_df, self._risk_disc_rates)

        metric_df = self._calc_averted(metric_df)
        metric_df[MEASURE_IMPL_COST_NAME] = metric_df[MEASURE_COL_NAME].map(
            meas_impl_cost
        )
        metric_df = reorder_dataframe_columns(metric_df, DEFAULT_DF_COLUMN_PRIORITY)
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
