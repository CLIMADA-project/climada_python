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

import numpy as np
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

            return -self.measure_set[measure_name].cost_income.init_cost

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

    @staticmethod
    def _to_timestamp(value, *, end: bool) -> pd.Timestamp:
        """Convert a year (int) or any pandas-parsable date to a Timestamp.

        Integer years are inclusive: a start year is mapped to 1 January of that
        year, an end year to 1 January of the following year.
        """
        if isinstance(value, (int, np.integer)):
            return pd.Timestamp(year=int(value) + int(end), month=1, day=1)
        return pd.Timestamp(value)

    def cost_benefit_over_period(
        self,
        period: tuple[int | pd.Timestamp, int | pd.Timestamp],
        disc_rates: DiscRates | None = None,
    ) -> pd.DataFrame:
        """Benefit, net cost and benefit-cost ratio of each measure over a period.

        Each year of the period is attributed to the nearest snapshot (ties go to
        the later one), whose averted risk is taken as representative of that year.
        Years only partially covered by the period are weighted accordingly.
        Measures are assumed to be implemented at the start of the period.

        Parameters
        ----------
        period : tuple
            Start and end of the period. Either integers (years, both included),
            or dates parsable by `pandas.Timestamp` (end excluded).
        disc_rates : DiscRates, optional
            Used to discount benefits to the first year of the period.
            Defaults to None (no discounting).

        Returns
        -------
        pd.DataFrame
            One row per measure, with the benefit, the costs, the incomes, the
            net cost and the benefit-cost ratio (benefit divided by net cost).

        Raises
        ------
        ValueError
            If the period is empty, or if `risk_disc_rates` was set at instantiation
            (it would discount the benefits twice).
        NotImplementedError
            If `cost_disc_rates` is set (not applied by `calc_total`).
        """
        if self._risk_disc_rates is not None:
            raise ValueError(
                "Instantiate without `risk_disc_rates`; pass `disc_rates` here."
            )
        if self._cost_disc_rates is not None:
            raise NotImplementedError("`cost_disc_rates` is not supported here yet.")

        start = self._to_timestamp(period[0], end=False)
        end = self._to_timestamp(period[1], end=True)
        if end <= start:
            raise ValueError(f"Empty period: {start} -> {end}.")

        # Fraction of each calendar year covered by the period
        years = np.arange(start.year, end.year + 1)
        y_start = np.array([f"{y}-01-01" for y in years], dtype="datetime64[ns]")
        y_next = np.array([f"{y + 1}-01-01" for y in years], dtype="datetime64[ns]")
        covered = np.minimum(y_next, end.to_datetime64()) - np.maximum(
            y_start, start.to_datetime64()
        )
        weights = np.clip(covered / (y_next - y_start), 0.0, 1.0)

        # Nearest snapshot to the middle of each year
        df = self.aai_metrics()
        df = df[
            (df[MEASURE_COL_NAME] != NO_MEASURE_VALUE)
            & (df[GROUP_COL_NAME] == self._all_groups_name)
        ]
        snap_dates = pd.DatetimeIndex(sorted(df[DATE_COL_NAME].unique()))
        if start < snap_dates[0] or end > snap_dates[-1]:
            LOGGER.warning(
                "Period extends beyond the snapshots: extrapolating nearest values."
            )
        mid_years = pd.DatetimeIndex(y_start + (y_next - y_start) // 2)
        nearest = snap_dates.get_indexer(mid_years, method="nearest")

        rows = {}
        for meas_name, sub in df.groupby(MEASURE_COL_NAME, observed=True):
            averted = (
                sub.set_index(DATE_COL_NAME)[AVERTED_RISK_NAME]
                .reindex(snap_dates)
                .to_numpy()[nearest]
                * weights
            )
            benefit = (
                disc_rates.net_present_value(start.year, end.year, averted)
                if disc_rates
                else averted.sum()
            )
            net, costs, incomes = self.measure_set[meas_name].cost_income.calc_total(
                start, start, end
            )
            rows[meas_name] = {
                "benefit": benefit,
                "costs": -costs,
                "incomes": incomes,
                "net_cost": -net,
            }

        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = MEASURE_COL_NAME
        out["benefit_cost_ratio"] = out["benefit"] / out["net_cost"]
        return out
