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
import warnings
from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from climada.engine.option_appraisal.constants import (
    AVERTED_RISK_NAME,
    BENEFIT_COL_NAME,
    MEASURE_IMPL_COST_NAME,
    REFERENCE_RISK_NAME,
)
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures import MeasureSet
from climada.entity.measures.cost_income import (
    COST_COL_NAME,
    INCOME_COL_NAME,
    NET_COL_NAME,
)
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
            return cast(
                pd.Timestamp, pd.Timestamp(year=int(value) + int(end), month=1, day=1)
            )
        return cast(pd.Timestamp, pd.Timestamp(value))

    def _period_bounds(self, period: tuple) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the period as two Timestamps (end excluded)."""
        start = self._to_timestamp(period[0], end=False)
        end = self._to_timestamp(period[1], end=True)
        if end <= start:
            raise ValueError(f"Empty period: {start} -> {end}.")
        return start, end

    @staticmethod
    def _check_rates_cover(
        disc_rates: DiscRates | None, ref_year: int, last_year: int, name: str
    ) -> None:
        """Raise if `disc_rates` misses years needed to discount to `ref_year`.

        `_calc_npv_cash_flows` silently uses a factor of 1 beyond the years
        of the discount rates.
        """
        if disc_rates is None:
            return
        missing = sorted(
            set(range(ref_year + 1, last_year + 1)) - {int(y) for y in disc_rates.years}
        )
        if missing:
            raise ValueError(
                f"`{name}` do not cover the years {missing[0]}-{missing[-1]} "
                f"needed to discount to {ref_year}."
            )

    def _discount_factors(
        self, disc_rates: DiscRates | None, dates, ref_date: pd.Timestamp
    ) -> np.ndarray:
        """Discount factor of each date, relative to `ref_date`.

        Uses the same convention as the discounting of the risk metrics.
        """
        ones = pd.Series(1.0, index=pd.DatetimeIndex(dates))
        return np.asarray(self._calc_npv_cash_flows(ones, ref_date, disc_rates))

    def _yearly_benefits(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        risk_disc_rates: DiscRates | None,
    ) -> pd.DataFrame:
        """Averted risk of each measure for each calendar year of the period.

        Each year is attributed to the snapshot nearest to its middle (ties
        go to the later one), weighted by the fraction of the year covered
        by the period, and discounted to the start of the period.

        If `risk_disc_rates` was set at instantiation, the tidy values are net
        present values at the snapshot dates: this is undone first.

        Returns
        -------
        pd.DataFrame
            Indexed by the first day of each year, one column per measure.
        """
        years = np.arange(start.year, end.year + 1)
        y_start = pd.DatetimeIndex(
            [pd.Timestamp(year=y, month=1, day=1) for y in years]
        ).as_unit("ns")
        y_next = pd.DatetimeIndex(
            [pd.Timestamp(year=y + 1, month=1, day=1) for y in years]
        ).as_unit("ns")
        lo = y_start.where(y_start > start, start)
        hi = y_next.where(y_next < end, end)
        weights = np.clip(np.asarray((hi - lo) / (y_next - y_start), dtype=float), 0, 1)
        keep = weights > 0
        y_start, y_next, weights = y_start[keep], y_next[keep], weights[keep]

        df = self.aai_metrics()
        df = df.assign(
            **{
                MEASURE_COL_NAME: df[MEASURE_COL_NAME].astype(str),
                GROUP_COL_NAME: df[GROUP_COL_NAME].astype(str),
            }
        )
        df = df[
            (df[MEASURE_COL_NAME] != NO_MEASURE_VALUE)
            & (df[GROUP_COL_NAME] == str(self._all_groups_name))
        ]
        table = df.pivot(
            index=DATE_COL_NAME, columns=MEASURE_COL_NAME, values=AVERTED_RISK_NAME
        ).sort_index()
        table = table[[n for n in self.measure_set.names if n in table.columns]]

        snap_dates = pd.DatetimeIndex(table.index).as_unit("ns")
        last_year = int((end - pd.Timedelta(days=1)).year)
        if start.year < snap_dates[0].year or last_year > snap_dates[-1].year:
            warnings.warn(
                "Period extends beyond the snapshots: using the nearest snapshot."
            )
        if self._risk_disc_rates is not None:
            table = table.div(
                self._discount_factors(
                    self._risk_disc_rates, snap_dates, snap_dates[0]
                ),
                axis=0,
            )

        bounds = snap_dates[:-1] + (snap_dates[1:] - snap_dates[:-1]) / 2
        mid_years = y_start + (y_next - y_start) / 2
        nearest = np.searchsorted(bounds.to_numpy(), mid_years.to_numpy(), side="right")

        self._check_rates_cover(
            risk_disc_rates, start.year, last_year, "risk_disc_rates"
        )
        factors = self._discount_factors(risk_disc_rates, y_start, start)
        values = table.to_numpy()[nearest] * (weights * factors)[:, None]
        return pd.DataFrame(values, index=y_start, columns=table.columns)

    def _cost_flows(
        self,
        measure_name: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        cost_disc_rates: DiscRates | None,
    ) -> pd.DataFrame:
        """Dated cash flows of a measure over the period, discounted to `start`.

        The measure is implemented at `start`. Cash flows are those of the
        `CostIncome`, at its own frequency, with the `CostIncome` sign
        convention (costs are negative, net = income + cost).

        Returns
        -------
        pd.DataFrame
            Indexed by period, with columns "net", "costs" and "incomes".
        """
        cost_income = self.measure_set[measure_name].cost_income
        # `calc_cash_flows` includes the period containing its end date
        last = cast(pd.Timestamp, end - pd.Timedelta(days=1))
        flows = cost_income.to_dataframe(start, start, last).set_index(DATE_COL_NAME)
        self._check_rates_cover(
            cost_disc_rates, start.year, last.year, "cost_disc_rates"
        )
        npv = flows.transform(
            self._calc_npv_cash_flows,
            start_date=flows.index.min(),
            disc_rates=cost_disc_rates,
        )
        return npv

    def cost_benefit_over_period(
        self,
        period: tuple[int | pd.Timestamp, int | pd.Timestamp],
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
    ) -> pd.DataFrame:
        """Benefit, net cost and benefit-cost ratio of each measure over a period.

        Each year of the period is attributed to the nearest snapshot, whose
        averted risk is taken as representative of that year. Years only
        partially covered by the period are weighted accordingly for the
        benefit (not for the costs, which follow the cash flows of the
        measure). Measures are assumed to be implemented at the start of the
        period.

        Parameters
        ----------
        period : tuple
            Start and end of the period. Either integers (years, both
            included), or dates parsable by `pandas.Timestamp` (end excluded).
        risk_disc_rates : DiscRates, optional
            Discount rates for the benefits, overriding those given at
            instantiation. Defaults to those given at instantiation, if any.
        cost_disc_rates : DiscRates, optional
            Discount rates for the costs and incomes, overriding those given at
            instantiation. Defaults to those given at instantiation, if any.

        Returns
        -------
        pd.DataFrame
            One row per measure, with the benefit, the cost, the income, the
            net cost (cost minus income) and the benefit-cost ratio (benefit
            divided by net cost, larger is better). Costs and incomes are
            positive amounts.

        Raises
        ------
        ValueError
            If the period is empty or the discount rates do not cover it.
        """
        risk_rates = (
            risk_disc_rates if risk_disc_rates is not None else self._risk_disc_rates
        )
        cost_rates = (
            cost_disc_rates if cost_disc_rates is not None else self._cost_disc_rates
        )
        start, end = self._period_bounds(period)
        benefits = self._yearly_benefits(start, end, risk_rates)

        rows = {}
        for name in benefits.columns:
            flows = self._cost_flows(name, start, end, cost_rates)
            rows[name] = {
                BENEFIT_COL_NAME: benefits[name].sum(),
                COST_COL_NAME: -flows[COST_COL_NAME].sum(),
                INCOME_COL_NAME: flows[INCOME_COL_NAME].sum(),
                NET_COL_NAME: -flows[NET_COL_NAME].sum(),
            }
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.index.name = MEASURE_COL_NAME
        out["benefit_cost_ratio"] = out[BENEFIT_COL_NAME] / out[NET_COL_NAME]
        return out

    def cumulative_net_benefit(
        self,
        period: tuple[int | pd.Timestamp, int | pd.Timestamp],
        risk_disc_rates: DiscRates | None = None,
        cost_disc_rates: DiscRates | None = None,
    ) -> pd.DataFrame:
        """Cumulated benefit plus cumulated net cash flow, at the end of each year.

        Parameters
        ----------
        period : tuple
            See `cost_benefit_over_period`.
        risk_disc_rates : DiscRates, optional
            See `cost_benefit_over_period`.
        cost_disc_rates : DiscRates, optional
            See `cost_benefit_over_period`.

        Returns
        -------
        pd.DataFrame
            Indexed by date (the start of the period, then the end of each
            year), one column per measure. The year at which a column
            becomes positive is the payback time of the measure.
        """
        risk_rates = (
            risk_disc_rates if risk_disc_rates is not None else self._risk_disc_rates
        )
        cost_rates = (
            cost_disc_rates if cost_disc_rates is not None else self._cost_disc_rates
        )
        start, end = self._period_bounds(period)
        benefits = self._yearly_benefits(start, end, risk_rates)

        dates = pd.DatetimeIndex(
            [start, *(min(t + pd.DateOffset(years=1), end) for t in benefits.index)],
            name=DATE_COL_NAME,
        )
        out = {}
        for name in benefits.columns:
            net = self._cost_flows(name, start, end, cost_rates)["net"]
            net_per_year = net.groupby(net.index.start_time.year).sum()
            cum_net = net_per_year.reindex(
                benefits.index.year, fill_value=0.0
            ).cumsum()  # ty: ignore
            out[name] = np.r_[
                net.iloc[0],  # at the start: what is spent at implementation
                benefits[name].cumsum().to_numpy() + cum_net.to_numpy(),
            ]
        return pd.DataFrame(out, index=dates)
