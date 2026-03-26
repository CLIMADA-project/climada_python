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

This file implements abstract trajectory objects, to factorise the code common to
interpolated and static trajectories.

"""

import datetime
import logging
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.constants import (
    DATE_COL_NAME,
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_RP,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    PERIOD_COL_NAME,
    RISK_COL_NAME,
    UNIT_COL_NAME,
)
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)

__all__ = ["RiskTrajectory"]

DEFAULT_DF_COLUMN_PRIORITY = [
    DATE_COL_NAME,
    PERIOD_COL_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    UNIT_COL_NAME,
]
INDEXING_COLUMNS = [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME]


class RiskTrajectory(ABC):
    """Base abstract class for risk trajectory objects.

    See concrete implementation :class:`StaticRiskTrajectory` and
    :class:`InterpolatedRiskTrajectory` for more details.

    """

    _grouper = [MEASURE_COL_NAME, METRIC_COL_NAME]
    """Results dataframe grouper used in most `groupby()` calls."""

    POSSIBLE_METRICS = []
    """Class variable listing the risk metrics that can be computed."""

    def __init__(
        self,
        snapshots_list: Iterable[Snapshot],
        *,
        return_periods: Iterable[int] = DEFAULT_RP,
        all_groups_name: str = DEFAULT_ALLGROUP_NAME,
        risk_disc_rates: DiscRates | None = None,
    ):
        self._reset_metrics()
        self._snapshots = sorted(snapshots_list, key=lambda snap: snap.date)
        self._all_groups_name = all_groups_name
        self._return_periods = return_periods
        self.start_date = min((snapshot.date for snapshot in snapshots_list))
        self.end_date = max((snapshot.date for snapshot in snapshots_list))
        self._risk_disc_rates = risk_disc_rates

    def _reset_metrics(self) -> None:
        """Resets the computed metrics to None.

        This method is called to inititialize the `POSSIBLE_METRICS` to `None` during
        the initialisation.

        It is also called when properties that would change the results of
        computed metrics (for instance changing the time resolution in
        :class:`InterpolatedRiskMetrics`)

        """
        for metric in self.POSSIBLE_METRICS:
            setattr(self, "_" + metric + "_metrics", None)

    @abstractmethod
    def _generic_metrics(
        self, /, metric_name: str, metric_meth: str, **kwargs
    ) -> pd.DataFrame:
        """Main method to return the results of a specific metric.

        This method should call the `_generic_metrics()` of its parent and
        define the part of the computation and treatment that
        is specific to a child class of :class:`RiskTrajectory`.

        See also
        --------

        - :method:`_compute_metrics`

        """
        raise NotImplementedError(
            f"'_generic_metrics' must be implemented by subclasses of {self.__class__.__name__}"
        )

    def _compute_metrics(
        self, /, metric_name: str, metric_meth: str, **kwargs
    ) -> pd.DataFrame:
        """Helper method to compute metrics.

        Notes
        -----

        This method exists for the sake of the children classes for option appraisal, for which
        `_generic_metrics` can have a different signature and extend on its
        parent method. This method can stay the same (same signature) for all classes.
        """
        return self._generic_metrics(
            metric_name=metric_name, metric_meth=metric_meth, **kwargs
        )

    @property
    def return_periods(self) -> Iterable[int]:
        """The return period values to use when computing risk period metrics.

        Notes
        -----

        Changing its value resets the corresponding metric.
        """
        return self._return_periods

    @return_periods.setter
    def return_periods(self, value, /):
        if not isinstance(value, Iterable):
            raise ValueError("Return periods need to be a list of int.")
        if any(not isinstance(i, int) for i in value):
            raise ValueError("Return periods need to be a list of int.")
        self._return_periods_metrics = None
        self._return_periods = value

    @property
    def risk_disc_rates(self) -> DiscRates | None:
        """The discount rate applied to compute net present values.
        None means no discount rate.

        Notes
        -----

        Changing its value resets all the metrics.
        """
        return self._risk_disc_rates

    @risk_disc_rates.setter
    def risk_disc_rates(self, value, /):
        if value is not None and not isinstance(value, (DiscRates)):
            raise ValueError(
                "The discount rate applied to risk values needs to be a `DiscRates` object."
            )

        self._reset_metrics()
        self._risk_disc_rates = value

    @classmethod
    def npv_transform(
        cls, metric_df: pd.DataFrame, risk_disc_rates: DiscRates
    ) -> pd.DataFrame:
        """Apply provided discount rate to the provided metric `DataFrame`.

        Parameters
        ----------
        metric_df : pd.DataFrame
            The `DataFrame` of the metric to discount.
        risk_disc_rates : DiscRate
            The discount rate to apply.

        Returns
        -------
        pd.DataFrame
            The discounted risk metric.

        """

        def _npv_group(group, disc):
            start_date = group.index.get_level_values(DATE_COL_NAME).min()
            return cls._calc_npv_cash_flows(group, start_date, disc)

        metric_df = metric_df.set_index(DATE_COL_NAME)
        grouper = cls._grouper
        if GROUP_COL_NAME in metric_df.columns:
            grouper = [GROUP_COL_NAME] + grouper

        metric_df[RISK_COL_NAME] = metric_df.groupby(
            grouper,
            dropna=False,
            as_index=False,
            group_keys=False,
            observed=True,
        )[RISK_COL_NAME].transform(_npv_group, risk_disc_rates)
        metric_df = metric_df.reset_index()
        return metric_df

    @staticmethod
    def _calc_npv_cash_flows(
        cash_flows: pd.Series,
        start_date: datetime.date,
        disc_rates: DiscRates | None = None,
    ) -> pd.Series:
        """Apply discount rate to cash flows.

        If it is defined, applies a discount rate `disc` to a given cash flow
        `cash_flows` using `start_date` as the reference year.

        Parameters
        ----------
        cash_flows : pd.DataFrame
            The cash flow to apply the discount rate to.
        start_date : datetime.date
            The date representing the present.
        disc : DiscRates, optional
            The discount rates to apply.

        Returns
        -------

        A Series (copy) of `cash_flows` where values are discounted according to `disc`.

        """

        if disc_rates is None:
            return cash_flows

        if not isinstance(cash_flows.index, (pd.PeriodIndex, pd.DatetimeIndex)):
            raise ValueError(
                "cash_flows must be a pandas Series with a PeriodIndex or DatetimeIndex"
            )

        growth_factors = (
            pd.Series(disc_rates.rates, index=disc_rates.years)
            .loc[lambda x: x.index > start_date.year]
            .add(1)
            .cumprod()
        )
        discount_factors = 1 / cash_flows.index.year.map(growth_factors).fillna(1.0)
        return cash_flows.multiply(discount_factors, axis=0)
