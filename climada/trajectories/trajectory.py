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
from abc import ABC

import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.riskperiod import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


class RiskTrajectory(ABC):
    _grouper = ["measure", "metric"]
    """Results dataframe grouper"""

    POSSIBLE_METRICS = []
    DEFAULT_RP = []

    def __init__(
        self,
        snapshots_list: list[Snapshot],
        *,
        all_groups_name: str = "All",
        risk_disc: DiscRates | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        self._reset_metrics()
        self._snapshots = snapshots_list
        self._all_groups_name = all_groups_name
        self._default_rp = self.DEFAULT_RP
        self.start_date = min([snapshot.date for snapshot in snapshots_list])
        self.end_date = max([snapshot.date for snapshot in snapshots_list])
        self._risk_disc = risk_disc
        self._impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        self._risk_periods_calculators = None

    def _reset_metrics(self):
        for metric in self.POSSIBLE_METRICS:
            setattr(self, "_" + metric + "_metrics", None)

        self._all_risk_metrics = None

    @property
    def default_rp(self) -> list[int]:
        """The default return period values to use when computing risk period metrics.

        Notes
        -----

        Changing its value resets the corresponding metric.
        """
        return self._default_rp

    @default_rp.setter
    def default_rp(self, value):
        if not isinstance(value, list):
            raise ValueError("Return periods need to be a list of int.")
        if any(not isinstance(i, int) for i in value):
            raise ValueError("Return periods need to be a list of int.")
        self._return_periods_metrics = None
        self._all_risk_metrics = None
        self._default_rp = value

    @property
    def risk_disc(self) -> DiscRates | None:
        """The discount rate applied to compute net present values.
        None means no discount rate.

        Notes
        -----

        Changing its value resets the metrics.
        """
        return self._risk_disc

    @risk_disc.setter
    def risk_disc(self, value, /):
        if not isinstance(value, DiscRates):
            raise ValueError("Risk discount needs to be a `DiscRates` object.")

        self._reset_metrics()
        self._risk_disc = value

    @classmethod
    def npv_transform(cls, df: pd.DataFrame, risk_disc: DiscRates) -> pd.DataFrame:
        """Apply discount rate to a metric `DataFrame`.

        Parameters
        ----------
        df : pd.DataFrame
            The `DataFrame` of the metric to discount.
        risk_disc : DiscRate
            The discount rate to apply.

        Returns
        -------
        pd.DataFrame
            The discounted risk metric.


        """

        def _npv_group(group, disc):
            start_date = group.index.get_level_values("date").min()
            return cls._calc_npv_cash_flows(group, start_date, disc)

        df = df.set_index("date")
        grouper = cls._grouper
        if "group" in df.columns:
            grouper = ["group"] + grouper

        df["risk"] = df.groupby(
            grouper,
            dropna=False,
            as_index=False,
            group_keys=False,
            observed=True,
        )["risk"].transform(_npv_group, risk_disc)
        df = df.reset_index()
        return df

    @staticmethod
    def _calc_npv_cash_flows(
        cash_flows: pd.DataFrame,
        start_date: datetime.date,
        disc: DiscRates | None = None,
    ):
        """Apply discount rate to cash flows.

        If it is defined, applies a discount rate `disc` to a given cash flow
        `cash_flows` assuming present year corresponds to `start_date`.

        Parameters
        ----------
        cash_flows : pd.DataFrame
            The cash flow to apply the discount rate to.
        start_date : datetime.date
            The date representing the present.
        end_date : datetime.date, optional
        disc : DiscRates, optional
            The discount rate to apply.

        Returns
        -------

        A dataframe (copy) of `cash_flows` where values are discounted according to `disc`
        """

        if not disc:
            return cash_flows

        if not isinstance(cash_flows.index, pd.DatetimeIndex):
            raise ValueError("cash_flows must be a pandas Series with a datetime index")

        df = cash_flows.to_frame(name="cash_flow")
        df["year"] = df.index.year

        # Merge with the discount rates based on the year
        tmp = df.merge(
            pd.DataFrame({"year": disc.years, "rate": disc.rates}),
            on="year",
            how="left",
        )
        tmp.index = df.index
        df = tmp.copy()
        start = pd.Timestamp(start_date)
        df["discount_factor"] = (1 / (1 + df["rate"])) ** (
            (df.index - start).days // 365
        )

        # Apply the discount factors to the cash flows
        df["npv_cash_flow"] = df["cash_flow"] * df["discount_factor"]

        return df["npv_cash_flow"]
