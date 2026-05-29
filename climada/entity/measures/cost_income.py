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

Define the CostIncome class to handle the cash flow of measures.
"""

from datetime import datetime
from typing import Any, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from climada.entity.measures.measure_config import CostIncomeConfig


class CostIncome:
    """
    Manages costs and incomes related to a measure over time.

    Income are stored a positive numbers and costs as negative
    ones.

    Attributes
    ----------
    mkt_price_year : int, default to today's year.
        The reference year for market prices.
    init_cost : float
        Initial implementation cost (stored as negative).
    periodic_cost : float
        Recurring cost per period (stored as negative).
    periodic_income : float
        Recurring income per period.
    cost_yearly_growth_rate : float
        Yearly growth rate of costs.
    income_yearly_growth_rate : float
        Yearly growth rate of income.
    custom_cash_flows : pd.DataFrame, optional
        User-defined cash flows indexed by date.
    freq : str
        Periodicity (frequency) of the cash flows (e.g., 'Y', '3M', '7D').
        All pandas aliases are possible. See [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases).

    """

    def __init__(
        self,
        *,
        mkt_price_year: Optional[int] = None,
        init_cost: float = 0.0,
        periodic_cost: float = 0.0,
        periodic_income: float = 0.0,
        cost_yearly_growth_rate: float = 0.0,
        income_yearly_growth_rate: float = 0.0,
        custom_cash_flows: Optional[pd.DataFrame] = None,
        freq: str = "Y",
    ):
        """Initialize CostIncome with parameters.

        Parameters
        ----------
        mkt_price_year : int, default to today's year.
            The reference year for market prices.
        init_cost : float
            Initial implementation cost (stored as negative).
        periodic_cost : float
            Recurring cost per period (stored as negative).
        periodic_income : float
            Recurring income per period.
        cost_yearly_growth_rate : float
            Yearly growth rate of costs.
        income_yearly_growth_rate : float
            Yearly growth rate of income.
        custom_cash_flows : pd.DataFrame, optional
            User-defined cash flows indexed by date.
        freq : str
            Frequency of the cash flows (e.g., 'Y', '3M', '7D').
        """

        self.freq = freq
        self.mkt_price_year = datetime(mkt_price_year or datetime.today().year, 1, 1)
        self.cost_growth_rate = cost_yearly_growth_rate

        self.init_cost = -abs(init_cost)
        self.periodic_cost = -abs(periodic_cost)
        self.periodic_income = abs(periodic_income)

        self.income_growth_rate = income_yearly_growth_rate
        self.custom_cash_flows = custom_cash_flows

    def __repr__(self) -> str:
        lines = [
            "CostIncome(",
            f"  mkt_price_year          = {self.mkt_price_year.year}",
            f"  freq                    = {self.freq!r}",
            f"  init_cost               = {self.init_cost:,.2f}",
            f"  periodic_cost           = {self.periodic_cost:,.2f}",
            f"  periodic_income         = {self.periodic_income:,.2f}",
            f"  cost_yearly_growth_rate = {self.cost_growth_rate:.2%}",
            f"  income_yearly_growth_rate = {self.income_growth_rate:.2%}",
            "  custom_cash_flows       = "
            f"{None if self.custom_cash_flows is None else f'DataFrame({len(self.custom_cash_flows)} rows)'}",
            ")",
        ]
        return "\n".join(lines)

    @property
    def custom_cash_flows(self) -> pd.DataFrame | None:
        """:obj:`pd.DataFrame` : Get or set the optional user-defined cash
        flows.

        Input cash flow have to contain a "date" column as well as at least one
        of "cost" and "income". The custom cash flow is coerced to the internal
        period frequency.
        """
        return self._custom_cash_flows

    @custom_cash_flows.setter
    def custom_cash_flows(self, value, /):
        if value is None:
            self._custom_cash_flows = None

        else:
            if not isinstance(value, pd.DataFrame):
                raise ValueError("Custom cash flows only accept pandas DataFrame.")

            self._custom_cash_flows = self._prepare_custom_flows(value)

    def _prepare_custom_flows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and resample custom cash flow dataframe

        Enforce costs as negative numbers and date to the correct frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Custom cashflow

        Returns
        -------
        pd.DataFrame
            Processed custom cashflow
        """

        if "date" not in df.columns:
            raise ValueError("No 'date' column found in custom cash flow DataFrame.")

        if "cost" not in df.columns and "income" not in df.columns:
            raise ValueError(
                "No 'cost' or 'income' column found in custom cash flow DataFrame."
            )

        df = df.copy()
        df["cost"] = -df["cost"].abs()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")

        freq = self._make_offset_compat(self.freq)
        return df.resample(freq).sum()

    @staticmethod
    def _make_offset_compat(freq: str, start=True) -> str:
        suffix = "S" if start else "E"
        match freq:
            case "Y":
                return "Y" + suffix
            case "M":
                return "M" + suffix
            case "Q":
                return "Q" + suffix
            case _:
                return freq

    @classmethod
    def from_config(cls, config: CostIncomeConfig) -> "CostIncome":
        """Create a `CostIncome` from a `CostIncomeConfig`.

        Parameters
        ----------
        config : CostIncomeConfig

        Returns
        -------
        CostIncome
        """

        df = None
        if config.custom_cash_flows is not None:
            df = pd.DataFrame(config.custom_cash_flows)
            df["date"] = pd.to_datetime(df["date"])
        return cls(
            mkt_price_year=config.mkt_price_year,
            init_cost=config.init_cost,
            periodic_cost=config.periodic_cost,
            periodic_income=config.periodic_income,
            cost_yearly_growth_rate=config.cost_yearly_growth_rate,
            income_yearly_growth_rate=config.income_yearly_growth_rate,
            custom_cash_flows=df,
            freq=config.freq,
        )

    @classmethod
    def from_dict(cls, args_dict: dict) -> "CostIncome":
        """Create a `CostIncome` from a dictionary.

        Parameters
        ----------
        args_dict : dict

        Returns
        -------
        CostIncome
        """

        return cls.from_config(
            CostIncomeConfig(
                mkt_price_year=args_dict.get("mkt_price_year"),
                init_cost=args_dict.get("init_cost", 0.0),
                periodic_cost=args_dict.get("periodic_cost", 0.0),
                periodic_income=args_dict.get("periodic_income", 0.0),
                cost_yearly_growth_rate=args_dict.get("cost_yearly_growth_rate", 0.0),
                income_yearly_growth_rate=args_dict.get(
                    "income_yearly_growth_rate", 0.0
                ),
                freq=args_dict.get("freq", "Y"),
                custom_cash_flows=args_dict.get("custom_cash_flows"),
            )
        )

    @classmethod
    def from_yaml(cls, path: str) -> "CostIncome":
        """Create a `CostIncome` from a yaml file of a `MeasureConfig`
        object.

        Parameters
        ----------
        path : str
            Path to the yaml file.

        Returns
        -------
        CostIncome
        """

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f)["cost_income"])

    @classmethod
    def _freq_to_days(cls, freq: str) -> str:
        """
        Convert a frequency string to the equivalent number of days.

        Parameters:
        -----------
        freq : str
            A frequency string (e.g., 'D' for daily, 'M' for monthly, 'Y' for yearly).

        Returns:
        --------
        float
            The equivalent number of days for the given frequency string.
        """

        try:
            # Convert the frequency string to a DateOffset object
            freq = cls._make_offset_compat(freq, start=False)
            offset = pd.tseries.frequencies.to_offset(freq)

            # Calculate the number of days by applying the offset to a base date
            base_date = pd.Timestamp("2000-01-01")
            end_date = base_date + offset

            # Return the difference in days
            return f"{(end_date - base_date).days}d"
        except ValueError as exc:
            raise ValueError(f"Invalid frequency string: {freq}") from exc

    def _get_width_days(self) -> float:
        """Return the number of days in the current frequency."""

        ref = pd.Timestamp("2000-01-01")
        freq = self._make_offset_compat(self.freq, start=False)
        offset = pd.tseries.frequencies.to_offset(freq)
        return float(((ref + offset) - ref).days)

    def calc_at_date(
        self, impl_date: pd.Timestamp, curr_date: pd.Timestamp
    ) -> Tuple[float, float, float]:
        r"""Calculate cash flows for a single timestamp.

        Computes the total cash flow, total cost, and total income for a given
        evaluation date, accounting for growth rates applied to base costs and
        incomes, as well as the custom cash flow if provided.

        The calculation applies compound growth to both costs and incomes based
        on the number of years elapsed since the market price reference date
        (`self.mkt_price_year`).

        Parameters
        ----------
        impl_date : pd.Timestamp
            The implementation date that determines which cost/income regime
            applies. Dates before this have no cost or income; "at the date" uses
            the implementation cost, and dates after use the initialized or
            periodic amounts respectively.
            Note that if a custom cash flow is defined for `curr_date`,
            before `impl_date`, it will be accounted for.
        curr_date : pd.Timestamp
            The evaluation date for which cash flows are being calculated. This
            is compared against `impl_date` to determine the applicable base
            amounts and is also used to index into `custom_cash_flows` if present.

        Returns
        -------
        Tuple[float, float, float]
            A tuple containing:

            * total_cash_flow : float
                Net cash flow for the period, calculated as `total_income + total_cost`.
                Note: Costs are typically negative values in financial contexts,
                so this represents the net position.
            * total_cost : float
                Total cost amount for the period, including both standard and
                custom cost components.
            * total_income : float
                Total income amount for the period, including both standard and
                custom income components.

        Notes
        -----
        Growth calculations use compound interest formula:

        .. math::
            factor = (1 + rate)^{years\_passed}

        where `years_passed` is computed as `(curr_date - mkt_price_year).days / 365.0`.

        Cost and income regimes:

        - **Before impl_date**: Both base cost and income are zero
        - **At impl_date**: Uses `init_cost` for cost
        - **After impl_date**: Uses `periodic_cost` for cost and `periodic_income` for income

        Custom cash flows (if `self.custom_cash_flows` is not None) are added
        on top of the calculated standard amounts. Missing dates in the custom
        cash flow DataFrame will raise a KeyError.
        """
        # Calculate growth factor based on years from market price reference
        years_passed = (curr_date - self.mkt_price_year).days / 365.0

        cost_factor = (1 + self.cost_growth_rate) ** years_passed
        inc_factor = (1 + self.income_growth_rate) ** years_passed

        if curr_date < impl_date:
            cost, income = 0.0, 0.0
        elif curr_date == impl_date:
            cost = self.init_cost * cost_factor
            income = 0.0
        else:
            cost = self.periodic_cost * cost_factor
            income = self.periodic_income * inc_factor

        if (
            self.custom_cash_flows is not None
            and curr_date in self.custom_cash_flows.index
        ):
            c_cost = cast(float, self.custom_cash_flows.loc[curr_date, "cost"])
            c_inc = cast(float, self.custom_cash_flows.loc[curr_date, "income"])
        else:
            c_cost, c_inc = 0.0, 0.0

        total_cost = cost + c_cost
        total_inc = income + c_inc
        return (total_inc + total_cost), total_cost, total_inc

    def calc_cash_flows(
        self, impl_date: pd.Timestamp, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate net cash flows, costs, and incomes over a period.

        Computes cash flow metrics across a specified date range by iterating
        through each period.

        The method creates a period range based on the configured frequency
        (`self.freq`) and evaluates cash flows at the start time of each period.
        Results are returned as NumPy arrays for efficient downstream processing.

        Parameters
        ----------
        impl_date : pd.Timestamp
            The implementation date that determines which cost/income regime
            applies.
        start_date : pd.Timestamp
            The beginning of the calculation period.
        end_date : pd.Timestamp
            The end of the calculation period.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing three NumPy arrays of equal length:

            * net : np.ndarray
                Net cash flow for each period (income + cost).
            * costs : np.ndarray
                Total costs for each period.
            * incomes : np.ndarray
                Total incomes for each period.
        """

        # 'Trick' to make sure that e.g., impl_date "2020-01-05" falls in
        # period "2020-01" if freq is "M"
        impl_ts = pd.Timestamp(str(impl_date)).to_period(self.freq).start_time
        periods = pd.period_range(start=start_date, end=end_date, freq=self.freq)

        results = [self.calc_at_date(impl_ts, p.start_time) for p in periods]
        net, costs, incs = map(np.array, zip(*results))
        return net, costs, incs

    def calc_total(self, impl_date, start_date, end_date) -> Tuple[float, float, float]:
        """
        Calculate the total value of the cash flows over a given period.

        Parameters:
        -----------
        impl_date:
            The date the measure is implemented.
        start_year:
            The start date of the period.
        end_year: int
            The end year of the period.

        Returns:
        --------
        Tuple[float, float, float]
            the total net, cost, and income values over the given period.
        """

        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_date, start_date, end_date
        )
        return np.sum(net_cash_flows), np.sum(costs), np.sum(incomes)

    def plot_cash_flows(
        self,
        impl_date: Any,
        start_date: Any,
        end_date: Any,
        figsize: Tuple[int, int] = (12, 7),
        title: Optional[str] = None,
    ):
        """Plot periodic and cumulative cash flows over a given period.

        Displays a two-panel figure:
        - Top panel: stacked bar chart of costs and incomes per period,
          with a net cash flow line overlay.
        - Bottom panel: cumulative net cash flow over time.

        Parameters
        ----------
        impl_date :
            The date the measure is implemented.
        start_date :
            Start of the evaluation period.
        end_date :
            End of the evaluation period.
        figsize : tuple, optional
            Figure size as (width, height). Default is (12, 7).
        title : str, optional
            Overall figure title. Defaults to 'Cash Flow Analysis'.

        Returns
        -------
        tuple(plt.Axis, plt.Axis)
        """
        net, costs, incomes = self.calc_cash_flows(impl_date, start_date, end_date)
        periods = pd.period_range(start=start_date, end=end_date, freq=self.freq)
        dates = [p.start_time for p in periods]
        cumulative_net = np.cumsum(net)

        width = pd.Timedelta(days=self._get_width_days() * 0.6)

        fig, (ax_bar, ax_cum) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        # --- top panel: stacked bars + net line ---
        ax_bar.bar(
            dates, incomes, width=width, color="#4C9BE8", label="Income", zorder=2
        )
        ax_bar.bar(dates, costs, width=width, color="#E8604C", label="Cost", zorder=2)
        ax_bar.plot(
            dates,
            net,
            color="black",
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Net",
            zorder=3,
        )
        ax_bar.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)

        ax_bar.set_ylabel("Cash flow")
        ax_bar.legend(frameon=False, fontsize=9)
        ax_bar.spines[["top", "right"]].set_visible(False)
        ax_bar.tick_params(axis="x", which="both", bottom=False)
        ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # --- bottom panel: cumulative net ---
        # dates = dates.append()
        positive = np.maximum(cumulative_net, 0)
        negative = np.minimum(cumulative_net, 0)
        ax_cum.fill_between(dates, positive, alpha=0.25, color="#4C9BE8", step="mid")
        ax_cum.fill_between(dates, negative, alpha=0.25, color="#E8604C", step="mid")
        # ax_cum.plot(dates, cumulative_net, color="black", linewidth=1.5, zorder=3)
        ax_cum.axhline(0, color="black", linewidth=0.6, linestyle="--", zorder=1)

        ax_cum.set_ylabel("Cumulative net")
        ax_cum.spines[["top", "right"]].set_visible(False)
        ax_cum.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.autofmt_xdate(rotation=30, ha="right")
        fig.suptitle(title or "Cash Flow Analysis", fontsize=13, y=1.01)
        return (ax_bar, ax_cum)

    def to_dataframe(self, impl_date, start_date, end_date) -> pd.DataFrame:
        """Return cash flows as a formatted DataFrame."""
        net, costs, incs = self.calc_cash_flows(impl_date, start_date, end_date)
        periods = pd.period_range(start=start_date, end=end_date, freq=self.freq)

        return pd.DataFrame(
            {"date": periods, "net": net, "cost": costs, "income": incs}
        )

    @staticmethod
    def comb_cost_income(cost_incomes: list["CostIncome"]) -> "CostIncome":
        """Combine multiple CostIncomes together.

        Combination sums the costs and incomes from all provided CostIncome
        objects.
        """

        first_ci = cost_incomes[0]

        if not all(
            (
                first_ci.mkt_price_year.year == c.mkt_price_year.year
                for c in cost_incomes
            )
        ):
            raise ValueError(
                "Measure cost incomes have different market price years,"
                " combination is not possible."
            )

        if not all(
            first_ci.cost_growth_rate == c.cost_growth_rate for c in cost_incomes
        ):
            raise ValueError(
                "Measure cost incomes have different cost_growth_rate,"
                " combination is not possible."
            )

        if not all(
            first_ci.income_growth_rate == c.income_growth_rate for c in cost_incomes
        ):
            raise ValueError(
                "Measure cost incomes have different income_growth_rate,"
                " combination is not possible."
            )

        if not all(first_ci.freq == c.freq for c in cost_incomes):
            raise ValueError(
                "Measure cost incomes have different period frequencies,"
                " combination is not possible."
            )

        try:
            custom_cash_flows = cast(
                pd.DataFrame,
                pd.concat([c.custom_cash_flows for c in cost_incomes])  # type: ignore
                .groupby(level=0)
                .sum()
                .reset_index(),
            )
        except ValueError as err:
            if str(err) == "All objects passed were None":
                custom_cash_flows = None
            else:
                raise err

        return CostIncome(
            mkt_price_year=first_ci.mkt_price_year.year,
            init_cost=sum(c.init_cost for c in cost_incomes),
            periodic_cost=sum(c.periodic_cost for c in cost_incomes),
            periodic_income=sum(c.periodic_income for c in cost_incomes),
            cost_yearly_growth_rate=first_ci.cost_growth_rate,
            income_yearly_growth_rate=first_ci.income_growth_rate,
            freq=first_ci.freq,
            custom_cash_flows=custom_cash_flows,
        )
