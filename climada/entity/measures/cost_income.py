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
from typing import Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_config import CostIncomeConfig


class CostIncome:
    """
    Manages costs and incomes related to a measure over time.

    Income are stored a positive numbers and costs as negative
    ones.

    Attributes
    ----------
    mkt_price_year : datetime, default to today's year.
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
        mkt_price_year : datetime, default to today's year.
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

        if custom_cash_flows is not None:
            self.custom_cash_flows = self._prepare_custom_flows(custom_cash_flows)
        else:
            self.custom_cash_flows = None

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

        df = df.copy()
        if "cost" in df.columns:
            df["cost"] = -df["cost"].abs()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        match self.freq:
            case "Y":
                resampling_freq = "YS"
            case "M":
                resampling_freq = "MS"
            case "Q":
                resampling_freq = "QS"
            case _:
                resampling_freq = self.freq

        return df.resample(resampling_freq).sum()

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
        """Create a `CostIncome` from a yaml file.

        Parameters
        ----------
        path : str
            Path to the yaml file.

        Returns
        -------
        CostIncome
        """

        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f)["cost_income"])

    @staticmethod
    def _freq_to_days(freq: str) -> str:
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
            offset = pd.tseries.frequencies.to_offset(freq)

            # Calculate the number of days by applying the offset to a base date
            base_date = pd.Timestamp("2000-01-01")
            end_date = base_date + offset

            # Return the difference in days
            return f"{(end_date - base_date).days}d"
        except ValueError:
            raise ValueError(f"Invalid frequency string: {freq}")

    def _get_width_days(self) -> float:
        """Return the number of days in the current frequency."""

        ref = pd.Timestamp("2000-01-01")
        offset = pd.tseries.frequencies.to_offset(self.freq)
        return float(((ref + offset) - ref).days)

    def _calc_at_date(
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
            applies. Dates before this use have no cost or income; "at the date" uses
            the implementation cost, and dates after use the initialized or
            periodic amounts respectively.
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
            income = self.periodic_income * inc_factor
        else:
            cost = self.periodic_cost * cost_factor
            income = self.periodic_income * inc_factor

        if self.custom_cash_flows is not None:
            c_cost = cast(float, self.custom_cash_flows.loc[curr_date, "cost"])
            c_inc = cast(float, self.custom_cash_flows.loc[curr_date, "income"])
        else:
            c_cost, c_inc = 0.0, 0.0

        total_cost = cost + c_cost
        total_inc = income + c_inc
        return (total_inc + total_cost), total_cost, total_inc

    def calc_cash_flows(
        self, impl_date, start_date, end_date, disc_rates: Optional[DiscRates] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate net cash flows, costs, and incomes over a period.

        Computes cash flow metrics across a specified date range by iterating
        through each period. Optionally
        applies discounting to calculate Net Present Value (NPV) when discount
        rates are provided.

        The method creates a period range based on the configured frequency
        (`self.freq`) and evaluates cash flows at the start time of each period.
        Results are returned as NumPy arrays for efficient downstream processing.

        Parameters
        ----------
        impl_date :
            The implementation date that determines which cost/income regime
            applies.
        start_date :
            The beginning of the calculation period.
        end_date :
            The end of the calculation period.
        disc_rates : DiscRates, optional
            An object containing discount rate information for NPV calculations.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple containing three NumPy arrays of equal length:

            * net : np.ndarray
                Net cash flow for each period (income + cost). May be discounted
                if `disc_rates` is provided.
            * costs : np.ndarray
                Total costs for each period. May be discounted if `disc_rates`
                is provided.
            * incomes : np.ndarray
                Total incomes for each period. May be discounted if `disc_rates`
                is provided.
        """

        impl_ts = pd.Timestamp(impl_date)
        periods = pd.period_range(start=start_date, end=end_date, freq=self.freq)

        results = [self._calc_at_date(impl_ts, p.start_time) for p in periods]
        net, costs, incs = map(np.array, zip(*results))

        if disc_rates is not None:
            # Vectorized discounting
            years = np.array([p.year for p in periods])
            # Note: Assumes disc.rates is indexed/aligned with disc.years
            rate_map = dict(zip(disc_rates.years, disc_rates.rates))
            factors = np.array(
                [
                    1
                    / (1 + rate_map.get(yr, 0.0))
                    ** (yr - pd.Timestamp(start_date).year)
                    for yr in years
                ]
            )
            # Handle potential length mismatch if years aren't in disc
            valid = np.array([yr in rate_map for yr in years])
            return (
                net[valid] * factors[valid],
                costs[valid] * factors[valid],
                incs[valid] * factors[valid],
            )

        return net, costs, incs

    def calc_total(
        self, impl_date, start_date, end_date, disc: Optional[DiscRates] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate the total or net present value of the cash flows over a given period.

        Parameters:
        -----------
        impl_date:
            The date the measure is implemented.
        start_year:
            The start date of the period.
        end_year: int
            The end year of the period.
        disc: DiscRates, optional
            The discount rates to apply.

        Returns:
        --------
        Tuple[float, float, float]
            the total net, cost, and income present values over the given period
        """

        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_date, start_date, end_date, disc_rates=disc
        )
        return np.sum(net_cash_flows), np.sum(costs), np.sum(incomes)

    def plot_cash_flows(
        self,
        impl_date: Any,
        start_date: Any,
        end_date: Any,
        disc: Optional[Any] = None,
        to_plot: List[str] = ["net", "cost", "income"],
    ):
        """
        Plot the cash flows over a given period.

        Parameters:
        -----------
        impl_date: datetime
            The date the measure is implemented.
        start_date: datetime
            The start date of the period.
        end_date: datetime
            The end date of the period.
        disc: DiscRates object
            The discount rates (optional).
        to_plot: list
            List of strings indicating which cash flows to plot. Options are 'net', 'cost', 'income'.
        """

        # Calculate the cash flows over the given period
        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_date, start_date, end_date, disc_rates=disc
        )

        # Plot the cash flows with colors
        date_range = pd.date_range(
            start=start_date, end=end_date, freq=self._freq_to_days(self.freq)
        )
        width = self._get_width_days() * 0.8  # 80% width for visibility
        fig, ax = plt.subplots()

        if "cost" in to_plot:
            ax.bar(date_range, costs, color="red", label="Cost", width=width)
        if "income" in to_plot:
            ax.bar(
                date_range,
                incomes,
                color="blue",
                label="Income",
                alpha=0.7,
                width=width,
            )
        if "net" in to_plot:
            ax.bar(
                date_range,
                net_cash_flows,
                color="blue",
                edgecolor="red",
                hatch="//",
                label="Net",
                alpha=0.5,
                width=width,
            )

        ax.xaxis_date()  # <---- treat x-ticks as datetime
        fig.autofmt_xdate()
        plt.xlabel("Date")
        plt.ylabel("Cash Flow")
        plt.title("Discounted Cash Flows" if disc else "Cash Flows")
        plt.legend()
        plt.show()
        return ax

    def to_dataframe(
        self, impl_date, start_date, end_date, disc_rates: Optional[DiscRates] = None
    ) -> pd.DataFrame:
        """Return cash flows as a formatted DataFrame."""
        net, costs, incs = self.calc_cash_flows(
            impl_date, start_date, end_date, disc_rates=disc_rates
        )
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
            [
                first_ci.mkt_price_year.year == c.mkt_price_year.year
                for c in cost_incomes
            ]
        ):
            raise ValueError(
                "Measure cost incomes have different market price years, combination is not possible."
            )

        if not all(
            [first_ci.cost_growth_rate == c.cost_growth_rate for c in cost_incomes]
        ):
            raise ValueError(
                "Measure cost incomes have different cost_growth_rate, combination is not possible."
            )

        if not all(
            [first_ci.income_growth_rate == c.income_growth_rate for c in cost_incomes]
        ):
            raise ValueError(
                "Measure cost incomes have different income_growth_rate, combination is not possible."
            )

        return CostIncome(
            mkt_price_year=first_ci.mkt_price_year.year,
            cost_yearly_growth_rate=first_ci.cost_growth_rate,
            init_cost=sum(c.init_cost for c in cost_incomes),
            periodic_cost=sum(c.periodic_cost for c in cost_incomes),
            periodic_income=sum(c.periodic_income for c in cost_incomes),
            income_yearly_growth_rate=first_ci.income_growth_rate,
        )
