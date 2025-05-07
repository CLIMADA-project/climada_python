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

Define the Cash Flows class.
"""

# Define default discount rates
# DISC_RATES = DiscRates(years=np.arange(1900, 2100), rates=np.ones(np.arange(1900, 2100).size))

from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CostIncome:
    def __init__(
        self,
        *,
        mkt_price_year: int = datetime.today().year,
        init_cost: float = 0.0,
        periodic_cost: float = 0.0,
        periodic_income: float = 0.0,
        cost_yearly_growth_rate: float = 0.0,
        income_yearly_growth_rate: float = 0.0,
        custom_cash_flows: Optional[pd.DataFrame] = None,
        freq: str = "YS",
    ):
        self.freq = freq  # CostIncome._freq_to_days(freq)
        self.mkt_price_year = datetime(mkt_price_year, 1, 1)
        self.cost_growth_rate = cost_yearly_growth_rate
        self.init_cost = -abs(init_cost)
        self.periodic_cost = -abs(periodic_cost)
        self.periodic_income = periodic_income
        self.income_growth_rate = income_yearly_growth_rate

        if custom_cash_flows is not None and "cost" in custom_cash_flows.columns:
            custom_cash_flows["cost"] = -abs(custom_cash_flows["cost"])
            custom_cash_flows["date"] = pd.to_datetime(custom_cash_flows["date"])
            custom_cash_flows = (
                custom_cash_flows.set_index("date").resample(self.freq).sum()
            )

        self.custom_cash_flows = custom_cash_flows

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

    def _get_custom_cash_flow(self, date, column):
        return self.custom_cash_flows.loc[date, column]

    def _calc_cash_flow_at_date(self, impl_date, current_date):
        delta = (current_date - self.mkt_price_year) / pd.Timedelta("365d")

        cost_incr = (1 + self.cost_growth_rate) ** delta
        income_incr = (1 + self.income_growth_rate) ** delta

        if current_date < impl_date:
            cost = 0
            income = 0
        elif current_date == impl_date:
            cost = self.init_cost * cost_incr
            income = self.periodic_income * income_incr
        else:
            cost = self.periodic_cost * cost_incr
            income = self.periodic_income * income_incr

        custom_cost = (
            self._get_custom_cash_flow(current_date, "cost")
            if self.custom_cash_flows is not None
            else 0
        )
        custom_income = (
            self._get_custom_cash_flow(current_date, "income")
            if self.custom_cash_flows is not None
            else 0
        )
        net = custom_income + income + custom_cost + cost

        return net, custom_cost + cost, custom_income + income

    def calc_cash_flows(self, impl_date, start_date, end_date, disc=None):
        """
        Calculate the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        disc: DiscRates object
            the discount rates (required if discounted is True)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            the net cash flows, costs, and incomes over the given period
        """
        impl_date = pd.Timestamp(impl_date)
        dates = pd.date_range(start=start_date, end=end_date, freq=self.freq)

        net_list, cost_list, income_list = [], [], []

        for d in dates:
            net, cost, income = self._calc_cash_flow_at_date(impl_date, d)
            net_list.append(net)
            cost_list.append(cost)
            income_list.append(income)

        net_cash_flows = np.array(net_list)
        costs = np.array(cost_list)
        incomes = np.array(income_list)

        if disc:
            # Get the discount factors for the dates in the period
            years = np.array([d.year for d in dates])
            disc_years = np.intersect1d(years, disc.years)
            disc_rates = disc.rates[np.isin(disc.years, disc_years)]
            years = np.array([year for year in years if year in disc_years])
            discount_factors = np.array(
                [
                    1
                    / (1 + disc_rates[disc_years == year][0])
                    ** (year - start_date.year)
                    for year in years
                ]
            )
            return (
                net_cash_flows[: len(years)] * discount_factors,
                costs[: len(years)] * discount_factors,
                incomes[: len(years)] * discount_factors,
            )

        return net_cash_flows, costs, incomes

    def calc_total(self, impl_year, start_year, end_year, disc=None):
        """
        Calculate the total or net present value of the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        disc: DiscRates object
            the discount rates (required if discounted is True)

        Returns:
        --------
        Tuple[float, float, float]
            the total net, cost, and income present values over the given period
        """

        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_year, start_year, end_year, disc=disc
        )
        return np.sum(net_cash_flows), np.sum(costs), np.sum(incomes)

    def plot_cash_flows(
        self,
        impl_date: date,
        start_date: date,
        end_date: date,
        disc=None,
        to_plot=["net", "cost", "income"],
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
        offset: timedelta or str
            The offset for the period (e.g., timedelta(days=1) or 'M' for month).
        disc: DiscRates object
            The discount rates (optional).
        to_plot: list
            List of strings indicating which cash flows to plot. Options are 'net', 'cost', 'income'.
        """
        # Calculate the cash flows over the given period
        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_date, start_date, end_date, disc=disc
        )

        # Plot the cash flows with colors
        fig, ax = plt.subplots()
        date_range = pd.date_range(
            start=start_date, end=end_date, freq=self._freq_to_days(self.freq)
        )
        width = pd.tseries.frequencies.to_offset(self.freq).delta.days

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
        plt.ylabel("Cash Flow [CHF]")
        plt.title("Discounted Cash Flows" if disc else "Cash Flows")
        plt.legend()
        plt.show()
        return ax

    def calc_cashflows(
        self, impl_date: datetime, start_date: datetime, end_date: datetime, disc=None
    ) -> pd.DataFrame:
        """
        Calculate the cash flows over a given period and return them as a DataFrame.

        Parameters:
        -----------
        impl_date: datetime
            The date the measure is implemented.
        start_date: datetime
            The start date of the period.
        end_date: datetime
            The end date of the period.
        offset: timedelta or str
            The offset for the period (e.g., timedelta(days=1) or 'M' for month).
        disc: DiscRates object
            The discount rates (optional).

        Returns:
        --------
        cash_flows: pd.DataFrame
            The cash flows over the given period.
        """
        # Make a DataFrame to store the cash flows
        cash_flows = pd.DataFrame(columns=["date", "net", "cost", "income"])

        # Calculate the cash flows for each date
        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_date, start_date, end_date, disc=disc
        )

        # Add the cash flows to the DataFrame
        date_range = pd.date_range(start=start_date, end=end_date, freq=self.freq)
        cash_flows["date"] = date_range
        cash_flows["net"] = net_cash_flows
        cash_flows["cost"] = costs
        cash_flows["income"] = incomes

        return cash_flows
