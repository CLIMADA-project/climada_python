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


from datetime import datetime
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CostIncome:

    def __init__(
        self,
        mkt_price_year: float = datetime.today().year,  # Get the year of today as integer
        cost_growth_rate: float = 0.0,
        init_cost: float = 0.0,
        annual_cost: float = 0.0,
        annual_income: float = 0.0,
        income_growth_rate: float = 0.0,
        custom_cash_flows: Optional[pd.DataFrame] = None,
    ):

        self.mkt_price_year = mkt_price_year
        # Cost parameters
        self.cost_growth_rate = cost_growth_rate
        self.init_cost = -abs(init_cost)
        self.annual_cost = -abs(annual_cost)
        # Income parameters
        self.annual_income = annual_income
        self.income_growth_rate = income_growth_rate
        # Custom cash flows
        # Update the cost columns to be negative
        if custom_cash_flows is not None and "cost" in custom_cash_flows.columns:
            custom_cash_flows["cost"] = -abs(custom_cash_flows["cost"])
        self.custom_cash_flows = custom_cash_flows

        # Custom cash flows
        if self.custom_cash_flows is not None:
            self.custom_cash_flows = self.custom_cash_flows.groupby("year").sum()
            if (
                "cost" not in self.custom_cash_flows.columns
                and "income" not in self.custom_cash_flows.columns
            ):
                raise ValueError(
                    "Custom cash flows DataFrame must contain 'cost' or 'income' column"
                )

    def _calc_cash_flow_at_year(self, impl_year, year):
        """
        Calculate the net cash flow at a specific year

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented
        year: int
            the year to calculate the cash flow

        Returns:
        --------
        Tuple[float, float, float]
            the net cash flow, cost, and income at the given year
        """

        # Calculate the increased cost and income multipliers
        cost_incr_multi = (1 + self.cost_growth_rate) ** (year - self.mkt_price_year)
        income_incr_multi = (1 + self.income_growth_rate) ** (
            year - self.mkt_price_year
        )

        # Calculate the cash flows at the given year
        if year < impl_year:
            cost = 0
            income = 0
        elif year == impl_year:
            cost = self.init_cost * cost_incr_multi
            income = self.annual_income * income_incr_multi
        else:
            cost = self.annual_cost * cost_incr_multi
            income = self.annual_income * income_incr_multi

        # Custom cash flows
        custom_cost = self._get_custom_cash_flow(year, "cost")
        custom_income = self._get_custom_cash_flow(year, "income")

        # Net cash flow
        net_cash_flow = custom_income + income + custom_cost + cost

        return net_cash_flow, custom_cost + cost, custom_income + income

    def _get_custom_cash_flow(self, year, column):
        if self.custom_cash_flows is not None:
            if year in self.custom_cash_flows.index:
                if column in self.custom_cash_flows.columns:
                    return self.custom_cash_flows.loc[year, column]
                else:
                    raise ValueError(
                        f"Column '{column}' not found in custom cash flows DataFrame"
                    )
        return 0

    def calc_cash_flows(self, impl_year, start_year, end_year, disc=None):
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

        net_cash_flows = []
        costs = []
        incomes = []

        # Calculate the cash flows for each year
        for year in range(start_year, end_year + 1):
            net_cash_flow, cost, income = self._calc_cash_flow_at_year(impl_year, year)
            net_cash_flows.append(net_cash_flow)
            costs.append(cost)
            incomes.append(income)

        net_cash_flows = np.array(net_cash_flows)
        costs = np.array(costs)
        incomes = np.array(incomes)

        # Discount the cash flows if discount rates are provided
        if disc:
            # Get the discount factors for the years in the period
            years = np.array(list(range(start_year, end_year + 1)))
            disc_years = np.intersect1d(years, disc.years)
            disc_rates = disc.rates[np.isin(disc.years, disc_years)]
            years = np.array([year for year in years if year in disc_years])
            discount_factors = np.array(
                [
                    1 / (1 + disc_rates[disc_years == year][0]) ** (year - start_year)
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
            impl_year, start_year, end_year, disc=None
        )
        return np.sum(net_cash_flows), np.sum(costs), np.sum(incomes)

    def plot_cash_flows(
        self,
        impl_year,
        start_year,
        end_year,
        disc=None,
        to_plot=["net", "cost", "income"],
    ):
        """
        Plot the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        disc: DiscRates object
            the discount rates (optional)

        to_plot: list
            list of strings indicating which cash flows to plot. Options are 'net', 'cost', 'income'.
        """

        # Calculate the cash flows over the given period
        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_year, start_year, end_year, disc=disc
        )

        # Plot the cash flows with colors
        plt.figure()
        years = range(start_year, end_year + 1)
        if "cost" in to_plot:
            plt.bar(years, costs, color="red", label="Cost")
        if "income" in to_plot:
            plt.bar(years, incomes, color="blue", label="Income", alpha=0.7)
        if "net" in to_plot:
            plt.bar(
                years,
                net_cash_flows,
                color="blue",
                edgecolor="red",
                hatch="//",
                label="Net",
                alpha=0.5,
            )
        plt.xlabel("Year")
        plt.ylabel("Cash Flow [CHF]")
        plt.title("Discounted Cash Flows" if disc else "Cash Flows")
        plt.legend()
        plt.show()

    def calc_cashflows(self, impl_year, start_year, end_year, disc=None):
        """
        Calculate the cash flows over a given period and return them as a DataFrame

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented
        start_year: int
            the start year of the period
        end_year: int
            the end year of the period
        disc: DiscRates object
            the discount rates (optional)

        Returns:
        --------
        cash_flows: pd.DataFrame
            the cash flows over the given period
        """

        # Make a DataFrame to store the cash flows
        cash_flows = pd.DataFrame(columns=["year", "net", "cost", "income"])

        # Calculate the cash flows for each year
        net_cash_flows, costs, incomes = self.calc_cash_flows(
            impl_year, start_year, end_year, disc=disc
        )

        # Add the cash flows to the DataFrame
        cash_flows["year"] = range(start_year, end_year + 1)
        cash_flows["net"] = net_cash_flows
        cash_flows["cost"] = costs
        cash_flows["income"] = incomes

        return cash_flows
