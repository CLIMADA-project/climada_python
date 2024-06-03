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

# # Example usage
# import numpy as np
# from climada.entity import DiscRates

# # Define custom costs and incomes
# custom_cash_flows = pd.DataFrame({
#     'year': [2021, 2021, 2023, 2023],
#     'Cost': [200, 300, 150, 150],
#     'Income': [600, 700, 800, 900]
# })

# cost = Cost(mkt_price_year=2020, initial_cash_flow=1000, yearly_cash_flow=100, annual_growth_rate=0.02, custom_cash_flows_df=custom_cash_flows)
# income = Income(mkt_price_year=2020, initial_cash_flow=0, yearly_cash_flow=500, annual_growth_rate=0.02, custom_cash_flows_df=custom_cash_flows)

# # Define discount rates
# years = np.arange(1950, 2100)
# rates = np.ones(years.size) * 0.014
# rates[51:55] = 0.025
# rates[95:120] = 0.035
# disc = DiscRates(years=years, rates=rates)

# # Create an instance of CostIncomeAnalysis
# analysis = CostIncomeAnalysis(cost, income)

# # Plot the cash flows
# analysis.plot_cost_and_income(impl_year=2020, start_year=2020, end_year=2040, discounted=False, disc=disc)

# # Calculate the total net present value
# total_npv = analysis.calc_total(impl_year=2020, start_year=2020, end_year=2040, discounted=False, disc=disc)
# print("Total Net Present Value:", total_npv)


from typing import Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class CashFlow(ABC):

    def __init__(self,
                 mkt_price_year: float,  # year the market prices are given 
                 initial_cash_flow: float = 0.0,  # initial cash flow (e.g., implementation cost)
                 yearly_cash_flow: float = 0.0,  # yearly cash flow (e.g., maintenance cost or yearly income)
                 annual_growth_rate: Optional[float] = 0.0,  # annual growth rate of the cash flow
                 custom_cash_flows_df: Optional[pd.DataFrame] = None  # custom cash flows per year
                 ):

        self.annual_growth_rate = annual_growth_rate
        self.mkt_price_year = mkt_price_year
        self.initial_cash_flow = initial_cash_flow
        self.yearly_cash_flow = yearly_cash_flow
        self.custom_cash_flows_df = custom_cash_flows_df

        if self.custom_cash_flows_df is not None:
            self.custom_cash_flows_df = self.custom_cash_flows_df.groupby('year').sum()
            if 'Cost' not in self.custom_cash_flows_df.columns and 'Income' not in self.custom_cash_flows_df.columns:
                raise ValueError("Custom cash flows DataFrame must contain 'Cost' or 'Income' column")

    # Abstract method to calculate the cash flow at a given year
    @abstractmethod
    def _calc_cash_flow_at_year(self, impl_year, year):
        pass

    # Method to get custom cash flow for a given year
    def _get_custom_cash_flow(self, year, column):
        if self.custom_cash_flows_df is not None:
            if year in self.custom_cash_flows_df.index:
                if column in self.custom_cash_flows_df.columns:
                    return self.custom_cash_flows_df.loc[year, column]
                else:
                    raise ValueError(f"Column '{column}' not found in custom cash flows DataFrame")
        return 0

    # Calculate the cash flows over a given period
    def calc_cash_flows(self, impl_year, start_year, end_year, discounted=False, disc=None):
        '''
        Calculate the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        discounted: bool
            whether to return discounted cash flows

        disc: DiscRates object
            the discount rates (required if discounted is True)

        Returns:
        --------
        cash_flows: np.array
            the cash flows over the given period
        '''

        cash_flows = []
        column = 'Cost' if isinstance(self, Cost) else 'Income'
        for year in range(start_year, end_year + 1):
            base_cash_flow = self._calc_cash_flow_at_year(impl_year, year)
            custom_cash_flow = self._get_custom_cash_flow(year, column)
            cash_flows.append(base_cash_flow + custom_cash_flow)

        cash_flows = np.array(cash_flows)

        if discounted:
            if disc is None:
                raise ValueError("Discount rates must be provided if discounted is True.")
            years = np.array(list(range(start_year, end_year + 1)))
            disc_years = np.intersect1d(years, disc.years)
            disc_rates = disc.rates[np.isin(disc.years, disc_years)]
            years = np.array([year for year in years if year in disc_years])
            discount_factors = np.array([1 / (1 + disc_rates[disc_years == year][0])**(year - impl_year) for year in years])
            return cash_flows[:len(years)] * discount_factors

        return cash_flows

    # Calculate the total or net present value of the cash flows over a given period
    def calc_total(self, impl_year, start_year, end_year, disc=None, discounted=False):
        '''
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

        discounted: bool
            whether to calculate the total discounted value

        Returns:
        --------
        total: float
            the total or net present value of the cash flows over the given period
        '''

        cash_flows = self.calc_cash_flows(impl_year, start_year, end_year, discounted=discounted, disc=disc)

        return np.sum(cash_flows)

    # Plot the cash flows over a given period
    def plot_cash_flows(self, impl_year, start_year, end_year, discounted=False, disc=None):
        '''
        Plot the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        discounted: bool
            whether to plot discounted cash flows

        disc: DiscRates object
            the discount rates (required if discounted is True)
        '''

        # Calculate the cash flows over the given period
        cash_flows = self.calc_cash_flows(impl_year, start_year, end_year, discounted=discounted, disc=disc)

        # Adjust sign for costs
        if isinstance(self, Cost):
            cash_flows = -cash_flows

        # Plot the cash flows with colors
        plt.figure()
        color = 'red' if isinstance(self, Cost) else 'blue'
        plt.bar(range(start_year, end_year + 1), cash_flows, color=color)
        plt.xlabel('Year')
        plt.ylabel('Cash Flow')
        plt.title('Discounted Cash Flows' if discounted else 'Cash Flows')
        plt.legend(['Discounted Cash Flow' if discounted else 'Cash Flow'])
        plt.show()

class Cost(CashFlow):

    def __init__(self, mkt_price_year: float, 
                 initial_cash_flow: float = 0.0, 
                 yearly_cash_flow: float = 0.0, 
                 annual_growth_rate: float = 0.0, 
                 custom_cash_flows_df: pd.DataFrame = None):
        
        super().__init__(mkt_price_year, initial_cash_flow=initial_cash_flow, yearly_cash_flow=yearly_cash_flow, annual_growth_rate=annual_growth_rate, custom_cash_flows_df=custom_cash_flows_df)

    # Calculate the cash flow at a given year
    def _calc_cash_flow_at_year(self, impl_year, year):
        '''
        Calculate the cash flow of the measure at a specific year

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented
        year: int
            the year to calculate the cash flow of the measure (being implemented or maintenance)

        Returns:
        --------
        cash_flow: float
            the cash flow of the measure at the given year
        '''

        # Calculate the increased cost multiplier
        cost_incr_multi = (1 + self.annual_growth_rate)**(year - self.mkt_price_year)

        # Calculate the cash flow of the measure at the given year
        if year < impl_year:
            cash_flow = 0
        elif year == impl_year:
            cash_flow = self.initial_cash_flow * cost_incr_multi
        else:
            cash_flow = self.yearly_cash_flow * cost_incr_multi

        return cash_flow

class Income(CashFlow):

    def __init__(self, mkt_price_year: float, 
                 initial_cash_flow: float = 0.0, 
                 yearly_cash_flow: float = 0.0, 
                 annual_growth_rate: float = 0.0, 
                 custom_cash_flows_df: pd.DataFrame = None):
        
        super().__init__(mkt_price_year, initial_cash_flow=initial_cash_flow, yearly_cash_flow=yearly_cash_flow, annual_growth_rate=annual_growth_rate, custom_cash_flows_df=custom_cash_flows_df)

    # Calculate the cash flow at a given year
    def _calc_cash_flow_at_year(self, impl_year, year):
        '''
        Calculate the cash flow at a specific year

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented
        year: int
            the year to calculate the cash flow

        Returns:
        --------
        cash_flow: float
            the cash flow at the given year
        '''

        # Calculate the increased gain multiplier
        gain_incr_multi = (1 + self.annual_growth_rate)**(year - self.mkt_price_year)

        # Calculate the cash flow at the given year
        if year < impl_year:
            cash_flow = 0
        else:
            cash_flow = self.yearly_cash_flow * gain_incr_multi

        return cash_flow


class CostIncomeAnalysis:

    def __init__(self, cost: Optional[Cost] = None, 
                 income: Optional[Income] = None):
        self.cost = cost
        self.income = income

    def calc_cost_income(self, impl_year: int, start_year: int, end_year: int, discounted=False, disc=None):
        '''
        Calculate the cost, income, and net cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        discounted: bool
            whether to calculate discounted cash flows

        disc: DiscRates object
            the discount rates (required if discounted is True)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            the cost, income, and net cash flows over the given period
        '''

        # Calculate the cash flows for cost and income
        cost_cash_flows = self.cost.calc_cash_flows(impl_year, start_year, end_year, discounted=discounted, disc=disc) if self.cost else np.zeros(end_year - start_year + 1)
        income_cash_flows = self.income.calc_cash_flows(impl_year, start_year, end_year, discounted=discounted, disc=disc) if self.income else np.zeros(end_year - start_year + 1)

        # Adjust sign for costs
        cost_cash_flows = -cost_cash_flows

        # Calculate net cash flows
        net_cash_flows = income_cash_flows + cost_cash_flows

        return cost_cash_flows, income_cash_flows, net_cash_flows

    def calc_total(self, impl_year: int, start_year: int, end_year: int, discounted=False, disc=None):
        '''
        Calculate the total or net present value of the cash flows over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        discounted: bool
            whether to calculate the total discounted value

        disc: DiscRates object
            the discount rates (required if discounted is True)

        Returns:
        --------
        total: float
            the total or net present value of the cash flows over the given period
        '''

        _, _, net_cash_flows = self.calc_cost_income(impl_year, start_year, end_year, discounted=discounted, disc=disc)
        return np.sum(net_cash_flows)

    def plot_cost_and_income(self, impl_year: int, start_year: int, end_year: int, discounted=False, disc=None):
        '''
        Plot the cash flows of both cost and income objects over a given period

        Parameters:
        -----------
        impl_year: int
            the year the measure is implemented

        start_year: int
            the start year of the period

        end_year: int
            the end year of the period

        discounted: bool
            whether to plot discounted cash flows

        disc: DiscRates object
            the discount rates (required if discounted is True)
        '''

        # Calculate the cost, income, and net cash flows
        cost_cash_flows, income_cash_flows, net_cash_flows = self.calc_cost_income(impl_year, start_year, end_year, discounted, disc)

        # Plot the cash flows with colors
        plt.figure()
        if self.cost:
            plt.bar(range(start_year, end_year + 1), cost_cash_flows, color='red', label='Cost')
        if self.income:
            plt.bar(range(start_year, end_year + 1), income_cash_flows, color='blue', label='Income', alpha=0.7)
        plt.bar(range(start_year, end_year + 1), net_cash_flows, color='blue', edgecolor='red', hatch='//', label='Netto', alpha=0.5)
        plt.xlabel('Year')
        plt.ylabel('Cash Flow')
        plt.title('Discounted Cash Flows' if discounted else 'Cash Flows')
        plt.legend()
        plt.show()
