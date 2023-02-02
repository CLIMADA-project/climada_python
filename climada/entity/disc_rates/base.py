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

Define DiscRates class.
"""

__all__ = ['DiscRates']

import copy
from array import array
import logging
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter

import climada.util.checker as u_check
from climada.entity.tag import Tag
import climada.util.finance as u_fin
import climada.util.hdf5_handler as u_hdf5

LOGGER = logging.getLogger(__name__)

"""MATLAB variable names"""
DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'discount',
               'var_name': {'year': 'year',
                            'disc': 'discount_rate'
                           }
              }

"""Excel variable names"""
DEF_VAR_EXCEL = {'sheet_name': 'discount',
                 'col_name': {'year': 'year',
                              'disc': 'discount_rate'
                             }
                }


class DiscRates():
    """
    Defines discount rates and basic methods. Loads from
    files with format defined in FILE_EXT.

    Attributes
    ---------
    tag: climada.entity.tag.Tag
        information about the source data
    years: np.array
        list of years
    rates: np.array
        list of discount rates for each year (between 0 and 1)
    """

    def __init__(
        self,
        years : Optional[np.ndarray] = None,
        rates : Optional[np.ndarray] = None,
        tag : Optional[Tag] = None
        ):
        """
        Fill discount rates with values and check consistency data

        Parameters
        ----------
        years : numpy.ndarray(int)
            Array of years. Default is numpy.array([]).
        rates : numpy.ndarray(float)
            Discount rates for each year in years.
            Default is numpy.array([]).
            Note: rates given in float, e.g., to set 1% rate use 0.01
        tag : climate.entity.tag
            Metadata. Default is None.
        """
        self.years = np.array([]) if years is None else years
        self.rates = np.array([]) if rates is None else rates
        self.tag = Tag() if tag is None else tag

    def clear(self):
        """Reinitialize attributes."""

        self.tag = Tag()
        # Following values are given for each defined year
        self.years = np.array([], int)
        self.rates = np.array([], float)

    def check(self):
        """
        Check attributes consistency.

        Raises
        ------
        ValueError
        """
        u_check.size(len(self.years), self.rates, 'DiscRates.rates')

    def select(self, year_range):
        """
        Select discount rates in given years.

        Parameters
        ----------
        year_range: np.array(int)
            continuous sequence of selected years.

        Returns: climada.entity.DiscRates
            The selected discrates in the year_range
        """
        pos_year = np.isin(year_range, self.years)
        if not np.all(pos_year):
            LOGGER.info('No discount rates for given years.')
            return None
        pos_year = np.isin(self.years, year_range)

        return DiscRates(years=self.years[pos_year],
                         rates=self.rates[pos_year],
                         tag=self.tag)

    def append(self, disc_rates):
        """
        Check and append discount rates to current DiscRates. Overwrite
        discount rate if same year.

        Parameters
        ----------
        disc_rates: climada.entity.DiscRates
            DiscRates instance to append

        Raises
        ------
        ValueError
        """
        disc_rates.check()
        if self.years.size == 0:
            self.__dict__ = copy.deepcopy(disc_rates.__dict__)
            return

        self.tag.append(disc_rates.tag)

        new_year = array('l')
        new_rate = array('d')
        for year, rate in zip(disc_rates.years, disc_rates.rates):
            found = np.where(year == self.years)[0]
            if found.size > 0:
                self.rates[found[0]] = rate
            else:
                new_year.append(year)
                new_rate.append(rate)

        self.years = np.append(self.years, new_year).astype(int, copy=False)
        self.rates = np.append(self.rates, new_rate)

    def net_present_value(self, ini_year, end_year, val_years):
        """
        Compute net present value between present year and future year.

        Parameters
        ----------
        ini_year: float
            initial year
        end_year: float
            end year
        val_years: np.array
            cash flow at each year btw ini_year and end_year (both included)

        Returns
        -------
            net_present_value: float
                net present value between present year and future year.

        """
        year_range = np.arange(ini_year, end_year + 1)
        if year_range.size != val_years.size:
            raise ValueError('Wrong size of yearly values.')
        sel_disc = self.select(year_range)
        if sel_disc is None:
            raise ValueError('No information of discount rates for provided years:'
                             f' {ini_year} - {end_year}')
        return u_fin.net_present_value(sel_disc.years, sel_disc.rates,
                                       val_years)

    def plot(self, axis=None, figsize=(6, 8), **kwargs):
        """
        Plot discount rates per year.

        Parameters
        ----------
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: tuple(int, int), optional
            size of the figure. The default is (6,8)
        kwargs: optional
            keyword arguments  passed to plotting function axis.plot

        Returns
        -------
        axis: matplotlib.axes._subplots.AxesSubplot
            axis handles of the plot
        """
        if not axis:
            _, axis = plt.subplots(1, 1, figsize=figsize)

        axis.set_title('Discount rates')
        axis.set_xlabel('Year')
        axis.set_ylabel('discount rate (%)')
        axis.plot(self.years, self.rates * 100, **kwargs)
        axis.set_xlim((self.years.min(), self.years.max()))
        return axis

    @classmethod
    def from_mat(cls, file_name, description='', var_names=None):
        """
        Read MATLAB file generated with previous MATLAB CLIMADA version.

        Parameters
        ----------
        file_name: str
            filename including path and extension
        description: str, optional
            description of the data. The default is ''
        var_names: dict, optional
            name of the variables in the file. Default:

            >>> DEF_VAR_MAT = {
            ...     'sup_field_name': 'entity',
            ...     'field_name': 'discount',
            ...     'var_name': {
            ...         'year': 'year',
            ...         'disc': 'discount_rate',
            ...     }
            ... }

        Returns
        -------
        climada.entity.DiscRates :
            The disc rates from matlab
        """
        if var_names is None:
            var_names = DEF_VAR_MAT
        disc = u_hdf5.read(file_name)
        tag = Tag(file_name=str(file_name), description=description)
        try:
            disc = disc[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            disc = disc[var_names['field_name']]
            years = np.squeeze(disc[var_names['var_name']['year']]). \
                astype(int, copy=False)
            rates = np.squeeze(disc[var_names['var_name']['disc']])
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

        return cls(years=years, rates=rates, tag=tag)

    def read_mat(self, *args, **kwargs):
        """This function is deprecated, use DiscRates.from_mats instead."""
        LOGGER.warning("The use of DiscRates.read_mats is deprecated."
                       "Use DiscRates.from_mats instead.")
        self.__dict__ = DiscRates.from_mat(*args, **kwargs).__dict__

    @classmethod
    def from_excel(cls, file_name, description='', var_names=None):
        """
        Read excel file following template and store variables.

        Parameters
        ----------
        file_name: str
            filename including path and extension
        description: str, optional
            description of the data. The default is ''
        var_names: dict, optional
            name of the variables in the file. The Default is

            >>> DEF_VAR_EXCEL = {
            ...     'sheet_name': 'discount',
            ...     'col_name': {
            ...         'year': 'year',
            ...         'disc': 'discount_rate',
            ...     }
            ... }

        Returns
        -------
        climada.entity.DiscRates :
            The disc rates from excel
        """
        if var_names is None:
            var_names = DEF_VAR_EXCEL
        dfr = pd.read_excel(file_name, var_names['sheet_name'])
        tag = Tag(file_name=str(file_name), description=description)
        try:
            years = dfr[var_names['col_name']['year']].values. \
                astype(int, copy=False)
            rates = dfr[var_names['col_name']['disc']].values
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

        return cls(years=years, rates=rates, tag=tag)

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use DiscRates.from_excel instead."""
        LOGGER.warning("The use of DiscRates.read_excel is deprecated."
                       "Use DiscRates.from_excel instead.")
        self.__dict__ = DiscRates.from_mat(*args, **kwargs).__dict__


    def write_excel(self, file_name, var_names=None):
        """
        Write excel file following template.

        Parameters
        ----------
        file_name: str
            filename including path and extension
        var_names: dict, optional
            name of the variables in the file. The Default is

            >>> DEF_VAR_EXCEL = {
            ...     'sheet_name': 'discount',
            ...     'col_name': {
            ...         'year': 'year',
            ...         'disc': 'discount_rate',
            ...     }
            ... }
        """
        if var_names is None:
            var_names = DEF_VAR_EXCEL
        disc_wb = xlsxwriter.Workbook(file_name)
        disc_ws = disc_wb.add_worksheet(var_names['sheet_name'])

        header = [var_names['col_name']['year'], var_names['col_name']['disc']]
        for icol, head_dat in enumerate(header):
            disc_ws.write(0, icol, head_dat)
        for i_yr, (disc_yr, disc_rt) in enumerate(zip(self.years, self.rates), 1):
            disc_ws.write(i_yr, 0, disc_yr)
            disc_ws.write(i_yr, 1, disc_rt)
        disc_wb.close()
