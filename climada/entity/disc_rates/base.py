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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter

import climada.util.checker as u_check
from climada.entity.tag import Tag
import climada.util.finance as u_fin
import climada.util.hdf5_handler as u_hdf5

LOGGER = logging.getLogger(__name__)

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'discount',
               'var_name': {'year': 'year',
                            'disc': 'discount_rate'
                           }
              }
"""MATLAB variable names"""

DEF_VAR_EXCEL = {'sheet_name': 'discount',
                 'col_name': {'year': 'year',
                              'disc': 'discount_rate'
                             }
                }
"""Excel variable names"""

class DiscRates():
    """Defines discount rates and basic methods. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        years (np.array): years
        rates (np.array): discount rates for each year (between 0 and 1)
    """

    def __init__(self):
        """Empty initialization.

        Examples:
            Fill discount rates with values and check consistency data:

            >>> disc_rates = DiscRates()
            >>> disc_rates.years = np.array([2000, 2001])
            >>> disc_rates.rates = np.array([0.02, 0.02])
            >>> disc_rates.check()

            Read discount rates from year_2050.mat and checks consistency data.

            >>> disc_rates = DiscRates(ENT_TEMPLATE_XLS)
        """
        self.clear()

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        # Following values are given for each defined year
        self.years = np.array([], int)
        self.rates = np.array([], float)

    def check(self):
        """Check attributes consistency.

        Raises:
            ValueError
        """
        u_check.size(len(self.years), self.rates, 'DiscRates.rates')

    def select(self, year_range):
        """Select discount rates in given years.

        Parameters:
            year_range (np.array): continuous sequence of selected years.

        Returns:
            DiscRates
        """
        pos_year = np.isin(year_range, self.years)
        if not np.all(pos_year):
            LOGGER.info('No discount rates for given years.')
            return None
        pos_year = np.isin(self.years, year_range)
        sel_disc = self.__class__()
        sel_disc.tag = self.tag
        sel_disc.years = self.years[pos_year]
        sel_disc.rates = self.rates[pos_year]

        return sel_disc

    def append(self, disc_rates):
        """Check and append discount rates to current DiscRates. Overwrite
        discount rate if same year.

        Parameters:
            disc_rates (DiscRates): DiscRates instance to append

        Raises:
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
        """Compute net present value between present year and future year.

        Parameters:
            ini_year (float): initial year
            end_year (float): end year
            val_years (np.array): cash flow at each year btw ini_year and
                end_year (both included)
        Returns:
            float
        """
        year_range = np.arange(ini_year, end_year + 1)
        if year_range.size != val_years.size:
            LOGGER.error('Wrong size of yearly values.')
            raise ValueError
        sel_disc = self.select(year_range)
        if sel_disc is None:
            LOGGER.error('No information of discount rates for provided years:'
                         ' %s - %s', ini_year, end_year)
            raise ValueError
        return u_fin.net_present_value(sel_disc.years, sel_disc.rates,
                                       val_years)

    def plot(self, axis=None, **kwargs):
        """Plot discount rates per year.

        Parameters:
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for plot matplotlib function, e.g. marker='x'

        Returns:
            matplotlib.axes._subplots.AxesSubplot
        """
        if not axis:
            _, axis = plt.subplots(1, 1)

        axis.set_title('Discount rates')
        axis.set_xlabel('Year')
        axis.set_ylabel('discount rate (%)')
        axis.plot(self.years, self.rates * 100, **kwargs)
        axis.set_xlim((self.years.min(), self.years.max()))
        return axis

    def read_mat(self, file_name, description='', var_names=DEF_VAR_MAT):
        """Read MATLAB file generated with previous MATLAB CLIMADA version.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            var_names (dict, optional): name of the variables in the file
        """
        disc = u_hdf5.read(file_name)
        self.clear()
        self.tag.file_name = str(file_name)
        self.tag.description = description
        try:
            disc = disc[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            disc = disc[var_names['field_name']]
            self.years = np.squeeze(disc[var_names['var_name']['year']]). \
                astype(int, copy=False)
            self.rates = np.squeeze(disc[var_names['var_name']['disc']])
        except KeyError as err:
            LOGGER.error("Not existing variable: %s", str(err))
            raise err

    def read_excel(self, file_name, description='', var_names=DEF_VAR_EXCEL):
        """Read excel file following template and store variables.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            var_names (dict, optional): name of the variables in the file
        """
        dfr = pd.read_excel(file_name, var_names['sheet_name'])
        self.clear()
        self.tag.file_name = str(file_name)
        self.tag.description = description
        try:
            self.years = dfr[var_names['col_name']['year']].values. \
                astype(int, copy=False)
            self.rates = dfr[var_names['col_name']['disc']].values
        except KeyError as err:
            LOGGER.error("Not existing variable: %s", str(err))
            raise err

    def write_excel(self, file_name, var_names=DEF_VAR_EXCEL):
        """Write excel file following template.

        Parameters:
            file_name (str): absolute file name to write
            var_names (dict, optional): name of the variables in the file
        """
        disc_wb = xlsxwriter.Workbook(file_name)
        disc_ws = disc_wb.add_worksheet(var_names['sheet_name'])

        header = [var_names['col_name']['year'], var_names['col_name']['disc']]
        for icol, head_dat in enumerate(header):
            disc_ws.write(0, icol, head_dat)
        for i_yr, (disc_yr, disc_rt) in enumerate(zip(self.years, self.rates), 1):
            disc_ws.write(i_yr, 0, disc_yr)
            disc_ws.write(i_yr, 1, disc_rt)
        disc_wb.close()
