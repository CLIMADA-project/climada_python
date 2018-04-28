"""
Define DiscRates class.
"""

__all__ = ['DiscRates',
           'FILE_EXT']

import os
import copy
from array import array
import logging
import numpy as np

from climada.entity.disc_rates.source import READ_SET
from climada.util.files_handler import to_list, get_file_names
import climada.util.checker as check
from climada.entity.tag import Tag
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

class DiscRates(object):
    """Defines discount rates and basic methods. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        years (np.array): years
        rates (np.array): discount rates for each year
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters:
            file_name (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises:
            ValueError

        Examples:
            Fill discount rates with values and check consistency data:

            >>> disc_rates = DiscRates()
            >>> disc_rates.years = np.array([2000, 2001])
            >>> disc_rates.rates = np.array([0.02, 0.02])
            >>> disc_rates.check()

            Read discount rates from year_2050.mat and checks consistency data.

            >>> disc_rates = DiscRates(ENT_TEST_XLS)
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description)

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
        check.size(len(self.years), self.rates, 'DiscRates.rates')

    def read(self, files, descriptions='', var_names=None):
        """Read and check discount rates.

        Parameters:
            files (str or list(str)): absolute file name(s) or folder name
                containing the files to read
            descriptions (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises:
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, var in zip(all_files, desc_list, var_list):
            self.append(self._read_one(file, desc, var))

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
            self.__dict__ = disc_rates.__dict__.copy()
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

    @staticmethod
    def get_sup_file_format():
        """ Get supported file extensions that can be read.

        Returns:
            list(str)
        """
        return list(FILE_EXT.keys())

    def plot(self):
        """Plot discount rates per year."""
        graph = plot.Graph2D('Discount rates')
        graph.add_subplot('Year', 'discount rate (%)', '')
        graph.add_curve(self.years, self.rates * 100, 'b')
        graph.set_x_lim(self.years)
        return graph.get_elems()

    @staticmethod
    def get_def_file_var_names(src_format):
        """Get default variable names for given file format.

        Parameters:
            src_format (str): extension of the file, e.g. '.xls', '.mat'.

        Returns:
            dict: dictionary with variable names
        """
        try:
            if '.' not in src_format:
                src_format = '.' + src_format
            return copy.deepcopy(READ_SET[FILE_EXT[src_format]][0])
        except KeyError:
            LOGGER.error('File extension not supported: %s.', src_format)
            raise ValueError

    @staticmethod
    def _read_one(file_name, description='', var_names=None):
        """Read one file and fill attributes.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            var_names (dict, optional): name of the variables in the file

        Raises:
            ValueError

        Returns:
            DiscRates
        """
        LOGGER.info('Reading file: %s', file_name)
        new_disc = DiscRates()
        new_disc.tag = Tag(file_name, description)

        extension = os.path.splitext(file_name)[1]
        try:
            reader = READ_SET[FILE_EXT[extension]][1]
        except KeyError:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        reader(new_disc, file_name, var_names)

        return new_disc

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
