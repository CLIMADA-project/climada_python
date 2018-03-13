"""
Define DiscRates.
"""

__all__ = ['DiscRates']

from array import array
import logging
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np

from climada.entity.disc_rates.source import read as read_source
from climada.util.files_handler import to_str_list, get_file_names
import climada.util.checker as check
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

class DiscRates(object):
    """Defines discount rates and basic methods.

    Attributes
    ----------
        tag (Tag): information about the source data
        years (np.array): years
        rates (np.array): discount rates for each year
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises
        ------
            ValueError

        Examples
        --------
            >>> disc_rates = DiscRates()
            >>> disc_rates.years = np.array([2000, 2001])
            >>> disc_rates.rates = np.array([0.02, 0.02])
            >>> disc_rates.check()
            Fill discount rates with values and check consistency data.
            >>> disc_rates = DiscRates('year_2050.mat')
            Read discount rates from year_2050.mat and checks consistency data.
        """
        self.clear()
        # Load values from file_name if provided
        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""        
        self.tag = Tag()
        # Following values are given for each defined year
        self.years = np.array([], np.int64)
        self.rates = np.array([], np.float64)

    def check(self):
        """Check attributes consistency.

        Raises
        ------
            ValueError
        """
        check.size(len(self.years), self.rates, 'DiscRates.rates')

    def read(self, files, descriptions='', var_names=None):
        """Read and check discount rates in parallel through files.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in 
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        desc_list = to_str_list(len(all_files), descriptions, 'descriptions')
        var_list = to_str_list(len(all_files), var_names, 'var_names')
        self.clear()
        disc_part = Pool().map(self._read_one, all_files, desc_list, var_list)
        for disc, file in zip(disc_part, all_files):
            LOGGER.info('Read file: %s', file)    
            self.append(disc)

    def append(self, disc_rates):
        """Check and append discount rates to current DiscRates. Overwrite 
        discount rate if same year.
        
        Parameters
        ----------
            disc_rates (DiscRates): DiscRates instance to append

        Raises
        ------
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
        
        self.years = np.append(self.years, new_year).astype(int)
        self.rates = np.append(self.rates, new_rate)

    @staticmethod
    def _read_one(file_name, description, var_names):
        """Read one file and fill attributes.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str): description of the source data
            var_names (dict): name of the variables in the file (e.g. 
                      DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError

        Returns
        ------
            DiscRates
        """
        new_disc = DiscRates()
        read_source(new_disc, file_name, description, var_names)
        return new_disc
