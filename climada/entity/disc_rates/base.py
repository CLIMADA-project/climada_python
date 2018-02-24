"""
Define DiscRates.
"""

__all__ = ['DiscRates']

import os
import numpy as np

from climada.entity.disc_rates.source_excel import read as read_excel
from climada.entity.disc_rates.source_mat import read as read_mat
import climada.util.checker as check
from climada.entity.tag import Tag

class DiscRates(object):
    """Contains discount rates.

    Attributes
    ----------
        tag (Taf): information about the source data
        years (np.array): years
        rates (np.array): discount rates for each year
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

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
        """
        self.tag = Tag(file_name, description)
        # Following values are given for each defined year
        self.years = np.array([], np.int64)
        self.rates = np.array([], np.float64)

        # Load values from file_name if provided
        if file_name != '':
            self.load(file_name, description)

    def check(self):
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        check.size(len(self.years), self.rates, 'DiscRates.rates')

    def read(self, file_name, description=''):
        """Read input file.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError
        """
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(self, file_name, description)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(self, file_name, description)
        else:
            raise TypeError('Input file extension not supported: %s.' % \
                            extension)

    def load(self, file_name, description=''):
        """Read and check.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError
        """
        self.read(file_name, description)
        self.check()
