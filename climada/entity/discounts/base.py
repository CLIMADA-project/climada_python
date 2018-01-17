"""
Define Discounts.
"""

import numpy as np

from climada.entity.loader import Loader
import climada.util.auxiliar as aux
from climada.entity.tag import Tag

class Discounts(Loader):
    """Contains discount rates.

    Attributes
    ----------
        tag (Taf): information about the source data
        years (np.array): years
        rates (np.array): discount rates for each year
    """

    def __init__(self, file_name=None, description=None):
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
            >>> disc_rates = Discounts()
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
        if file_name is not None:
            self.load(file_name, description)

    def check(self):
        """ Override Loader check."""
        aux.check_size(len(self.years), self.rates, 'Discounts.rates')
