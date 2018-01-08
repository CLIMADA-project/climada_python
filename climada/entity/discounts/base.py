"""
Define Discounts class.
"""

import abc
import pickle
import numpy as np

from climada.entity.tag import Tag

class Discounts(metaclass=abc.ABCMeta):
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
            This is an abstract class, it can't be instantiated.
        """
        self.tag = Tag(file_name, description)
        # Following values are given for each defined year
        self.years = np.array([], np.int64)
        self.rates = np.array([])

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def is_discounts(self):
        """ Checks if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        # TODO: raise Error if instance is not well filled

    def load(self, file_name, description=None, out_file_name=None):
        """Read, check and save as pkl, if output file name.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            out_file_name (str, optional): output file name to save as pkl

        Raises
        ------
            ValueError
        """
        self._read(file_name, description)
        self.is_discounts()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    @abc.abstractmethod
    def _read(self, file_name, description=None):
        """ Read input file. Abstract method. To be implemented by subclass.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
        """
