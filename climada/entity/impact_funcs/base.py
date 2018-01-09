"""
Define ImpactFunc class and ImpactFuncs ABC.
"""

import abc
import pickle
import numpy as np

from climada.entity.tag import Tag

class ImpactFunc(object):
    """Contains the definition of one Damage Function.

    Attributes
    ----------
        id (int): id of the function
        name (str): name of the function
        intensity_unit (str): unit of the intensity
        intensity (np.array): intensity values
        mdd (np.array): mean damage (impact) degree for each intensity
        paa (np.array): percentage of affected assets (exposures) for each
            intensity
    """

    def __init__(self):
        """ Empty initialization."""
        self.id = 0
        self.name = ''
        self.intensity_unit = 'NA'
        # Followng values defined for each intensity value
        self.intensity = np.array([])
        self.mdd = np.array([])
        self.paa = np.array([])

    def interpolate(self, inten, attribute):
        """ Interpolate impact function to a given intensity.

        Parameters
        ----------
            inten (float or np.array): intensity, the x-coordinate of the
                interpolated values.
            attribute (str): defines the impact function attribute to
                interpolate. Possbile values: 'mdd' or 'paa'.

        Raises
        ------
            ValueError
        """
        if attribute == 'mdd':
            return np.interp(inten, self.intensity, self.mdd)
        elif attribute == 'paa':
            return np.interp(inten, self.intensity, self.paa)
        else:
            raise ValueError('Attribute of the impact function ' + \
                             attribute + 'not found.')

class ImpactFuncs(metaclass=abc.ABCMeta):
    """Contains impact functions of type ImpactFunc.

    Attributes
    ----------
        tag (Taf): information about the source data
        data (dict): dictionary of impact functions. Keys are the impact
            functions' id and values are instances of ImpactFunc.
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
        self.data = {} # {hazard_id : {id:ImpactFunc}}

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def is_impactfuncs(self):
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
        self.is_impactfuncs()
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
