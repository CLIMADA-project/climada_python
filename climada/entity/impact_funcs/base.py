"""
Define ImpactFunc class and ImpactFuncs ABC.
"""

import numpy as np
import matplotlib.pyplot as plt

from climada.entity.loader import Loader
import climada.util.checker as check
from climada.entity.tag import Tag

class ImpactFuncs(Loader):
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
            >>> fun_1 = ImpactFunc()
            >>> fun_1.id = 3
            >>> fun_1.intensity = np.array([0, 20])
            >>> fun_1.paa = np.array([0, 1])
            >>> fun_1.mdd = np.array([0, 0.5])
            >>> imp_fun = ImpactFuncs()
            >>> imp_fun.data['TC'] = {fun_1.id : fun_1}
            >>> imp_fun.check()
            Fill impact functions with values and check consistency data.
        """
        self.tag = Tag(file_name, description)
        self.data = {} # {hazard_id : {id:ImpactFunc}}

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def check(self):
        """ Override Loader check."""
        for key_haz, fun in self.data.items():
            for key, val in fun.items():
                if key != val.id:
                    raise ValueError('Wrong ImpactFunc.id: %s != %s' %\
                                     (key, val.id))
                if key_haz != val.haz_type:
                    raise ValueError('Wrong ImpactFunc.haz_type: %s != %s' %\
                                     (key_haz, val.haz_type))
                val.check()

class ImpactFunc(object):
    """Contains the definition of one Damage Function.

    Attributes
    ----------
        id (int): id of the function
        name (str): name of the function
        haz_type (str): hazard type
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
        self.haz_type = 'NA'
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
            raise ValueError('Attribute of the impact function %s not found.'\
                             % (attribute))

    def plot(self):
        """Plot the impact functions MDD, MDR and PAA in one graph."""
        fig, ax = plt.subplots()
        ax.plot(self.intensity, self.mdd * 100, 'b', label='MDD')
        ax.plot(self.intensity, self.paa * 100, 'r', label='PAA')
        ax.plot(self.intensity, self.mdd * self.paa * 100, 'k--', label='MDR')
        ax.grid()
        ax.legend(loc='upper left')
        ax.set_xlabel('Intensity (%s)' % self.intensity_unit)
        ax.set_ylabel('Percentage (%)')
        ax.set_xlim([np.min(self.intensity), np.max(self.intensity)])
        fig.suptitle('%s %s %s' % (self.haz_type, str(self.id), self.name))

        plt.show()

    def check(self):
        """ Check consistent instance data.

        Raises
        ------
            ValueError
        """
        num_exp = len(self.intensity)
        check.size(num_exp, self.mdd, 'ImpactFunc.mdd')
        check.size(num_exp, self.paa, 'ImpactFunc.paa')
