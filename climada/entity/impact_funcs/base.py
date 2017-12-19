"""
=====================
ImpactFuncs
=====================

The class ImpactFunc defines an impact function and the class ImpactFuncs
contains multiple ImpactFunc put in a dictionary.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Nov 13 09:21:06 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import abc
import numpy as np

from climada.entity.tag import Tag

class ImpactFunc(object):
    """ Contains the definition of one Damage Function """

    def __init__(self):
        self.id = 0
        self.name = ''
        self.intensity_unit = 'NA'
        # Followng values defined for each intensity value
        self.intensity = np.array([])
        self.mdd = np.array([])
        self.paa = np.array([])

    def interpolate(self, inten, attribute):
        """ Interpolate impact function to a given intensity
        INPUT:
            -inten: the x-coordinate of the interpolated values. numpy array.
            -attribute: string defining the impact function attribute to \
            interpolate. Possbile values: 'mdd' or 'paa'.
        """
        if attribute == 'mdd':
            return np.interp(inten, self.intensity, self.mdd)
        elif attribute == 'paa':
            return np.interp(inten, self.intensity, self.paa)
        else:
            raise ValueError('Attribute of the impact function ' + \
                             attribute + 'not found.')

class ImpactFuncs(metaclass=abc.ABCMeta):
    """ Contains Damage Functions definitions """

    def __init__(self, file_name=None, description=None):
        self.tag = Tag(file_name, description)
        self.data = {} # {hazard_id : {id:ImpactFunc}}

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description)

    @abc.abstractmethod
    def read(self, file_name, description=None):
        """ Virtual class. Needs to be defined for each child."""
