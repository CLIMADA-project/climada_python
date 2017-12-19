"""
=====================
base module
=====================

Define the class Discount and the class Dicounts containing dicount rates.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Nov 13 09:21:32 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import abc
import numpy as np

from climada.entity.tag import Tag

class Discounts(metaclass=abc.ABCMeta):
    """ Contains the definition of one Discount rate"""

    def __init__(self, file_name=None, description=None):
        self.tag = Tag(file_name, description)
        # Following values are given for each defined year
        self.years = np.array([], np.int64)
        self.rates = np.array([])

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description)

    @abc.abstractmethod
    def read(self, file_name, description=None):
        """ Virtual class. Needs to be defined for each child."""
