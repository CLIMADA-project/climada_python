"""
=====================
Measures
=====================

Define the class Measure and the class Measures containing different measures.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Nov 13 09:21:36 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import abc
import numpy as np

from climada.entity.tag import Tag

class Measure(object):
    """ Contains the definition of one Measure """

    def __init__(self):
        self.name = ""
        self.color_rgb = np.array([0, 0, 0])
        self.cost = 0
        self.hazard_freq_cutoff = 0
        self.hazard_event_set = 'NA'
        self.hazard_intensity = () # parameter a and b
        self.mdd_impact = () # parameter a and b
        self.paa_impact = () # parameter a and b
        self.risk_transf_attach = 0
        self.risk_transf_cover = 0

class Measures(metaclass=abc.ABCMeta):
    """ Contains Measures definitions """

    def __init__(self, file_name=None, description=None):
        self.tag = Tag(file_name, description)
        self.data = {} # {id: Measure()}

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description)

    @abc.abstractmethod
    def read(self, file_name, description=None):
        """ Virtual class. Needs to be defined for each child."""
