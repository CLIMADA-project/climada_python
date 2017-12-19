"""
=====================
exposures_base
=====================

Define the class Exposures.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Nov 10 10:00:03 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import abc
import numpy as np

from climada.entity.tag import Tag
from climada.util.interpolation import Interpolator

class Exposures(metaclass=abc.ABCMeta):
    """ Contains Assets definitions """

    def __init__(self, file_name=None, description=None):
        # Tag defining Exposures contents
        self.tag = Tag(file_name, description)
        # reference year considered
        self.ref_year = 0
        # unit used for the values if the Exposures
        self.value_unit = 'NA'
        # Followng values defined for each exposure
        self.id = np.array([], np.int64)
        self.coord = np.array([])
        self.value = np.array([])
        self.deductible = np.array([])
        self.cover = np.array([])
        self.impact_id = np.array([], np.int64)
        self.category_id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)

        # Assignement of centroid neighbors to each coordinate
        # Computed in function assign
        self.assigned = np.array([])

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description)

    @abc.abstractmethod
    def read(self, file_name, description=None):
        """ Virtual class. Needs to be defined for each child."""

    def assign(self, hazard, method='NN', dist='approx', threshold=100):
        """ Compute the hazard centroids id used at each exposure"""
        interp = Interpolator(threshold)
        self.assigned = interp.interpol_index(hazard.centroids.coord, \
                                              self.coord, method, dist)

    def geo_coverage(self):
        """ Get geographic coverage of the assets"""
