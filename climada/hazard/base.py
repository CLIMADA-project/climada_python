"""
=====================
base module
=====================

Define Hazard Base Class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Nov 13 10:42:54 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

#from datetime import date
import abc
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids import Centroids

class Hazard(metaclass=abc.ABCMeta):
    """ Contains hazards of the same type """

    def __init__(self, file_name=None, description=None):
        self.tag = TagHazard(file_name, description)
        self.id = 0
        self.units = 'NA'
        # following values are defined for each event
        self.centroids = Centroids()
        self.event_id = np.array([], np.int64) # id for each hazard event, size: num_events
        self.frequency = np.array([])  # size: num_events
        #self.name = [""]  # size: num_events
        #self.date = [date(1,1,1)]  # size: num_events
        self.intensity = sparse.csr_matrix([]) # sparse matrix events x centr
        self.fraction = sparse.csr_matrix([])  # sparse matrix events x centr

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description)

    def calc_future(self, conf):
        """ Compute the future assets following the configuration """

    @abc.abstractmethod
    def read(self, file_name, description=None, centroids=None,
             out_file_name=None):
        """ Virtual class. Needs to be defined for each child."""
