"""
Define Hazard ABC.
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
    """Contains events of the same hazard type. Describes interface. 
    
    Attributes
    ----------
        tag (TagHazard): information about the source
        id (int): hazard id
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id of each event
        frequency (np.array): frequency of each event
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, file_name=None, description=None, haztype=None):
        """Initialize values from given file, if given.
    
        Parameters
        ----------
            file_name (str, optional): file name to read
            description (str, optional): description of the data
            haztype (str, optional): acronym of the hazard type (e.g. 'TC')

        Raises
        ------
            ValueError
        """
        self.tag = TagHazard(file_name, description, haztype)
        self.id = 0
        self.units = 'NA'
        # following values are defined for each event
        self.centroids = Centroids()
        self.event_id = np.array([], np.int64) 
        self.frequency = np.array([])  
        #self.name = [""]
        #self.date = [date(1,1,1)]  # size: num_events
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix([]) # events x centroids
        self.fraction = sparse.csr_matrix([])  # events x centroids

        # Load values from file_name if provided
        if file_name is not None:
            self.read(file_name, description, haztype)

    def calc_future(self, conf):
        """ Compute the future assets following the configuration """

    @abc.abstractmethod
    def read(self, file_name, description=None, haztype=None, centroids=None,
             out_file_name=None):
        """ Virtual class. Needs to be defined for each child."""
