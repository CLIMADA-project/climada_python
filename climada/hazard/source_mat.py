"""
=====================
source_mat module
=====================

Define HazardMat class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Dec 11 08:50:42 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import pickle
import numpy as np

from climada.hazard.base import Hazard
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard
import climada.util.hdf5_handler as hdf5

class HazardMat(Hazard):
    """Class that loads the exposures from an excel file"""

    def __init__(self, file_name=None, description=None):
        """Define the name of the sheet and columns names where the exposures
        are defined"""
        # Initialize
        Hazard.__init__(self, file_name, description)

    def read(self, file_name, description=None, centroids=None,
             out_file_name=None):
        """Virtual class. Needs to be defined for each child"""

        # Load hazard data
        hazard = hdf5.read(file_name)
        try:
            hazard = hazard['hazard']
        except KeyError:
            pass

        # Fill hazard tag
        # Get hazard type
        haz_type = hdf5.get_string(hazard['peril_ID'])
        self.tag = TagHazard(file_name, description, haz_type)

        # Set the centroids if given, otherwise load them from the same file
        if centroids is None:
            self.centroids.tag = Tag(file_name, description)
            cen_lat = hazard['lat'].reshape(len(hazard['lat']),)
            cen_lon = hazard['lon'].reshape(len(hazard['lon']),)
            self.centroids.coord = np.array([cen_lat, cen_lon]).transpose()
            self.centroids.id = hazard['centroid_ID'].astype(int, copy=False)
        else:
            self.centroids = centroids

        # reshape from shape (x,1) to 1d array shape (x,)
        self.frequency = hazard['frequency']. \
        reshape(len(hazard['frequency']),)
        self.event_id = hazard['event_ID'].astype(int, copy=False). \
        reshape(len(hazard['event_ID']),)
        self.units = hdf5.get_string(hazard['units'])

        # number of centroids and events
        n_cen = len(self.centroids.id)
        n_event = len(self.event_id)

        # intensity and fraction
        try:
            self.intensity = hdf5.get_sparse_mat(hazard['intensity'], \
                                                 (n_event, n_cen))
        except ValueError:
            print('Size missmatch in intensity matrix.')
            raise

        try:
            self.fraction = hdf5.get_sparse_mat(hazard['fraction'], \
                                     (n_event, n_cen))
        except ValueError:
            print('Size missmatch in fraction matrix.')
            raise

        # Save results if output filename is given
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)
