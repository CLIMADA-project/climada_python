"""
Define Centroids class.
"""

import pickle
import pandas
import numpy as np

from climada.entity.tag import Tag

class Centroids(object):
    """Definition of the irregular grid."""

    def __init__(self):
        self.tag = Tag()
        self.coord = np.array([])
        self.id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)
        #self.mask = 0
        # Define tha name of the sheet that is read
        self.sheet_name = 'centroids'
        # Define the names of the columns that are read
        self.col_names = {'cen_id' : 'centroid_ID',
                          'lat' : 'Latitude',
                          'lon' : 'Longitude'
                         }

    def read_excel(self, file_name, description=None, out_file_name=None):
        """ Read centroids from an excel file"""

        # load Excel data of the Centroids
        dfr = pandas.read_excel(file_name, self.sheet_name)

        self.tag = Tag(file_name, description)
        coord_cols = [self.col_names['lat'], self.col_names['lon']]

        self.coord = np.array(dfr[coord_cols])
        self.id = dfr[self.col_names['cen_id']].values

        # Save results if output filename is given
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    def read_mat(self, file_name, description=None, out_file_name=None):
        """ Read from matlab file."""
        #TODO

    def is_centroids(self):
        """ Check if attributes are coherent."""
        # TODO
