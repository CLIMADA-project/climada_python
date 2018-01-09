"""
Define HazardExcel class.
"""

import pandas
from scipy import sparse
import numpy as np

from climada.hazard.base import Hazard
from climada.hazard.centroids.source_excel import CentroidsExcel
from climada.hazard.tag import Tag as TagHazard

class HazardExcel(Hazard):
    """Hazard class loaded from an excel file."""

    def __init__(self, file_name=None, description=None, haztype=None):
        """Extend Hazard __init__ method."""
        # Define tha name of the sheet that is read
        self.sheet_names = {'centroid' : 'centroids',
                            'inten' : 'hazard_intensity',
                            'freq' : 'hazard_frequency'
                           }
        # Define the names of the columns that are read
        self.col_names = {'cen_id' : 'centroid_ID',
                          'even_id' : 'event_ID',
                          'freq' : 'frequency'
                         }
        # Define tha names of the columns describing the centroids.
        # Used only when the centroids are not provided and have to be read
        # from the same file as the hazard
        self.col_centroids = {'cen_id' : 'centroid_ID',
                              'lat' : 'Latitude',
                              'lon' : 'Longitude'
                             }

        # Initialize
        Hazard.__init__(self, file_name, description, haztype)

    def _read(self, file_name, description=None, haztype=None, centroids=None):
        """Override _read Hazard method."""
        # append the file name and description into the instance class.
        # Put type TC as default
        self.tag = TagHazard(file_name, description, 'TC')

        # Set the centroids if given, otherwise load them from the same file
        if centroids is None:
            self.centroids = CentroidsExcel()
            self.centroids.sheet_name = self.sheet_names['centroid']
            self.centroids.col_names = self.col_centroids
            self.centroids._read(file_name, description)
        else:
            self.centroids = centroids

        # number of centroids
        num_cen = len(self.centroids.id)

        # Load hazard frequency
        dfr = pandas.read_excel(file_name, self.sheet_names['freq'])
        # number of events
        num_events = dfr.shape[0]
        self.frequency = dfr[self.col_names['freq']].values
        self.event_id = dfr[self.col_names['even_id']].values

        # Load hazard intensity
        dfr = pandas.read_excel(file_name, self.sheet_names['inten'])
        # number of events (ignore centroid_ID column)
        # check the number of events is the same as the one in the frequency
        if dfr.shape[1] - 1 is not num_events:
            raise ValueError('Hazard intensity is given for a number of \
                  events different from the number of defined in its \
                  frequency: ', dfr.shape[1] - 1, ' != ', num_events)
        # check number of centroids is the same as retrieved before
        if dfr.shape[0] is not num_cen:
            raise ValueError('Hazard intensity is given for a number of \
                              centroids different from the number of \
                              centroids defined: %s != %s', dfr.shape[0], \
                              num_cen)
        # check centroids ids are correct
        if not np.array_equal(dfr[self.col_names['cen_id']].values,
                              self.centroids.id[-num_cen:]):
            raise ValueError('Hazard intensity centroids ids do not match \
                             previously defined centroids.')

        self.intensity = dfr.values[:, 1:num_events+1].transpose()
        # make the intensity a sparse matrix
        self.intensity = sparse.csr_matrix(self.intensity)

        # Set fraction matrix to default value of 1
        self.fraction = sparse.csr_matrix(np.ones(self.intensity.shape, \
                                          dtype=np.float))
