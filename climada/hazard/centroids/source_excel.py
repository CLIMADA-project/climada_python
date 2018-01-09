"""
Define CentroidsExcel class.
"""

import pandas
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.entity.tag import Tag

class CentroidsExcel(Centroids):
    """Centroids class loaded from an excel file."""

    def __init__(self, file_name=None, description=None):
        """Extend Centroids __init__ method."""
        # Define tha name of the sheet that is read
        self.sheet_name = 'centroids'
        # Define the names of the columns that are read
        self.col_names = {'cen_id' : 'centroid_ID',
                          'lat' : 'Latitude',
                          'lon' : 'Longitude'
                         }

        # Initialize
        Centroids.__init__(self, file_name, description)

    def _read(self, file_name, description=None):
        """Override _read Centroids method."""
        dfr = pandas.read_excel(file_name, self.sheet_name)

        self.tag = Tag(file_name, description)
        coord_cols = [self.col_names['lat'], self.col_names['lon']]

        self.coord = np.array(dfr[coord_cols])
        self.id = dfr[self.col_names['cen_id']].values
