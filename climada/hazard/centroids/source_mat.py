"""
Define CentroidsMat class.
"""

import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

class CentroidsMat(Centroids):
    """Centroids class loaded from a mat file produced by climada.

    Attributes
    ----------
        field_name (str): name of variable containing the data
        var_names (dict): name of the variables in field_name
    """

    def __init__(self, file_name=None, description=None):
        """Extend Centroids __init__ method."""
        # Define tha name of the field that is read
        self.field_name = 'centroids'
        # Define the names of the variables in field_name that are read
        self.var_names = {'cen_id' : 'centroid_ID',
                          'lat' : 'lat',
                          'lon' : 'lon'
                         }

        # Initialize
        Centroids.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Centroids method."""
        cent = hdf5.read(file_name)
        try:
            cent = cent[self.field_name]
        except KeyError:
            pass

        self.tag = Tag(file_name, description)
        cen_lat = np.squeeze(cent[self.var_names['lat']])
        cen_lon = np.squeeze(cent[self.var_names['lon']])
        self.coord = np.array([cen_lat, cen_lon]).transpose()
        self.id = np.squeeze(cent[self.var_names['cen_id']]. \
        astype(int, copy=False))
