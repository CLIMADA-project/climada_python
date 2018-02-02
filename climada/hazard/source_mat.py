"""
Define HazardMat class.
"""

__all__ = ['HazardMat']

import numpy as np

from climada.hazard.base import Hazard
from climada.hazard.centroids.source_mat import CentroidsMat
from climada.hazard.tag import Tag as TagHazard
import climada.util.hdf5_handler as hdf5

class HazardMat(Hazard):
    """Hazard class loaded from a mat file produced by climada.

    Attributes
    ----------
        field_name (str): name of variable containing the data
        var_names (dict): name of the variables in field_name
    """

    def __init__(self, file_name=None, description=None, haztype=None):
        """Extend Hazard __init__ method."""
        # Define tha name of the field that is read
        self.field_name = 'hazard'
        # Define the names of the variables in field_namethat are read
        self.var = {'per_id' : 'peril_ID',
                    'even_id' : 'event_ID',
                    'ev_name' : 'name',
                    'freq' : 'frequency',
                    'inten': 'intensity',
                    'unit': 'units',
                    'frac': 'fraction'
                   }
        # Define tha names of the variables describing the centroids.
        # Used only when the centroids are not provided and have to be read
        # from the same file as the hazard
        self.var_cent = {'cen_id' : 'centroid_ID',
                         'lat' : 'lat',
                         'lon' : 'lon'
                        }
        # Initialize
        Hazard.__init__(self, file_name, description, haztype)

    def read(self, file_name, description=None, haztype=None, centroids=None):
        """Override read Hazard method."""
        # Load hazard data
        hazard = hdf5.read(file_name)
        try:
            hazard = hazard[self.field_name]
        except KeyError:
            pass

        # Fill hazard tag
        haz_type = hdf5.get_string(hazard[self.var['per_id']])
        # Raise error if provided hazard type does not match with read one
        if haztype is not None and haz_type != haztype:
            raise ValueError('Hazard read is not of type: ' + haztype)
        self.tag = TagHazard(file_name, description, haz_type)

        # Set the centroids if given, otherwise load them from the same file
        self._read_centroids(centroids)

        # reshape from shape (x,1) to 1d array shape (x,)
        self.frequency = np.squeeze(hazard[self.var['freq']])
        self.event_id = np.squeeze(hazard[self.var['even_id']]. \
                                   astype(int, copy=False))
        self.units = hdf5.get_string(hazard[self.var['unit']])

        # number of centroids and events
        n_cen = len(self.centroids.id)
        n_event = len(self.event_id)

        # intensity
        try:
            self.intensity = hdf5.get_sparse_mat(hazard[self.var['inten']], \
                                                 (n_event, n_cen))
        except ValueError:
            print('Size missmatch in intensity matrix.')
            raise
        # fraction
        try:
            self.fraction = hdf5.get_sparse_mat(hazard[self.var['frac']], \
                                     (n_event, n_cen))
        except ValueError:
            print('Size missmatch in fraction matrix.')
            raise
        # Event names: set as event_id if no provided
        try:
            self.event_name = hdf5.get_list_str_from_ref(
                file_name, hazard[self.var['ev_name']])
        except KeyError:
            self.event_name = list(self.event_id)

    def _read_centroids(self, centroids=None):
        """Read centroids file if no centroids provided"""
        if centroids is None:
            self.centroids = CentroidsMat()
            self.centroids.field_name = 'hazard'
            self.centroids.var_names = self.var_cent
            self.centroids.read(self.tag.file_name, self.tag.description)
        else:
            self.centroids = centroids
