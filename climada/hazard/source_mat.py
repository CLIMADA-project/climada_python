"""
Define HazardMat class.
"""

import numpy as np

from climada.hazard.base import Hazard
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard
import climada.util.hdf5_handler as hdf5

class HazardMat(Hazard):
    """Hazard class loaded from a mat file."""

    def __init__(self, file_name=None, description=None, haztype=None):
        """Extend Hazard __init__ method."""
        # Initialize
        Hazard.__init__(self, file_name, description, haztype)

    def _read(self, file_name, description=None, haztype=None, centroids=None):
        """Override _read Hazard method."""
        # Load hazard data
        hazard = hdf5.read(file_name)
        try:
            hazard = hazard['hazard']
        except KeyError:
            pass

        # Fill hazard tag
        haz_type = hdf5.get_string(hazard['peril_ID'])
        # Raise error if provided hazard type does not match with read one
        if haztype is not None and haz_type != haztype:
            raise ValueError('Hazard read is not of type: ' + haztype)
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
