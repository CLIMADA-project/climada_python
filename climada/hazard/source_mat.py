"""
Define Hazard reader function from a MATLAB file.
"""

import warnings
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.hazard.tag import Tag as TagHazard
import climada.util.hdf5_handler as hdf5

# Define tha name of the field that is read
FIELD_NAME = 'hazard'
# Define the names of the variables in field_namethat are read
VAR_NAME = {'per_id' : 'peril_ID',
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
VAR_CENT = {'cen_id' : 'centroid_ID',
            'lat' : 'lat',
            'lon' : 'lon'
           }

def read(hazard, file_name, haztype, description=None, centroids=None):
    """Read MATLAB file and store variables in hazard. """
    # Load hazard data
    data = hdf5.read(file_name)
    try:
        data = data[FIELD_NAME]
    except KeyError:
        warnings.warn("Variables are not under: %s." % FIELD_NAME)

    # Fill hazard tag
    haz_type = hdf5.get_string(data[VAR_NAME['per_id']])
    # Raise error if provided hazard type does not match with read one
    if haztype is not None and haz_type != haztype:
        raise ValueError('Hazard read is not of type: ' + haztype)
    hazard.tag = TagHazard(file_name, description, haz_type)

    # Set the centroids if given, otherwise load them from the same file
    read_centroids(hazard, centroids)

    # reshape from shape (x,1) to 1d array shape (x,)
    hazard.frequency = np.squeeze(data[VAR_NAME['freq']])
    hazard.event_id = np.squeeze(data[VAR_NAME['even_id']]. \
                               astype(int, copy=False))
    hazard.units = hdf5.get_string(data[VAR_NAME['unit']])

    # number of centroids and events
    n_cen = len(hazard.centroids.id)
    n_event = len(hazard.event_id)

    # intensity
    try:
        hazard.intensity = hdf5.get_sparse_mat(data[VAR_NAME['inten']], \
                                             (n_event, n_cen))
    except ValueError:
        print('Size missmatch in intensity matrix.')
        raise
    # fraction
    try:
        hazard.fraction = hdf5.get_sparse_mat(data[VAR_NAME['frac']], \
                                 (n_event, n_cen))
    except ValueError:
        print('Size missmatch in fraction matrix.')
        raise
    # Event names: set as event_id if no provided
    try:
        hazard.event_name = hdf5.get_list_str_from_ref(
            file_name, data[VAR_NAME['ev_name']])
    except KeyError:
        hazard.event_name = list(hazard.event_id)

def read_centroids(hazard, centroids=None):
    """Read centroids file if no centroids provided"""
    if centroids is None:
        hazard.centroids = Centroids()
        hazard.centroids.field_name = 'hazard'
        hazard.centroids.var_names = VAR_CENT
        hazard.centroids.read(hazard.tag.file_name, hazard.tag.description)
    else:
        hazard.centroids = centroids
