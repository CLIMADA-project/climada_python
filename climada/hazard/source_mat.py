"""
Define Hazard reader function from a MATLAB file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import warnings
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.hazard.tag import Tag as TagHazard
import climada.util.hdf5_handler as hdf5

# name of the enclosing variable, if present. 
# name of each variable in the source file.
# Define tha names of the variables describing the centroids.
# Used only when the centroids are not provided and have to be read
# from the same file as the hazard
DEF_VAR_NAME = {'field_name': 'hazard',
                'var_name': {'per_id' : 'peril_ID',
                             'even_id' : 'event_ID',
                             'ev_name' : 'name',
                             'freq' : 'frequency',
                             'inten': 'intensity',
                             'unit': 'units',
                             'frac': 'fraction'
                            },
                'var_cent': {'cen_id' : 'centroid_ID',
                             'lat' : 'lat',
                             'lon' : 'lon'
                            }
               }

def read(hazard, file_name, haztype, description='', centroids=None, \
         var_names=None):
    """Read MATLAB file and store variables in hazard. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # Load hazard data
    data = hdf5.read(file_name)
    try:
        data = data[var_names['field_name']]
    except KeyError:
        warnings.warn("Variables are not under: %s." % var_names['field_name'])

    # Fill hazard tag
    haz_type = hdf5.get_string(data[var_names['var_name']['per_id']])
    # Raise error if provided hazard type does not match with read one
    if haztype is not None and haz_type != haztype:
        raise ValueError('Hazard read is not of type: ' + haztype)
    hazard.tag = TagHazard(file_name, haz_type, description)

    # Set the centroids if given, otherwise load them from the same file
    read_centroids(hazard, centroids, var_names)

    # reshape from shape (x,1) to 1d array shape (x,)
    hazard.frequency = np.squeeze(data[var_names['var_name']['freq']])
    hazard.event_id = np.squeeze(data[var_names['var_name']['even_id']]. \
                               astype(int, copy=False))
    hazard.units = hdf5.get_string(data[var_names['var_name']['unit']])

    # number of centroids and events
    n_cen = len(hazard.centroids.id)
    n_event = len(hazard.event_id)

    # intensity
    try:
        hazard.intensity = hdf5.get_sparse_csr_mat( \
                data[var_names['var_name']['inten']], (n_event, n_cen))
    except ValueError:
        print('Size missmatch in intensity matrix.')
        raise
    # fraction
    try:
        hazard.fraction = hdf5.get_sparse_csr_mat( \
                data[var_names['var_name']['frac']], (n_event, n_cen))
    except ValueError:
        print('Size missmatch in fraction matrix.')
        raise
    # Event names: set as event_id if no provided
    try:
        hazard.event_name = hdf5.get_list_str_from_ref(
            file_name, data[var_names['var_name']['ev_name']])
    except KeyError:
        hazard.event_name = list(hazard.event_id)

def read_centroids(hazard, centroids, var_names):
    """Read centroids file if no centroids provided"""
    if centroids is None:
        hazard.centroids = Centroids()
        hazard.centroids.field_name = 'hazard'
        hazard.centroids.var_names = var_names['var_cent']
        hazard.centroids.read_one(hazard.tag.file_name, hazard.tag.description)
    else:
        hazard.centroids = centroids
