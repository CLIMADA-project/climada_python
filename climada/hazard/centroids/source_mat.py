"""
Define Centroids reader function from a MATLAB file.
"""

import warnings
import numpy as np

from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

# Define tha name of the field that is read
FIELD_NAMES = ['centroids', 'hazard']
# Define the names of the variables in field_name that are read
VAR_NAME = {'cen_id' : 'centroid_ID',
            'lat' : 'lat',
            'lon' : 'lon'
           }

def read(centroids, file_name, description=''):
    """Read MATLAB file and store variables in hazard. """
    cent = hdf5.read(file_name)
    # Try open encapsulating variable FIELD_NAMES
    num_try = 0
    for field in FIELD_NAMES:
        try:
            cent = cent[field]
            break
        except KeyError:
            pass
        num_try += 1
    if num_try == len(FIELD_NAMES):
        warnings.warn("Variables are not under: %s." % FIELD_NAMES)

    centroids.tag = Tag(file_name, description)
    cen_lat = np.squeeze(cent[VAR_NAME['lat']])
    cen_lon = np.squeeze(cent[VAR_NAME['lon']])
    centroids.coord = np.array([cen_lat, cen_lon]).transpose()
    centroids.id = np.squeeze(cent[VAR_NAME['cen_id']]. \
    astype(int, copy=False))
