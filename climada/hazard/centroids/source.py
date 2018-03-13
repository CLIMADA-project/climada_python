"""
Define Centroids general reader functions from specific format files.
"""

import os
import logging
import pandas
import numpy as np

from climada.hazard.centroids.tag import Tag
import climada.util.hdf5_handler as hdf5

DEF_VAR_MAT = {'field_names': ['centroids', 'hazard'],
               'var_name': {'cen_id' : 'centroid_ID',
                            'lat' : 'lat',
                            'lon' : 'lon'
                           }
              }

DEF_VAR_EXCEL = {'sheet_name': 'centroids',
                 'col_name': {'cen_id' : 'centroid_ID',
                              'lat' : 'Latitude',
                              'lon' : 'Longitude'
                             }
                }

LOGGER = logging.getLogger(__name__)

def read(centroids, file_name, description, var_names):
    """Read file and store variables in Centroids. """
    centroids.tag = Tag(file_name, description)
    
    extension = os.path.splitext(file_name)[1]
    if extension == '.mat':
        try:
            read_mat(centroids, file_name, var_names)
        except (TypeError, KeyError) as var_err:
            LOGGER.error("Not existing variable. " + str(var_err))
            raise var_err
    elif (extension == '.xlsx') or (extension == '.xls'):
        try:
            read_excel(centroids, file_name, var_names)
        except (TypeError, KeyError) as var_err:
            LOGGER.error("Not existing variable. " + str(var_err))
            raise var_err
    else:
        LOGGER.error("Input file extension not supported: %s.", extension)
        raise ValueError

def read_excel(centroids, file_name, var_names):
    """Read excel file and store variables in centroids. """
    if var_names is None:
        var_names = DEF_VAR_EXCEL
         
    dfr = pandas.read_excel(file_name, var_names['sheet_name'])
    
    coord_cols = [var_names['col_name']['lat'], \
                  var_names['col_name']['lon']]

    centroids.coord = np.array(dfr[coord_cols])
    centroids.id = dfr[var_names['col_name']['cen_id']].values

def read_mat(centroids, file_name, var_names):
    """Read MATLAB file and store variables in hazard. """
    if var_names is None:
        var_names = DEF_VAR_MAT
    
    cent = hdf5.read(file_name)
    # Try open encapsulating variable FIELD_NAMES
    num_try = 0
    for field in var_names['field_names']:
        try:
            cent = cent[field]
            break
        except KeyError:
            pass
        num_try += 1
    if num_try == len(var_names['field_names']):
        LOGGER.warning("Variables are not under: %s.", \
                       var_names['field_names'])

    cen_lat = np.squeeze(cent[var_names['var_name']['lat']])
    cen_lon = np.squeeze(cent[var_names['var_name']['lon']])
    centroids.coord = np.array([cen_lat, cen_lon]).transpose()
    centroids.id = np.squeeze(cent[var_names['var_name']['cen_id']]. \
                              astype(int, copy=False))
