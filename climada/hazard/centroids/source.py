"""
Define Centroids reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['DEF_VAR_EXCEL',
           'DEF_VAR_MAT',
           'FILE_EXT',
           'read']

import os
import logging
import pandas as pd
import numpy as np

from climada.hazard.centroids.tag import Tag
import climada.util.hdf5_handler as hdf5
from climada.util.coordinates import IrregularGrid

DEF_VAR_MAT = {'field_names': ['centroids', 'hazard'],
               'var_name': {'cen_id' : 'centroid_ID',
                            'lat' : 'lat',
                            'lon' : 'lon',
                            'dist_coast': 'distance2coast_km',
                            'admin0_name': 'admin0_name',
                            'admin0_iso3': 'admin0_ISO3',
                            'comment': 'comment',
                            'region_id': 'NatId'
                           }
              }
""" MATLAB variable names """

DEF_VAR_EXCEL = {'sheet_name': 'centroids',
                 'col_name': {'cen_id' : 'centroid_ID',
                              'lat' : 'Latitude',
                              'lon' : 'Longitude',
                             }
                }
""" Excel variable names """

FILE_EXT = {'MAT':  '.mat',
            'XLS':  '.xls',
            'XLSX': '.xlsx'
           }
""" Supported files format to read from """

LOGGER = logging.getLogger(__name__)

def read(centroids, file_name, description='', var_names=None):
    """Read file and fill centroids.

    Parameters:
        centroids (Centroids): hazard to fill
        file_name (str): absolute path of the file to read
        description (str, optional): description of the data
        var_names (dict, optional): names of the variables in the file

    Raises:
        TypeError, KeyError, ValueError
    """
    centroids.tag = Tag(file_name, description)

    extension = os.path.splitext(file_name)[1]
    if extension == FILE_EXT['MAT']:
        try:
            read_mat(centroids, file_name, var_names)
        except (TypeError, KeyError) as var_err:
            LOGGER.error("Not existing variable. " + str(var_err))
            raise var_err
    elif (extension == FILE_EXT['XLS']) or (extension == FILE_EXT['XLSX']):
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

    dfr = pd.read_excel(file_name, var_names['sheet_name'])

    coord_cols = [var_names['col_name']['lat'], \
                  var_names['col_name']['lon']]

    centroids.coord = IrregularGrid(np.array(dfr[coord_cols]))
    centroids.id = dfr[var_names['col_name']['cen_id']].values. \
                    astype(int, copy=False)

def read_mat(centroids, file_name, var_names):
    """Read MATLAB file and store variables in centroids. """
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
            num_try += 1
    if num_try == len(var_names['field_names']):
        LOGGER.warning("Variables are not under: %s.", \
                       var_names['field_names'])

    cen_lat = np.squeeze(cent[var_names['var_name']['lat']])
    cen_lon = np.squeeze(cent[var_names['var_name']['lon']])
    centroids.coord = IrregularGrid(np.array([cen_lat, cen_lon]).transpose())
    centroids.id = np.squeeze(cent[var_names['var_name']['cen_id']]. \
                              astype(int, copy=False))

    try:
        centroids.dist_coast = \
                        np.squeeze(cent[var_names['var_name']['dist_coast']])
    except KeyError:
        pass
    try:
        centroids.admin0_name = hdf5.get_string(\
                                    cent[var_names['var_name']['admin0_name']])
    except KeyError:
        pass
    try:
        centroids.admin0_iso3 = hdf5.get_string(\
                                    cent[var_names['var_name']['admin0_ISO3']])
    except KeyError:
        pass
    try:
        comment = hdf5.get_string(cent[var_names['var_name']['comment']])
        if num_try == 0:
            centroids.tag.description += ' ' + comment
    except KeyError:
        pass
    try:
        centroids.region_id = \
                        np.squeeze(cent[var_names['var_name']['region_id']])
    except KeyError:
        pass
