"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Centroids reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['READ_SET']

import logging
import pandas as pd
import numpy as np

import climada.util.hdf5_handler as hdf5

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
                 'col_name': {'cen_id' : 'centroid_id',
                              'lat' : 'latitude',
                              'lon' : 'longitude',
                             }
                }
""" Excel variable names """

DEF_VAR_CSV = {'lat': 'X',
               'lon': 'Y',
               'region_id': 'iso_n3',
              }
""" CSV variable names """

LOGGER = logging.getLogger(__name__)

def read_excel(centroids, file_name, var_names):
    """Read excel file and store variables in centroids. """
    if var_names is None:
        var_names = DEF_VAR_EXCEL

    try:
        dfr = pd.read_excel(file_name, var_names['sheet_name'])
        coord_cols = [var_names['col_name']['lat'], \
                  var_names['col_name']['lon']]

        centroids.coord = np.array(dfr[coord_cols])
        centroids.id = dfr[var_names['col_name']['cen_id']].values. \
                    astype(int, copy=False)
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

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

    try:
        read_att_mat(centroids, cent, num_try, var_names)
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

def read_att_mat(centroids, cent, num_try, var_names):
    """Read impact functions' attributes from MATLAB file"""
    cen_lat = np.squeeze(cent[var_names['var_name']['lat']])
    cen_lon = np.squeeze(cent[var_names['var_name']['lon']])
    centroids.coord = np.array([cen_lat, cen_lon]).transpose()
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

def read_csv(centroids, file_name, var_names):
    """ Read csv centroids representations. Currently only supports lat/lon
        and region_id.
    """
    if var_names is None:
        var_names = DEF_VAR_CSV

    cent_pd = pd.read_csv(file_name)

    centroids.id = np.array(cent_pd.index)
    centroids.coord = np.array(cent_pd[[var_names['lat'],
                                        var_names['lon'],]])
    centroids.region_id = np.array(
        cent_pd[[var_names['region_id']]]
    )

    centroids.tag.file_name = file_name
    centroids.tag.description = 'Read from csv'

READ_SET = {'XLS': (DEF_VAR_EXCEL, read_excel),
            'MAT': (DEF_VAR_MAT, read_mat),
            'CSV': (DEF_VAR_CSV, read_csv),
           }
