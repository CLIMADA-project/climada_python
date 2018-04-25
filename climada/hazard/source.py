"""
Define Hazard reader functions from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['READ_SET']

import logging
import datetime as dt
import numpy as np
import pandas
from scipy import sparse

from climada.hazard.centroids.base import Centroids
import climada.util.hdf5_handler as hdf5

DEF_VAR_MAT = {'field_name': 'hazard',
               'var_name': {'per_id' : 'peril_ID',
                            'even_id' : 'event_ID',
                            'ev_name' : 'name',
                            'freq' : 'frequency',
                            'inten': 'intensity',
                            'unit': 'units',
                            'frac': 'fraction',
                            'comment': 'comment',
                            'datenum': 'datenum'
                           },
               'var_cent': {'field_names': ['centroids', 'hazard'],
                            'var_name': {'cen_id' : 'centroid_ID',
                                         'lat' : 'lat',
                                         'lon' : 'lon'
                                        }
                           }
              }
""" MATLAB variable names """

DEF_VAR_EXCEL = {'sheet_name': {'inten' : 'hazard_intensity',
                                'freq' : 'hazard_frequency'
                               },
                 'col_name': {'cen_id' : 'centroid_ID',
                              'even_id' : 'event_ID',
                              'freq' : 'frequency'
                             },
                 'col_centroids': {'sheet_name': 'centroids',
                                   'col_name': {'cen_id' : 'centroid_ID',
                                                'lat' : 'Latitude',
                                                'lon' : 'Longitude'
                                               }
                                  }
                }
""" Excel variable names """

LOGGER = logging.getLogger(__name__)

def read_mat(hazard, file_name, centroids, var_names):
    """Read MATLAB file and store variables in hazard."""
    if var_names is None:
        var_names = DEF_VAR_MAT
    try:
        data = hdf5.read(file_name)
        try:
            data = data[var_names['field_name']]
        except KeyError:
            pass

        new_haz = hdf5.get_string(data[var_names['var_name']['per_id']])
        hazard.tag.haz_type = new_haz

        read_centroids(hazard, centroids, var_names['var_cent'])

        read_att_mat(hazard, data, file_name, var_names)
    except KeyError as var_err:
        LOGGER.error("Not existing variable. " + str(var_err))
        raise var_err

def read_excel(hazard, file_name, centroids, var_names):
    """Read excel file and store variables in hazard. """
    if var_names is None:
        var_names = DEF_VAR_EXCEL
    try:
        read_centroids(hazard, centroids, var_names['col_centroids'])
        read_att_excel(hazard, file_name, var_names)
    except KeyError as var_err:
        LOGGER.error("Not existing variable. " + str(var_err))
        raise var_err

def read_att_mat(hazard, data, file_name, var_names):
    """ Read MATLAB hazard's attributes. """
    hazard.frequency = np.squeeze(data[var_names['var_name']['freq']])
    hazard.event_id = np.squeeze(data[var_names['var_name']['even_id']]. \
                                 astype(np.int, copy=False))
    hazard.units = hdf5.get_string(data[var_names['var_name']['unit']])

    n_cen = len(hazard.centroids.id)
    n_event = len(hazard.event_id)

    try:
        hazard.intensity = hdf5.get_sparse_csr_mat( \
                data[var_names['var_name']['inten']], (n_event, n_cen))
    except ValueError as err:
        LOGGER.error('Size missmatch in intensity matrix.')
        raise err
    try:
        hazard.fraction = hdf5.get_sparse_csr_mat( \
                data[var_names['var_name']['frac']], (n_event, n_cen))
    except ValueError as err:
        LOGGER.error('Size missmatch in fraction matrix.')
        raise err
    # Event names: set as event_id if no provided
    try:
        hazard.event_name = hdf5.get_list_str_from_ref(
            file_name, data[var_names['var_name']['ev_name']])
    except KeyError:
        hazard.event_name = list(hazard.event_id)
    try:
        comment = hdf5.get_string(data[var_names['var_name']['comment']])
        hazard.tag.description += ' ' + comment
    except KeyError:
        pass

    datenum = data[var_names['var_name']['datenum']].squeeze()
    hazard.date = np.array([(dt.datetime.fromordinal(int(date)) + \
         dt.timedelta(days=date%1)- \
         dt.timedelta(days=366)).toordinal() for date in datenum])

def read_att_excel(hazard, file_name, var_names):
    """ Read Excel hazard's attributes. """
    num_cen = len(hazard.centroids.id)

    dfr = pandas.read_excel(file_name, var_names['sheet_name']['freq'])

    num_events = dfr.shape[0]

    hazard.frequency = dfr[var_names['col_name']['freq']].values
    hazard.event_id = dfr[var_names['col_name']['even_id']].values. \
                        astype(int, copy=False)

    dfr = pandas.read_excel(file_name, var_names['sheet_name']['inten'])
    hazard.event_name = dfr.keys().values[1:].tolist()
    # number of events (ignore centroid_ID column)
    # check the number of events is the same as the one in the frequency
    if dfr.shape[1] - 1 is not num_events:
        LOGGER.error('Hazard intensity is given for a number of events ' \
                'different from the number of defined in its frequency: ' \
                '%s != %s', dfr.shape[1] - 1, num_events)
        raise ValueError
    # check number of centroids is the same as retrieved before
    if dfr.shape[0] is not num_cen:
        LOGGER.error('Hazard intensity is given for a number of centroids ' \
                'different from the number of centroids defined: %s != %s', \
                dfr.shape[0], num_cen)
        raise ValueError
    # check centroids ids are correct
    if not np.array_equal(dfr[var_names['col_name']['cen_id']].values,
                          hazard.centroids.id[-num_cen:]):
        LOGGER.error('Hazard intensity centroids ids do not match ' \
                     'previously defined centroids.')
        raise ValueError

    hazard.intensity = dfr.values[:, 1:num_events+1].transpose()
    hazard.intensity = sparse.csr_matrix(hazard.intensity)

    # Set fraction matrix to default value of 1
    hazard.fraction = sparse.csr_matrix(np.ones(hazard.intensity.shape, \
                                      dtype=np.float))

def read_centroids(hazard, centroids, var_names):
    """Read centroids file if no centroids provided"""
    if centroids is None:
        hazard.centroids = Centroids()
        hazard.centroids.read(hazard.tag.file_name, hazard.tag.description,
                              var_names)
    else:
        hazard.centroids = centroids

READ_SET = {'XLS': (DEF_VAR_EXCEL, read_excel),
            'MAT': (DEF_VAR_MAT, read_mat)
           }
