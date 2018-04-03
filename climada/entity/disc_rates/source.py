"""
Define DiscRates reader function from MATLAB file.
"""

__all__ = ['DEF_VAR_MAT',
           'DEF_VAR_EXCEL',
           'read'
          ]

import os
import logging
import numpy as np
import pandas

import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'discount',
               'var_name': {'year' : 'year',
                            'disc' : 'discount_rate'
                           }
              }

DEF_VAR_EXCEL = {'sheet_name': 'discount',
                 'col_name': {'year' : 'year',
                              'disc' : 'discount_rate'
                             }
                }

LOGGER = logging.getLogger(__name__)

def read(disc_rates, file_name, description, var_names):
    """Read file and store variables in disc_rates. """
    disc_rates.tag = Tag(file_name, description)
    
    extension = os.path.splitext(file_name)[1]
    if extension == '.mat':
        try:
            read_mat(disc_rates, file_name, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable. " + str(var_err))
            raise var_err   
    elif (extension == '.xlsx') or (extension == '.xls'):
        try:
            read_excel(disc_rates, file_name, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable. " + str(var_err))
            raise var_err   
    else:
        LOGGER.error('Not supported file extension: %s.', extension)
        raise ValueError

def read_mat(disc_rates, file_name, var_names):
    """Read MATLAB file and store variables in disc_rates. """
    if var_names is None:
        var_names = DEF_VAR_MAT
    
    disc = hdf5.read(file_name)
    try:
        disc = disc[var_names['sup_field_name']]
    except KeyError:
        pass
    disc = disc[var_names['field_name']]

    disc_rates.years = np.squeeze(disc[var_names['var_name']['year']]). \
                        astype(int, copy=False)
    disc_rates.rates = np.squeeze(disc[var_names['var_name']['disc']])

def read_excel(disc_rates, file_name, var_names):
    """Read excel file and store variables in disc_rates. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_EXCEL

    dfr = pandas.read_excel(file_name, var_names['sheet_name'])

    disc_rates.years = dfr[var_names['col_name']['year']].values. \
                        astype(int, copy=False)
    disc_rates.rates = dfr[var_names['col_name']['disc']].values
    