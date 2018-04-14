"""
Define DiscRates reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['DEF_VAR_MAT',
           'DEF_VAR_EXCEL',
           'read_mat',
           'read_excel'
          ]

import logging
import numpy as np
import pandas

import climada.util.hdf5_handler as hdf5

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'discount',
               'var_name': {'year' : 'year',
                            'disc' : 'discount_rate'
                           }
              }
""" MATLAB variable names """

DEF_VAR_EXCEL = {'sheet_name': 'discount',
                 'col_name': {'year' : 'year',
                              'disc' : 'discount_rate'
                             }
                }
""" Excel variable names """

LOGGER = logging.getLogger(__name__)

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
