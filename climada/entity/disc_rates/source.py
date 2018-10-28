"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along 
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define DiscRates reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['READ_SET']

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

    try:
        disc = disc[var_names['field_name']]
        disc_rates.years = np.squeeze(disc[var_names['var_name']['year']]). \
                        astype(int, copy=False)
        disc_rates.rates = np.squeeze(disc[var_names['var_name']['disc']])
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

def read_excel(disc_rates, file_name, var_names):
    """Read excel file and store variables in disc_rates. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_EXCEL

    try:
        dfr = pandas.read_excel(file_name, var_names['sheet_name'])
        disc_rates.years = dfr[var_names['col_name']['year']].values. \
                            astype(int, copy=False)
        disc_rates.rates = dfr[var_names['col_name']['disc']].values
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

READ_SET = {'XLS': (DEF_VAR_EXCEL, read_excel),
            'MAT': (DEF_VAR_MAT, read_mat)
           }
