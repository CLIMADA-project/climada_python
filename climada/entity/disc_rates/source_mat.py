"""
Define DiscRates reader function from MATLAB file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

# name of the enclosing variable, if present       
# name of variable containing the data
# name of the variables in field_name
DEF_VAR_NAME = {'sup_field_name': 'entity',
                'field_name': 'discount',
                'var_name': {'year' : 'year',
                             'disc' : 'discount_rate'
                            }
               }

def read(disc_rates, file_name, description='', var_names=None):
    """Read MATLAB file and store variables in disc_rates. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME
        
   # append the file name and description into the instance class
    disc_rates.tag = Tag(file_name, description)

    # Load mat data
    disc = hdf5.read(file_name)
    try:
        disc = disc[var_names['sup_field_name']]
    except KeyError:
        pass
    disc = disc[var_names['field_name']]

    # get the discount rates years
    disc_rates.years = np.squeeze(disc[var_names['var_name']['year']]). \
                        astype(int)

    # get the discount rates for each year
    disc_rates.rates = np.squeeze(disc[var_names['var_name']['disc']])
