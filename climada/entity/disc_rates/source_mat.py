"""
Define DiscRates reader function from MATLAB file.
"""

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

# name of the enclosing variable, if present       
SUP_FIELD_NAME = 'entity'
# name of variable containing the data
FIELD_NAME = 'discount'
# name of the variables in field_name
VAR_NAMES = {'year' : 'year',
             'disc' : 'discount_rate'
            }

def read(disc_rates, file_name, description=''):
    """Read MATLAB file and store variables in disc_rates. """
   # append the file name and description into the instance class
    disc_rates.tag = Tag(file_name, description)

    # Load mat data
    disc = hdf5.read(file_name)
    try:
        disc = disc[SUP_FIELD_NAME]
    except KeyError:
        pass
    disc = disc[FIELD_NAME]

    # get the discount rates years
    disc_rates.years = np.squeeze(disc[VAR_NAMES['year']]).astype(int)

    # get the discount rates for each year
    disc_rates.rates = np.squeeze(disc[VAR_NAMES['disc']])
