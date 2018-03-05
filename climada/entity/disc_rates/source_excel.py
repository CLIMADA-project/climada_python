"""
Define DiscRates reader function from Excel file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import pandas

from climada.entity.tag import Tag

# Name of excel sheet containing the data
# Name of the table columns for each of the attributes
DEF_VAR_NAME = {'sheet_name': 'discount',
                'col_name': {'year' : 'year',
                             'disc' : 'discount_rate'
                            }
               }

def read(disc_rates, file_name, description='', var_names=None):
    """Read excel file and store variables in disc_rates. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # append the file name and description into the instance class
    disc_rates.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, var_names['sheet_name'])

    # get the discount rates years
    disc_rates.years = dfr[var_names['col_name']['year']].values

    # get the discount rates for each year
    disc_rates.rates = dfr[var_names['col_name']['disc']].values
