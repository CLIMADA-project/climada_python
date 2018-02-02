"""
Define DiscRates reader function from Excel file.
"""

import pandas

from climada.entity.tag import Tag

# Name of excel sheet containing the data
SHEET_NAME = 'discount'
# Name of the table columns for each of the attributes
COL_NAMES = {'year' : 'year',
             'disc' : 'discount_rate'
            }

def read(disc_rates, file_name, description=None):
    """Read excel file and store variables in disc_rates. """
    # append the file name and description into the instance class
    disc_rates.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, SHEET_NAME)

    # get the discount rates years
    disc_rates.years = dfr[COL_NAMES['year']].values

    # get the discount rates for each year
    disc_rates.rates = dfr[COL_NAMES['disc']].values
