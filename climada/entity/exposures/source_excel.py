"""
Define Exposures reader function from an Excel file.
"""

from xlrd import XLRDError
import numpy as np
import pandas

from climada.entity.tag import Tag
from climada.util.config import config

# name of excel sheet containing the data
SHEET_NAMES = {'exp': 'assets',
               'name': 'names'
              }
# name of the table columns for each of the attributes
COL_NAMES = {'lat' : 'Latitude',
             'lon' : 'Longitude',
             'val' : 'Value',
             'ded' : 'Deductible',
             'cov' : 'Cover',
             'imp' : 'DamageFunID',
             'cat' : 'Category_ID',
             'reg' : 'Region_ID',
             'uni' : 'Value unit',
             'ass' : 'centroid_index',
             'ref': 'reference_year',
             'item' : 'Item'
            }

def read(exposures, file_name, description=''):
    """Read excel file and store variables in exposures. """
    # append the file name and description into the instance class
    exposures.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, SHEET_NAMES['exp'])
    # get variables
    _read_obligatory(exposures, dfr)
    _read_default(exposures, dfr)
    _read_optional(exposures, dfr, file_name)

def _read_obligatory(exposures, dfr):
    """Fill obligatory variables."""
    exposures.value = dfr[COL_NAMES['val']].values

    coord_cols = [COL_NAMES['lat'], COL_NAMES['lon']]
    exposures.coord = np.array(dfr[coord_cols])

    exposures.impact_id = dfr[COL_NAMES['imp']].values

    # set exposures id according to appearance order
    num_exp = len(dfr.index)
    exposures.id = np.linspace(exposures.id.size, exposures.id.size + \
                          num_exp - 1, num_exp, dtype=int)

def _read_default(exposures, dfr):
    """Fill optional variables. Set default values."""
    # get the exposures deductibles as np.array float 64
    # if not provided set default zero values
    num_exp = len(dfr.index)
    exposures.deductible = _parse_default(dfr, COL_NAMES['ded'], \
                                          np.zeros(num_exp))
    # get the exposures coverages as np.array float 64
    # if not provided set default exposure values
    exposures.cover = _parse_default(dfr, COL_NAMES['cov'], exposures.value)

def _read_optional(exposures, dfr, file_name):
    """Fill optional parameters."""
    exposures.category_id = _parse_optional(dfr, exposures.category_id, \
                                            COL_NAMES['cat'])
    exposures.region_id = _parse_optional(dfr, exposures.region_id, \
                                          COL_NAMES['reg'])
    exposures.value_unit = _parse_optional(dfr, exposures.value_unit, \
                                           COL_NAMES['uni'])
    if not isinstance(exposures.value_unit, str):
        # Check all exposures have the same unit
        if len(np.unique(exposures.value_unit)) is not 1:
            raise ValueError('Different value units provided for \
                             exposures.')
        exposures.value_unit = exposures.value_unit[0]
    exposures.assigned = _parse_optional(dfr, exposures.assigned, \
                                         COL_NAMES['ass'])

    # check if reference year given under "names" sheet
    # if not, set default present reference year
    exposures.ref_year = _parse_ref_year(file_name)

def _parse_ref_year(file_name):
    """Retrieve reference year provided in the other sheet, if given."""
    try:
        dfr = pandas.read_excel(file_name, SHEET_NAMES['name'])
        dfr.index = dfr[COL_NAMES['item']]
        ref_year = dfr.loc[COL_NAMES['ref']]['name']
    except (XLRDError, KeyError):
        ref_year = config['present_ref_year']
    return ref_year

def _parse_optional(dfr, var, var_name):
    """Retrieve optional variable, leave its original value if fail."""
    try:
        var = dfr[var_name].values
    except KeyError:
        pass
    return var

def _parse_default(dfr, var_name, def_val):
    """Retrieve optional variable, set default value if fail."""
    try:
        res = dfr[var_name].values
    except KeyError:
        res = def_val
    return res
