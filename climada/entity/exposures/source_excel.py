"""
Define Exposures reader function from an Excel file.
"""

import logging
from xlrd import XLRDError
import numpy as np
import pandas

from climada.entity.tag import Tag
from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

# Name of excel sheet containing the data
# Name of the table columns for each of the attributes
DEF_VAR_NAME = {'sheet_name': {'exp': 'assets',
                               'name': 'names'
                              },
                'col_name': {'lat' : 'Latitude',
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
               }

def read(exposures, file_name, description='', var_names=None):
    """Read excel file and store variables in exposures. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME
    
    # append the file name and description into the instance class
    exposures.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, var_names['sheet_name']['exp'])
    # get variables
    _read_obligatory(exposures, dfr, var_names)
    _read_default(exposures, dfr, var_names)
    _read_optional(exposures, dfr, file_name, var_names)

def _read_obligatory(exposures, dfr, var_names):
    """Fill obligatory variables."""
    exposures.value = dfr[var_names['col_name']['val']].values

    coord_cols = [var_names['col_name']['lat'], var_names['col_name']['lon']]
    exposures.coord = np.array(dfr[coord_cols])

    exposures.impact_id = dfr[var_names['col_name']['imp']].values

    # set exposures id according to appearance order
    num_exp = len(dfr.index)
    exposures.id = np.linspace(exposures.id.size, exposures.id.size + \
                          num_exp - 1, num_exp, dtype=int)

def _read_default(exposures, dfr, var_names):
    """Fill optional variables. Set default values."""
    # get the exposures deductibles as np.array float 64
    # if not provided set default zero values
    num_exp = len(dfr.index)
    exposures.deductible = _parse_default(dfr, var_names['col_name']['ded'], \
                                          np.zeros(num_exp))
    # get the exposures coverages as np.array float 64
    # if not provided set default exposure values
    exposures.cover = _parse_default(dfr, var_names['col_name']['cov'], \
                                     exposures.value)

def _read_optional(exposures, dfr, file_name, var_names):
    """Fill optional parameters."""
    exposures.category_id = _parse_optional(dfr, exposures.category_id, \
                                            var_names['col_name']['cat'])
    exposures.region_id = _parse_optional(dfr, exposures.region_id, \
                                          var_names['col_name']['reg'])
    exposures.value_unit = _parse_optional(dfr, exposures.value_unit, \
                                           var_names['col_name']['uni'])
    if not isinstance(exposures.value_unit, str):
        # Check all exposures have the same unit
        if len(np.unique(exposures.value_unit)) is not 1:
            LOGGER.error("Different value units provided for exposures.")
            raise ValueError
        exposures.value_unit = exposures.value_unit[0]
    exposures.assigned = _parse_optional(dfr, exposures.assigned, \
                                         var_names['col_name']['ass'])

    # check if reference year given under "names" sheet
    # if not, set default present reference year
    exposures.ref_year = _parse_ref_year(file_name, var_names)

def _parse_ref_year(file_name, var_names):
    """Retrieve reference year provided in the other sheet, if given."""
    try:
        dfr = pandas.read_excel(file_name, var_names['sheet_name']['name'])
        dfr.index = dfr[var_names['col_name']['item']]
        ref_year = dfr.loc[var_names['col_name']['ref']]['name']
    except (XLRDError, KeyError):
        ref_year = CONFIG['present_ref_year']
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
