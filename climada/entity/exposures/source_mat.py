"""
Define Exposures reader function from a MATLAB file.
"""

__all__ = ['SUP_FIELD_NAME', 'FIELD_NAME', 'VAR_NAMES', 'read']

import numpy as np

from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

# name of the enclosing variable, if present
SUP_FIELD_NAME = 'entity'
# name of variable containing the data
FIELD_NAME = 'assets'
# name of the variables in FIELD_NAME
VAR_NAMES = {'lat' : 'lat',
             'lon' : 'lon',
             'val' : 'Value',
             'ded' : 'Deductible',
             'cov' : 'Cover',
             'imp' : 'DamageFunID',
             'cat' : 'Category_ID',
             'reg' : 'Region_ID',
             'uni' : 'Value_unit',
             'ass' : 'centroid_index',
             'ref' : 'reference_year'
            }

def read(exposures, file_name, description=''):
    """Read MATLAB file and store variables in exposures. """
   # append the file name and description into the instance class
    exposures.tag = Tag(file_name, description)

    # Load mat data
    data = hdf5.read(file_name)
    try:
        data = data[SUP_FIELD_NAME]
    except KeyError:
        pass
    data = data[FIELD_NAME]

    # Fill variables
    _read_obligatory(exposures, data)
    _read_default(exposures, data)
    _read_optional(exposures, data, file_name)
    
    return exposures

def _read_obligatory(exposures, data):
    """Fill obligatory variables."""
    exposures.value = np.squeeze(data[VAR_NAMES['val']])

    coord_lat = data[VAR_NAMES['lat']]
    coord_lon = data[VAR_NAMES['lon']]
    exposures.coord = np.concatenate((coord_lat, coord_lon), axis=1)

    exposures.impact_id = np.squeeze(data[VAR_NAMES['imp']]).astype(int)

    # set exposures id according to appearance order
    num_exp = len(exposures.value)
    exposures.id = np.linspace(exposures.id.size, exposures.id.size + \
                          num_exp - 1, num_exp, dtype=int)

def _read_default(exposures, data):
    """Fill optional variables. Set default values."""
    num_exp = len(data[VAR_NAMES['val']])
    # get the exposures deductibles as np.array float 64
    # if not provided set default zero values
    exposures.deductible = _parse_default(data, VAR_NAMES['ded'], \
                                           np.zeros(num_exp))
    # get the exposures coverages as np.array float 64
    # if not provided set default exposure values
    exposures.cover = _parse_default(data, VAR_NAMES['cov'], exposures.value)

def _read_optional(exposures, data, file_name):
    """Fill optional parameters."""
    exposures.ref_year = _parse_optional(data, exposures.ref_year, \
                                         VAR_NAMES['ref'])
    if not isinstance(exposures.ref_year, int):
        exposures.ref_year = int(exposures.ref_year)

    exposures.category_id = _parse_optional(data, exposures.category_id, \
                                            VAR_NAMES['cat']).astype(int)
    exposures.region_id = _parse_optional(data, exposures.region_id, \
                                            VAR_NAMES['reg']).astype(int)
    exposures.assigned = _parse_optional(data, exposures.assigned, \
                                            VAR_NAMES['ass']).astype(int)
    try:
        exposures.value_unit = hdf5.get_str_from_ref(file_name, \
            data[VAR_NAMES['uni']][0][0])
    except KeyError:
        pass

def _parse_optional(hdf, var, var_name):
    """Retrieve optional variable, leave its original value if fail."""
    try:
        var = np.squeeze(hdf[var_name])
    except KeyError:
        pass
    return var

def _parse_default(hdf, var_name, def_val):
    """Retrieve optional variable, set default value if fail."""
    try:
        res = np.squeeze(hdf[var_name])
    except KeyError:
        res = def_val
    return res
