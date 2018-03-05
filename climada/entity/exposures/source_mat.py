"""
Define Exposures reader function from a MATLAB file.
"""

__all__ = ['DEF_VAR_NAME', 'read']

import numpy as np

from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

# name of the enclosing variable, if present. 
# name of each variable in the source file.
DEF_VAR_NAME = {'sup_field_name': 'entity',
                'field_name': 'assets',
                'var_name': {'lat' : 'lat',
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
               }

def read(exposures, file_name, description='', var_names=None):
    """Read MATLAB file and store variables in exposures. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME
        
   # append the file name and description into the instance class
    exposures.tag = Tag(file_name, description)

    # Load mat data
    data = hdf5.read(file_name)
    try:
        data = data[var_names['sup_field_name']]
    except KeyError:
        pass
    data = data[var_names['field_name']]

    # Fill variables
    _read_obligatory(exposures, data, var_names)
    _read_default(exposures, data, var_names)
    _read_optional(exposures, data, file_name, var_names)

def _read_obligatory(exposures, data, var_names):
    """Fill obligatory variables."""
    exposures.value = np.squeeze(data[var_names['var_name']['val']])

    coord_lat = data[var_names['var_name']['lat']]
    coord_lon = data[var_names['var_name']['lon']]
    exposures.coord = np.concatenate((coord_lat, coord_lon), axis=1)

    exposures.impact_id = np.squeeze(data[var_names['var_name']['imp']]). \
        astype(int)

    # set exposures id according to appearance order
    num_exp = len(exposures.value)
    exposures.id = np.linspace(exposures.id.size, exposures.id.size + \
                          num_exp - 1, num_exp, dtype=int)

def _read_default(exposures, data, var_names):
    """Fill optional variables. Set default values."""
    num_exp = len(data[var_names['var_name']['val']])
    # get the exposures deductibles as np.array float 64
    # if not provided set default zero values
    exposures.deductible = _parse_default(data, var_names['var_name']['ded'], \
                                           np.zeros(num_exp))
    # get the exposures coverages as np.array float 64
    # if not provided set default exposure values
    exposures.cover = _parse_default(data, var_names['var_name']['cov'], \
                                     exposures.value)

def _read_optional(exposures, data, file_name, var_names):
    """Fill optional parameters."""
    exposures.ref_year = _parse_optional(data, exposures.ref_year, \
                                         var_names['var_name']['ref'])
    if not isinstance(exposures.ref_year, int):
        exposures.ref_year = int(exposures.ref_year)

    exposures.category_id = _parse_optional(data, exposures.category_id, \
                            var_names['var_name']['cat']).astype(int)
    exposures.region_id = _parse_optional(data, exposures.region_id, \
                            var_names['var_name']['reg']).astype(int)
    exposures.assigned = _parse_optional(data, exposures.assigned, \
                            var_names['var_name']['ass']).astype(int)
    try:
        exposures.value_unit = hdf5.get_str_from_ref(file_name, \
            data[var_names['var_name']['uni']][0][0])
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
