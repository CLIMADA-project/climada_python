"""
Define ImpactFuncs reader function from MATLAB file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import logging
import numpy as np

from climada.entity.impact_funcs.vulnerability import Vulnerability
import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

# name of the enclosing variable, if present. 
# name of each variable in the source file.
DEF_VAR_NAME = {'sup_field_name': 'entity',
                'field_name': 'damagefunctions',
                'var_name': {'fun_id' : 'DamageFunID',
                             'inten' : 'Intensity',
                             'mdd' : 'MDD',
                             'paa' : 'PAA',
                             'name' : 'name',
                             'unit' : 'Intensity_unit',
                             'peril' : 'peril_ID'
                            }
               }

def read(imp_funcs, file_name, description='', var_names=None):
    """Read MATLAB file and store variables in imp_funcs. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # append the file name and description into the instance class
    imp_funcs.tag = Tag(file_name, description)

    # Load mat data
    imp = hdf5.read(file_name)
    try:
        imp = imp[var_names['sup_field_name']]
    except KeyError:
        pass
    imp = imp[var_names['field_name']]

    # get the impact functions names and rows
    funcs_idx = _get_funcs_rows(imp, file_name, var_names)

    # iterate over each impact function
    for imp_key, imp_rows in funcs_idx.items():
        # get impact function values
        func = Vulnerability()
        func.haz_type = imp_key[0]
        func.id = imp_key[1]
        # check that this function only has one intensity unit, if provided
        try:
            func.intensity_unit = _get_imp_fun_unit(imp, imp_rows, \
                                                file_name, var_names)
        except KeyError:
            pass
        # check that this function only has one name
        func.name = _get_imp_fun_name(imp, imp_rows, \
                                                file_name, var_names)        
        func.intensity = np.take(imp[var_names['var_name']['inten']], imp_rows)
        func.mdd = np.take(imp[var_names['var_name']['mdd']], imp_rows)
        func.paa = np.take(imp[var_names['var_name']['paa']], imp_rows)

        imp_funcs.add_vulner(func)

def _get_funcs_rows(imp, file_name, var_names):
    """Get rows that fill every impact function and its name."""
    func_pos = dict()
    for row, (fun_id, fun_type) in enumerate(zip( \
    imp[var_names['var_name']['fun_id']], \
    imp[var_names['var_name']['peril']])):
        type_str = hdf5.get_str_from_ref(file_name, fun_type[0])
        key = (type_str, fun_id[0])
        if key not in func_pos:
            func_pos[key] = list()
        func_pos[key].append(row)
    return func_pos

def _get_imp_fun_unit(imp, idxs, file_name, var_names):
    """Get units of each value of an impact function. Check all the
    values are the same.

    Raises
    ------
        ValueError
    """
    prev_unit = ""
    for row in idxs:
        cur_unit = hdf5.get_str_from_ref(file_name, \
                imp[var_names['var_name']['unit']][row][0])
        if prev_unit == "":
            prev_unit = cur_unit
        elif prev_unit != cur_unit:
            LOGGER.error("Impact function with two different intensity units.")
            raise ValueError
    return prev_unit

def _get_imp_fun_name(imp, idxs, file_name, var_names):
    """Get name of each value of an impact function. Check all the
    values are the same.

    Raises
    ------
        ValueError
    """
    prev_name = ""
    for row in idxs:
        cur_name = hdf5.get_str_from_ref(file_name, \
                imp[var_names['var_name']['name']][row][0])
        if prev_name == "":
            prev_name = cur_name
        elif prev_name != cur_name:
            LOGGER.error("Impact function with two different names.")
            raise ValueError
    return prev_name
