"""
Define ImpactFuncs reader function from MATLAB file.
"""

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

# name of the enclosing variable, if present
SUP_FIELD_NAME = 'entity'
# name of excel sheet containing the data
FIELD_NAME = 'damagefunctions'
# name of the table columns for each of the attributes
VAR_NAMES = {'fun_id' : 'DamageFunID',
             'inten' : 'Intensity',
             'mdd' : 'MDD',
             'paa' : 'PAA',
             'name' : 'name',
             'unit' : 'Intensity_unit',
             'peril' : 'peril_ID'
            }

def read(imp_funcs, file_name, description=''):
    """Read MATLAB file and store variables in imp_funcs. """
    from climada.entity.impact_funcs.base import Vulnerability

    # append the file name and description into the instance class
    imp_funcs.tag = Tag(file_name, description)

    # Load mat data
    imp = hdf5.read(file_name)
    try:
        imp = imp[SUP_FIELD_NAME]
    except KeyError:
        pass
    imp = imp[FIELD_NAME]

    # get the impact functions names and rows
    funcs_idx = get_funcs_rows(imp, file_name)

    # iterate over each impact function
    for imp_name, imp_rows in funcs_idx.items():
        # get impact function values
        func = Vulnerability()
        func.name = imp_name

        # check that this function only represents one peril
        hazard = get_imp_fun_hazard(imp, imp_rows, file_name)
        func.haz_type = hazard
        # check that this function only has one id
        func.id = get_imp_fun_id(imp, imp_rows)
        # check that this function only has one intensity unit
        func.intensity_unit = get_imp_fun_unit(imp, imp_rows, \
                                                    file_name)

        func.intensity = np.take(imp[VAR_NAMES['inten']], imp_rows)
        func.mdd = np.take(imp[VAR_NAMES['mdd']], imp_rows)
        func.paa = np.take(imp[VAR_NAMES['paa']], imp_rows)

        imp_funcs.add_vulner(func)
        
    return imp_funcs

def get_funcs_rows(imp, file_name):
    """Get rows that fill every impact function and its name."""
    func_pos = dict()
    it_fun = np.nditer(imp[VAR_NAMES['name']], flags=['refs_ok', 'c_index'])
    while not it_fun.finished:
        str_aux = hdf5.get_str_from_ref(file_name, \
                                        it_fun.itviews[0][it_fun.index])
        if str_aux not in func_pos.keys():
            func_pos[str_aux] = [it_fun.index]
        else:
            func_pos[str_aux].append(it_fun.index)
        it_fun.iternext()
    return func_pos

def get_imp_fun_hazard(imp, idxs, file_name):
    """Get hazard id of each value of an impact function. Check all the
    values are the same.

    Raises
    ------
        ValueError
    """
    prev_haz = ""
    for row in idxs:
        cur_haz = hdf5.get_str_from_ref(file_name, \
                                        imp[VAR_NAMES['peril']][row][0])
        if prev_haz == "":
            prev_haz = cur_haz
        elif prev_haz != cur_haz:
            raise ValueError('Impact function with two different perils.')
    return prev_haz

def get_imp_fun_id(imp, idxs):
    """Get function id of each value of an impact function. Check all the
    values are the same.

    Raises
    ------
        ValueError
    """
    fun_id = np.unique(np.take(imp[VAR_NAMES['fun_id']], idxs))
    if len(fun_id) != 1:
        raise ValueError('Impact function with two different IDs.')
    else:
        return int(fun_id)

def get_imp_fun_unit(imp, idxs, file_name):
    """Get units of each value of an impact function. Check all the
    values are the same.

    Raises
    ------
        ValueError
    """
    prev_unit = ""
    for row in idxs:
        cur_unit = hdf5.get_str_from_ref(file_name, \
                                         imp[VAR_NAMES['unit']][row][0])
        if prev_unit == "":
            prev_unit = cur_unit
        elif prev_unit != cur_unit:
            raise ValueError('Impact function with two different \
                             intensity units.')
    return prev_unit
