"""
Define ImpactFuncSet reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['READ_SET']

import logging
import pandas
import numpy as np

from climada.entity.impact_funcs.impact_func import ImpactFunc
import climada.util.hdf5_handler as hdf5

DEF_VAR_EXCEL = {'sheet_name': 'damagefunctions',
                 'col_name': {'func_id' : 'DamageFunID',
                              'inten' : 'Intensity',
                              'mdd' : 'MDD',
                              'paa' : 'PAA',
                              'name' : 'name',
                              'unit' : 'Intensity_unit',
                              'peril' : 'peril_ID'
                             }
                }
""" Excel variable names """

DEF_VAR_MAT = {'sup_field_name': 'entity',
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
""" MATLAB variable names """

LOGGER = logging.getLogger(__name__)

def read_excel(imp_funcs, file_name, var_names):
    """Read excel file and store variables in imp_funcs. """
    if var_names is None:
        var_names = DEF_VAR_EXCEL

    try:
        dfr = pandas.read_excel(file_name, var_names['sheet_name'])
        read_att_excel(imp_funcs, dfr, var_names)
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

def read_mat(imp_funcs, file_name, var_names):
    """Read MATLAB file and store variables in imp_funcs. """
    if var_names is None:
        var_names = DEF_VAR_MAT

    imp = hdf5.read(file_name)

    try:
        imp = imp[var_names['sup_field_name']]
    except KeyError:
        pass

    try:
        imp = imp[var_names['field_name']]
        read_att_mat(imp_funcs, imp, file_name, var_names)
    except KeyError as err:
        LOGGER.error("Not existing variable: %s", str(err))
        raise err

def read_att_excel(imp_funcs, dfr, var_names):
    """Read impact functions' attributes from Excel file"""
    dist_func = _get_xls_funcs(dfr, var_names)
    for haz_type, imp_id in dist_func:
        df_func = dfr[dfr[var_names['col_name']['peril']] == haz_type]
        df_func = df_func[df_func[var_names['col_name']['func_id']] \
                          == imp_id]

        func = ImpactFunc()
        func.haz_type = haz_type
        func.id = imp_id
        # check that the unit of the intensity is the same
        try:
            if len(df_func[var_names['col_name']['name']].unique()) is not 1:
                raise ValueError('Impact function with two different names.')
            func.name = df_func[var_names['col_name']['name']].values[0]
        except KeyError:
            func.name = str(func.id)

        # check that the unit of the intensity is the same, if provided
        try:
            if len(df_func[var_names['col_name']['unit']].unique()) is not 1:
                raise ValueError('Impact function with two different \
                                 intensity units.')
            func.intensity_unit = \
                            df_func[var_names['col_name']['unit']].values[0]
        except KeyError:
            pass

        func.intensity = df_func[var_names['col_name']['inten']].values
        func.mdd = df_func[var_names['col_name']['mdd']].values
        func.paa = df_func[var_names['col_name']['paa']].values

        imp_funcs.add_func(func)

def read_att_mat(imp_funcs, imp, file_name, var_names):
    """Read impact functions' attributes from MATLAB file"""
    funcs_idx = _get_hdf5_funcs(imp, file_name, var_names)
    for imp_key, imp_rows in funcs_idx.items():
        func = ImpactFunc()
        func.haz_type = imp_key[0]
        func.id = imp_key[1]
        # check that this function only has one intensity unit, if provided
        try:
            func.intensity_unit = _get_hdf5_unit(imp, imp_rows, \
                                                file_name, var_names)
        except KeyError:
            pass
        # check that this function only has one name
        try:
            func.name = _get_hdf5_name(imp, imp_rows, file_name, var_names)
        except KeyError:
            func.name = str(func.id)
        func.intensity = np.take(imp[var_names['var_name']['inten']], imp_rows)
        func.mdd = np.take(imp[var_names['var_name']['mdd']], imp_rows)
        func.paa = np.take(imp[var_names['var_name']['paa']], imp_rows)

        imp_funcs.add_func(func)

def _get_xls_funcs(dfr, var_names):
    dist_func = []
    for (haz_type, imp_id) in zip(dfr[var_names['col_name']['peril']], \
    dfr[var_names['col_name']['func_id']]):
        if (haz_type, imp_id) not in dist_func:
            dist_func.append((haz_type, imp_id))
    return dist_func

def _get_hdf5_funcs(imp, file_name, var_names):
    """Get rows that fill every impact function and its name."""
    func_pos = dict()
    for row, (fun_id, fun_type) in enumerate(zip( \
    imp[var_names['var_name']['fun_id']].squeeze(), \
    imp[var_names['var_name']['peril']].squeeze())):
        type_str = hdf5.get_str_from_ref(file_name, fun_type)
        key = (type_str, int(fun_id))
        if key not in func_pos:
            func_pos[key] = list()
        func_pos[key].append(row)
    return func_pos

def _get_hdf5_unit(imp, idxs, file_name, var_names):
    """Get units of each value of an impact function. Check all the
    values are the same.

    Raises:
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

def _get_hdf5_name(imp, idxs, file_name, var_names):
    """Get name of each value of an impact function. Check all the
    values are the same.

    Raises:
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

READ_SET = {'XLS': (DEF_VAR_EXCEL, read_excel),
            'MAT': (DEF_VAR_MAT, read_mat)
           }
