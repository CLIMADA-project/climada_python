"""
Define ImpactFuncsMat class.
"""

__all__ = ['ImpactFuncsMat']

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.impact_funcs.base import Vulnerability, ImpactFuncs
from climada.entity.tag import Tag

class ImpactFuncsMat(ImpactFuncs):
    """ImpactFuncs class loaded from an excel file.

    Attributes
    ----------
        sup_field_name (str): name of the enclosing variable, if present
        sheet_name (str): name of excel sheet containing the data
        col_names (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """Extend ImpactFuncs __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> ImpactFuncsMat()
            Initializes empty attributes.
            >>> ImpactFuncsMat('filename')
            Loads data from the provided file.
            >>> ImpactFuncsMat('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sup_field_name = 'entity'
        self.field_name = 'damagefunctions'
        self.var = {'fun_id' : 'DamageFunID',
                    'inten' : 'Intensity',
                    'mdd' : 'MDD',
                    'paa' : 'PAA',
                    'name' : 'name',
                    'unit' : 'Intensity_unit',
                    'peril' : 'peril_ID'
                   }
        # Initialize
        ImpactFuncs.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # Load mat data
        imp = hdf5.read(file_name)
        try:
            imp = imp[self.sup_field_name]
        except KeyError:
            pass
        imp = imp[self.field_name]

        # get the impact functions names and rows
        funcs_idx = self.get_funcs_rows(imp, file_name)

        # iterate over each impact function
        for imp_name, imp_rows in funcs_idx.items():
            # get impact function values
            func = Vulnerability()
            func.name = imp_name

            # check that this function only represents one peril
            hazard = self.get_imp_fun_hazard(imp, imp_rows, file_name)
            func.haz_type = hazard
            # check that this function only has one id
            func.id = self.get_imp_fun_id(imp, imp_rows)
            # check that this function only has one intensity unit
            func.intensity_unit = self.get_imp_fun_unit(imp, imp_rows, \
                                                        file_name)

            func.intensity = np.take(imp[self.var['inten']], imp_rows)
            func.mdd = np.take(imp[self.var['mdd']], imp_rows)
            func.paa = np.take(imp[self.var['paa']], imp_rows)

            self.add_vulner(func)

    def get_funcs_rows(self, imp, file_name):
        """Get rows that fill every impact function and its name."""
        func_pos = dict()
        it_fun = np.nditer(imp[self.var['name']], flags=['refs_ok', 'c_index'])
        while not it_fun.finished:
            str_aux = hdf5.get_str_from_ref(file_name, \
                                               it_fun.itviews[0][it_fun.index])
            if str_aux not in func_pos.keys():
                func_pos[str_aux] = [it_fun.index]
            else:
                func_pos[str_aux].append(it_fun.index)
            it_fun.iternext()
        return func_pos

    def get_imp_fun_hazard(self, imp, idxs, file_name):
        """Get hazard id of each value of an impact function. Check all the
        values are the same.

        Raises
        ------
            ValueError
        """
        prev_haz = ""
        for row in idxs:
            cur_haz = hdf5.get_str_from_ref(file_name, \
                                               imp[self.var['peril']][row][0])
            if prev_haz == "":
                prev_haz = cur_haz
            elif prev_haz != cur_haz:
                raise ValueError('Impact function with two different perils.')
        return prev_haz

    def get_imp_fun_id(self, imp, idxs):
        """Get function id of each value of an impact function. Check all the
        values are the same.

        Raises
        ------
            ValueError
        """
        fun_id = np.unique(np.take(imp[self.var['fun_id']], idxs))
        if len(fun_id) != 1:
            raise ValueError('Impact function with two different IDs.')
        else:
            return int(fun_id)

    def get_imp_fun_unit(self, imp, idxs, file_name):
        """Get units of each value of an impact function. Check all the
        values are the same.

        Raises
        ------
            ValueError
        """
        prev_unit = ""
        for row in idxs:
            cur_unit = hdf5.get_str_from_ref(file_name, \
                                                imp[self.var['unit']][row][0])
            if prev_unit == "":
                prev_unit = cur_unit
            elif prev_unit != cur_unit:
                raise ValueError('Impact function with two different \
                                 intensity units.')
        return prev_unit
