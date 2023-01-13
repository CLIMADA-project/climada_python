"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define ImpactFuncSet class.
"""

__all__ = ['ImpactFuncSet']

import copy
import logging
from typing import Optional, Iterable
from itertools import repeat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.tag import Tag
import climada.util.plot as u_plot
import climada.util.hdf5_handler as u_hdf5

LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {'sheet_name': 'impact_functions',
                 'col_name': {'func_id': 'impact_fun_id',
                              'inten': 'intensity',
                              'mdd': 'mdd',
                              'paa': 'paa',
                              'name': 'name',
                              'unit': 'intensity_unit',
                              'peril': 'peril_id'
                             }
                }
"""Excel and csv variable names"""

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'damagefunctions',
               'var_name': {'fun_id': 'DamageFunID',
                            'inten': 'Intensity',
                            'mdd': 'MDD',
                            'paa': 'PAA',
                            'name': 'name',
                            'unit': 'Intensity_unit',
                            'peril': 'peril_ID'
                           }
              }
"""MATLAB variable names"""

class ImpactFuncSet:
    """Contains impact functions of type ImpactFunc. Loads from
    files with format defined in FILE_EXT.

    Attributes
    ----------
    tag : climada.entity.tag.Tag
        information about the source data
    _data : dict
        contains ImpactFunc classes. It's not suppossed to be
        directly accessed. Use the class methods instead.
    """

    def __init__(
        self,
        impact_funcs: Optional[Iterable[ImpactFunc]] = None,
        tag: Optional[Tag] = None
    ):
        """Initialization.

        Build an impact function set from an iterable of ImpactFunc.

        Parameters
        ----------
        impact_funcs : iterable of ImpactFunc, optional
            An iterable (list, set, array, ...) of ImpactFunc.
        tag : climada.entity.tag.Tag, optional
            The entity tag of this object.

        Examples
        --------
        Fill impact functions with values and check consistency data:

        >>> intensity = np.array([0, 20])
        >>> paa = np.array([0, 1])
        >>> mdd = np.array([0, 0.5])
        >>> fun_1 = ImpactFunc("TC", 3, intensity, mdd, paa)
        >>> imp_fun = ImpactFuncSet([fun_1])
        >>> imp_fun.check()

        Read impact functions from file and check data consistency.

        >>> imp_fun = ImpactFuncSet.from_excel(ENT_TEMPLATE_XLS)
        """
        # TODO: Automatically check this object if impact_funcs is not None.
        self.clear()
        if tag is not None:
            self.tag = tag
        if impact_funcs is not None:
            for impf in impact_funcs:
                self.append(impf)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self._data = dict()  # {hazard_type : {id:ImpactFunc}}

    def append(self, func):
        """Append a ImpactFunc. Overwrite existing if same id and haz_type.

        Parameters
        ----------
        func : ImpactFunc
            ImpactFunc instance

        Raises
        ------
        ValueError
        """
        if not isinstance(func, ImpactFunc):
            raise ValueError("Input value is not of type ImpactFunc.")
        if not func.haz_type:
            LOGGER.warning("Input ImpactFunc's hazard type not set.")
        if not func.id:
            LOGGER.warning("Input ImpactFunc's id not set.")
        if func.haz_type not in self._data:
            self._data[func.haz_type] = dict()
        self._data[func.haz_type][func.id] = func

    def remove_func(self, haz_type=None, fun_id=None):
        """Remove impact function(s) with provided hazard type and/or id.
        If no input provided, all impact functions are removed.

        Parameters
        ----------
        haz_type : str, optional
            all impact functions with this hazard
        fun_id : int, optional
            all impact functions with this id
        """
        if (haz_type is not None) and (fun_id is not None):
            try:
                del self._data[haz_type][fun_id]
            except KeyError:
                LOGGER.warning("No ImpactFunc with hazard %s and id %s.",
                               haz_type, fun_id)
        elif haz_type is not None:
            try:
                del self._data[haz_type]
            except KeyError:
                LOGGER.warning("No ImpactFunc with hazard %s.", haz_type)
        elif fun_id is not None:
            haz_remove = self.get_hazard_types(fun_id)
            if not haz_remove:
                LOGGER.warning("No ImpactFunc with id %s.", fun_id)
            for vul_haz in haz_remove:
                del self._data[vul_haz][fun_id]
        else:
            self._data = dict()

    def get_func(self, haz_type=None, fun_id=None):
        """Get ImpactFunc(s) of input hazard type and/or id.
        If no input provided, all impact functions are returned.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        fun_id : int, optional
            ImpactFunc id

        Returns
        -------
        ImpactFunc (if haz_type and fun_id),
        list(ImpactFunc) (if haz_type or fun_id),
        {ImpactFunc.haz_type : {ImpactFunc.id : ImpactFunc}} (if None)
        """
        if (haz_type is not None) and (fun_id is not None):
            try:
                return self._data[haz_type][fun_id]
            except KeyError:
                return list()
        elif haz_type is not None:
            try:
                return list(self._data[haz_type].values())
            except KeyError:
                return list()
        elif fun_id is not None:
            haz_return = self.get_hazard_types(fun_id)
            vul_return = []
            for vul_haz in haz_return:
                vul_return.append(self._data[vul_haz][fun_id])
            return vul_return
        else:
            return self._data

    def get_hazard_types(self, fun_id=None):
        """Get impact functions hazard types contained for the id provided.
        Return all hazard types if no input id.

        Parameters
        ----------
        fun_id : int, optional
            id of an impact function

        Returns
        -------
        list(str)
        """
        if fun_id is None:
            return list(self._data.keys())

        haz_types = []
        for vul_haz, vul_dict in self._data.items():
            if fun_id in vul_dict:
                haz_types.append(vul_haz)
        return haz_types

    def get_ids(self, haz_type=None):
        """Get impact functions ids contained for the hazard type provided.
        Return all ids for each hazard type if no input hazard type.

        Parameters
        ----------
        haz_type : str, optional
            hazard type from which to obtain the ids

        Returns
        -------
        list(ImpactFunc.id) (if haz_type provided),
        {ImpactFunc.haz_type : list(ImpactFunc.id)} (if no haz_type)
        """
        if haz_type is None:
            out_dict = dict()
            for vul_haz, vul_dict in self._data.items():
                out_dict[vul_haz] = list(vul_dict.keys())
            return out_dict

        try:
            return list(self._data[haz_type].keys())
        except KeyError:
            return list()

    def size(self, haz_type=None, fun_id=None):
        """Get number of impact functions contained with input hazard type and
        /or id. If no input provided, get total number of impact functions.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        fun_id : int, optional
            ImpactFunc id

        Returns
        -------
        int
        """
        if (haz_type is not None) and (fun_id is not None) and \
        (isinstance(self.get_func(haz_type, fun_id), ImpactFunc)):
            return 1
        if (haz_type is not None) or (fun_id is not None):
            return len(self.get_func(haz_type, fun_id))
        return sum(len(vul_list) for vul_list in self.get_ids().values())

    def check(self):
        """Check instance attributes.

        Raises
        ------
        ValueError
        """
        for key_haz, vul_dict in self._data.items():
            for fun_id, vul in vul_dict.items():
                if (fun_id != vul.id) | (fun_id == ''):
                    raise ValueError("Wrong ImpactFunc.id: %s != %s."
                                     % (fun_id, vul.id))
                if (key_haz != vul.haz_type) | (key_haz == ''):
                    raise ValueError("Wrong ImpactFunc.haz_type: %s != %s."
                                     % (key_haz, vul.haz_type))
                vul.check()

    def extend(self, impact_funcs):
        """Append impact functions of input ImpactFuncSet to current
        ImpactFuncSet. Overwrite ImpactFunc if same id and haz_type.

        Parameters
        ----------
        impact_funcs : ImpactFuncSet
            ImpactFuncSet instance to extend

        Raises
        ------
        ValueError
        """
        impact_funcs.check()
        if self.size() == 0:
            self.__dict__ = copy.deepcopy(impact_funcs.__dict__)
            return

        self.tag.append(impact_funcs.tag)

        new_func = impact_funcs.get_func()
        for _, vul_dict in new_func.items():
            for _, vul in vul_dict.items():
                self.append(vul)

    def plot(self, haz_type=None, fun_id=None, axis=None, **kwargs):
        """Plot impact functions of selected hazard (all if not provided) and
        selected function id (all if not provided).

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        fun_id : int, optional
            id of the function

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        num_plts = self.size(haz_type, fun_id)
        num_row, num_col = u_plot._get_row_col_size(num_plts)
        # Select all hazard types to plot
        if haz_type is not None:
            hazards = [haz_type]
        else:
            hazards = self._data.keys()

        if not axis:
            _, axis = plt.subplots(num_row, num_col)
        if num_plts > 1:
            axes = axis.flatten()
        else:
            axes = [axis]

        i_axis = 0
        for sel_haz in hazards:
            if fun_id is not None:
                self._data[sel_haz][fun_id].plot(axis=axes[i_axis], **kwargs)
                i_axis += 1
            else:
                for sel_id in self._data[sel_haz].keys():
                    self._data[sel_haz][sel_id].plot(axis=axes[i_axis], **kwargs)
                    i_axis += 1
        return axis

    @classmethod
    def from_excel(cls, file_name, description='', var_names=None):
        """Read excel file following template and store variables.

        Parameters
        ----------
        file_name : str
            absolute file name
        description : str, optional
            description of the data
        var_names : dict, optional
            name of the variables in the file

        Returns
        -------
        ImpactFuncSet
        """
        if var_names is None:
            var_names = DEF_VAR_EXCEL
        dfr = pd.read_excel(file_name, var_names['sheet_name'])

        imp_func_set = cls(tag=Tag(str(file_name), description))
        imp_func_set._fill_dfr(dfr, var_names)
        return imp_func_set

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use ImpactFuncSet.from_excel instead."""
        LOGGER.warning("The use of ImpactFuncSet.read_excel is deprecated."
                        " Use ImpactFuncSet.from_excel instead.")
        self.__dict__ = ImpactFuncSet.from_excel(*args, **kwargs).__dict__

    @classmethod
    def from_mat(cls, file_name, description='', var_names=None):
        """Read MATLAB file generated with previous MATLAB CLIMADA version.

        Parameters
        ----------
        file_name : str
            absolute file name
        description : str, optional
            description of the data
        var_names : dict, optional
            name of the variables in the file

        Return
        ------
        impf_set : climada.entity.impact_func_set.ImpactFuncSet
            Impact func set as defined in matlab file.
        """
        if var_names is None:
            var_names = DEF_VAR_MAT
        def _get_hdf5_funcs(imp, file_name, var_names):
            """Get rows that fill every impact function and its name."""
            func_pos = dict()
            for row, (fun_id, fun_type) in enumerate(
                    zip(imp[var_names['var_name']['fun_id']].squeeze(),
                        imp[var_names['var_name']['peril']].squeeze())):
                type_str = u_hdf5.get_str_from_ref(file_name, fun_type)
                key = (type_str, int(fun_id))
                if key not in func_pos:
                    func_pos[key] = list()
                func_pos[key].append(row)
            return func_pos

        def _get_hdf5_str(imp, idxs, file_name, var_name):
            """Get rows with same string in var_name."""
            prev_str = ""
            for row in idxs:
                cur_str = u_hdf5.get_str_from_ref(file_name, imp[var_name][row][0])
                if prev_str == "":
                    prev_str = cur_str
                elif prev_str != cur_str:
                    raise ValueError("Impact function with two different %s." % var_name)
            return prev_str

        imp = u_hdf5.read(file_name)

        try:
            imp = imp[var_names['sup_field_name']]
        except KeyError:
            pass
        try:
            imp = imp[var_names['field_name']]
            funcs_idx = _get_hdf5_funcs(imp, file_name, var_names)
            impact_funcs = []
            for imp_key, imp_rows in funcs_idx.items():
                # Store arguments in a dict (missing ones will be default)
                impf_kwargs = dict()
                impf_kwargs["haz_type"] = imp_key[0]
                impf_kwargs["id"] = imp_key[1]
                # check that this function only has one intensity unit, if provided
                try:
                    impf_kwargs["intensity_unit"] = _get_hdf5_str(
                        imp, imp_rows, file_name, var_names['var_name']['unit'])
                except KeyError:
                    pass
                # check that this function only has one name
                try:
                    impf_kwargs["name"] = _get_hdf5_str(
                        imp, imp_rows, file_name, var_names['var_name']['name'])
                except KeyError:
                    impf_kwargs["name"] = str(impf_kwargs["idx"])
                impf_kwargs["intensity"] = np.take(
                    imp[var_names['var_name']['inten']], imp_rows)
                impf_kwargs["mdd"] = np.take(imp[var_names['var_name']['mdd']], imp_rows)
                impf_kwargs["paa"] = np.take(imp[var_names['var_name']['paa']], imp_rows)
                impact_funcs.append(ImpactFunc(**impf_kwargs))
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

        return cls(impact_funcs, Tag(str(file_name), description))

    def read_mat(self, *args, **kwargs):
        """This function is deprecated, use ImpactFuncSet.from_mat instead."""
        LOGGER.warning("The use of ImpactFuncSet.read_mat  is deprecated."
                       "Use ImpactFuncSet.from_mat  instead.")
        self.__dict__ = ImpactFuncSet.from_mat(*args, **kwargs).__dict__

    def write_excel(self, file_name, var_names=None):
        """Write excel file following template.

        Parameters
        ----------
        file_name : str
            absolute file name to write
        var_names : dict, optional
            name of the variables in the file
        """
        if var_names is None:
            var_names = DEF_VAR_EXCEL
        def write_impf(row_ini, imp_ws, xls_data):
            """Write one impact function"""
            for icol, col_dat in enumerate(xls_data):
                for irow, data in enumerate(col_dat, row_ini):
                    imp_ws.write(irow, icol, data)

        imp_wb = xlsxwriter.Workbook(file_name)
        imp_ws = imp_wb.add_worksheet(var_names['sheet_name'])

        header = [var_names['col_name']['func_id'], var_names['col_name']['inten'],
                  var_names['col_name']['mdd'], var_names['col_name']['paa'],
                  var_names['col_name']['peril'], var_names['col_name']['unit'],
                  var_names['col_name']['name']]
        for icol, head_dat in enumerate(header):
            imp_ws.write(0, icol, head_dat)
        row_ini = 1
        for fun_haz_id, fun_haz in self._data.items():
            for fun_id, fun in fun_haz.items():
                n_inten = fun.intensity.size
                xls_data = [repeat(fun_id, n_inten), fun.intensity, fun.mdd,
                            fun.paa, repeat(fun_haz_id, n_inten),
                            repeat(fun.intensity_unit, n_inten),
                            repeat(fun.name, n_inten)]
                write_impf(row_ini, imp_ws, xls_data)
                row_ini += n_inten
        imp_wb.close()

    def _fill_dfr(self, dfr, var_names):

        def _get_xls_funcs(dfr, var_names):
            """Parse individual impact functions."""
            dist_func = []
            for (haz_type, imp_id) in zip(dfr[var_names['col_name']['peril']],
                                          dfr[var_names['col_name']['func_id']]):
                if (haz_type, imp_id) not in dist_func:
                    dist_func.append((haz_type, imp_id))
            return dist_func

        try:
            dist_func = _get_xls_funcs(dfr, var_names)
            for haz_type, imp_id in dist_func:
                df_func = dfr[dfr[var_names['col_name']['peril']] == haz_type]
                df_func = df_func[df_func[var_names['col_name']['func_id']]
                                  == imp_id]

                # Store arguments in a dict (missing ones will be default)
                impf_kwargs = dict()
                impf_kwargs["haz_type"] = haz_type
                impf_kwargs["id"] = imp_id
                # check that the unit of the intensity is the same
                try:
                    if len(df_func[var_names['col_name']['name']].unique()) != 1:
                        raise ValueError('Impact function with two different names.')
                    impf_kwargs["name"] = df_func[var_names['col_name']
                                                  ['name']].values[0]
                except KeyError:
                    impf_kwargs["name"] = str(impf_kwargs["id"])

                # check that the unit of the intensity is the same, if provided
                try:
                    if len(df_func[var_names['col_name']['unit']].unique()) != 1:
                        raise ValueError('Impact function with two different'
                                         ' intensity units.')
                    impf_kwargs["intensity_unit"] = df_func[var_names['col_name']
                                                            ['unit']].values[0]
                except KeyError:
                    pass

                impf_kwargs["intensity"] = df_func[var_names['col_name']['inten']].values
                impf_kwargs["mdd"] = df_func[var_names['col_name']['mdd']].values
                impf_kwargs["paa"] = df_func[var_names['col_name']['paa']].values

                self.append(ImpactFunc(**impf_kwargs))

        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err
