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

Define MeasureSet class.
"""

__all__ = ['MeasureSet']

import ast
import copy
import logging
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter

from climada.entity.measures.base import Measure
from climada.entity.tag import Tag
import climada.util.hdf5_handler as u_hdf5

LOGGER = logging.getLogger(__name__)

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'measures',
               'var_name': {'name': 'name',
                            'color': 'color',
                            'cost': 'cost',
                            'haz_int_a': 'hazard_intensity_impact_a',
                            'haz_int_b': 'hazard_intensity_impact_b',
                            'haz_frq': 'hazard_high_frequency_cutoff',
                            'haz_set': 'hazard_event_set',
                            'mdd_a': 'MDD_impact_a',
                            'mdd_b': 'MDD_impact_b',
                            'paa_a': 'PAA_impact_a',
                            'paa_b': 'PAA_impact_b',
                            'fun_map': 'damagefunctions_map',
                            'exp_set': 'assets_file',
                            'exp_reg': 'Region_ID',
                            'risk_att': 'risk_transfer_attachement',
                            'risk_cov': 'risk_transfer_cover',
                            'haz': 'peril_ID'
                           }
              }
"""MATLAB variable names"""

DEF_VAR_EXCEL = {'sheet_name': 'measures',
                 'col_name': {'name': 'name',
                              'color': 'color',
                              'cost': 'cost',
                              'haz_int_a': 'hazard intensity impact a',
                              'haz_int_b': 'hazard intensity impact b',
                              'haz_frq': 'hazard high frequency cutoff',
                              'haz_set': 'hazard event set',
                              'mdd_a': 'MDD impact a',
                              'mdd_b': 'MDD impact b',
                              'paa_a': 'PAA impact a',
                              'paa_b': 'PAA impact b',
                              'fun_map': 'damagefunctions map',
                              'exp_set': 'assets file',
                              'exp_reg': 'Region_ID',
                              'risk_att': 'risk transfer attachement',
                              'risk_cov': 'risk transfer cover',
                              'risk_fact': 'risk transfer cost factor',
                              'haz': 'peril_ID'
                             }
                }
"""Excel variable names"""

class MeasureSet():
    """Contains measures of type Measure. Loads from
    files with format defined in FILE_EXT.

    Attributes
    ----------
    tag : climada.entity.tag.Tag
        information about the source data
    _data : dict
        Contains Measure objects. This attribute is not suppossed to be accessed directly.
        Use the available methods instead.
    """

    def __init__(
        self,
        measure_list: Optional[List[Measure]] = None,
        tag: Optional[Tag] = None,
    ):
        """Initialize a new MeasureSet object with specified data.

        Parameters
        ----------
        measure_list : list of Measure objects, optional
            The measures to include in the MeasureSet
        tag : Tag, optional
            Information about the source data

        Examples
        --------
        Fill MeasureSet with values and check consistency data:

        >>> act_1 = Measure(
        ...     name='Seawall',
        ...     color_rgb=np.array([0.1529, 0.2510, 0.5451]),
        ...     hazard_intensity=(1, 0),
        ...     mdd_impact=(1, 0),
        ...     paa_impact=(1, 0),
        ... )
        >>> meas = MeasureSet(
        ...     measure_list=[act_1],
        ...     tag=Tag(description="my dummy MeasureSet.")
        ... )
        >>> meas.check()

        Read measures from file and checks consistency data:

        >>> meas = MeasureSet.from_excel(ENT_TEMPLATE_XLS)
        """
        self.clear(tag=tag)
        if measure_list is not None:
            for meas in measure_list:
                self.append(meas)

    def clear(self, tag: Optional[Tag] = None, _data: Optional[dict] = None):
        """Reinitialize attributes.

        Parameters
        ----------
        tag : Tag, optional
            Information about the source data. If not given, an empty Tag object is used.
        _data : dict, optional
            A dict containing the Measure objects. For internal use only: It's not suppossed to be
            set directly. Use the class methods instead.
        """
        self.tag = tag if tag is not None else Tag()
        self._data = _data if _data is not None else dict()  # {hazard_type : {name: Measure()}}

    def append(self, meas):
        """Append an Measure. Override if same name and haz_type.

        Parameters
        ----------
        meas : Measure
            Measure instance

        Raises
        ------
        ValueError
        """
        if not isinstance(meas, Measure):
            raise ValueError("Input value is not of type Measure.")
        if not meas.haz_type:
            LOGGER.warning("Input Measure's hazard type not set.")
        if not meas.name:
            LOGGER.warning("Input Measure's name not set.")
        if meas.haz_type not in self._data:
            self._data[meas.haz_type] = dict()
        self._data[meas.haz_type][meas.name] = meas

    def remove_measure(self, haz_type=None, name=None):
        """Remove impact function(s) with provided hazard type and/or id.
        If no input provided, all impact functions are removed.

        Parameters
        ----------
        haz_type : str, optional
            all impact functions with this hazard
        name : str, optional
            measure name
        """
        if (haz_type is not None) and (name is not None):
            try:
                del self._data[haz_type][name]
            except KeyError:
                LOGGER.info("No Measure with hazard %s and id %s.",
                            haz_type, name)
        elif haz_type is not None:
            try:
                del self._data[haz_type]
            except KeyError:
                LOGGER.info("No Measure with hazard %s.", haz_type)
        elif name is not None:
            haz_remove = self.get_hazard_types(name)
            if not haz_remove:
                LOGGER.info("No Measure with name %s.", name)
            for haz in haz_remove:
                del self._data[haz][name]
        else:
            self._data = dict()

    def get_measure(self, haz_type=None, name=None):
        """Get ImpactFunc(s) of input hazard type and/or id.
        If no input provided, all impact functions are returned.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        name : str, optional
            measure name

        Returns
        -------
        Measure (if haz_type and name),
        list(Measure) (if haz_type or name),
        {Measure.haz_type : {Measure.name : Measure}} (if None)
        """
        if (haz_type is not None) and (name is not None):
            try:
                return self._data[haz_type][name]
            except KeyError:
                LOGGER.info("No Measure with hazard %s and id %s.",
                            haz_type, name)
                return list()
        elif haz_type is not None:
            try:
                return list(self._data[haz_type].values())
            except KeyError:
                LOGGER.info("No Measure with hazard %s.", haz_type)
                return list()
        elif name is not None:
            haz_return = self.get_hazard_types(name)
            if not haz_return:
                LOGGER.info("No Measure with name %s.", name)
            meas_return = []
            for haz in haz_return:
                meas_return.append(self._data[haz][name])
            return meas_return
        else:
            return self._data

    def get_hazard_types(self, meas=None):
        """Get measures hazard types contained for the name provided.
        Return all hazard types if no input name.

        Parameters
        ----------
        name : str, optional
            measure name

        Returns
        -------
        list(str)
        """
        if meas is None:
            return list(self._data.keys())

        haz_return = []
        for haz, haz_dict in self._data.items():
            if meas in haz_dict:
                haz_return.append(haz)
        return haz_return

    def get_names(self, haz_type=None):
        """Get measures names contained for the hazard type provided.
        Return all names for each hazard type if no input hazard type.

        Parameters
        ----------
        haz_type : str, optional
            hazard type from which to obtain the names

        Returns
        -------
        list(Measure.name) (if haz_type provided),
        {Measure.haz_type : list(Measure.name)} (if no haz_type)
        """
        if haz_type is None:
            out_dict = dict()
            for haz, haz_dict in self._data.items():
                out_dict[haz] = list(haz_dict.keys())
            return out_dict

        try:
            return list(self._data[haz_type].keys())
        except KeyError:
            LOGGER.info("No Measure with hazard %s.", haz_type)
            return list()

    def size(self, haz_type=None, name=None):
        """Get number of measures contained with input hazard type and
        /or id. If no input provided, get total number of impact functions.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        name : str, optional
            measure name

        Returns
        -------
        int
        """
        if (haz_type is not None) and (name is not None) and \
        (isinstance(self.get_measure(haz_type, name), Measure)):
            return 1
        if (haz_type is not None) or (name is not None):
            return len(self.get_measure(haz_type, name))
        return sum(len(meas_list) for meas_list in self.get_names().values())

    def check(self):
        """Check instance attributes.

        Raises
        ------
        ValueError
        """
        for key_haz, meas_dict in self._data.items():
            def_color = plt.cm.get_cmap('Greys', len(meas_dict))
            for i_meas, (name, meas) in enumerate(meas_dict.items()):
                if (name != meas.name) | (name == ''):
                    raise ValueError("Wrong Measure.name: %s != %s."
                                     % (name, meas.name))
                if key_haz != meas.haz_type:
                    raise ValueError("Wrong Measure.haz_type: %s != %s."
                                     % (key_haz, meas.haz_type))
                # set default color if not set
                if np.array_equal(meas.color_rgb, np.zeros(3)):
                    meas.color_rgb = def_color(i_meas)
                meas.check()

    def extend(self, meas_set):
        """Extend measures of input MeasureSet to current
        MeasureSet. Overwrite Measure if same name and haz_type.

        Parameters
        ----------
        impact_funcs : MeasureSet
            ImpactFuncSet instance to extend

        Raises
        ------
        ValueError
        """
        meas_set.check()
        if self.size() == 0:
            self.__dict__ = copy.deepcopy(meas_set.__dict__)
            return

        self.tag.append(meas_set.tag)

        new_func = meas_set.get_measure()
        for _, meas_dict in new_func.items():
            for _, meas in meas_dict.items():
                self.append(meas)

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

        Returns
        -------
        meas_set: climada.entity.MeasureSet()
            Measure Set from matlab file
        """
        if var_names is None:
            var_names = DEF_VAR_MAT
        def read_att_mat(measures, data, file_name, var_names):
            """Read MATLAB measures attributes"""
            num_mes = len(data[var_names['var_name']['name']])
            for idx in range(0, num_mes):
                color_str = u_hdf5.get_str_from_ref(
                    file_name, data[var_names['var_name']['color']][idx][0])

                try:
                    hazard_inten_imp = (
                        data[var_names['var_name']['haz_int_a']][idx][0],
                        data[var_names['var_name']['haz_int_b']][0][idx])
                except KeyError:
                    hazard_inten_imp = (
                        data[var_names['var_name']['haz_int_a'][:-2]][idx][0], 0)

                meas_kwargs = dict(
                    name=u_hdf5.get_str_from_ref(
                        file_name, data[var_names['var_name']['name']][idx][0]),
                    color_rgb=np.fromstring(color_str, dtype=float, sep=' '),
                    cost=data[var_names['var_name']['cost']][idx][0],
                    haz_type=u_hdf5.get_str_from_ref(
                        file_name, data[var_names['var_name']['haz']][idx][0]),
                    hazard_freq_cutoff=data[var_names['var_name']['haz_frq']][idx][0],
                    hazard_set=u_hdf5.get_str_from_ref(
                        file_name, data[var_names['var_name']['haz_set']][idx][0]),
                    hazard_inten_imp=hazard_inten_imp,
                    # different convention of signs followed in MATLAB!
                    mdd_impact=(data[var_names['var_name']['mdd_a']][idx][0],
                                data[var_names['var_name']['mdd_b']][idx][0]),
                    paa_impact=(data[var_names['var_name']['paa_a']][idx][0],
                                data[var_names['var_name']['paa_b']][idx][0]),
                    imp_fun_map=u_hdf5.get_str_from_ref(
                        file_name, data[var_names['var_name']['fun_map']][idx][0]),
                    exposures_set=u_hdf5.get_str_from_ref(
                        file_name, data[var_names['var_name']['exp_set']][idx][0]),
                    risk_transf_attach=data[var_names['var_name']['risk_att']][idx][0],
                    risk_transf_cover=data[var_names['var_name']['risk_cov']][idx][0],
                )

                exp_region_id = data[var_names['var_name']['exp_reg']][idx][0]
                if exp_region_id:
                    meas_kwargs["exp_region_id"] = [exp_region_id]

                measures.append(Measure(**meas_kwargs))

        data = u_hdf5.read(file_name)
        meas_set = cls()
        meas_set.tag.file_name = str(file_name)
        meas_set.tag.description = description
        try:
            data = data[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            data = data[var_names['field_name']]
            read_att_mat(meas_set, data, file_name, var_names)
        except KeyError as var_err:
            raise KeyError("Variable not in MAT file: " + str(var_err)) from var_err

        return meas_set

    def read_mat(self, *args, **kwargs):
        """This function is deprecated, use MeasureSet.from_mat instead."""
        LOGGER.warning("The use of MeasureSet.read_mat is deprecated."
                       "Use MeasureSet.from_mat instead.")
        self.__dict__ = MeasureSet.from_mat(*args, **kwargs).__dict__

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
        meas_set : climada.entity.MeasureSet
            Measures set from Excel
        """
        if var_names is None:
            var_names = DEF_VAR_EXCEL
        def read_att_excel(measures, dfr, var_names):
            """Read Excel measures attributes"""
            num_mes = len(dfr.index)
            for idx in range(0, num_mes):
                # Search for (a, b) values, put a=1 otherwise
                try:
                    hazard_inten_imp = (dfr[var_names['col_name']['haz_int_a']][idx],
                                        dfr[var_names['col_name']['haz_int_b']][idx])
                except KeyError:
                    hazard_inten_imp = (1, dfr['hazard intensity impact'][idx])

                meas_kwargs = dict(
                    name=dfr[var_names['col_name']['name']][idx],
                    cost=dfr[var_names['col_name']['cost']][idx],
                    hazard_freq_cutoff=dfr[var_names['col_name']['haz_frq']][idx],
                    hazard_set=dfr[var_names['col_name']['haz_set']][idx],
                    hazard_inten_imp=hazard_inten_imp,
                    mdd_impact=(dfr[var_names['col_name']['mdd_a']][idx],
                                dfr[var_names['col_name']['mdd_b']][idx]),
                    paa_impact=(dfr[var_names['col_name']['paa_a']][idx],
                                dfr[var_names['col_name']['paa_b']][idx]),
                    imp_fun_map=dfr[var_names['col_name']['fun_map']][idx],
                    risk_transf_attach=dfr[var_names['col_name']['risk_att']][idx],
                    risk_transf_cover=dfr[var_names['col_name']['risk_cov']][idx],
                    color_rgb=np.fromstring(
                        dfr[var_names['col_name']['color']][idx], dtype=float, sep=' '),
                )

                try:
                    meas_kwargs["haz_type"] = dfr[var_names['col_name']['haz']][idx]
                except KeyError:
                    pass

                try:
                    meas_kwargs["exposures_set"] = dfr[var_names['col_name']['exp_set']][idx]
                except KeyError:
                    pass

                try:
                    meas_kwargs["exp_region_id"] = ast.literal_eval(
                        dfr[var_names['col_name']['exp_reg']][idx])
                except KeyError:
                    pass
                except ValueError:
                    meas_kwargs["exp_region_id"] = dfr[var_names['col_name']['exp_reg']][idx]

                try:
                    meas_kwargs["risk_transf_cost_factor"] = (
                        dfr[var_names['col_name']['risk_fact']][idx]
                    )
                except KeyError:
                    pass

                measures.append(Measure(**meas_kwargs))

        dfr = pd.read_excel(file_name, var_names['sheet_name'])
        dfr = dfr.fillna('')
        meas_set = cls()
        meas_set.tag.file_name = str(file_name)
        meas_set.tag.description = description
        try:
            read_att_excel(meas_set, dfr, var_names)
        except KeyError as var_err:
            raise KeyError("Variable not in Excel file: " + str(var_err)) from var_err

        return meas_set

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use MeasureSet.from_excel instead."""
        LOGGER.warning("The use ofMeasureSet.read_excel is deprecated."
                       "Use MeasureSet.from_excel instead.")
        self.__dict__ = MeasureSet.from_excel(*args, **kwargs).__dict__

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
        def write_meas(row_ini, imp_ws, xls_data):
            """Write one measure"""
            for icol, col_dat in enumerate(xls_data):
                imp_ws.write(row_ini, icol, col_dat)

        meas_wb = xlsxwriter.Workbook(file_name)
        mead_ws = meas_wb.add_worksheet(var_names['sheet_name'])

        header = [var_names['col_name']['name'], var_names['col_name']['color'],
                  var_names['col_name']['cost'], var_names['col_name']['haz_int_a'],
                  var_names['col_name']['haz_int_b'], var_names['col_name']['haz_frq'],
                  var_names['col_name']['haz_set'], var_names['col_name']['mdd_a'],
                  var_names['col_name']['mdd_b'], var_names['col_name']['paa_a'],
                  var_names['col_name']['paa_b'], var_names['col_name']['fun_map'],
                  var_names['col_name']['exp_set'], var_names['col_name']['exp_reg'],
                  var_names['col_name']['risk_att'], var_names['col_name']['risk_cov'],
                  var_names['col_name']['haz']]
        for icol, head_dat in enumerate(header):
            mead_ws.write(0, icol, head_dat)
        for row_ini, (_, haz_dict) in enumerate(self._data.items(), 1):
            for meas_name, meas in haz_dict.items():
                xls_data = [meas_name, ' '.join(list(map(str, meas.color_rgb))),
                            meas.cost, meas.hazard_inten_imp[0],
                            meas.hazard_inten_imp[1], meas.hazard_freq_cutoff,
                            meas.hazard_set, meas.mdd_impact[0], meas.mdd_impact[1],
                            meas.paa_impact[0], meas.paa_impact[1], meas.imp_fun_map,
                            meas.exposures_set, str(meas.exp_region_id), meas.risk_transf_attach,
                            meas.risk_transf_cover, meas.haz_type]
            write_meas(row_ini, mead_ws, xls_data)
        meas_wb.close()
