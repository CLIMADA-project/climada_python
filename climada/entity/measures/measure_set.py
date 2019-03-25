"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define MeasureSet class.
"""

__all__ = ['MeasureSet']

import copy
import logging
import numpy as np
import pandas as pd
import xlsxwriter

from climada.entity.measures.base import Measure
from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

LOGGER = logging.getLogger(__name__)

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'measures',
               'var_name': {'name' : 'name',
                            'color' : 'color',
                            'cost' : 'cost',
                            'haz_int_a' : 'hazard_intensity_impact_a',
                            'haz_int_b' : 'hazard_intensity_impact_b',
                            'haz_frq' : 'hazard_high_frequency_cutoff',
                            'haz_set' : 'hazard_event_set',
                            'mdd_a' : 'MDD_impact_a',
                            'mdd_b' : 'MDD_impact_b',
                            'paa_a' : 'PAA_impact_a',
                            'paa_b' : 'PAA_impact_b',
                            'fun_map' : 'damagefunctions_map',
                            'exp_set' : 'assets_file',
                            'exp_reg' : 'Region_ID',
                            'risk_att' : 'risk_transfer_attachement',
                            'risk_cov' : 'risk_transfer_cover',
                            'haz' : 'peril_ID'
                           }
              }
""" MATLAB variable names """

DEF_VAR_EXCEL = {'sheet_name': 'measures',
                 'col_name': {'name' : 'name',
                              'color' : 'color',
                              'cost' : 'cost',
                              'haz_int_a' : 'hazard intensity impact a',
                              'haz_int_b' : 'hazard intensity impact b',
                              'haz_frq' : 'hazard high frequency cutoff',
                              'haz_set' : 'hazard event set',
                              'mdd_a' : 'MDD impact a',
                              'mdd_b' : 'MDD impact b',
                              'paa_a' : 'PAA impact a',
                              'paa_b' : 'PAA impact b',
                              'fun_map' : 'damagefunctions map',
                              'exp_set' : 'assets file',
                              'exp_reg' : 'Region_ID',
                              'risk_att' : 'risk transfer attachement',
                              'risk_cov' : 'risk transfer cover',
                              'haz' : 'peril_ID'
                             }
                }
""" Excel variable names """

class MeasureSet():
    """Contains measures of type Measure. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        _data (dict): cotains Measure classes. It's not suppossed to be
            directly accessed. Use the class methods instead.
    """

    def __init__(self):
        """Empty initialization.

        Examples:
            Fill MeasureSet with values and check consistency data:

            >>> act_1 = Measure()
            >>> act_1.name = 'Seawall'
            >>> act_1.color_rgb = np.array([0.1529, 0.2510, 0.5451])
            >>> act_1.hazard_intensity = (1, 0)
            >>> act_1.mdd_impact = (1, 0)
            >>> act_1.paa_impact = (1, 0)
            >>> meas = MeasureSet()
            >>> meas.add_Measure(act_1)
            >>> meas.tag.description = "my dummy MeasureSet."
            >>> meas.check()

            Read measures from file and checks consistency data:

            >>> meas = MeasureSet()
            >>> meas.read_excel(ENT_TEMPLATE_XLS)
        """
        self.clear()

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self._data = dict() # {name: Measure()}

    def add_measure(self, meas):
        """Add an Measure.

        Parameters:
            meas (Measure): Measure instance

        Raises:
            ValueError
        """
        if not isinstance(meas, Measure):
            LOGGER.error("Input value is not of type Measure.")
            raise ValueError
        if not meas.name:
            LOGGER.error("Input Measure's name not set.")
            raise ValueError
        self._data[meas.name] = meas

    def remove_measure(self, name=None):
        """Remove Measure with provided name. Delete all Measures if no input
        name

        Parameters:
            name (str, optional): measure name

        Raises:
            ValueError
        """
        if name is not None:
            try:
                del self._data[name]
            except KeyError:
                LOGGER.warning('No Measure with name %s.', name)
        else:
            self._data = dict()

    def get_measure(self, name=None):
        """Get Measure with input name. Get all if no name provided.

        Parameters:
            name (str, optional): measure name

        Returns:
            list(Measure)
        """
        if name is not None:
            try:
                return self._data[name]
            except KeyError:
                return list()
        else:
            return list(self._data.values())

    def get_names(self):
        """Get all Measure names"""
        return list(self._data.keys())

    def num_measures(self):
        """Get number of measures contained """
        return len(self._data.keys())

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        for act_name, act in self._data.items():
            if (act_name != act.name) | (act.name == ''):
                raise ValueError('Wrong Measure.name: %s != %s' %\
                                (act_name, act.name))
            act.check()

    def append(self, meas):
        """Check and append measures of input MeasureSet to current MeasureSet.
        Overwrite Measure if same name.

        Parameters:
            meas (MeasureSet): MeasureSet instance to append

        Raises:
            ValueError
        """
        meas.check()
        if self.num_measures() == 0:
            self.__dict__ = copy.deepcopy(meas.__dict__)
            return

        self.tag.append(meas.tag)
        for measure in meas.get_measure():
            self.add_measure(measure)

    def read_mat(self, file_name, description='', var_names=DEF_VAR_MAT):
        """Read MATLAB file generated with previous MATLAB CLIMADA version.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            var_names (dict, optional): name of the variables in the file
        """
        def read_att_mat(measures, data, file_name, var_names):
            """Read MATLAB measures attributes"""
            num_mes = len(data[var_names['var_name']['name']])
            for idx in range(0, num_mes):
                meas = Measure()

                meas.name = hdf5.get_str_from_ref(
                    file_name, data[var_names['var_name']['name']][idx][0])

                color_str = hdf5.get_str_from_ref(
                    file_name, data[var_names['var_name']['color']][idx][0])
                meas.color_rgb = np.fromstring(color_str, dtype=float, sep=' ')
                meas.cost = data[var_names['var_name']['cost']][idx][0]
                meas.haz_type = hdf5.get_str_from_ref(
                    file_name, data[var_names['var_name']['haz']][idx][0])
                meas.hazard_freq_cutoff = data[var_names['var_name']['haz_frq']][idx][0]
                meas.hazard_set = hdf5.get_str_from_ref(file_name, \
                    data[var_names['var_name']['haz_set']][idx][0])
                try:
                    meas.hazard_inten_imp = ( \
                        data[var_names['var_name']['haz_int_a']][idx][0], \
                        data[var_names['var_name']['haz_int_b']][0][idx])
                except KeyError:
                    meas.hazard_inten_imp = ( \
                        data[var_names['var_name']['haz_int_a'][:-2]][idx][0], 0)

                # different convention of signes followed in MATLAB!
                meas.mdd_impact = (data[var_names['var_name']['mdd_a']][idx][0],
                                   data[var_names['var_name']['mdd_b']][idx][0])
                meas.paa_impact = (data[var_names['var_name']['paa_a']][idx][0],
                                   data[var_names['var_name']['paa_b']][idx][0])
                meas.imp_fun_map = hdf5.get_str_from_ref(file_name, \
                                   data[var_names['var_name']['fun_map']][idx][0])

                meas.exposures_set = hdf5.get_str_from_ref(
                    file_name, data[var_names['var_name']['exp_set']][idx][0])
                meas.exp_region_id = data[var_names['var_name']['exp_reg']][idx][0]

                meas.risk_transf_attach = data[var_names['var_name']['risk_att']][idx][0]
                meas.risk_transf_cover = data[var_names['var_name']['risk_cov']][idx][0]

                measures.add_measure(meas)

        data = hdf5.read(file_name)
        self.clear()
        self.tag.file_name = file_name
        self.tag.description = description
        try:
            data = data[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            data = data[var_names['field_name']]
            read_att_mat(self, data, file_name, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable %s", str(var_err))
            raise var_err

    def read_excel(self, file_name, description='', var_names=DEF_VAR_EXCEL):
        """Read excel file following template and store variables.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            var_names (dict, optional): name of the variables in the file
        """
        def read_att_excel(measures, dfr, var_names):
            """Read Excel measures attributes"""
            num_mes = len(dfr.index)
            for idx in range(0, num_mes):
                meas = Measure()

                meas.name = dfr[var_names['col_name']['name']][idx]
                try:
                    meas.haz_type = dfr[var_names['col_name']['haz']][idx]
                except KeyError:
                    pass
                meas.color_rgb = np.fromstring( \
                    dfr[var_names['col_name']['color']][idx], dtype=float, sep=' ')
                meas.cost = dfr[var_names['col_name']['cost']][idx]

                meas.hazard_freq_cutoff = dfr[var_names['col_name']['haz_frq']][idx]
                meas.hazard_set = dfr[var_names['col_name']['haz_set']][idx]
                # Search for (a, b) values, put a = 1 otherwise
                try:
                    meas.hazard_inten_imp = (dfr[var_names['col_name']['haz_int_a']][idx],\
                                             dfr[var_names['col_name']['haz_int_b']][idx])
                except KeyError:
                    meas.hazard_inten_imp = (1, dfr['hazard intensity impact'][idx])

                try:
                    meas.exposures_set = dfr[var_names['col_name']['exp_set']][idx]
                    meas.exp_region_id = dfr[var_names['col_name']['exp_reg']][idx]
                except KeyError:
                    pass

                meas.mdd_impact = (dfr[var_names['col_name']['mdd_a']][idx],
                                   dfr[var_names['col_name']['mdd_b']][idx])
                meas.paa_impact = (dfr[var_names['col_name']['paa_a']][idx],
                                   dfr[var_names['col_name']['paa_b']][idx])
                meas.imp_fun_map = dfr[var_names['col_name']['fun_map']][idx]
                meas.risk_transf_attach = dfr[var_names['col_name']['risk_att']][idx]
                meas.risk_transf_cover = dfr[var_names['col_name']['risk_cov']][idx]

                measures.add_measure(meas)

        dfr = pd.read_excel(file_name, var_names['sheet_name'])
        dfr = dfr.fillna('')
        self.clear()
        self.tag.file_name = file_name
        self.tag.description = description
        try:
            read_att_excel(self, dfr, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable: %s", str(var_err))
            raise var_err

    def write_excel(self, file_name, var_names=DEF_VAR_EXCEL):
        """ Write excel file following template.

        Parameters:
            file_name (str): absolute file name to write
            var_names (dict, optional): name of the variables in the file
        """
        def write_meas(row_ini, imp_ws, xls_data):
            """ Write one measure """
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
        row_ini = 1
        for meas_name, meas in self._data.items():
            xls_data = [meas_name, ' '.join(list(map(str, meas.color_rgb))),
                        meas.cost, meas.hazard_inten_imp[0],
                        meas.hazard_inten_imp[1], meas.hazard_freq_cutoff,
                        meas.hazard_set, meas.mdd_impact[0], meas.mdd_impact[1],
                        meas.paa_impact[0], meas.paa_impact[1], meas.imp_fun_map,
                        meas.exposures_set, meas.exp_region_id, meas.risk_transf_attach,
                        meas.risk_transf_cover, meas.haz_type]
            write_meas(row_ini, mead_ws, xls_data)
            row_ini += 1
        meas_wb.close()
