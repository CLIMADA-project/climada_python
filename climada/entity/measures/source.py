"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Measures reader function from a file with extension defined in
constant FILE_EXT.
"""

__all__ = ['READ_SET']

import logging
import numpy as np
import pandas

from climada.entity.measures.base import Measure
import climada.util.hdf5_handler as hdf5

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

LOGGER = logging.getLogger(__name__)

def read_mat(measures, file_name, var_names):
    """Read MATLAB file and store variables in measures."""
    if var_names is None:
        var_names = DEF_VAR_MAT

    data = hdf5.read(file_name)
    try:
        data = data[var_names['sup_field_name']]
    except KeyError:
        pass

    try:
        data = data[var_names['field_name']]
        read_att_mat(measures, data, file_name, var_names)
    except KeyError as var_err:
        LOGGER.error("Not existing variable %s", str(var_err))
        raise var_err

def read_excel(measures, file_name, var_names):
    """Read excel file and store variables in measures."""
    if var_names is None:
        var_names = DEF_VAR_EXCEL

    try:
        dfr = pandas.read_excel(file_name, var_names['sheet_name'])
        read_att_excel(measures, dfr, var_names)
    except KeyError as var_err:
        LOGGER.error("Not existing variable: %s", str(var_err))
        raise var_err

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

READ_SET = {'XLS': (DEF_VAR_EXCEL, read_excel),
            'MAT': (DEF_VAR_MAT, read_mat)
           }
