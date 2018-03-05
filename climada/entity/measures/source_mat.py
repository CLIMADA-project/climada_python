"""
Fill Measures from MATLAB file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import numpy as np

from climada.entity.measures.action import Action
import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

# name of the enclosing variable, if present. 
# name of each variable in the source file.
DEF_VAR_NAME = {'sup_field_name': 'entity',      
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
                             'risk_att' : 'risk_transfer_attachement',
                             'risk_cov' : 'risk_transfer_cover'
                            }
               }

def read(measures, file_name, description='', var_names=None):
    """Read MATLAB file and store variables in measures. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME
        
    # append the file name and description into the instance class
    measures.tag = Tag(file_name, description)

    # Load mat data
    meas = hdf5.read(file_name)
    try:
        meas = meas[var_names['sup_field_name']]
    except KeyError:
        pass
    meas = meas[var_names['field_name']]

    # number of measures
    num_mes = len(meas[var_names['var_name']['name']])

    # iterate over each measure
    for idx in range(0, num_mes):
        act = Action()

        act.name = hdf5.get_str_from_ref(
            file_name, meas[var_names['var_name']['name']][idx][0])

        color_str = hdf5.get_str_from_ref(
            file_name, meas[var_names['var_name']['color']][idx][0])
        act.color_rgb = np.fromstring(color_str, dtype=float, sep=' ')
        act.cost = meas[var_names['var_name']['cost']][idx][0]
        act.hazard_freq_cutoff = meas[var_names['var_name']['haz_frq']][idx][0]
        act.hazard_event_set = hdf5.get_str_from_ref(
            file_name, meas[var_names['var_name']['haz_set']][idx][0])
        act.hazard_intensity = ( \
                meas[var_names['var_name']['haz_int_a']][idx][0], \
                meas[var_names['var_name']['haz_int_b']][0][idx])
        act.mdd_impact = (meas[var_names['var_name']['mdd_a']][idx][0],
                          meas[var_names['var_name']['mdd_b']][idx][0])
        act.paa_impact = (meas[var_names['var_name']['paa_a']][idx][0],
                          meas[var_names['var_name']['paa_b']][idx][0])
        act.risk_transf_attach = \
                                meas[var_names['var_name']['risk_att']][idx][0]
        act.risk_transf_cover = meas[var_names['var_name']['risk_cov']][idx][0]

        measures.add_action(act)
