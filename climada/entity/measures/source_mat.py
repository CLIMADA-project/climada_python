"""
Define Measures from MATLAB file.
"""

import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.tag import Tag

# name of the enclosing variable, if present
SUP_FIELD_NAME = 'entity'
# name of excel sheet containing the data        
FIELD_NAME = 'measures'
# name of the table columns for each of the attributes
VAR_NAMES = {'name' : 'name',
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

def read(measures, file_name, description=None):
    """Read MATLAB file and store variables in measures. """
    from climada.entity.measures.base import Action

    # append the file name and description into the instance class
    measures.tag = Tag(file_name, description)

    # Load mat data
    meas = hdf5.read(file_name)
    try:
        meas = meas[SUP_FIELD_NAME]
    except KeyError:
        pass
    meas = meas[FIELD_NAME]

    # number of measures
    num_mes = len(meas[VAR_NAMES['name']])

    # iterate over each measure
    for idx in range(0, num_mes):
        act = Action()

        act.name = hdf5.get_str_from_ref(
            file_name, meas[VAR_NAMES['name']][idx][0])

        color_str = hdf5.get_str_from_ref(
            file_name, meas[VAR_NAMES['color']][idx][0])
        act.color_rgb = np.fromstring(color_str, dtype=float, sep=' ')
        act.cost = meas[VAR_NAMES['cost']][idx][0]
        act.hazard_freq_cutoff = meas[VAR_NAMES['haz_frq']][idx][0]
        act.hazard_event_set = hdf5.get_str_from_ref(
            file_name, meas[VAR_NAMES['haz_set']][idx][0])
        act.hazard_intensity = (meas[VAR_NAMES['haz_int_a']][idx][0], \
                                 meas[VAR_NAMES['haz_int_b']][0][idx])
        act.mdd_impact = (meas[VAR_NAMES['mdd_a']][idx][0],
                          meas[VAR_NAMES['mdd_b']][idx][0])
        act.paa_impact = (meas[VAR_NAMES['paa_a']][idx][0],
                          meas[VAR_NAMES['paa_b']][idx][0])
        act.risk_transf_attach = meas[VAR_NAMES['risk_att']][idx][0]
        act.risk_transf_cover = meas[VAR_NAMES['risk_cov']][idx][0]

        measures.add_action(act)
