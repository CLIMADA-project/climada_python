"""
Define Measures from Excel file.
"""

import numpy as np
import pandas

from climada.entity.tag import Tag

# name of the source file
SHEET_NAME = 'measures'
# description of the source data
COL_NAMES = {'name' : 'name',
             'color' : 'color',
             'cost' : 'cost',
             'haz_int' : 'hazard intensity impact',
             'haz_frq' : 'hazard high frequency cutoff',
             'haz_set' : 'hazard event set',
             'mdd_a' : 'MDD impact a',
             'mdd_b' : 'MDD impact b',
             'paa_a' : 'PAA impact a',
             'paa_b' : 'PAA impact b',
             'risk_att' : 'risk transfer attachement',
             'risk_cov' : 'risk transfer cover'
            }

def read(measures, file_name, description=''):
    """Read excel file and store variables in measures. """
    from climada.entity.measures.base import Action

    # append the file name and description into the instance class
    measures.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, SHEET_NAME)

    # number of measures
    num_mes = len(dfr.index)

    # iterate over each measure
    for idx in range(0, num_mes):
        act = Action()

        act.name = dfr[COL_NAMES['name']][idx]
        act.color_rgb = np.fromstring(dfr[COL_NAMES['color']][idx],
                                      dtype=float, sep=' ')
        act.cost = dfr[COL_NAMES['cost']][idx]
        act.hazard_freq_cutoff = dfr[COL_NAMES['haz_frq']][idx]
        act.hazard_event_set = dfr[COL_NAMES['haz_set']][idx]
        # Search for (a, b) values, put a = 1 otherwise
        try:
            act.hazard_intensity = (1, \
                                     dfr[COL_NAMES['haz_int']][idx])
        except KeyError:
            col_name_a = COL_NAMES['haz_int'] + ' a'
            col_name_b = COL_NAMES['haz_int'] + ' b'
            act.hazard_intensity = (dfr[col_name_a][idx], \
                                     dfr[col_name_b][idx])
        act.mdd_impact = (dfr[COL_NAMES['mdd_a']][idx],
                          dfr[COL_NAMES['mdd_b']][idx])
        act.paa_impact = (dfr[COL_NAMES['paa_a']][idx],
                          dfr[COL_NAMES['paa_b']][idx])
        act.risk_transf_attach = dfr[COL_NAMES['risk_att']][idx]
        act.risk_transf_cover = dfr[COL_NAMES['risk_cov']][idx]

        measures.add_action(act)
