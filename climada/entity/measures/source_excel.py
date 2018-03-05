"""
Fill Measures from Excel file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import numpy as np
import pandas

from climada.entity.measures.action import Action
from climada.entity.tag import Tag

# Name of excel sheet containing the data
# Name of the table columns for each of the attributes
DEF_VAR_NAME = {'sheet_name': 'measures',
                'col_name': {'name' : 'name',
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
               }

def read(measures, file_name, description='', var_names=None):
    """Read excel file and store variables in measures. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # append the file name and description into the instance class
    measures.tag = Tag(file_name, description)

    # load Excel data
    dfr = pandas.read_excel(file_name, var_names['sheet_name'])

    # number of measures
    num_mes = len(dfr.index)

    # iterate over each measure
    for idx in range(0, num_mes):
        act = Action()

        act.name = dfr[var_names['col_name']['name']][idx]
        act.color_rgb = np.fromstring(dfr[var_names['col_name']['color']][idx],
                                      dtype=float, sep=' ')
        act.cost = dfr[var_names['col_name']['cost']][idx]
        act.hazard_freq_cutoff = dfr[var_names['col_name']['haz_frq']][idx]
        act.hazard_event_set = dfr[var_names['col_name']['haz_set']][idx]
        # Search for (a, b) values, put a = 1 otherwise
        try:
            act.hazard_intensity = (1, dfr[var_names['col_name']['haz_int']]\
                                    [idx])
        except KeyError:
            col_name_a = var_names['col_name']['haz_int'] + ' a'
            col_name_b = var_names['col_name']['haz_int'] + ' b'
            act.hazard_intensity = (dfr[col_name_a][idx], \
                                     dfr[col_name_b][idx])
        act.mdd_impact = (dfr[var_names['col_name']['mdd_a']][idx],
                          dfr[var_names['col_name']['mdd_b']][idx])
        act.paa_impact = (dfr[var_names['col_name']['paa_a']][idx],
                          dfr[var_names['col_name']['paa_b']][idx])
        act.risk_transf_attach = dfr[var_names['col_name']['risk_att']][idx]
        act.risk_transf_cover = dfr[var_names['col_name']['risk_cov']][idx]

        measures.add_action(act)
