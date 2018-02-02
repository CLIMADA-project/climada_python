"""
Define MeasuresExcel class.
"""

__all__ = ['MeasuresExcel']

import numpy as np
import pandas

from climada.entity.measures.base import Action, Measures
from climada.entity.tag import Tag

class MeasuresExcel(Measures):
    """Measures class loaded from an excel file.

    Attributes
    ----------
        sheet_name (str): name of excel sheet containing the data
        col_names (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """Extend Measures __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> MeasuresExcel()
            Initializes empty attributes.
            >>> MeasuresExcel('filename')
            Loads data from the provided file.
            >>> MeasuresExcel('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sheet_name = 'measures'
        self.col_names = {'name' : 'name',
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
        # Initialize
        Measures.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # number of measures
        num_mes = len(dfr.index)

        # iterate over each measure
        for idx in range(0, num_mes):
            act = Action()

            act.name = dfr[self.col_names['name']][idx]
            act.color_rgb = np.fromstring(dfr[self.col_names['color']][idx],
                                          dtype=float, sep=' ')
            act.cost = dfr[self.col_names['cost']][idx]
            act.hazard_freq_cutoff = dfr[self.col_names['haz_frq']][idx]
            act.hazard_event_set = dfr[self.col_names['haz_set']][idx]
            # Search for (a, b) values, put a = 1 otherwise
            try:
                act.hazard_intensity = (1, \
                                         dfr[self.col_names['haz_int']][idx])
            except KeyError:
                col_name_a = self.col_names['haz_int'] + ' a'
                col_name_b = self.col_names['haz_int'] + ' b'
                act.hazard_intensity = (dfr[col_name_a][idx], \
                                         dfr[col_name_b][idx])
            act.mdd_impact = (dfr[self.col_names['mdd_a']][idx],
                              dfr[self.col_names['mdd_b']][idx])
            act.paa_impact = (dfr[self.col_names['paa_a']][idx],
                              dfr[self.col_names['paa_b']][idx])
            act.risk_transf_attach = dfr[self.col_names['risk_att']][idx]
            act.risk_transf_cover = dfr[self.col_names['risk_cov']][idx]

            self.add_action(act)
