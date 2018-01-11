"""
Define MeasuresExcelclass.
"""

import numpy as np
import pandas

from climada.entity.measures.base import Measure, Measures
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

        # number of exposures
        num_mes = len(dfr.index)

        # iterate over each measure
        for idx in range(0, num_mes):
            meas = Measure()

            meas.name = dfr[self.col_names['name']][idx]
            meas.color_rgb = np.fromstring(dfr[self.col_names['color']][idx],
                                           dtype=float, sep=' ')
            meas.cost = dfr[self.col_names['cost']][idx]
            meas.hazard_freq_cutoff = dfr[self.col_names['haz_frq']][idx]
            meas.hazard_event_set = dfr[self.col_names['haz_set']][idx]
            meas.hazard_intensity = (1, dfr[self.col_names['haz_int']][idx])
            meas.mdd_impact = (dfr[self.col_names['mdd_a']][idx],
                               dfr[self.col_names['mdd_b']][idx])
            meas.paa_impact = (dfr[self.col_names['paa_a']][idx],
                               dfr[self.col_names['paa_b']][idx])
            meas.risk_transf_attach = dfr[self.col_names['risk_att']][idx]
            meas.risk_transf_cover = dfr[self.col_names['risk_cov']][idx]

            self.data.append(meas)
