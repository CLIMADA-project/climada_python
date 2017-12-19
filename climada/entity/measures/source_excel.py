"""
=====================
source_excel module
=====================

Define MeasuresExcelclass.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 16:46:01 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import numpy as np
import pandas

from climada.entity.measures.base import Measure, Measures
from climada.entity.tag import Tag

class MeasuresExcel(Measures):
    """Class that loads the measures from an excel file."""

    def __init__(self, file_name=None, description=None):
        """ Define the name of the sheet and columns names where the exposures
        are defined"""
        # Define tha name of the sheet that is read
        self.sheet_name = 'measures'
        # Define the names of the columns that are read
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
        """ Virtual class. Needs to be defined for each child."""

        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # number of exposures
        num_mes = len(dfr.index)

        # iterate over each measure
        for idx in range(0, num_mes):
            meas = Measure()
            if not self.data.keys():
                meas_id = 0
            elif idx == 0:
                meas_id = np.array(list(self.data.keys())).max() + 1
            else:
                meas_id = meas_id + 1

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

            self.data[meas_id] = meas
