"""
=====================
source_csv module
=====================

Define ImpactFuncsCsv class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 16:46:01 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import pandas

from climada.entity.impact_funcs.base import ImpactFunc, ImpactFuncs
from climada.entity.tag import Tag

class ImpactFuncsExcel(ImpactFuncs):
    """Class that loads the impact functions from an excel file."""

    def __init__(self, file_name=None, description=None):
        """ Define the name of the sheet nad columns names where the exposures
        are defined"""

        # Define tha name of the sheet that is read
        self.sheet_name = 'damagefunctions'
        # Define the names of the columns that are read
        self.col_names = {'func_id' : 'DamageFunID',
                          'inten' : 'Intensity',
                          'mdd' : 'MDD',
                          'paa' : 'PAA',
                          'name' : 'name',
                          'unit' : 'Intensity_unit',
                          'peril' : 'peril_ID'
                         }

        # Initialize
        ImpactFuncs.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Virtual class. Needs to be defined for each child."""

        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # number of exposures
        names_func = dfr[self.col_names['name']].unique()
        num_func = len(names_func)

        # iterate over each measure
        for idx in range(0, num_func):

            # select an impact function
            df_func = dfr[dfr[self.col_names['name']] == names_func[idx]]
            func = ImpactFunc()
            # check that this function only represents one peril
            if len(df_func[self.col_names['peril']].unique()) is not 1:
                raise ValueError('Impact function with two different perils.')
            hazard = df_func[self.col_names['peril']].values[0]

            # load impact function values
            # check that the impact function has a unique id
            if len(df_func[self.col_names['func_id']].unique()) is not 1:
                raise ValueError('Impact function with two different IDs.')
            func.id = df_func[self.col_names['func_id']].values[0]
            func.name = names_func[idx]
            # check that the unit of the intensity is the same
            if len(df_func[self.col_names['unit']].unique()) is not 1:
                raise ValueError('Impact function with two different \
                                 intensity units.')
            func.intensity_unit = df_func[self.col_names['unit']].values[0]

            func.intensity = df_func[self.col_names['inten']].values
            func.mdd = df_func[self.col_names['mdd']].values
            func.paa = df_func[self.col_names['paa']].values

            # Save impact function
            if hazard not in self.data:
                self.data[hazard] = {}
            self.data[hazard][func.id] = func
