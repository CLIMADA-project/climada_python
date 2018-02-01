"""
Define ImpactFuncsExcel class.
"""

__all__ = ['ImpactFuncsExcel']

import pandas

from climada.entity.impact_funcs.base import Vulnerability, ImpactFuncs
from climada.entity.tag import Tag

class ImpactFuncsExcel(ImpactFuncs):
    """ImpactFuncs class loaded from an excel file.

    Attributes
    ----------
        sheet_name (str): name of excel sheet containing the data
        col_names (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """Extend ImpactFuncs __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> ImpactFuncsExcel()
            Initializes empty attributes.
            >>> ImpactFuncsExcel('filename')
            Loads data from the provided file.
            >>> ImpactFuncsExcel('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sheet_name = 'damagefunctions'
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
        """Override read Loader method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # number of impact functions
        names_func = dfr[self.col_names['name']].unique()
        num_func = len(names_func)

        # iterate over each impact function
        for idx in range(0, num_func):

            # select an impact function
            df_func = dfr[dfr[self.col_names['name']] == names_func[idx]]
            func = Vulnerability()
            # check that this function only represents one peril
            if len(df_func[self.col_names['peril']].unique()) is not 1:
                raise ValueError('Impact function with two different perils.')
            hazard = df_func[self.col_names['peril']].values[0]

            # load impact function values
            func.haz_type = hazard
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
