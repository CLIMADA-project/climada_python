"""
Define ImpactFuncs reader function from Excel file.
"""

import pandas

from climada.entity.tag import Tag

# name of excel sheet containing the data
SHEET_NAME = 'damagefunctions'
# name of the table columns for each of the attributes
COL_NAMES = {'func_id' : 'DamageFunID',
             'inten' : 'Intensity',
             'mdd' : 'MDD',
             'paa' : 'PAA',
             'name' : 'name',
             'unit' : 'Intensity_unit',
             'peril' : 'peril_ID'
            }
        
def read(imp_funcs, file_name, description=None):
    """Read excel file and store variables in imp_funcs. """
    from climada.entity.impact_funcs.base import Vulnerability

    # append the file name and description into the instance class
    imp_funcs.tag = Tag(file_name, description)
    
    # load Excel data
    dfr = pandas.read_excel(file_name, SHEET_NAME)
    
    # number of impact functions
    names_func = dfr[COL_NAMES['name']].unique()
    num_func = len(names_func)
    
    # iterate over each impact function
    for idx in range(0, num_func):
    
        # select an impact function
        df_func = dfr[dfr[COL_NAMES['name']] == names_func[idx]]

        func = Vulnerability()
        # check that this function only represents one peril
        if len(df_func[COL_NAMES['peril']].unique()) is not 1:
            raise ValueError('Impact function with two different perils.')
        hazard = df_func[COL_NAMES['peril']].values[0]
    
        # load impact function values
        func.haz_type = hazard
        # check that the impact function has a unique id
        if len(df_func[COL_NAMES['func_id']].unique()) is not 1:
            raise ValueError('Impact function with two different IDs.')
        func.id = df_func[COL_NAMES['func_id']].values[0]
        func.name = names_func[idx]
        # check that the unit of the intensity is the same
        if len(df_func[COL_NAMES['unit']].unique()) is not 1:
            raise ValueError('Impact function with two different \
                             intensity units.')
        func.intensity_unit = df_func[COL_NAMES['unit']].values[0]
    
        func.intensity = df_func[COL_NAMES['inten']].values
        func.mdd = df_func[COL_NAMES['mdd']].values
        func.paa = df_func[COL_NAMES['paa']].values
    
        imp_funcs.add_vulner(func)
