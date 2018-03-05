"""
Define ImpactFuncs reader function from Excel file.
"""

__all__ = ['DEF_VAR_NAME',
           'read'
          ]

import pandas

from climada.entity.impact_funcs.vulnerability import Vulnerability
from climada.entity.tag import Tag

# Name of excel sheet containing the data
# Name of the table columns for each of the attributes
DEF_VAR_NAME = {'sheet_name': 'damagefunctions',
                'col_name': {'func_id' : 'DamageFunID',
                             'inten' : 'Intensity',
                             'mdd' : 'MDD',
                             'paa' : 'PAA',
                             'name' : 'name',
                             'unit' : 'Intensity_unit',
                             'peril' : 'peril_ID'
                            }
               }

def read(imp_funcs, file_name, description='', var_names=None):
    """Read excel file and store variables in imp_funcs. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # append the file name and description into the instance class
    imp_funcs.tag = Tag(file_name, description)
    
    # load Excel data
    dfr = pandas.read_excel(file_name, var_names['sheet_name'])
    
    # number of impact functions
    names_func = dfr[var_names['col_name']['name']].unique()
    num_func = len(names_func)
    
    # iterate over each impact function
    for idx in range(0, num_func):
    
        # select an impact function
        df_func = dfr[dfr[var_names['col_name']['name']] == names_func[idx]]

        func = Vulnerability()
        # check that this function only represents one peril
        if len(df_func[var_names['col_name']['peril']].unique()) is not 1:
            raise ValueError('Impact function with two different perils.')
        hazard = df_func[var_names['col_name']['peril']].values[0]
    
        # load impact function values
        func.haz_type = hazard
        # check that the impact function has a unique id
        if len(df_func[var_names['col_name']['func_id']].unique()) is not 1:
            raise ValueError('Impact function with two different IDs.')
        func.id = df_func[var_names['col_name']['func_id']].values[0]
        func.name = names_func[idx]
        # check that the unit of the intensity is the same
        if len(df_func[var_names['col_name']['unit']].unique()) is not 1:
            raise ValueError('Impact function with two different \
                             intensity units.')
        func.intensity_unit = df_func[var_names['col_name']['unit']].values[0]
    
        func.intensity = df_func[var_names['col_name']['inten']].values
        func.mdd = df_func[var_names['col_name']['mdd']].values
        func.paa = df_func[var_names['col_name']['paa']].values
    
        imp_funcs.add_vulner(func)
