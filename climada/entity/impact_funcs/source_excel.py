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
    dist_func = _distinct_funcs(dfr, var_names)
    
    # iterate over each impact function
    for haz_type, imp_id in dist_func:
        # hazard type
        
        # select an impact function
        df_func = dfr[dfr[DEF_VAR_NAME['col_name']['peril']] == haz_type]
        df_func = df_func[df_func[DEF_VAR_NAME['col_name']['func_id']] \
                          == imp_id]

        # load impact function values
        func = Vulnerability()
        func.haz_type = haz_type
        func.id = imp_id
        # check that the unit of the intensity is the same
        if len(df_func[var_names['col_name']['name']].unique()) is not 1:
            raise ValueError('Impact function with two different names.')
        func.name = df_func[var_names['col_name']['name']].values[0]
        # check that the unit of the intensity is the same, if provided
        try:
            if len(df_func[var_names['col_name']['unit']].unique()) is not 1:
                raise ValueError('Impact function with two different \
                                 intensity units.')
            func.intensity_unit = \
                            df_func[var_names['col_name']['unit']].values[0]
        except KeyError:
            pass
    
        func.intensity = df_func[var_names['col_name']['inten']].values
        func.mdd = df_func[var_names['col_name']['mdd']].values
        func.paa = df_func[var_names['col_name']['paa']].values
    
        imp_funcs.add_vulner(func)

def _distinct_funcs(dfr, var_names):
    dist_func = []
    for (haz_type, imp_id) in zip(dfr[var_names['col_name']['peril']], \
    dfr[var_names['col_name']['func_id']]):
        if (haz_type, imp_id) not in dist_func:
            dist_func.append((haz_type, imp_id))
    return dist_func
