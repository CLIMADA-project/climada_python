"""
Define constants.
"""

__all__ = ['WORKING_DIR',
           'DATA_DIR',
           'HAZ_DEMO_XLS',
           'HAZ_DEMO_MAT',
           'ENT_DEMO_XLS',
           'ENT_TEMPLATE_XLS',
           'ENT_DEMO_MAT',
           'ONE_LAT_KM',
           'EARTH_RADIUS']

import os

# working directory
WORKING_DIR = os.path.dirname(__file__) + '/../../'
# folder containing the demo data
DATA_DIR = WORKING_DIR + 'data/'

# Files containg data used as demo, e.g. for unit testing
HAZ_DEMO_XLS = DATA_DIR + 'demo/Excel_hazard.xlsx'
HAZ_DEMO_MAT = DATA_DIR + 'demo/atl_prob.mat'
ENT_DEMO_XLS = DATA_DIR + 'demo/demo_today.xlsx'
ENT_TEMPLATE_XLS = DATA_DIR + 'demo/entity_template.xlsx'
ENT_DEMO_MAT = DATA_DIR + 'demo/demo_today.mat'

SHAPES_MAT = DATA_DIR + 'system/admin0.mat'

# mean one latitude in km
ONE_LAT_KM = 111.12
# earth radius in km
EARTH_RADIUS = 6371
