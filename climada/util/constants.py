"""
Define constants.
"""

__all__ = ['SOURCE_DIR',
           'DATA_DIR',
           'HAZ_DEMO_XLS',
           'HAZ_DEMO_MAT',
           'HAZ_TEST_MAT',
           'ENT_DEMO_XLS',
           'ENT_TEMPLATE_XLS',
           'ENT_DEMO_MAT',
           'ONE_LAT_KM',
           'EARTH_RADIUS']

import os

# climada directory
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                           os.pardir))

# folder containing the data
DATA_DIR = os.path.abspath(os.path.join(SOURCE_DIR, 'test', 'data'))

# Files containg data used as demo and unit testing
HAZ_DEMO_XLS = os.path.join(DATA_DIR, 'demo', 'Excel_hazard.xlsx')
HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
HAZ_TEST_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob_no_name.mat')
ENT_DEMO_XLS = os.path.join(DATA_DIR, 'demo', 'demo_today.xlsx')
ENT_TEMPLATE_XLS = os.path.join(DATA_DIR, 'demo', 'entity_template.xlsx')
ENT_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'demo_today.mat')

# Auxiliary files
SHAPES_MAT = os.path.join(DATA_DIR, 'system', 'admin0.mat')

# Mean one latitude in km
ONE_LAT_KM = 111.12

# Earth radius in km
EARTH_RADIUS = 6371
