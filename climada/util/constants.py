"""
Define constants.
"""

import os

# working directory
WORKING_DIR = os.path.dirname(__file__) + '/../../'
# folder containing the demo data
DATA_DIR = WORKING_DIR + 'data/'

# Files containg data used as demo, e.g. for unit testing
HAZ_DEMO_XLS = DATA_DIR + 'demo/Excel_hazard.xlsx'
HAZ_DEMO_MAT = DATA_DIR + 'demo/atl_prob.mat'
ENT_DEMO_XLS = DATA_DIR + 'demo/demo_today.xlsx'

# mean one latitude in km
ONE_LAT_KM = 111.12
# earth radius in km
EARTH_RADIUS = 6371
