"""
Define configuration parameters.
"""

from climada.util.constants import DATA_DIR, ENT_TEMPLATE_XLS

# Define folder containg repository data
REPO_DIR = DATA_DIR

HAZ_DEF_XLS = DATA_DIR + 'demo/Excel_hazard.xlsx'
HAZ_DEF_MAT = DATA_DIR + 'demo/atl_prob.mat'
ENT_DEF_XLS = ENT_TEMPLATE_XLS

# TODO JSON configuration file
config = {
    "present_ref_year": 2016,
    "future_ref_year": 2030
}
