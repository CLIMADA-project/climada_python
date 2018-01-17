"""
Define configuration parameters.
"""

from climada.util.constants import DATA_DIR, ENT_TEMPLATE_XLS

HAZ_DEF_XLS = DATA_DIR + 'demo/Excel_hazard.xlsx'
HAZ_DEF_MAT = DATA_DIR + 'demo/atl_prob.mat'
ENT_DEF_XLS = ENT_TEMPLATE_XLS

config = {
    "present_ref_year": 2016,
    "future_ref_year": 2030
}
