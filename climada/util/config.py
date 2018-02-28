"""
Define configuration parameters.
"""

__all__ = ['REPO_DIR',
           'setup_logging']

import os
import json
import logging.config

from climada.util.constants import DATA_DIR, ENT_TEMPLATE_XLS

# Define folder containg repository data
REPO_DIR = DATA_DIR
CONFIG_DIR = DATA_DIR + '/config/'

HAZ_DEF_XLS = DATA_DIR + 'demo/Excel_hazard.xlsx'
HAZ_DEF_MAT = DATA_DIR + 'demo/atl_prob.mat'
ENT_DEF_XLS = ENT_TEMPLATE_XLS

# TODO JSON configuration file
config = {
    "present_ref_year": 2016,
    "future_ref_year": 2030
}

def setup_logging(default_path=CONFIG_DIR + 'logging.json', \
                  default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as file:
            config_log = json.load(file)
        logging.config.dictConfig(config_log)
    else:
        logging.basicConfig(level=default_level)
