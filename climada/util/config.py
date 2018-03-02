"""
Define configuration parameters.
"""

__all__ = ['CONFIG',
           'setup_logging',
           'setup_conf_user'
          ]

import os
import json
import logging
import logging.config
from pkg_resources import Requirement, resource_filename

from climada.util.constants import SOURCE_DIR, DATA_DIR

WORKING_DIR = os.getcwd()
logging.basicConfig(level=logging.DEBUG)

def check_conf():
    """Check configuration files presence and generate folders if needed."""
    for key, path in CONFIG['local_data'].items():
        abspath = path
        if not os.path.isabs(abspath):
            abspath = os.path.abspath(os.path.join(WORKING_DIR, \
                                                   os.path.expanduser(path)))    
        if (key == "entity_def") and (path == ""):
            abspath = os.path.join(DATA_DIR, 'demo', 'entity_template.xlsx')
        CONFIG['local_data'][key] = abspath

CONFIG_DIR = os.path.abspath(os.path.join(SOURCE_DIR, 'conf'))

DEFAULT_PATH = os.path.abspath(os.path.join(CONFIG_DIR, 'defaults.conf'))
if not os.path.isfile(DEFAULT_PATH):
    DEFAULT_PATH = resource_filename(Requirement.parse('climada'), \
                                     'defaults.conf')
logging.debug('Loading config default file: %s', DEFAULT_PATH)

with open(DEFAULT_PATH) as def_file:
    CONFIG = json.load(def_file)

check_conf()


def setup_logging(default_level=logging.INFO):
    """Setup logging configuration"""
    default_path = os.path.abspath(os.path.join(CONFIG_DIR, 'logging.conf'))
    if not os.path.isfile(default_path):
        default_path = resource_filename(Requirement.parse('climada'), \
                                         'logging.conf')
    logging.debug('Loading logging config default file: %s', DEFAULT_PATH)

    path = default_path
    user_file = os.path.abspath(os.path.join(WORKING_DIR, 'climada_log.conf'))
    while not os.path.isfile(user_file) and user_file != '/climada_log.conf':
        user_file = os.path.abspath(os.path.join(user_file, os.pardir, \
                                            os.pardir, 'climada_log.conf'))
    if os.path.isfile(user_file):
        path = user_file
        logging.debug('Loading user logging config: %s ...', user_file)
    
    if os.path.exists(path):
        with open(path, 'rt') as log_file:
            config_log = json.load(log_file)
        logging.config.dictConfig(config_log)
    else:
        logging.basicConfig(level=default_level)

def setup_conf_user():
    """Setup climada configuration"""
    user_file = os.path.abspath(os.path.join(WORKING_DIR, 'climada.conf'))
    while not os.path.isfile(user_file) and user_file != '/climada.conf':
        user_file = os.path.abspath(os.path.join(user_file, os.pardir, \
                                                 os.pardir, 'climada.conf'))
    
    if os.path.isfile(user_file):
        logging.debug('Loading user config: %s ...', user_file)
        
        with open(user_file) as conf_file:
            userconfig = json.load(conf_file)
    
        if 'local_data' in userconfig.keys():
            CONFIG['local_data'].update(userconfig['local_data'])
    
        if 'present_ref_year' in userconfig.keys():
            CONFIG['present_ref_year'] = userconfig['present_ref_year']

        if 'future_ref_year' in userconfig.keys():
            CONFIG['future_ref_year'] = userconfig['future_ref_year']

        check_conf()  
