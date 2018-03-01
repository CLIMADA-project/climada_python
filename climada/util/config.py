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

from climada.util.constants import WORKING_DIR

def check_conf():
    """Check configuration files presence and generate folders if needed."""
    for key, path in CONFIG['local_data'].items():
        abspath = os.path.abspath(os.path.expanduser(path))
        CONFIG['local_data'][key] = abspath
        if "_dir" in key:
            logging.debug('CONFIG:%s: Checking presence of %s ...', key, \
                          abspath)
            if not os.path.isdir(abspath):
                os.mkdir(abspath)
                logging.debug('Created folder for %s (%s).', key, abspath)


CONFIG_DIR = os.path.abspath(os.path.join(WORKING_DIR, 'climada', 'config'))

DEFAULT_PATH = os.path.abspath(os.path.join(CONFIG_DIR, 'defaults.conf'))
if not os.path.isfile(DEFAULT_PATH):
    DEFAULT_PATH = resource_filename(Requirement.parse('climada'), \
                                     'defaults.conf')

logging.debug('Loading CONFIGFILE=%s', DEFAULT_PATH)

with open(DEFAULT_PATH) as def_file:
    CONFIG = json.load(def_file)

check_conf()


def setup_logging(default_level=logging.INFO):
    """Setup logging configuration"""
    default_path = os.path.abspath(os.path.join(CONFIG_DIR, 'logging.conf'))
    if not os.path.isfile(default_path):
        default_path = resource_filename(Requirement.parse('climada'), \
                                         'logging.conf')
        
    path = default_path
    user_file = os.path.abspath(os.path.join(WORKING_DIR, 'climada', \
                                             'climada_log.conf'))
    if os.path.isfile(user_file):
        path = user_file
    
    if os.path.exists(path):
        with open(path, 'rt') as log_file:
            config_log = json.load(log_file)
        logging.config.dictConfig(config_log)
    else:
        logging.basicConfig(level=default_level)

def setup_conf_user():
    """Setup climada configuration"""
    user_file = os.path.abspath(os.path.join(WORKING_DIR, 'climada', \
                                             'climada.conf'))
    
    if os.path.isfile(user_file):
        logging.debug('Loading user config: %s ...', user_file)
        
        with open(user_file) as conf_file:
            userconfig = json.load(conf_file)
    
        if 'local_data' in userconfig.keys():
            CONFIG['local_data'].update(userconfig['local_data'])
    
        if 'present_ref_year' in userconfig.keys():
            CONFIG['present_ref_year'].update(userconfig['present_ref_year'])

        if 'future_ref_year' in userconfig.keys():
            CONFIG['present_ref_year'].update(userconfig['present_ref_year'])

        check_conf()  
