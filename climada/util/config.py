"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define configuration parameters.
"""

__all__ = [
    'CONFIG',
    'setup_logging',
    'setup_conf_user',
]

import sys
import os
import json
import logging
from pkg_resources import Requirement, resource_filename

from climada.util.constants import SOURCE_DIR


WORKING_DIR = os.getcwd()
WINDOWS_END = WORKING_DIR[0:3]
UNIX_END = '/'

def remove_handlers(logger):
    """Remove logger handlers."""
    if logger.hasHandlers():
        for handler in logger.handlers:
            logger.removeHandler(handler)

LOGGER = logging.getLogger('climada')
LOGGER.setLevel(logging.DEBUG)
LOGGER.propagate = False
remove_handlers(LOGGER)
FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
CONSOLE = logging.StreamHandler(stream=sys.stdout)
CONSOLE.setFormatter(FORMATTER)
LOGGER.addHandler(CONSOLE)

def check_conf():
    """Check configuration files presence and generate folders if needed."""
    for key, path in CONFIG['local_data'].items():
        abspath = path
        if not os.path.isabs(abspath):
            abspath = os.path.abspath(os.path.join(WORKING_DIR,
                                                   os.path.expanduser(path)))
        if key == "save_dir":
            def_file = os.path.join(WORKING_DIR, "results")
        else:
            LOGGER.error("Configuration option %s not found.", key)

        if path == "":
            abspath = def_file
        elif not os.path.exists(abspath) and key != "save_dir":
            LOGGER.warning('Path %s not found. Default used: %s', abspath,
                           def_file)
            abspath = def_file

        CONFIG['local_data'][key] = abspath

CONFIG_DIR = os.path.abspath(os.path.join(SOURCE_DIR, 'conf'))

DEFAULT_PATH = os.path.abspath(os.path.join(CONFIG_DIR, 'defaults.conf'))
if not os.path.isfile(DEFAULT_PATH):
    DEFAULT_PATH = resource_filename(Requirement.parse('climada'),
                                     'defaults.conf')
with open(DEFAULT_PATH) as def_conf:
    LOGGER.debug('Loading default config file: %s', DEFAULT_PATH)
    CONFIG = json.load(def_conf)

check_conf()

def setup_logging(log_level='DEBUG'):
    """Setup logging configuration"""
    remove_handlers(LOGGER)
    LOGGER.propagate = False
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.addHandler(CONSOLE)

def setup_conf_user():
    """Setup climada configuration"""
    conf_name = 'climada.conf'
    user_file = os.path.abspath(os.path.join(WORKING_DIR, conf_name))
    while not os.path.isfile(user_file) and user_file != UNIX_END + conf_name \
            and user_file != WINDOWS_END + conf_name:
        user_file = os.path.abspath(os.path.join(user_file, os.pardir,
                                                 os.pardir, conf_name))

    if os.path.isfile(user_file):
        LOGGER.debug('Loading user config file: %s', user_file)

        with open(user_file) as conf_file:
            userconfig = json.load(conf_file)

        if 'local_data' in userconfig.keys():
            CONFIG['local_data'].update(userconfig['local_data'])

        if 'global' in userconfig.keys():
            CONFIG['global'] = userconfig['global']

        if 'entity' in userconfig.keys():
            CONFIG['entity'].update(userconfig['entity'])

        if 'trop_cyclone' in userconfig.keys():
            CONFIG['trop_cyclone'].update(userconfig['trop_cyclone'])

        if 'cost_benefit' in userconfig.keys():
            CONFIG['cost_benefit'] = userconfig['cost_benefit']

        check_conf()
