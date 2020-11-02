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
import re
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


class Config(object):
    """Convenience Class.
    A Config object is a slow JSON object like dictonary who's values can be accessed by their names right away.
    E.g.: `a.b.c.str()` instead of `a['b']['c']`
    """
    def __str__(self):
        # pylint: disable=bare-except,multiple-statements
        try: return self.str()
        except: pass
        try: return str(self.int())
        except: pass
        try: return str(self.float())
        except: pass
        try: return str(self.list())
        except: pass
        return '{{{}}}'.format(", ".join([
            f'{k}: {v}' for (k, v) in self.__dict__.items() if not k=='_root'
        ]))

    def __repr__(self):
        return self.__str__()

    def __init__(self, val=None, root=None):
        """
        Parameters
        ----------
        val : [float, int, str, list], optional
            the value of the Config in case it's basic, by default None,
            when a dictionary like object is created
        """
        if val is not None:
            self._val = val
        if root is None:
            self._root = self
        else:
            self._root = root

    def str(self, index=None):
        """
        Returns
        -------
        str
            the value of this Config if it is a string

        Raises
        ------
        Exception
            if it is not a string
        """
        def feval(root, cstr):
            def expand(dct, lst):
                if len(lst) == 1:
                    return dct.__getattribute__(lst[0]).str()
                else:
                    return expand(dct.__getattribute__(lst[0]), lst[1:])
            def msub(m):
                cpath = m.group(1).split('.')
                return expand(root, cpath)
            return re.sub(r'{([\w\.]+)}', msub, cstr)

        if index is None:
            if self._val.__class__ is str:
                return feval(self._root, self._val)
            raise Exception(f"{self._val.__class__}, not str")
        else:
            if self._val.__class__ is list:
                return self._val[index].str()
            raise Exception(f"{self._val.__class__}, not list")

    def int(self, index=None):
        """
        Returns
        -------
        int
            the value of this Config if it is an integer

        Raises
        ------
        Exception
            if it is not an integer
        """
        if index is None:
            if self._val.__class__ is int:
                return self._val
            raise Exception(f"{self._val.__class__}, not int")
        else:
            if self._val.__class__ is list:
                return self._val[index].int()
            raise Exception(f"{self._val.__class__}, not list")

    def list(self, index=None):
        """
        Returns
        -------
        int
            the value of this Config if it is a list

        Raises
        ------
        Exception
            if it is not an list
        """
        if index is None:
            if self._val.__class__ is list:
                return self._val
            raise Exception(f"{self._val.__class__}, not list")
        else:
            if self._val.__class__ is list:
                return self._val[index].list()
            raise Exception(f"{self._val.__class__}, not list")

    def float(self, index=None):
        """
        Returns
        -------
        float
            the value of this Config if it is a float

        Raises
        ------
        Exception
            if it is not a float
        """
        if index is None:
            if self._val.__class__ is float:
                return self._val
            raise Exception(f"{self._val.__class__}, not float")
        else:
            if self._val.__class__ is list:
                return self._val[index].float()
            raise Exception(f"{self._val.__class__}, not list")

    def get(self, *args):
        """
        Parameters
        ----------
        indices : list of int, optional
            i for getting the i-th item in the list
        Returns
        -------
        Config
            the i-th Config object in the list
        Raises
        ------
        Exception
            if it is not a list
        """
        
        if self._val.__class__ is list:
            if len(list(args)) == 1:
                return self._val[args[0]]
            return self._val[args[0]].get(*list(args)[1:])
        raise Exception(f"{self._val.__class__}, not list")

    @classmethod
    def _objectify_dict(cls, dct, root):
        obj = Config(root=root)
        for key, val in dct.items():
            if val.__class__ is dict:
                obj.__setattr__(key, cls._objectify_dict(val, obj._root))
            elif val.__class__ is list:
                obj.__setattr__(key, cls._objectify_list(val, obj._root))
            else:
                obj.__setattr__(key, Config(val, root=obj._root))
        return obj

    @classmethod
    def _objectify_list(cls, lst, root):
        objs = list()
        for item in lst:
            if item.__class__ is dict:
                objs.append(cls._objectify_dict(item, root))
            elif item.__class__ is list:
                objs.append(cls._objectify_list(item, root))
            else:
                objs.append(Config(item, root=root))
        return Config(objs, root)

    @classmethod
    def from_dict(cls, dct):
        """Creates a Config object from a json object like dictionary.
        Parameters
        ----------
        dct : dict
            keys must be of type str.
            values can be one of these: int, float, str, dict, list.
        Returns
        -------
        Config
            contaning the same data as the input parameter `dct`
        """
        return cls._objectify_dict(dct, root=None)
