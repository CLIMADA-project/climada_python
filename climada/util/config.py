"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define configuration parameters.
"""

__all__ = [
    "CONFIG",
]

import json
import logging
import re
import sys
from pathlib import Path


class Config:
    """Convenience Class. A Config object is a slow JSON object like nested dictonary who's values
    can be accessed by their names right away. E.g.: `a.b.c.str()` instead of `a['b']['c']`
    """

    SOURCE_DIR = None
    """If a config value is a path and the root directory is given as '...' it will be replaced by
    the path to the installation directory.
    Like this it's possible to refer to e.g. test files in the climada sources.
    """

    def __getattribute__(self, __name):
        # the only reason for overriding __getattribute__: better error messages in case a
        # configuration value is asked for but never defined in any of the possible config files
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            conf_files = [
                (
                    Path(_find_in_parents(conf_dir, CONFIG_NAME))
                    if _find_in_parents(conf_dir, CONFIG_NAME)
                    else conf_dir / CONFIG_NAME
                )
                for conf_dir in CONFIG_DIRS[::-1]
            ]
            raise AttributeError(  # pylint: disable=raise-missing-from
                f"there is no '{__name}' configured for '{super().__getattribute__('_name')}'."
                f" check your config files: {conf_files}"
            )

    def __str__(self):
        # pylint: disable=bare-except,multiple-statements,too-complex
        try:
            return self.str()
        except:
            pass
        try:
            return str(self.int())
        except:
            pass
        try:
            return str(self.float())
        except:
            pass
        try:
            return str(self.bool())
        except:
            pass
        try:
            return str(self.list())
        except:
            pass
        return "{{{}}}".format(
            ", ".join(
                [
                    f"{k}: {v}"
                    for (k, v) in self.__dict__.items()
                    if not k in {"_name", "_root"}
                ]
            )
        )

    def __repr__(self):
        return self.__str__()

    def __init__(self, name=None, val=None, root=None):
        """
        Parameters
        ----------
        name : str, optional
            the name of the configuration, used only for providing sensible error messages.
            Default: None
        root : Config, optional
            the top Config object, required for self referencing str objects,
            if None, it is pointing to self, otherwise it's passed from containing to
            contained.
            Default: None
        val : [float, int, bool, str, list], optional
            the value of the Config in case it's basic, when a dictionary like object is created
            Default: None
        """
        self._name = name
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
                return expand(dct.__getattribute__(lst[0]), lst[1:])

            def msub(match):
                cpath = match.group(1).split(".")
                return expand(root, cpath)

            return re.sub(r"{([\w\.]+)}", msub, cstr)

        if index is None:
            if self._val.__class__ is str:
                return feval(self._root, self._val)
            raise Exception(f"{self._val.__class__}, not str")
        if self._val.__class__ is list:
            return self._val[index].str()
        raise Exception(f"{self._val.__class__}, not list")

    def bool(self, index=None):
        """
        Returns
        -------
        bool
            the value of this Config interpreted as boolean

        Raises
        ------
        Exception
            if it cannot be take as boolean
        """
        if index is None:
            if self._val.__class__ is bool:
                return self._val
            raise Exception(f"{self._val.__class__}, not bool")
        if self._val.__class__ is list:
            return self._val[index].bool()
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
        if self._val.__class__ is list:
            return self._val[index].int()
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
        if self._val.__class__ is list:
            return self._val[index].float()
        raise Exception(f"{self._val.__class__}, not list")

    def list(self, index=None):
        """
        Returns
        -------
        list
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
        if self._val.__class__ is list:
            return self._val[index].list()
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

    def dir(self, index=None, create=True):
        """Convenience method to get this configuration value as a Path object.
        If the respective directory does not exist it is created upon the call
        unless the flag `create` is set to False.

        Parameters
        ----------
        index: int, optional
            the index of the item if the addressed Config object is a list
        create: bool, optional
            flag to indictate whether the directory is going to be created
            default: True

        Returns
        -------
        pathlib.Path
            the absolute path to the directory of this config's value (if it is a string)

        Raises
        ------
        Exception
            if the value is not a string or if the directory cannot be created
        """
        path = self._expand_source_dir(Path(self.str(index)).expanduser())
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path.absolute()

    @classmethod
    def _expand_source_dir(cls, path):
        parts = path.parts
        if parts[0] == "...":
            return Path(cls.SOURCE_DIR, *parts[1:])
        return Path(*parts)

    @classmethod
    def _objectify_dict(cls, name, dct, root):
        # pylint: disable=protected-access
        obj = Config(name=name, root=root)
        for key, val in dct.items():
            if val.__class__ is dict:
                obj.__setattr__(
                    key, cls._objectify_dict(name=key, dct=val, root=obj._root)
                )
            elif val.__class__ is list:
                obj.__setattr__(
                    key, cls._objectify_list(name=key, lst=val, root=obj._root)
                )
            else:
                obj.__setattr__(key, Config(name=key, val=val, root=obj._root))
        return obj

    @classmethod
    def _objectify_list(cls, name, lst, root):
        objs = list()
        for item in lst:
            if item.__class__ is dict:
                objs.append(cls._objectify_dict(name, item, root))
            elif item.__class__ is list:
                objs.append(cls._objectify_list(name, item, root))
            else:
                objs.append(Config(name=name, val=item, root=root))
        return Config(name=name, val=objs, root=root)

    @classmethod
    def from_dict(cls, dct):
        """Creates a Config object from a json object like dictionary.
        Parameters
        ----------
        dct : dict
            keys must be of type str.
            values can be one of these: int, float, bool, str, dict, list.
        Returns
        -------
        Config
            contaning the same data as the input parameter `dct`
        """
        return cls._objectify_dict("climada.CONFIG", dct, root=None)


def _supersede(nested, addendum):
    for key, val in addendum.items():
        if nested.get(key).__class__ is dict:
            if val.__class__ is dict:
                nested[key] = _supersede(nested[key], val)
            else:
                nested[key] = val
        else:
            nested[key] = val
    return nested


def _find_in_parents(directory, filename):
    if Path(directory, filename).is_file():
        return str(Path(directory, filename))
    for dirpath in Path(directory).parents:
        if Path(dirpath, filename).is_file():
            return str(Path(dirpath, filename))
    return None


def _fetch_conf(directories, config_name):
    superseding_configs = [_find_in_parents(path, config_name) for path in directories]
    conf_dct = dict()
    for conf_path in superseding_configs:
        if conf_path is None:
            continue
        with open(conf_path, encoding="utf-8") as conf:
            dct = json.load(conf)
            conf_dct = _supersede(conf_dct, dct)

    return conf_dct


SOURCE_DIR = Path(__file__).absolute().parent.parent.parent
CONFIG_NAME = "climada.conf"
CONFIG_DIRS = [
    Path(SOURCE_DIR, "climada", "conf"),  # default config from the climada repository
    Path(Path.home(), "climada", "conf"),  # ~/climada/conf directory
    Path(Path.home(), ".config"),  # ~/.config directory
    Path.cwd(),  # current working directory
]

CONFIG = Config.from_dict(_fetch_conf(CONFIG_DIRS, CONFIG_NAME))
Config.SOURCE_DIR = SOURCE_DIR


# set climada style logging
if CONFIG.logging.managed.bool():
    LOGGER = logging.getLogger("climada")
    LOGGER.propagate = False
    FORMATTER = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    CONSOLE = logging.StreamHandler(stream=sys.stdout)
    CONSOLE.setFormatter(FORMATTER)
    LOGGER.addHandler(CONSOLE)

    LOGGER.setLevel(getattr(logging, CONFIG.log_level.str()))
