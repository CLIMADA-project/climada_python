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

init util
"""
import logging
from pint import UnitRegistry

from .config import *
from .constants import *
from .coordinates import *
from .save import *

ureg = UnitRegistry()

class log_level:
    """Context manager that sets all loggers with names starting with
    name_prefix (default is "") to a given specified level.

    Examples
    --------
    Set ALL loggers temporarily to the level 'WARNING'
    >>> with log_level(level='WARNING'):
    >>>     ...

    Set all the climada loggers temporarily to the level 'ERROR'
    >>> with log_level(level='ERROR', name_prefix='climada'):
    >>>     ...

    """

    def __init__(self, level, name_prefix=""):
        self.level = level
        self.loggers = {
            name: (logger, logger.level)
            for name, logger in logging.root.manager.loggerDict.items()
            if isinstance(logger, logging.Logger) and name.startswith(name_prefix)
            }
        if name_prefix == "":
            self.loggers[""] = (logging.getLogger(), logging.getLogger().level)

    def __enter__(self):
        for logger, _ in self.loggers.values():
            logger.setLevel(self.level)

    def __exit__(self, exception_type, exception, traceback):
        for logger, previous_level in self.loggers.values():
            logger.setLevel(previous_level)
