"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Exposures reader function from a file with extension defined in
constant FILE_EXT.
"""

import logging
import numpy as np

from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5
from climada.entity.exposures.base import INDICATOR_IF, INDICATOR_CENTR,\
DEF_REF_YEAR, DEF_VALUE_UNIT



DEF_HAZ_TYPE = ''
""" Hazard type used for the impact functions. Used for compatibility."""

LOGGER = logging.getLogger(__name__)


