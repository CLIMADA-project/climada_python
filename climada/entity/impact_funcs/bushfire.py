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
Define impact functions for BushFires.
"""

__all__ = ['IFBushfire']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class IFBushfire(ImpactFunc):
    """Impact function for bushfire."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'BF'
        
    def set_default(self, threshold):
        self.haz_type = "BF"
        self.id = 1
        self.name = "bushfire default"
        self.intensity_unit = "K"
        self.intensity = np.array([295, threshold, threshold, 367])
        self.mdd = np.array([0, 0, 1, 1])
        self.paa = np.array([1, 1, 1, 1])