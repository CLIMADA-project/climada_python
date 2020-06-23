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
Define impact functions for AgriculturalDroughts.
"""


__all__ = ['IFCropPotential']

import logging
import numpy as np
from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class IFCropPotential(ImpactFunc):
    """Impact functions for agricultural droughts."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'CP'
        self.intensity_unit = ''
        #self.continent = ''

    def set_relativeyield(self):
        """Impact functions defining the impact as (intensity-1)"""
        self.haz_type = 'CP'
        self.id = 1
        self.name = 'Crop Potential ISIMIP'
        self.intensity_unit = ''
        self.intensity = np.arange(0, 11)
        self.mdr = (self.intensity - 1)
        self.mdd = (self.intensity - 1)
        self.paa = np.ones(len(self.intensity))
