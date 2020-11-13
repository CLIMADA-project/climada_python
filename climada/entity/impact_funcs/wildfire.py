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

__all__ = ['IFWildFire']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class IFBushfire(ImpactFunc):
    """Impact function for bushfire."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'WF'
        
    def set_step(self, threshold):
        self.haz_type = "WF"
        self.id = 1
        self.name = "wildfire default"
        self.intensity_unit = "K"
        self.intensity = np.array([295, threshold, threshold, 367])
        self.mdd = np.array([0, 0, 1, 1])
        self.paa = np.array([1, 1, 1, 1])
        
    def set_sigmoid(self, int_range=np.arange(295,500,5), sig_mid=320, sig_shape=0.1, sig_max=1.0):
        self.haz_type = "WF"
        self.id = 1
        self.name = "wildfire default"
        self.intensity_unit = "K"
        self.intensity = int_range
        self.mdd = sig_max/(1+np.exp(-sig_shape*(int_range-sig_mid)))
        self.paa = np.ones(len(int_range))
        
    def set_default(self, i_half=523.8):
        self.haz_type = "WF"
        self.id = 1
        self.name = "wildfire default"
        self.intensity_unit = "K"
        self.intensity = np.arange(295,500,5)
        i_thresh = 295
        i_n = (self.intensity-i_thresh)/(i_half-i_thresh)
        self.paa = i_n**3/(1+i_n**3)
        self.mdd = np.ones(len(self.intensity))