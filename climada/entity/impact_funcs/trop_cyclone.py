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

Define impact functions for tropical cyclnes .
"""

__all__ = ['IFTropCyclone']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class IFTropCyclone(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'TC'

    def set_emanuel_usa(self, if_id=1, intensity=np.arange(0, 121, 5),
                        v_thresh=25.7, v_half=74.7, scale=1.0):
        """Using the formula of Emanuele 2011.

        Parameters:
            if_id (int, optional): impact function id. Default: 1
            intensity (np.array, optional): intensity array in m/s. Default:
                5 m/s step array from 0 to 120m/s
            v_thresh (float, optional): first shape parameter, wind speed in
                m/s below which there is no damage. Default: 25.7(Emanuel 2011)
            v_half (float, optional): second shape parameter, wind speed in m/s
                at which 50% of max. damage is expected. Default:
                v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
            scale (float, optional): scale parameter, linear scaling of MDD.
                0<=scale<=1. Default: 1.0

        Raises:
            ValueError
        """
        if v_half <= v_thresh:
            LOGGER.error('Shape parameters out of range: v_half <= v_thresh.')
            raise ValueError
        if  v_thresh < 0 or v_half < 0:
            LOGGER.error('Negative shape parameter.')
            raise ValueError
        if scale > 1 or scale <= 0:
            LOGGER.error('Scale parameter out of range.')
            raise ValueError

        self.name = 'Emanuel 2011'
        self.id = if_id
        self.intensity_unit = 'm/s'
        self.intensity = intensity
        self.paa = np.ones(intensity.shape)
        v_temp = (self.intensity - v_thresh) / (v_half - v_thresh)
        v_temp[v_temp < 0] = 0
        self.mdd = v_temp**3 / (1 + v_temp**3)
        self.mdd *= scale
