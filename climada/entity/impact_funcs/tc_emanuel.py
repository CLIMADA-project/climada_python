"""
Define impact function for tropical cyclnes using the formula of Emanuele 2011.
"""

__all__ = ['IFEmanuele']

import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

class IFEmanuele(ImpactFunc):
    """Impact function for tropical cyclones according to Emanuele 2011."""

    def __init__(self, if_id=1, intensity=np.arange(0, 121, 5), v_thresh=25.7,
                 v_half=74.7, scale=1.0):
        """ Empty initialization.
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
        """
        ImpactFunc.__init__(self)
        self.name = 'Emanuel 2011'
        self.haz_type = 'TC'
        self.id = if_id
        self.intensity_unit = 'm/s'
        self.intensity = intensity
        self.paa = np.ones(intensity.shape)
        v_temp = (self.intensity - v_thresh) / (v_half - v_thresh)
        v_temp[v_temp < 0] = 0
        self.mdd = scale * v_temp**3 / (1 + v_temp**3)
