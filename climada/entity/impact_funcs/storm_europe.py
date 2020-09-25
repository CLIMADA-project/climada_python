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

Define impact functions for extratropical storms (mainly windstorms in Europe).
"""

__all__ = ['IFStormEurope']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc
from climada.engine.calibration_opt import init_if


LOGGER = logging.getLogger(__name__)

class IFStormEurope(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'WS'

    def set_schwierz(self, if_id=1):
        """
        Using the impact functions of Schwierz et al. 2011.
        """

        self.name = 'Schwierz 2011'
        self.id = if_id
        self.intensity_unit = 'm/s'
        self.intensity = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100])
        self.paa = np.array([0., 0., 0.001, 0.00676, 
                             0.03921, 0.10707, 0.25357, 0.48869, 
                             0.82907, 1., 1., 1.])
        self.mdd = np.array([0., 0., 0.001, 0.00177515,
                             0.00367253, 0.00749977, 0.01263556, 0.01849639,
                             0.02370487, 0.037253, 0.037253, 0.037253])
        self.check()
    
    def set_welker(self, if_id=1):
        """
        Using the impact functions of Welker et al. 2020 (in submission).
        It is the schwierz function, calibrated with a simple multiplicative
        factor to minimize RMSE between modelled damages and reported damages.
        """
        
        temp_if = IFStormEurope()
        temp_if.set_schwierz()
        scaling_factor = {'paa_scale': 1.332518, 'mdd_scale': 1.332518}
        temp_if = init_if(temp_if, scaling_factor)[0]
        self.name = 'Welker 2020'
        self.id = if_id
        self.intensity_unit = 'm/s'
        self.intensity = temp_if.intensity
        self.paa = temp_if.paa
        self.mdd = temp_if.mdd
        self.check()
        
        
        
        

