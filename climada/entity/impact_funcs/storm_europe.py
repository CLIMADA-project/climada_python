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

Define impact functions for extratropical storms (mainly windstorms in Europe).
"""

__all__ = ['ImpfStormEurope', 'IFStormEurope']

import logging
from deprecation import deprecated
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc
from climada.engine.calibration_opt import init_impf


LOGGER = logging.getLogger(__name__)

class ImpfStormEurope(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'WS'

    def set_schwierz(self, impf_id=1):
        """
        Using the impact functions of Schwierz et al. 2010, doi:10.1007/s10584-009-9712-1
        """

        self.name = 'Schwierz 2010'
        self.id = impf_id
        self.intensity_unit = 'm/s'
        self.intensity = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100])
        self.paa = np.array([0., 0., 0.001, 0.00676,
                             0.03921, 0.10707, 0.25357, 0.48869,
                             0.82907, 1., 1., 1.])
        self.mdd = np.array([0., 0., 0.001, 0.00177515,
                             0.00367253, 0.00749977, 0.01263556, 0.01849639,
                             0.02370487, 0.037253, 0.037253, 0.037253])
        self.check()

    def set_welker(self, Impf_id=1):
        """
        Using the impact functions of Welker et al. 2020, doi:10.5194/nhess-21-279-2021
        It is the Schwierz function, calibrated with a simple multiplicative
        factor to minimize RMSE between modelled damages and reported damages.
        """

        temp_Impf = ImpfStormEurope()
        temp_Impf.set_schwierz()
        scaling_factor = {'paa_scale': 1.332518, 'mdd_scale': 1.332518}
        temp_Impf = init_impf(temp_Impf, scaling_factor)[0]
        self.name = 'Welker 2020'
        self.id = Impf_id
        self.intensity_unit = 'm/s'
        self.intensity = temp_Impf.intensity
        self.paa = temp_Impf.paa
        self.mdd = temp_Impf.mdd
        self.check()


@deprecated(details="The class name IFStormEurope is deprecated and won't be supported in a future "
                   +"version. Use ImpfStormEurope instead")
class IFStormEurope(ImpfStormEurope):
    """Is ImpfStormEurope now"""
