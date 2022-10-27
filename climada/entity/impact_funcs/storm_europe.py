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

    @classmethod
    def from_schwierz(cls, impf_id=1):
        """
        Generate the impact function of Schwierz et al. 2010, doi:10.1007/s10584-009-9712-1

        Returns
        -------
        impf : climada.entity.impact_funcs.storm_europe.ImpfStormEurope:
            impact function for asset damages due to storm defined in Schwierz et al. 2010
        """

        impf = cls()
        impf.name = 'Schwierz 2010'
        impf.id = impf_id
        impf.intensity_unit = 'm/s'
        impf.intensity = np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100])
        impf.paa = np.array([0., 0., 0.001, 0.00676,
                             0.03921, 0.10707, 0.25357, 0.48869,
                             0.82907, 1., 1., 1.])
        impf.mdd = np.array([0., 0., 0.001, 0.00177515,
                             0.00367253, 0.00749977, 0.01263556, 0.01849639,
                             0.02370487, 0.037253, 0.037253, 0.037253])
        impf.check()
        return impf

    @classmethod
    def from_welker(cls, impf_id=1):
        """
        Return the impact function of Welker et al. 2021, doi:10.5194/nhess-21-279-2021
        It is the Schwierz function, calibrated with a simple multiplicative
        factor to minimize RMSE between modelled damages and reported damages.

        Returns
        -------
        impf: climada.entity.impact_funcs.storm_europe.ImpfStormEurope:
            impact function for asset damages due to storm defined in Welker et al. 2021
        """

        temp_Impf = ImpfStormEurope.from_schwierz()
        scaling_factor = {'paa_scale': 1.332518, 'mdd_scale': 1.332518}
        temp_Impf = init_impf(temp_Impf, scaling_factor)[0]
        temp_Impf.name = 'Welker 2021'
        temp_Impf.id = impf_id
        temp_Impf.intensity_unit = 'm/s'
        temp_Impf.check()
        return temp_Impf

    def set_schwierz(self, impf_id=1):
        """
        This function is deprecated, use ImpfStormEurope.from_schwierz
        instead.
        """
        LOGGER.warning("The use of ImpfStormEurope.set_schwierz is deprecated."
                       "Use ImpfStormEurope.from_schwierz instead.")
        self.__dict__ = ImpfStormEurope.from_schwierz(impf_id=impf_id).__dict__

    def set_welker(self, impf_id=1):
        """
        This function is deprecated, use ImpfStormEurope.from_welker
        instead.
        """
        LOGGER.warning("The use of ImpfStormEurope.set_welker is deprecated."
                       "Use ImpfStormEurope.from_welker instead.")
        self.__dict__ = ImpfStormEurope.from_welker(impf_id=impf_id).__dict__


@deprecated(details="The class name IFStormEurope is deprecated and won't be supported in a future "
                   +"version. Use ImpfStormEurope instead")
class IFStormEurope(ImpfStormEurope):
    """Is ImpfStormEurope now"""
