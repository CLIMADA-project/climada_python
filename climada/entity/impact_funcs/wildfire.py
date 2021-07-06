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

Define impact functions for WildFires.
"""

__all__ = ['ImpfWildfire']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class ImpfWildfire(ImpactFunc):
    """Impact function for wildfire."""

    def __init__(self, haz_type = 'WFsingle'):
        ImpactFunc.__init__(self)
        self.haz_type = haz_type
        LOGGER.warning('haz_type is set to %s.', self.haz_type)
        

    def set_default_FIRMS(self, i_half=295.01, impf_id=1):

        """ This function sets the impact curve to a sigmoid type shape, as
        common in impact modelling. We adapted the function as proposed by
        Emanuel et al. (2011) which hinges on two parameters (intercept (i_thresh)
        and steepness (i_half) of the sigmoid).

        .. math::
            f = \frac{i_{n}^{3}}{1+i_{n}^{3}}
        with

        .. math::
            i_n = \frac{MAX[(I_{lat, lon}-I_{thresh}), 0]}{I_{half}-I_{thresh}}

        The intercept is defined at the minimum intensity of a FIRMS value
        (295K) which leaves the steepness (i_half) the only parameter that
        needs to be calibrated.

        Here, i_half is set to 295 K as a result of the calibration
        performed by Lüthi et al. (in prep). This value is suited for an
        exposure resolution of 1 km.
        
        Calibration was further performed for:
            - 4 km: resulting i_half = 409.4 K
            - 10 km: resulting i_half = 484.4 K

        Calibration has been performed globally (using EMDAT data) and is
        based on 84 damage records since 2001.
        
        Intensity range is set between 295 K and 500 K as this is the typical
        range of FIRMS intensities.

        Parameters
        ----------
        i_half : float, optional, default = 295.01
            steepnes of the IF, [K] at which 50% of max. damage is expected
        if_id : int, optional, default = 1
            impact function id

        Returns
        -------
        self : climada.entity.impact_funcs.ImpfWildfire instance

        """

        self.id = impf_id
        self.name = "wildfire default 1 km"
        self.intensity_unit = "K"
        self.intensity = np.arange(295, 500, 5)
        i_thresh = 295
        i_n = (self.intensity-i_thresh)/(i_half-i_thresh)
        self.paa = i_n**3 / (1 + i_n**3)
        self.mdd = np.ones(len(self.intensity))
