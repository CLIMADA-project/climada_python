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

__all__ = ['IFWildfire']

import logging
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class IFWildfire(ImpactFunc):
    """Impact function for wildfire."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'WFsingle'

    def set_default_FIRMS(self, i_half=535.8, if_id=1):

        """Using the formula of Emanuele 2011 but adjusted to wildfire.
        As 295K resembles the lowest number of a FIRMS data set, i_thresh can
        be set constant to 295 K leaving i_half the only parameter that needs
        calibration. i_half is set to 523.8 as a result of the calibration
        performed by Lüthi (2021).

        Calibration has been performed globally (using EMDAT data) on a 10 km
        resolution. Thus, function is optimised to estimate impacts on 10 km.

        Intensity range is set between 295 K and 500 K as this is the typical
        range of FIRMS intensities.

        Parameters:
            i_half (float, optional): steepnes of the IF, [K] at which 50% of
            max. damage is expected
            if_id (int, optional): impact function id. Default: 1

        Raises:
            ValueError
        """

        self.id = if_id
        self.name = "wildfire default 10 km"
        self.intensity_unit = "K"
        self.intensity = np.arange(295, 500, 5)
        i_thresh = 295
        i_n = (self.intensity-i_thresh)/(i_half-i_thresh)
        self.paa = i_n**3/(1+i_n**3)
        self.mdd = np.ones(len(self.intensity))

    def set_step(self, threshold=295., if_id=1):

        """ Step function type impact function. Everything is destroyed above
        threshold. Can be usefull for high resolution modelling.

        Defaults are not calibrated

        Intensity range is set between 295 K and 500 K as this is the typical
        range of FIRMS intensities.

        Parameters:
            threshold (float, optional): threshold over which exposure is fully
            destroyed
            if_id (int, optional): impact function id. Default: 1

        Raises:
            ValueError
        """

        self.id = if_id
        self.name = "wildfire step"
        self.intensity_unit = "K"
        self.intensity = np.array([295, threshold, threshold, 500])
        self.mdd = np.array([0, 0, 1, 1])
        self.paa = np.array([1, 1, 1, 1])

    def set_sigmoid(self, sig_mid=320, sig_shape=0.1, sig_max=1.0, if_id=1):

        """ Sigmoid type impact function hinging on three parameter. This type
        of impact function is very flexible for any sort of study/resolution.
        Parameters can be thought of as intercept (sig_mid), slope (sig_shape)
        and top (sig_max) of a sigmoid.

        Default values are not calibrated.

        Parameters:
            int_range (array, optional): x-axis range of imapct function [K]
            sig_mid (float, optional): "intercept"
            sig_shape (float, optional): "slope"
            sig_max (float, optional): "top", between 0. and 1.
            if_id (int, optional): impact function id. Default: 1

        Raises:
            ValueError
        """
        self.id = if_id
        self.name = "wildfire sigmoid"
        self.intensity_unit = "K"
        self.intensity = np.arange(295, 500, 5)
        self.mdd = np.ones(len(self.intensity))
        self.paa = sig_max/(1+np.exp(-sig_shape*(self.intensity-sig_mid)))

