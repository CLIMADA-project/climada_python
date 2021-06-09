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

Define impact function for droughts.
"""

__all__ = ['ImpfDrought', 'IFDrought']

import logging
from deprecation import deprecated
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

LOGGER = logging.getLogger(__name__)

class ImpfDrought(ImpactFunc):
    """Impact function for droughts."""

    def __init__(self):
        """Empty initialization.

        Parameters:
            impf_id (int, optional): impact function id. Default: 1
            intensity (np.array, optional): intensity array SPEI [-].
                default: intensity defintion 1 (minimum)
                default_sum: intensity definition 3 (sum over all drought months)

        Raises:
            ValueError
        """
        ImpactFunc.__init__(self)

    def set_default(self):
        self.haz_type = "DR"
        self.id = 1
        self.name = "drought default"
        self.intensity_unit = "NA"
        self.intensity = [-6.5, -4, -1, 0]
        self.mdd = [1, 1, 0, 0]
        self.paa = [1, 1, 0, 0]

    def set_default_sum(self):
        self.haz_type = "DR_sum"
        self.id = 1
        self.name = "drought default sum"
        self.intensity_unit = "NA"
        self.intensity = [-15, -12, -9, -7, -5, 0]
        self.mdd = [1, 0.65, 0.5, 0.3, 0, 0]
        self.paa = [1, 1, 1, 1, 0, 0]

    def set_default_sumthr(self):
        self.haz_type = "DR_sumthr"
        self.id = 1
        self.name = "drought default sum - thr"
        self.intensity_unit = "NA"
        self.intensity = [-8, -5, -2, 0]
        self.mdd = [0.7, 0.3, 0, 0]
        self.paa = [1, 1, 0, 0]

    def set_step(self):
        self.haz_type = "DR"
        self.id = 1
        self.name = "step"
        self.intensity_unit = "NA"
        self.intensity = np.arange(-4, 0)
        self.mdd = np.ones(self.intensity.size)
        self.paa = np.ones(self.mdd.size)


@deprecated(details="The class name IFDrought is deprecated and won't be supported in a future "
                   +"version. Use ImpfDrought instead")
class IFDrought(ImpfDrought):
    """Is ImpfDrought now"""
