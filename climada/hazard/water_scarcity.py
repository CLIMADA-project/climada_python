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

Define WaterScarcity (WS) class.
WORK IN PROGRESS
"""

__all__ = ['WaterScarcity']

import logging


from climada.hazard.base import Hazard
from climada.hazard.isimip_data import _read_one_nc # example



LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
""" Hazard type acronym for Water Scarcity """


class WaterScarcity(Hazard):
    """Contains water scarcity events.

    Attributes:
        ...
    """

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_hist_events(self, centroids=None):
        """ 

        Parameters:
            ...: ...
        """
        LOGGER.info('Setting up historical events.')
        self.clear()

