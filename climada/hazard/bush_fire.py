"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along 
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define BushFire class.
"""

__all__ = ['BushFire']

import logging

from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

DIST_INTER_FIRE = 50 #km
""" Distance between two different fires/events (in km)"""

class BushFire(Hazard):
    """Contains bush fire events.

    Attributes:
    """

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
