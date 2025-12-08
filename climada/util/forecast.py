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

Define Forecast base class.
"""

import numpy as np


class Forecast:
    def __init__(self, lead_time=None, member=None, *args, **kwargs):
        if lead_time is None:
            self.lead_time = np.array([])
        else:
            self.lead_time = np.array(lead_time)

        if member is None:
            self.member = np.array([])
        else:
            self.member = member

        super().__init__(*args, **kwargs)
