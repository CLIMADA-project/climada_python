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

This module implements risk trajectory objects which enable computation and
possibly interpolation of risk metric over multiple dates.

"""

from .interpolation import AllLinearStrategy, ExponentialExposureStrategy
from .snapshot import Snapshot

__all__ = [
    "AllLinearStrategy",
    "ExponentialExposureStrategy",
    "Snapshot",
]
