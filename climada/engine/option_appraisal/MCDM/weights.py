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

"""

from abc import ABC


class Weights(ABC):
    def __init__(self, weights) -> None:
        super().__init__()
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value, /):
        self._weights = self._normalize(value)

    def _normalize(self, value):
        return value / value.sum()
