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

from climada.engine.option_appraisal.MCDM.constants import IMPORTANCE_MATCH


class WeightedItem:
    def __init__(self, weight) -> None:
        self.weight = weight

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value, /):
        if value is None:
            value = 0.0

        if isinstance(value, str):
            try:
                value = IMPORTANCE_MATCH[value]
            except KeyError as err:
                err.add_note(
                    f"Importance '{value}' is not defined. It must be defined within {list( IMPORTANCE_MATCH.keys() )}"
                )
                raise

        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Weight needs to be between 0 and 1 (received {value}.")

        self._weight = value
