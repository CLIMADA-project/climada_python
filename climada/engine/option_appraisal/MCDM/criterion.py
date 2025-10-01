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

from dataclasses import dataclass

import pandas as pd

from climada.engine.option_appraisal.MCDM.category import CategorizedObject


@dataclass
class Criterion(CategorizedObject):
    name: str
    column_name: str
    obj_maximise: bool
    group: None | str = None


class CriteriaSet:

    def __init__(self, criteria: list[Criterion]) -> None:
        self.criteria = criteria

    @property
    def criteria_matrix(self):
        return self._criteria_matrix

    @property
    def criteria(self):
        return self._criteria

    @criteria.setter
    def criteria(self, value, /):
        self._check_consistency(value)
        self._criteria = value
        self._criteria_matrix = pd.concat([item.data for item in value], axis=1)

    @staticmethod
    def _check_consistency(criteria):
        if not isinstance(criteria, list):
            raise ValueError("Criteria must be a list of Criterion.")

        if not all([isinstance(criterion, Criterion) for criterion in criteria]):
            raise ValueError("Criteria must be a list of Criterion.")

        if not all(
            [
                criteria[0].data.index.equals(criterion.data.index)
                for criterion in criteria
            ]
        ):
            raise ValueError("Criteria do not all have the same index (options).")
