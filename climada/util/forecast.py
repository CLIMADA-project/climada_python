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

from abc import ABC, abstractmethod
from typing import Literal, Self

import numpy as np
from scipy.sparse import block_array, csr_matrix


def reduce_mat(mat: csr_matrix, func):
    """Reduce a matrix and return the CSR representation"""
    return func(mat).tocsr()


def concat_matrices_per_event(*matrices: csr_matrix):
    """Concatenate matrices by event"""
    return block_array([[mat] for mat in matrices], format="csr")


def matrix_event_padding(mat: csr_matrix, num_events):
    """Pad zero events"""
    pad_events = mat.shape[0] - num_events
    if pad_events < 1:
        return mat
    return block_array(
        [[mat], csr_matrix((pad_events, mat.shape[1], mat.dtype))], format="csr"
    )


class Forecast(ABC):
    lead_time: np.ndarray[np.timedelta64]
    member: np.ndarray[int]
    forecast_date: np.datetime64 | None
    num_members: int
    num_lead_times: int

    def __init__(self, lead_time, member, forecast_date: np.datetime64 | None = None):
        """Store members"""
        pass

    # --- Selection --- #

    @abstractmethod
    def _select_by_index(self, index: tuple[np.ndarray, ...]) -> Self:
        """Return a new object with the index used for selecting events"""
        raise NotImplementedError

    def _select_member(self, member: int | None) -> np.ndarray:
        """Return boolean array where self.member == member"""
        ...

    def _select_lead_time(self, lead_time: np.timedelta64 | None) -> np.ndarray:
        """Return boolean array where self.lead_time == lead_time"""
        ...

    def select(self, *, member: int | None, lead_time: np.timedelta64 | None) -> Self:
        index = np.nonzero(
            self._select_member(member) & self._select_lead_time(lead_time)
        )
        return self._select_by_index(index)

    # --- Generic reduction --- #

    @classmethod
    @abstractmethod
    def concat(cls, *obj: Self) -> Self:
        """Concatenate multiple object instances"""
        raise NotImplementedError

    @abstractmethod
    def _reduce(self, func) -> Self:
        """Apply the reduction function in the derived class and return the result

        Note: The derived class will likely need to pad matrices!
        """
        raise NotImplementedError

    def reduce(self, func, dim: Literal["member", "lead_time"] | None = None) -> Self:
        """Reduce along a given dimension with func"""
        if dim is None:
            # TODO: Check if we selected a specific member or lead time.
            #       Pad events accordingly!
            return self._reduce(func=func)  # Derived class specialization

        return self.concat(
            *(
                self.select(**{dim: val}).reduce(func=func, dim=None)
                for val in np.unique(getattr(self, dim))
            )
        )

    # --- Specializations --- #

    @abstractmethod
    def _max(self) -> Self:
        """Apply the maximum function in the derived class and return the result"""
        raise NotImplementedError

    def _reduce_attr(
        self, attr: str, dim: Literal["member", "lead_time"] | None = None
    ) -> Self:
        """Reduce along a given dimension with attribute attr"""
        if dim is None:
            # TODO: Check if we selected a specific member or lead time.
            #       Pad events accordingly!
            return getattr(self, "_" + attr)()  # Derived class specialization

        return self.concat(
            *(
                getattr(self.select(**{dim: val}), attr)(dim=None)
                for val in np.unique(getattr(self, dim))
            )
        )

    def max(self, dim):
        return self._reduce_attr(attr="max", dim=dim)
