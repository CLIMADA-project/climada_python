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

This modules implements different sparce matrices interpolation approaches.

"""

import logging
from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

LOGGER = logging.getLogger(__name__)


class InterpolationStrategy(ABC):
    """Interface for interpolation strategies."""

    @abstractmethod
    def interpolate(self, imp_E0, imp_E1, time_points: int) -> list: ...


class LinearInterpolation(InterpolationStrategy):
    """Linear interpolation strategy."""

    def interpolate(self, imp_E0, imp_E1, time_points: int):
        try:
            return self.interpolate_imp_mat(imp_E0, imp_E1, time_points)
        except ValueError as e:
            if str(e) == "inconsistent shapes":
                raise ValueError(
                    "Interpolation between impact matrices of different shapes"
                )
            else:
                raise e

    @staticmethod
    def interpolate_imp_mat(imp0, imp1, time_points):
        """Interpolate between two impact matrices over a specified time range.

        Parameters
        ----------
        imp0 : ImpactCalc
            The impact calculation for the starting time.
        imp1 : ImpactCalc
            The impact calculation for the ending time.
        time_points:
            The number of points to interpolate.

        Returns
        -------
        list of np.ndarray
            List of interpolated impact matrices for each time points in the specified range.
        """

        def interpolate_sm(mat_start, mat_end, time, time_points):
            """Perform linear interpolation between two matrices for a specified time point."""
            if time > time_points:
                raise ValueError("time point must be within the range")

            ratio = time / (time_points - 1)

            # Convert the input matrices to a format that allows efficient modification of its elements
            mat_start = lil_matrix(mat_start)
            mat_end = lil_matrix(mat_end)

            # Perform the linear interpolation
            mat_interpolated = mat_start + ratio * (mat_end - mat_start)

            return csr_matrix(mat_interpolated)

        LOGGER.debug(f"imp0: {imp0.imp_mat.data[0]}, imp1: {imp1.imp_mat.data[0]}")
        return [
            interpolate_sm(imp0.imp_mat, imp1.imp_mat, time, time_points)
            for time in range(time_points)
        ]


class ExponentialInterpolation(InterpolationStrategy):
    """Exponential interpolation strategy."""

    def interpolate(self, imp_E0, imp_E1, time_points: int):
        return self.interpolate_imp_mat(imp_E0, imp_E1, time_points)

    @staticmethod
    def interpolate_imp_mat(imp0, imp1, time_points):
        """Interpolate between two impact matrices over a specified time range.

        Parameters
        ----------
        imp0 : ImpactCalc
            The impact calculation for the starting time.
        imp1 : ImpactCalc
            The impact calculation for the ending time.
        time_points:
            The number of points to interpolate.

        Returns
        -------
        list of np.ndarray
            List of interpolated impact matrices for each time points in the specified range.
        """

        def interpolate_sm(mat_start, mat_end, time, time_points):
            """Perform exponential interpolation between two matrices for a specified time point."""
            if time > time_points:
                raise ValueError("time point must be within the range")

            # Convert matrices to logarithmic domain
            log_mat_start = np.log(mat_start.toarray() + np.finfo(float).eps)
            log_mat_end = np.log(mat_end.toarray() + np.finfo(float).eps)

            # Perform linear interpolation in the logarithmic domain
            ratio = time / (time_points - 1)
            log_mat_interpolated = log_mat_start + ratio * (log_mat_end - log_mat_start)

            # Convert back to the original domain using the exponential function
            mat_interpolated = np.exp(log_mat_interpolated)

            return csr_matrix(mat_interpolated)

        return [
            interpolate_sm(imp0.imp_mat, imp1.imp_mat, time, time_points)
            for time in range(time_points)
        ]
