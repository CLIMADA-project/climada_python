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
from abc import ABC
from typing import Callable

import numpy as np

LOGGER = logging.getLogger(__name__)


def linear_interp_imp_mat(mat_start, mat_end, interpolation_range) -> list:
    """Linearly interpolates between two impact matrices over an interpolation range.

    Returns a list of `interpolation_range` matrices linearly interpolated between
    `mat_start` and `mat_end`.
    """
    res = []
    for point in range(interpolation_range):
        ratio = point / (interpolation_range - 1)
        mat_interpolated = mat_start + ratio * (mat_end - mat_start)
        res.append(mat_interpolated)
    return res


def exponential_interp_imp_mat(mat_start, mat_end, interpolation_range, rate) -> list:
    """Exponentially interpolates between two impact matrices over an interpolation range with a growth rate `rate`.

    Returns a list of `interpolation_range` matrices exponentially (with growth rate `rate`) interpolated between
    `mat_start` and `mat_end`.
    """
    # Convert matrices to logarithmic domain
    if rate <= 0:
        raise ValueError("Rate for exponential interpolation must be positive")

    mat_start = mat_start.copy()
    mat_end = mat_end.copy()
    mat_start.data = np.log(mat_start.data + np.finfo(float).eps) / np.log(rate)
    mat_end.data = np.log(mat_end.data + np.finfo(float).eps) / np.log(rate)

    # Perform linear interpolation in the logarithmic domain
    res = []
    for point in range(interpolation_range):
        ratio = point / (interpolation_range - 1)
        mat_interpolated = mat_start * (1 - ratio) + ratio * mat_end
        mat_interpolated.data = np.exp(mat_interpolated.data * np.log(rate))
        res.append(mat_interpolated)
    return res


def linear_interp_arrays(arr_start, arr_end):
    """Perform linear interpolation between two arrays of `n` dates of one or multiple scalar metrics.

    Returns a `n` sized arrays where the values linearly change from `arr_start` to `arr_end` over the `n` dates.
    """
    if arr_start.shape != arr_end.shape:
        raise ValueError(
            f"Cannot interpolate arrays of different shapes: {arr_start.shape} and {arr_end.shape}."
        )
    interpolation_range = arr_start.shape[0]
    prop1 = np.linspace(0, 1, interpolation_range)
    prop0 = 1 - prop1
    if arr_start.ndim > 1:
        prop0, prop1 = prop0.reshape(-1, 1), prop1.reshape(-1, 1)

    return np.multiply(arr_start, prop0) + np.multiply(arr_end, prop1)


def exponential_interp_arrays(arr_start, arr_end, rate):
    """Perform exponential interpolation between two arrays of `n` dates of one or multiple scalar metrics.

    Returns a `n` sized arrays where the values exponentially change from `arr_start` to `arr_end` over the `n` dates.
    """

    if rate <= 0:
        raise ValueError("Rate for exponential interpolation must be positive")

    if arr_start.shape != arr_end.shape:
        raise ValueError(
            f"Cannot interpolate arrays of different shapes: {arr_start.shape} and {arr_end.shape}."
        )
    interpolation_range = arr_start.shape[0]

    prop1 = np.linspace(0, 1, interpolation_range)
    prop0 = 1 - prop1
    if arr_start.ndim > 1:
        prop0, prop1 = prop0.reshape(-1, 1), prop1.reshape(-1, 1)

    return np.exp(
        (
            np.multiply(np.log(arr_start + np.finfo(float).eps) / np.log(rate), prop0)
            + np.multiply(np.log(arr_end + np.finfo(float).eps) / np.log(rate), prop1)
        )
        * np.log(rate)
    )


class InterpolationStrategyBase(ABC):
    exposure_interp: Callable
    hazard_interp: Callable
    vulnerability_interp: Callable

    def interp_exposure_dim(
        self, imp_E0, imp_E1, interpolation_range: int, **kwargs
    ) -> list:
        """Interpolates along the exposure change between two impact matrices.

        Returns a list of `interpolation_range` matrices linearly interpolated between
        `mat_start` and `mat_end`.
        """
        try:
            res = self.exposure_interp(imp_E0, imp_E1, interpolation_range, **kwargs)
        except ValueError as err:
            if str(err) == "inconsistent shapes":
                raise ValueError(
                    "Tried to interpolate impact matrices of different shape. A possible reason could be Exposures of different shapes."
                )

            raise err

        return res

    def interp_hazard_dim(self, metric_0, metric_1, **kwargs) -> np.ndarray:
        return self.hazard_interp(metric_0, metric_1, **kwargs)

    def interp_vulnerability_dim(self, metric_0, metric_1, **kwargs) -> np.ndarray:
        return self.vulnerability_interp(metric_0, metric_1, **kwargs)


class InterpolationStrategy(InterpolationStrategyBase):
    """Interface for interpolation strategies."""

    def __init__(self, exposure_interp, hazard_interp, vulnerability_interp) -> None:
        super().__init__()
        self.exposure_interp = exposure_interp
        self.hazard_interp = hazard_interp
        self.vulnerability_interp = vulnerability_interp


class AllLinearStrategy(InterpolationStrategyBase):
    """Linear interpolation strategy."""

    def __init__(self) -> None:
        super().__init__()
        self.exposure_interp = linear_interp_imp_mat
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays


class ExponentialExposureInterpolation(InterpolationStrategyBase):
    """Exponential interpolation strategy."""

    def __init__(self) -> None:
        super().__init__()
        self.exposure_interp = exponential_interp_imp_mat
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays
