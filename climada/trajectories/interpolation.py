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

This modules implements different sparce matrices and numpy arrays
interpolation approaches.

"""

import logging
from abc import ABC
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import sparse

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AllLinearStrategy",
    "ExponentialExposureStrategy",
    "linear_interp_arrays",
    "linear_interp_imp_mat",
    "exponential_interp_arrays",
    "exponential_interp_imp_mat",
]


def linear_interp_imp_mat(
    mat_start: sparse.csr_matrix,
    mat_end: sparse.csr_matrix,
    number_of_interpolation_points: int,
) -> List[sparse.csr_matrix]:
    r"""
    Linearly interpolates between two sparse impact matrices.

    Creates a sequence of matrices representing a linear transition from a starting
    matrix to an ending matrix. The interpolation includes both the start and end
    points.

    Parameters
    ----------
    mat_start : scipy.sparse.csr_matrix
        The starting impact matrix. Must have a shape compatible with `mat_end`
        for arithmetic operations.
    mat_end : scipy.sparse.csr_matrix
        The ending impact matrix. Must have a shape compatible with `mat_start`
        for arithmetic operations.
    number_of_interpolation_points : int
        The total number of matrices to return, including the start and end points.
        Must be $\ge 2$.

    Returns
    -------
    list of scipy.sparse.csr_matrix
        A list of matrices, where the first element is `mat_start` and the last
        element is `mat_end`. The total length of the list is
        `number_of_interpolation_points`.

    Notes
    -----
    The formula used for interpolation at proportion $p$ is:
    $$M_p = M_{start} \cdot (1 - p) + M_{end} \cdot p$$
    The proportions $p$ range from 0 to 1, inclusive.
    """

    return [
        mat_start + prop * (mat_end - mat_start)
        for prop in np.linspace(0, 1, number_of_interpolation_points)
    ]


def exponential_interp_imp_mat(
    mat_start: sparse.csr_matrix,
    mat_end: sparse.csr_matrix,
    number_of_interpolation_points: int,
) -> List[sparse.csr_matrix]:
    r"""
    Exponentially interpolates between two "impact matrices".

    This function performs interpolation in a logarithmic space, effectively
    achieving an exponential-like transition between `mat_start` and `mat_end`.
    It is designed for objects that wrap NumPy arrays and expose them via a
    `.data` attribute.

    Parameters
    ----------
    mat_start : object
        The starting matrix object. Must have a `.data` attribute that is a
        NumPy array of positive values.
    mat_end : object
        The ending matrix object. Must have a `.data` attribute that is a
        NumPy array of positive values and have a compatible shape with `mat_start`.
    number_of_interpolation_points : int
        The total number of matrix objects to return, including the start and
        end points. Must be $\ge 2$.

    Returns
    -------
    list of object
        A list of interpolated matrix objects. The first element corresponds to
        `mat_start` and the last to `mat_end` (after the conversion/reversion).
        The list length is `number_of_interpolation_points`.

    Notes
    -----
    The interpolation is achieved by:

    1. Mapping the matrix data to a transformed logarithmic space:
       $$M'_{i} = \ln(M_{i})}$$
       (where $\ln$ is the natural logarithm, and $\epsilon$ is added to $M_{i}$
       to prevent $\ln(0)$).
    2. Performing standard linear interpolation on the transformed matrices
       $M'_{start}$ and $M'_{end}$ to get $M'_{interp}$:
       $$M'_{interp} = M'_{start} \cdot (1 - \text{ratio}) + M'_{end} \cdot \text{ratio}$$
    3. Mapping the result back to the original domain:
       $$M_{interp} = \exp(M'_{interp}$$
    """

    mat_start = mat_start.copy()
    mat_end = mat_end.copy()
    mat_start.data = np.log(mat_start.data + np.finfo(float).eps)
    mat_end.data = np.log(mat_end.data + np.finfo(float).eps)

    # Perform linear interpolation in the logarithmic domain
    res = []
    num_points = number_of_interpolation_points
    for point in range(num_points):
        ratio = point / (num_points - 1)
        mat_interpolated = mat_start * (1 - ratio) + ratio * mat_end
        mat_interpolated.data = np.exp(mat_interpolated.data)
        res.append(mat_interpolated)
    return res


def linear_interp_arrays(arr_start: np.ndarray, arr_end: np.ndarray) -> np.ndarray:
    r"""
    Performs linear interpolation between two NumPy arrays over their first dimension.

    This function interpolates each metric (column) linearly across the time steps
    (rows), including both the start and end states.

    Parameters
    ----------
    arr_start : numpy.ndarray
        The starting array of metrics. The first dimension (rows) is assumed to
        represent the interpolation steps (e.g., dates/time points).
    arr_end : numpy.ndarray
        The ending array of metrics. Must have the exact same shape as `arr_start`.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as `arr_start` and `arr_end`. The values
        in the first dimension transition linearly from those in `arr_start`
        to those in `arr_end`.

    Raises
    ------
    ValueError
        If `arr_start` and `arr_end` do not have the same shape.

    Notes
    -----
    The interpolation is performed element-wise along the first dimension
    (axis 0). For each row $i$ and proportion $p_i$, the result $R_i$ is calculated as:

    $$R_i = arr\_start_i \cdot (1 - p_i) + arr\_end_i \cdot p_i$$

    where $p_i$ is generated by $\text{np.linspace}(0, 1, n)$ and $n$ is the
    size of the first dimension ($\text{arr\_start.shape}[0]$).
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


def exponential_interp_arrays(arr_start: np.ndarray, arr_end: np.ndarray) -> np.ndarray:
    r"""
    Performs exponential interpolation between two NumPy arrays over their first dimension.

    This function achieves an exponential-like transition by performing linear
    interpolation in the logarithmic space, suitable to interpolate over a dimension which has
    a growth factor.

    Parameters
    ----------
    arr_start : numpy.ndarray
        The starting array of metrics. Values must be positive.
    arr_end : numpy.ndarray
        The ending array of metrics. Must have the exact same shape as `arr_start`.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as `arr_start` and `arr_end`. The values
        in the first dimension transition exponentially from those in `arr_start`
        to those in `arr_end`.

    Raises
    ------
    ValueError
        If `arr_start` and `arr_end` do not have the same shape.

    Notes
    -----
    The interpolation is performed by transforming the arrays to a logarithmic
    domain, linearly interpolating, and then transforming back.

    The formula for the interpolated result $R$ at proportion $\text{prop}$ is:
    $$
    R = \exp \left(
        \ln(A_{start}) \cdot (1 - \text{prop}) +
        \ln(A_{end}) \cdot \text{prop}
    \right)
    $$
    where $A_{start}$ and $A_{end}$ are the input arrays (with $\epsilon$ added
    to prevent $\ln(0)$) and $\text{prop}$ ranges from 0 to 1.
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

    # Perform log transformation, linear interpolation, and exponential back-transformation
    log_arr_start = np.log(arr_start + np.finfo(float).eps)
    log_arr_end = np.log(arr_end + np.finfo(float).eps)

    interpolated_log_arr = np.multiply(log_arr_start, prop0) + np.multiply(
        log_arr_end, prop1
    )

    return np.exp(interpolated_log_arr)


class InterpolationStrategyBase(ABC):
    r"""
    Base abstract class for defining a set of interpolation strategies.

    This class serves as a blueprint for implementing specific interpolation
    methods (e.g., 'Linear', 'Exponential') across different impact dimensions:
    Exposure (matrices), Hazard, and Vulnerability (arrays/metrics).

    Attributes
    ----------
    exposure_interp : Callable
        The function used to interpolate sparse impact matrices over the
        exposure dimension.
        Signature: (mat_start, mat_end, num_points, **kwargs) -> list[sparse.csr_matrix].
    hazard_interp : Callable
        The function used to interpolate NumPy arrays of metrics over the
        hazard dimension.
        Signature: (arr_start, arr_end, **kwargs) -> np.ndarray.
    vulnerability_interp : Callable
        The function used to interpolate NumPy arrays of metrics over the
        vulnerability dimension.
        Signature: (arr_start, arr_end, **kwargs) -> np.ndarray.
    """

    exposure_interp: Callable
    hazard_interp: Callable
    vulnerability_interp: Callable

    def interp_over_exposure_dim(
        self,
        imp_E0: sparse.csr_matrix,
        imp_E1: sparse.csr_matrix,
        interpolation_range: int,
        /,
        **kwargs: Optional[Dict[str, Any]],
    ) -> List[sparse.csr_matrix]:
        """
        Interpolates between two impact matrices using the defined exposure strategy.

        This method calls the function assigned to :attr:`exposure_interp` to generate
        a sequence of matrices.

        Parameters
        ----------
        imp_E0 : scipy.sparse.csr_matrix
            A sparse matrix of the impacts at the start of the range.
        imp_E1 : scipy.sparse.csr_matrix
            A sparse matrix of the impacts at the end of the range.
        interpolation_range : int
            The total number of time points to interpolate, including the start and end.
        **kwargs : Optional[Dict[str, Any]]
            Keyword arguments to pass to the underlying :attr:`exposure_interp` function.

        Returns
        -------
        list of scipy.sparse.csr_matrix
            A list of ``interpolation_range`` interpolated impact matrices.

        Raises
        ------
        ValueError
            If the underlying interpolation function raises a ``ValueError``
            indicating incompatible matrix shapes.
        """
        try:
            res = self.exposure_interp(imp_E0, imp_E1, interpolation_range, **kwargs)
        except ValueError as err:
            if str(err) == "inconsistent shapes":
                raise ValueError(
                    "Tried to interpolate impact matrices of different shapes. "
                    "A possible reason could be Exposures of different shapes."
                ) from err

            raise err

        return res

    def interp_over_hazard_dim(
        self,
        metric_0: np.ndarray,
        metric_1: np.ndarray,
        /,
        **kwargs: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Interpolates between two metric arrays using the defined hazard strategy.

        This method calls the function assigned to :attr:`hazard_interp`.

        Parameters
        ----------
        metric_0 : numpy.ndarray
            The starting array of metrics.
        metric_1 : numpy.ndarray
            The ending array of metrics. Must have the same shape as ``metric_0``.
        **kwargs : Optional [Dict[str, Any]]
            Keyword arguments to pass to the underlying :attr:`hazard_interp` function.

        Returns
        -------
        numpy.ndarray
            The resulting interpolated array.
        """
        return self.hazard_interp(metric_0, metric_1, **kwargs)

    def interp_over_vulnerability_dim(
        self,
        metric_0: np.ndarray,
        metric_1: np.ndarray,
        /,
        **kwargs: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Interpolates between two metric arrays using the defined vulnerability strategy.

        This method calls the function assigned to :attr:`vulnerability_interp`.

        Parameters
        ----------
        metric_0 : numpy.ndarray
            The starting array of metrics.
        metric_1 : numpy.ndarray
            The ending array of metrics. Must have the same shape as ``metric_0``.
        **kwargs : Optional[Dict[str, Any]]
            Keyword arguments to pass to the underlying :attr:`vulnerability_interp` function.

        Returns
        -------
        numpy.ndarray
            The resulting interpolated array.
        """
        # Note: Assuming the Callable takes the exact positional arguments
        return self.vulnerability_interp(metric_0, metric_1, **kwargs)


class InterpolationStrategy(InterpolationStrategyBase):
    r"""Interface for interpolation strategies.

    This is the class to use to define your own custom interpolation strategy.
    """

    def __init__(
        self,
        exposure_interp: Callable,
        hazard_interp: Callable,
        vulnerability_interp: Callable,
    ) -> None:
        super().__init__()
        self.exposure_interp = exposure_interp
        self.hazard_interp = hazard_interp
        self.vulnerability_interp = vulnerability_interp


class AllLinearStrategy(InterpolationStrategyBase):
    r"""Linear interpolation strategy over all dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.exposure_interp = linear_interp_imp_mat
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays


class ExponentialExposureStrategy(InterpolationStrategyBase):
    r"""Exponential interpolation strategy for exposure and linear for Hazard and Vulnerability."""

    def __init__(self) -> None:
        super().__init__()
        self.exposure_interp = (
            lambda mat_start, mat_end, points: exponential_interp_imp_mat(
                mat_start, mat_end, points
            )
        )
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays
