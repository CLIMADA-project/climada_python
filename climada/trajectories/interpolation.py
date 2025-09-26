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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import sparse

LOGGER = logging.getLogger(__name__)

# Define a type alias for the expected signature of the metric interpolation functions
# (e.g., linear_interp_arrays)
MetricInterpFunc = Callable[
    [np.ndarray, np.ndarray, Optional[Dict[str, Any]]], np.ndarray
]

# Define a type alias for the expected signature of the matrix interpolation function
# (e.g., linear_interp_imp_mat)
MatrixInterpFunc = Callable[
    [sparse.csr_matrix, sparse.csr_matrix, int, Optional[Dict[str, Any]]],
    List[sparse.csr_matrix],
]


def linear_interp_imp_mat(
    mat_start: sparse.csr_matrix,
    mat_end: sparse.csr_matrix,
    number_of_interpolation_points: int,
) -> List[sparse.csr_matrix]:
    """
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


# Assuming the matrix object type is complex and not easily type-hinted beyond 'Any'
# If a specific custom type exists (e.g., 'ImpactMatrix'), that should be used instead of 'Any'.


def exponential_interp_imp_mat(
    mat_start: Any, mat_end: Any, number_of_interpolation_points: int, rate: float
) -> List[Any]:
    """
    Exponentially interpolates between two "impact matrices" using a specified rate.

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
    rate : float
        The base rate used for the exponential scaling. Must be positive ($> 0$).
        It determines the scaling factor used in the logarithmic conversion.

    Returns
    -------
    list of object
        A list of interpolated matrix objects. The first element corresponds to
        `mat_start` and the last to `mat_end` (after the conversion/reversion).
        The list length is `number_of_interpolation_points`.

    Raises
    ------
    ValueError
        If `rate` is less than or equal to zero.

    Notes
    -----
    The interpolation is achieved by:

    1. Mapping the matrix data to a transformed logarithmic space:
       $$M'_{i} = \frac{\ln(M_{i})}{\ln(\text{rate})}$$
       (where $\ln$ is the natural logarithm, and $\epsilon$ is added to $M_{i}$
       to prevent $\ln(0)$).
    2. Performing standard linear interpolation on the transformed matrices
       $M'_{start}$ and $M'_{end}$ to get $M'_{interp}$:
       $$M'_{interp} = M'_{start} \cdot (1 - \text{ratio}) + M'_{end} \cdot \text{ratio}$$
    3. Mapping the result back to the original domain:
       $$M_{interp} = \exp(M'_{interp} \cdot \ln(\text{rate}))$$

    This process assumes the values in the original matrices are growth/impact
    factors that should be compounded.
    """
    # ... function body remains the same ...
    # Convert matrices to logarithmic domain
    if rate <= 0:
        raise ValueError("Rate for exponential interpolation must be positive")

    mat_start = mat_start.copy()
    mat_end = mat_end.copy()
    log_rate = np.log(rate)
    mat_start.data = np.log(mat_start.data + np.finfo(float).eps) / log_rate
    mat_end.data = np.log(mat_end.data + np.finfo(float).eps) / log_rate

    # Perform linear interpolation in the logarithmic domain
    res = []
    num_points = number_of_interpolation_points
    for point in range(num_points):
        ratio = point / (num_points - 1)
        mat_interpolated = mat_start * (1 - ratio) + ratio * mat_end
        mat_interpolated.data = np.exp(mat_interpolated.data * log_rate)
        res.append(mat_interpolated)
    return res


def linear_interp_arrays(arr_start: np.ndarray, arr_end: np.ndarray) -> np.ndarray:
    """
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


def exponential_interp_arrays(
    arr_start: np.ndarray, arr_end: np.ndarray, rate: float
) -> np.ndarray:
    """
    Performs exponential interpolation between two NumPy arrays over their first dimension.

    This function achieves an exponential-like transition by performing linear
    interpolation in the logarithmic space, suitable for metrics that represent
    growth factors.

    Parameters
    ----------
    arr_start : numpy.ndarray
        The starting array of metrics. Values must be positive.
    arr_end : numpy.ndarray
        The ending array of metrics. Must have the exact same shape as `arr_start`.
    rate : float
        The base rate used for the exponential scaling. Must be positive ($> 0$).
        It defines the base for the logarithmic transformation.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as `arr_start` and `arr_end`. The values
        in the first dimension transition exponentially from those in `arr_start`
        to those in `arr_end`.

    Raises
    ------
    ValueError
        If `arr_start` and `arr_end` do not have the same shape, or if `rate` is
        less than or equal to zero.

    Notes
    -----
    The interpolation is performed by transforming the arrays to a logarithmic
    domain, linearly interpolating, and then transforming back.

    The formula for the interpolated result $R$ at proportion $\text{prop}$ is:
    $$
    R = \exp \left( \left(
        \frac{\ln(A_{start})}{\ln(\text{rate})} \cdot (1 - \text{prop}) +
        \frac{\ln(A_{end})}{\ln(\text{rate})} \cdot \text{prop}
    \right) \cdot \ln(\text{rate}) \right)
    $$
    where $A_{start}$ and $A_{end}$ are the input arrays (with $\epsilon$ added
    to prevent $\ln(0)$) and $\text{prop}$ ranges from 0 to 1.
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

    # Perform log transformation, linear interpolation, and exponential back-transformation
    log_rate = np.log(rate)
    log_arr_start = np.log(arr_start + np.finfo(float).eps) / log_rate
    log_arr_end = np.log(arr_end + np.finfo(float).eps) / log_rate

    interpolated_log_arr = np.multiply(log_arr_start, prop0) + np.multiply(
        log_arr_end, prop1
    )

    return np.exp(interpolated_log_arr * log_rate)


class InterpolationStrategyBase(ABC):
    """
    Base abstract class for defining a set of interpolation strategies.

    This class serves as a blueprint for implementing specific interpolation
    methods (e.g., 'Linear', 'Exponential') across different impact dimensions:
    Exposure (matrices), Hazard, and Vulnerability (arrays/metrics).

    Attributes
    ----------
    exposure_interp : MatrixInterpFunc
        The function used to interpolate sparse impact matrices over the
        exposure dimension.
        Signature: (mat_start, mat_end, num_points, **kwargs) -> list[sparse.csr_matrix].
    hazard_interp : MetricInterpFunc
        The function used to interpolate NumPy arrays of metrics over the
        hazard dimension.
        Signature: (arr_start, arr_end, **kwargs) -> np.ndarray.
    vulnerability_interp : MetricInterpFunc
        The function used to interpolate NumPy arrays of metrics over the
        vulnerability dimension.
        Signature: (arr_start, arr_end, **kwargs) -> np.ndarray.
    """

    exposure_interp: MatrixInterpFunc
    hazard_interp: MetricInterpFunc
    vulnerability_interp: MetricInterpFunc

    def interp_over_exposure_dim(
        self,
        imp_E0: sparse.csr_matrix,
        imp_E1: sparse.csr_matrix,
        interpolation_range: int,
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
        **kwargs : Optional[ Dict[str, Any]]
            Keyword arguments (e.g., 'rate' for exponential interpolation) to pass
            to the underlying :attr:`exposure_interp` function.

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
            # Note: Assuming the Callable takes the exact positional arguments
            res = self.exposure_interp(imp_E0, imp_E1, interpolation_range, **kwargs)
        except ValueError as err:
            # Specific error handling for clarity
            if str(err) == "inconsistent shapes":
                raise ValueError(
                    "Tried to interpolate impact matrices of different shape. "
                    "A possible reason could be Exposures of different shapes."
                ) from err  # Use 'from err' to chain the exception

            raise err

        return res

    def interp_over_hazard_dim(
        self,
        metric_0: np.ndarray,
        metric_1: np.ndarray,
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
        **kwargs : Optional[ Dict[str, Any]]
            Keyword arguments to pass to the underlying :attr:`hazard_interp` function.

        Returns
        -------
        numpy.ndarray
            The resulting interpolated array.
        """
        # Note: Assuming the Callable takes the exact positional arguments
        return self.hazard_interp(metric_0, metric_1, **kwargs)

    def interp_over_vulnerability_dim(
        self,
        metric_0: np.ndarray,
        metric_1: np.ndarray,
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
        **kwargs : Optional[ Dict[str, Any]]
            Keyword arguments to pass to the underlying :attr:`vulnerability_interp` function.

        Returns
        -------
        numpy.ndarray
            The resulting interpolated array.
        """
        # Note: Assuming the Callable takes the exact positional arguments
        return self.vulnerability_interp(metric_0, metric_1, **kwargs)


class InterpolationStrategy(InterpolationStrategyBase):
    """Interface for interpolation strategies."""

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
    """Linear interpolation strategy."""

    def __init__(self) -> None:
        super().__init__()
        self.exposure_interp = linear_interp_imp_mat
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays


class ExponentialExposureStrategy(InterpolationStrategyBase):
    """Exponential interpolation strategy."""

    def __init__(self, rate) -> None:
        super().__init__()
        self.rate = rate
        self.exposure_interp = (
            lambda mat_start, mat_end, points: exponential_interp_imp_mat(
                mat_start, mat_end, points, self.rate
            )
        )
        self.hazard_interp = linear_interp_arrays
        self.vulnerability_interp = linear_interp_arrays
