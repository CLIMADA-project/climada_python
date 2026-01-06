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

Tests for interpolation

"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from climada.trajectories.interpolation import (
    AllLinearStrategy,
    CustomImpactInterpolationStrategy,
    ExponentialExposureStrategy,
    exponential_convex_combination,
    exponential_interp_matrix_elemwise,
    linear_convex_combination,
    linear_interp_matrix_elemwise,
)

# --- Fixtures ---


@pytest.fixture
def interpolation_data():
    """Provides common matrices and constants for interpolation tests."""
    return {
        "imp_mat0": csr_matrix(np.array([[1, 2], [3, 4]])),
        "imp_mat1": csr_matrix(np.array([[5, 6], [7, 8]])),
        "imp_mat2": csr_matrix(np.array([[5, 6, 7], [8, 9, 10]])),
        "time_points": 5,
        "rtol": 1e-5,
        "atol": 1e-8,
        "dummy_metric_0": np.array([10, 20, 30]),
        "dummy_metric_1": np.array([100, 200, 300]),
        "dummy_matrix_0": csr_matrix([[1, 2], [3, 4]]),
        "dummy_matrix_1": csr_matrix([[10, 20], [30, 40]]),
    }


# --- Tests for Interpolation Functions ---


def test_linear_interp_arrays(interpolation_data):
    arr_start = np.array([10, 50, 100])
    arr_end = np.array([20, 100, 200])
    expected = np.array([10.0, 75.0, 200.0])
    result = linear_convex_combination(arr_start, arr_end)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=interpolation_data["rtol"],
        atol=interpolation_data["atol"],
    )


@pytest.mark.parametrize(
    "func", [linear_convex_combination, exponential_convex_combination]
)
def test_convex_combination_shape_error(func):
    arr_start = np.array([10, 100, 5])
    arr_end = np.array([20, 200])
    with pytest.raises(ValueError, match="different shapes"):
        func(arr_start, arr_end)


def test_exponential_convex_combination_2d(interpolation_data):
    arr_start = np.array([[1, 10, 100]] * 3)
    arr_end = np.array([[2, 20, 200]] * 3)
    expected = np.array(
        [[1.0, 10.0, 100.0], [1.4142136, 14.142136, 141.42136], [2, 20, 200]]
    )
    result = exponential_convex_combination(arr_start, arr_end)
    np.testing.assert_allclose(
        result,
        expected,
        rtol=interpolation_data["rtol"],
        atol=interpolation_data["atol"],
    )


@pytest.mark.parametrize(
    "func", [linear_convex_combination, exponential_convex_combination]
)
def test_convex_combinations_start_equals_end(interpolation_data, func):
    """Test that if start and end are identical, the result is the same array."""
    arr = np.array([5.0, 5.0])
    result = func(arr, arr)
    np.testing.assert_allclose(result, arr, rtol=interpolation_data["rtol"])


@pytest.mark.parametrize(
    "func,expected",
    [
        (
            linear_interp_matrix_elemwise,
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                    [[3.0, 4.0], [5.0, 6.0]],
                    [[4.0, 5.0], [6.0, 7.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
        ),
        (
            exponential_interp_matrix_elemwise,
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[1.49534878, 2.63214803], [3.70779275, 4.75682846]],
                    [[2.23606798, 3.46410162], [4.58257569, 5.65685425]],
                    [[3.34370152, 4.55901411], [5.66374698, 6.72717132]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
        ),
    ],
)
def test_impmat_interpolate(interpolation_data, func, expected):
    data = interpolation_data
    result = func(data["imp_mat0"], data["imp_mat1"], data["time_points"])

    assert len(result) == data["time_points"]
    assert all(isinstance(mat, csr_matrix) for mat in result)

    dense = np.array([r.todense() for r in result])
    np.testing.assert_array_almost_equal(dense, expected)


# --- Tests for Interpolation Strategies ---


def test_custom_strategy_init():
    mock_func = lambda a, b, r: a + b
    strategy = CustomImpactInterpolationStrategy(mock_func, mock_func, mock_func)
    assert strategy.exposure_interp == mock_func
    assert strategy.hazard_interp == mock_func
    assert strategy.vulnerability_interp == mock_func


def test_custom_strategy_exposure_dim_error(interpolation_data):
    mock_exposure = MagicMock(side_effect=ValueError("inconsistent shapes"))
    strategy = CustomImpactInterpolationStrategy(
        mock_exposure, linear_convex_combination, linear_convex_combination
    )

    with pytest.raises(
        ValueError, match="Tried to interpolate impact matrices of different shape"
    ):
        strategy.interp_over_exposure_dim(
            interpolation_data["dummy_matrix_0"], csr_matrix(np.array([[1]])), 3
        )


# --- Tests for Concrete Strategies ---


def test_all_linear_strategy(interpolation_data):
    data = interpolation_data
    strategy = AllLinearStrategy()

    # Test property assignment
    assert strategy.exposure_interp == linear_interp_matrix_elemwise

    # Test Hazard dim
    result_haz = strategy.interp_over_hazard_dim(
        data["dummy_metric_0"], data["dummy_metric_1"]
    )
    expected_haz = linear_convex_combination(
        data["dummy_metric_0"], data["dummy_metric_1"]
    )
    np.testing.assert_allclose(result_haz, expected_haz)

    # Test Exposure dim
    result_exp = strategy.interp_over_exposure_dim(
        data["dummy_matrix_0"], data["dummy_matrix_1"], 3
    )
    assert len(result_exp) == 3
    # Check midpoint (index 1) manually
    expected_mid = csr_matrix([[5.5, 11], [16.5, 22]])
    np.testing.assert_allclose(result_exp[1].data, expected_mid.data)


def test_exponential_exposure_strategy(interpolation_data):
    data = interpolation_data
    strategy = ExponentialExposureStrategy()

    result_exp = strategy.interp_over_exposure_dim(
        data["dummy_matrix_0"], data["dummy_matrix_1"], 3
    )

    # Midpoint should be geometric mean for exponential strategy
    # sqrt(1*10) = 3.162278
    expected_mid_data = np.array([3.162278, 6.324555, 9.486833, 12.649111])
    np.testing.assert_allclose(
        result_exp[1].data, expected_mid_data, rtol=data["rtol"], atol=data["atol"]
    )
