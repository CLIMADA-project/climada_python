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

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
from scipy.sparse import csr_matrix

from climada.trajectories.interpolation import (
    AllLinearStrategy,
    ExponentialExposureInterpolation,
    InterpolationStrategy,
    exponential_interp_arrays,
    exponential_interp_imp_mat,
    linear_interp_arrays,
    linear_interp_imp_mat,
)


class TestInterpolationFuncs(unittest.TestCase):
    def setUp(self):
        # Create mock impact matrices for testing
        self.imp_mat0 = csr_matrix(np.array([[1, 2], [3, 4]]))
        self.imp_mat1 = csr_matrix(np.array([[5, 6], [7, 8]]))
        self.imp_mat2 = csr_matrix(np.array([[5, 6, 7], [8, 9, 10]]))  # Different shape
        self.time_points = 5
        self.interpolation_range_5 = 5
        self.interpolation_range_1 = 1
        self.interpolation_range_2 = 2
        self.rtol = 1e-5
        self.atol = 1e-8

    def test_linear_interp_arrays(self):
        arr_start = np.array([10, 100])
        arr_end = np.array([20, 200])
        expected = np.array([10.0, 200.0])
        result = linear_interp_arrays(arr_start, arr_end)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_linear_interp_arrays2D(self):
        arr_start = np.array([[10, 100], [10, 100]])
        arr_end = np.array([[20, 200], [20, 200]])
        expected = np.array([[10.0, 100.0], [20, 200]])
        result = linear_interp_arrays(arr_start, arr_end)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_linear_interp_arrays_shape(self):
        arr_start = np.array([10, 100, 5])
        arr_end = np.array([20, 200])
        with self.assertRaises(ValueError):
            linear_interp_arrays(arr_start, arr_end)

    def test_linear_interp_arrays_start_equals_end(self):
        arr_start = np.array([5, 5])
        arr_end = np.array([5, 5])
        expected = np.array([5.0, 5.0])
        result = linear_interp_arrays(arr_start, arr_end)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_exponential_interp_arrays_1d(self):
        arr_start = np.array([1, 10, 100])
        arr_end = np.array([2, 20, 200])
        rate = 10
        expected = np.array([1.0, 14.142136, 200.0])
        result = exponential_interp_arrays(arr_start, arr_end, rate)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_exponential_interp_arrays_shape(self):
        arr_start = np.array([10, 100, 5])
        arr_end = np.array([20, 200])
        rate = 10
        with self.assertRaises(ValueError):
            exponential_interp_arrays(arr_start, arr_end, rate)

    def test_exponential_interp_arrays_2d(self):
        arr_start = np.array(
            [
                [1, 10, 100],  # date 1 metric a,b,c
                [1, 10, 100],  # date 2 metric a,b,c
                [1, 10, 100],
            ]
        )  # date 3 metric a,b,c
        arr_end = np.array([[2, 20, 200], [2, 20, 200], [2, 20, 200]])
        rate = 10
        expected = np.array(
            [[1.0, 10.0, 100.0], [1.4142136, 14.142136, 141.42136], [2, 20, 200]]
        )
        result = exponential_interp_arrays(arr_start, arr_end, rate)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_exponential_interp_arrays_start_equals_end(self):
        arr_start = np.array([5, 5])
        arr_end = np.array([5, 5])
        rate = 2
        expected = np.array([5.0, 5.0])
        result = exponential_interp_arrays(arr_start, arr_end, rate)
        np.testing.assert_allclose(result, expected, rtol=self.rtol, atol=self.atol)

    def test_exponential_interp_arrays_invalid_rate(self):
        arr_start = np.array([10, 100])
        arr_end = np.array([20, 200])
        # Test rate <= 0
        with self.assertRaises(ValueError) as cm:
            exponential_interp_arrays(arr_start, arr_end, 0)
        self.assertIn(
            "Rate for exponential interpolation must be positive", str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            exponential_interp_arrays(arr_start, arr_end, -2)
        self.assertIn(
            "Rate for exponential interpolation must be positive", str(cm.exception)
        )

    def test_linear_impmat_interpolate(self):
        result = linear_interp_imp_mat(self.imp_mat0, self.imp_mat1, self.time_points)
        self.assertEqual(len(result), self.time_points)
        for mat in result:
            self.assertIsInstance(mat, csr_matrix)

        dense = np.array([r.todense() for r in result])
        expected = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 3.0], [4.0, 5.0]],
                [[3.0, 4.0], [5.0, 6.0]],
                [[4.0, 5.0], [6.0, 7.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        np.testing.assert_array_equal(dense, expected)

    def test_linear_impmat_interpolate_inconsistent_shape(self):
        with self.assertRaises(ValueError):
            linear_interp_imp_mat(self.imp_mat0, self.imp_mat2, self.time_points)

    def test_exp_impmat_interpolate(self):
        result = exponential_interp_imp_mat(
            self.imp_mat0, self.imp_mat1, self.time_points, 1.1
        )
        self.assertEqual(len(result), self.time_points)
        for mat in result:
            self.assertIsInstance(mat, csr_matrix)

        dense = np.array([r.todense() for r in result])
        expected = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.49534878, 2.63214803], [3.70779275, 4.75682846]],
                [[2.23606798, 3.46410162], [4.58257569, 5.65685425]],
                [[3.34370152, 4.55901411], [5.66374698, 6.72717132]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        np.testing.assert_array_almost_equal(dense, expected)

    def test_exp_impmat_interpolate_inconsistent_shape(self):
        with self.assertRaises(ValueError):
            exponential_interp_imp_mat(
                self.imp_mat0, self.imp_mat2, self.time_points, 1.1
            )


class TestInterpolationStrategies(unittest.TestCase):

    def setUp(self):
        self.interpolation_range = 3
        self.dummy_metric_0 = np.array([10, 20])
        self.dummy_metric_1 = np.array([100, 200])
        self.dummy_matrix_0 = np.array([[1, 2], [3, 4]])
        self.dummy_matrix_1 = np.array([[10, 20], [30, 40]])

    def test_InterpolationStrategy_init(self):
        mock_exposure = lambda a, b, r: a + b
        mock_hazard = lambda a, b, r: a * b
        mock_vulnerability = lambda a, b, r: a / b

        strategy = InterpolationStrategy(mock_exposure, mock_hazard, mock_vulnerability)
        self.assertEqual(strategy.exposure_interp, mock_exposure)
        self.assertEqual(strategy.hazard_interp, mock_hazard)
        self.assertEqual(strategy.vulnerability_interp, mock_vulnerability)

    def test_InterpolationStrategy_interp_exposure_dim(self):
        mock_exposure = MagicMock(return_value=["mock_result"])
        strategy = InterpolationStrategy(
            mock_exposure, linear_interp_arrays, linear_interp_arrays
        )

        result = strategy.interp_exposure_dim(
            self.dummy_matrix_0, self.dummy_matrix_1, self.interpolation_range
        )
        mock_exposure.assert_called_once_with(
            self.dummy_matrix_0, self.dummy_matrix_1, self.interpolation_range
        )
        self.assertEqual(result, ["mock_result"])

    def test_InterpolationStrategy_interp_exposure_dim_inconsistent_shapes(self):
        mock_exposure = MagicMock(side_effect=ValueError("inconsistent shapes"))
        strategy = InterpolationStrategy(
            mock_exposure, linear_interp_arrays, linear_interp_arrays
        )

        with self.assertRaisesRegex(
            ValueError, "Tried to interpolate impact matrices of different shape"
        ):
            strategy.interp_exposure_dim(
                self.dummy_matrix_0, np.array([[1]]), self.interpolation_range
            )
        mock_exposure.assert_called_once()  # Ensure it was called

    def test_InterpolationStrategy_interp_hazard_dim(self):
        mock_hazard = MagicMock(return_value=np.array([1, 2, 3]))
        strategy = InterpolationStrategy(
            linear_interp_imp_mat, mock_hazard, linear_interp_arrays
        )

        result = strategy.interp_hazard_dim(self.dummy_metric_0, self.dummy_metric_1)
        mock_hazard.assert_called_once_with(self.dummy_metric_0, self.dummy_metric_1)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_InterpolationStrategy_interp_vulnerability_dim(self):
        mock_vulnerability = MagicMock(return_value=np.array([4, 5, 6]))
        strategy = InterpolationStrategy(
            linear_interp_imp_mat, linear_interp_arrays, mock_vulnerability
        )

        result = strategy.interp_vulnerability_dim(
            self.dummy_metric_0, self.dummy_metric_1
        )
        mock_vulnerability.assert_called_once_with(
            self.dummy_metric_0, self.dummy_metric_1
        )
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))


class TestConcreteInterpolationStrategies(unittest.TestCase):

    def setUp(self):
        self.interpolation_range = 3
        self.dummy_metric_0 = np.array([10, 20, 30])
        self.dummy_metric_1 = np.array([100, 200, 300])
        self.dummy_matrix_0 = csr_matrix([[1, 2], [3, 4]])
        self.dummy_matrix_1 = csr_matrix([[10, 20], [30, 40]])
        self.dummy_matrix_0_1_lin = csr_matrix([[5.5, 11], [16.5, 22]])
        self.dummy_matrix_0_1_exp = csr_matrix(
            [[3.162278, 6.324555], [9.486833, 12.649111]]
        )
        self.rtol = 1e-5
        self.atol = 1e-8

    def test_AllLinearStrategy_init_and_methods(self):
        strategy = AllLinearStrategy()
        self.assertEqual(strategy.exposure_interp, linear_interp_imp_mat)
        self.assertEqual(strategy.hazard_interp, linear_interp_arrays)
        self.assertEqual(strategy.vulnerability_interp, linear_interp_arrays)

        # Test hazard interpolation
        expected_hazard_interp = linear_interp_arrays(
            self.dummy_metric_0, self.dummy_metric_1
        )
        result_hazard = strategy.interp_hazard_dim(
            self.dummy_metric_0, self.dummy_metric_1
        )
        np.testing.assert_allclose(
            result_hazard, expected_hazard_interp, rtol=self.rtol, atol=self.atol
        )

        # Test vulnerability interpolation
        expected_vulnerability_interp = linear_interp_arrays(
            self.dummy_metric_0, self.dummy_metric_1
        )
        result_vulnerability = strategy.interp_vulnerability_dim(
            self.dummy_metric_0, self.dummy_metric_1
        )
        np.testing.assert_allclose(
            result_vulnerability,
            expected_vulnerability_interp,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Test exposure interpolation (using mock for linear_interp_imp_mat)
        result_exposure = strategy.interp_exposure_dim(
            self.dummy_matrix_0, self.dummy_matrix_1, self.interpolation_range
        )
        # Verify the structure/first/last elements of the mock output
        self.assertEqual(len(result_exposure), self.interpolation_range)
        np.testing.assert_allclose(result_exposure[0].data, self.dummy_matrix_0.data)
        np.testing.assert_allclose(
            result_exposure[1].data, self.dummy_matrix_0_1_lin.data
        )
        np.testing.assert_allclose(result_exposure[2].data, self.dummy_matrix_1.data)

    def test_ExponentialExposureInterpolation_init_and_methods(self):
        strategy = ExponentialExposureInterpolation()
        self.assertEqual(strategy.exposure_interp, exponential_interp_imp_mat)
        self.assertEqual(strategy.hazard_interp, linear_interp_arrays)
        self.assertEqual(strategy.vulnerability_interp, linear_interp_arrays)

        # Test hazard interpolation (should be linear)
        expected_hazard_interp = linear_interp_arrays(
            self.dummy_metric_0, self.dummy_metric_1
        )
        result_hazard = strategy.interp_hazard_dim(
            self.dummy_metric_0, self.dummy_metric_1
        )
        np.testing.assert_allclose(
            result_hazard, expected_hazard_interp, rtol=self.rtol, atol=self.atol
        )

        # Test vulnerability interpolation (should be linear)
        expected_vulnerability_interp = linear_interp_arrays(
            self.dummy_metric_0, self.dummy_metric_1
        )
        result_vulnerability = strategy.interp_vulnerability_dim(
            self.dummy_metric_0, self.dummy_metric_1
        )
        np.testing.assert_allclose(
            result_vulnerability,
            expected_vulnerability_interp,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Test exposure interpolation (using mock for exponential_interp_imp_mat)
        result_exposure = strategy.interp_exposure_dim(
            self.dummy_matrix_0, self.dummy_matrix_1, self.interpolation_range, rate=1.1
        )
        # Verify the structure/first/last elements of the mock output
        self.assertEqual(len(result_exposure), self.interpolation_range)
        np.testing.assert_allclose(result_exposure[0].data, self.dummy_matrix_0.data)
        np.testing.assert_allclose(
            result_exposure[1].data,
            self.dummy_matrix_0_1_exp.data,
            rtol=self.rtol,
            atol=self.atol,
        )
        np.testing.assert_allclose(result_exposure[-1].data, self.dummy_matrix_1.data)


if __name__ == "__main__":
    unittest.main()
