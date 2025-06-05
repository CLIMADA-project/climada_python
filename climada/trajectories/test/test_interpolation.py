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

import unittest
from unittest.mock import MagicMock

import numpy as np
from scipy.sparse import csr_matrix

from climada.trajectories.interpolation import (
    AllLinearInterpolation,
    ExponentialExposureInterpolation,
)


class TestLinearInterpolation(unittest.TestCase):
    def setUp(self):
        # Create mock impact matrices for testing
        self.imp0 = MagicMock()
        self.imp1 = MagicMock()
        self.imp0.imp_mat = csr_matrix(np.array([[1, 2], [3, 4]]))
        self.imp1.imp_mat = csr_matrix(np.array([[5, 6], [7, 8]]))
        self.time_points = 5

        # Create an instance of LinearInterpolation
        self.linear_interpolation = AllLinearInterpolation()

    def test_interpolate(self):
        result = self.linear_interpolation.interpolate(
            self.imp0, self.imp1, self.time_points
        )
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

    def test_interpolate_inconsistent_shape(self):
        imp0 = MagicMock()
        imp1 = MagicMock()
        imp0.imp_mat = csr_matrix(np.array([[1, 2], [3, 4]]))
        imp1.imp_mat = csr_matrix(np.array([[5, 6, 7], [8, 9, 10]]))  # Different shape

        with self.assertRaises(ValueError):
            self.linear_interpolation.interpolate(imp0, imp1, self.time_points)


class TestExponentialInterpolation(unittest.TestCase):
    def setUp(self):
        # Create mock impact matrices for testing
        self.imp0 = MagicMock()
        self.imp1 = MagicMock()
        self.imp0.imp_mat = csr_matrix(np.array([[1, 2], [3, 4]]))
        self.imp1.imp_mat = csr_matrix(np.array([[5, 6], [7, 8]]))
        self.time_points = 5

        # Create an instance of ExponentialInterpolation
        self.exponential_interpolation = ExponentialExposureInterpolation()

    def test_interpolate(self):
        result = self.exponential_interpolation.interpolate(
            self.imp0, self.imp1, self.time_points
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

    def test_interpolate_inconsistent_shape(self):
        imp0 = MagicMock()
        imp1 = MagicMock()
        imp0.imp_mat = csr_matrix(np.array([[1, 2], [3, 4]]))
        imp1.imp_mat = csr_matrix(np.array([[5, 6, 7], [8, 9, 10]]))  # Different shape

        with self.assertRaises(ValueError):
            self.exponential_interpolation.interpolate(imp0, imp1, self.time_points)


if __name__ == "__main__":
    unittest.main()
