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

Test of fit_methods module
"""

import unittest
import numpy as np

from climada.util.fit_methods import calc_fit_interp, group_frequency


class TestFitMethods(unittest.TestCase):
    """Test different fit configurations"""

    def test_calc_fit_interp_linear_fit(self):
        """Test linear fit"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 6.])
        x_test = np.array([0., 3., 4.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit'),
            np.array([1., 4., 5.])
        )

    def test_calc_fit_interp_linear_interp(self):
        """Test linear interpolation"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 8.])
        x_test = np.array([0., 3., 4.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp'),
            np.array([2., 4., 6.])
        )
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp', left = -1.),
            np.array([-1., 4., 6.])
        )

    def test_calc_fit_interp_threshold_parameters(self):
        """Test input threshold parameters"""
        x_train = np.array([0., 3., 6.])
        y_train = np.array([4., 1., 4.])
        x_test = np.array([-1., 3., 4.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit'),
            np.array([3., 3., 3.])
        )
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit', x_thres=1.),
            np.array([-3., 1., 2.])
        )
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit', y_thres=2.),
            np.array([4., 4., 4.])
        )
    
    def test_calc_fit_interp_scale_parameters(self):
        """Test log scale parameters"""
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1., 3.])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit', x_scale='log'),
            np.array([0., 2.])
        )
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp', x_scale='log'),
            np.array([1., 2.])
        )
        x_train = np.array([1., 3.])
        y_train = np.array([1e1, 1e3])
        x_test = np.array([0., 2.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit', y_scale='log'),
            np.array([1e0, 1e2])
        )
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp', y_scale='log'),
            np.array([1e1, 1e2])
        )
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1e1, 1e5])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit', x_scale='log', y_scale='log'),
            np.array([1e-1, 1e3])
        )

    def test_calc_fit_interp_stepfunction(self):
        """Test stepfunction method"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 8.])
        x_test = np.array([0., 3., 4., 6.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='stepfunction'),
            np.array([2., 4., 8., np.nan])
        )

    def test_calc_fit_interp_degenerate_input(self):
        """Test interp to constant zeros"""
        x_train = np.array([1., 3., 5.])
        x_test = np.array([0., 2., 4.])
        y_train = np.zeros(3)
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp'),
            np.zeros(3)
        )

    def test_calc_fit_empty_input(self):
        """Test small input"""
        x_train = np.array([1.])
        y_train = np.array([2.])
        x_test = np.array([0., 1.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='interp'),
            np.array([2., 2.])
        )
        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0., 1.])
        np.testing.assert_allclose(
            calc_fit_interp(x_test, x_train, y_train, method='fit'),
            np.array([0., 0.])
        )
    
    def test_frequency_group(self):
        """Test frequency grouping method"""
        frequency = np.ones(6)
        intensity = [3., 3., 2., 1., 1., 1]
        np.testing.assert_allclose(
            group_frequency(frequency, intensity), 
            ([2, 1, 3], [3, 2, 1])
        )
        np.testing.assert_allclose(
            group_frequency([], []), 
            ([], [])
        )
        with self.assertRaises(ValueError):
            group_frequency(frequency, intensity[::-1])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFitMethods)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
