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

from climada.util.fit_methods import interpolate_ev, group_frequency


class TestFitMethods(unittest.TestCase):
    """Test different fit configurations"""

    def test_interpolate_ev_linear_interp(self):
        """Test linear interpolation"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 8.])
        x_test = np.array([0., 3., 4.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', bounds_error=False),
            np.array([np.nan, 4., 6.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', fill_value = -1, bounds_error=False),
            np.array([-1., 4., 6.])
        )

    def test_interpolate_ev_threshold_parameters(self):
        """Test input threshold parameters"""
        x_train = np.array([0., 3., 6.])
        y_train = np.array([4., 1., 4.])
        x_test = np.array([-1., 3., 4.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', bounds_error=False),
            np.array([np.nan, 1., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', x_threshold=1., bounds_error=False),
            np.array([np.nan, 1., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', y_threshold=2., bounds_error=False),
            np.array([np.nan, 4., 4.])
        )
    
    def test_interpolate_ev_scale_parameters(self):
        """Test log scale parameters"""
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1., 3.])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', x_scale='log', fill_value='extrapolate'),
            np.array([0., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', x_scale='log', bounds_error=False),
            np.array([np.nan, 2.])
        )
        x_train = np.array([1., 3.])
        y_train = np.array([1e1, 1e3])
        x_test = np.array([0., 2.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', y_scale='log', fill_value='extrapolate'),
            np.array([1e0, 1e2])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', y_scale='log', bounds_error=False),
            np.array([np.nan, 1e2])
        )
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1e1, 1e5])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', x_scale='log', y_scale='log', fill_value='extrapolate'),
            np.array([1e-1, 1e3])
        )

    def test_interpolate_ev_stepfunction(self):
        """Test stepfunction method"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 8.])
        x_test = np.array([0., 3., 4., 6.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='stepfunction'),
            np.array([2., 4., 8., np.nan])
        )

    def test_interpolate_ev_degenerate_input(self):
        """Test interp to constant zeros"""
        x_train = np.array([1., 3., 5.])
        x_test = np.array([0., 2., 4.])
        y_train = np.zeros(3)
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate', bounds_error=False),
            np.array([np.nan, 0., 0.])
        )

    def test_interpolate_ev(self):
        """Test small input"""
        x_train = np.array([1.])
        y_train = np.array([2.])
        x_test = np.array([0., 1.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, method='interpolate'),
            np.array([2., 2.])
        )
    
    def test_frequency_group(self):
        """Test frequency grouping method"""
        frequency = np.ones(6)
        intensity = np.array([3., 3., 2., 1., 1., 1])
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
