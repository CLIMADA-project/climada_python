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

from climada.util.interpolation import interpolate_ev, stepfunction_ev, group_frequency


class TestFitMethods(unittest.TestCase):
    """Test different fit configurations"""

    def test_interpolate_ev_linear_interp(self):
        """Test linear interpolation"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([2., 4., 8.])
        x_test = np.array([0., 3., 4.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 4., 6.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, fill_value = -1),
            np.array([-1., 4., 6.])
        )

    def test_interpolate_ev_threshold_parameters(self):
        """Test input threshold parameters"""
        x_train = np.array([0., 3., 6.])
        y_train = np.array([4., 1., 4.])
        x_test = np.array([-1., 3., 4.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 1., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, x_threshold=1.),
            np.array([np.nan, 1., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, y_threshold=2.),
            np.array([np.nan, 4., 4.])
        )
    
    def test_interpolate_ev_scale_parameters(self):
        """Test log scale parameters"""
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1., 3.])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, logx=True, fill_value='extrapolate'),
            np.array([0., 2.])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, logx=True),
            np.array([np.nan, 2.])
        )
        x_train = np.array([1., 3.])
        y_train = np.array([1e1, 1e3])
        x_test = np.array([0., 2.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, logy=True, fill_value='extrapolate'),
            np.array([1e0, 1e2])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, logy=True),
            np.array([np.nan, 1e2])
        )
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1e1, 1e5])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, logx=True, logy=True, fill_value='extrapolate'),
            np.array([1e-1, 1e3])
        )

    def test_interpolate_ev_degenerate_input(self):
        """Test interp to constant zeros"""
        x_train = np.array([1., 3., 5.])
        x_test = np.array([0., 2., 4.])
        y_train = np.zeros(3)
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 0., 0.])
        )

    def test_interpolate_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.])
        y_train = np.array([2.])
        x_test = np.array([0., 1., 2.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train),
            np.array([2., 2., np.nan])
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.array([2., 2., 0.])
        )
        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0., 1., 2.])
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train),
            np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            interpolate_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.zeros(3)
        )

    def test_stepfunction_ev(self):
        """Test stepfunction method"""
        x_train = np.array([1., 3., 5.])
        y_train = np.array([8., 4., 2.])
        x_test = np.array([0., 3., 4., 6.])
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train),
            np.array([8., 4., 2., np.nan])
        )
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0.),
            np.array([8., 4., 2., 0.])
        )

    def test_stepfunction_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.])
        y_train = np.array([2.])
        x_test = np.array([0., 1., 2.])
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train),
            np.array([2., 2., np.nan])
        )
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.array([2., 2., 0.])
        )
        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0., 1., 2.])
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train),
            np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.zeros(3)
        )
    
    def test_frequency_group(self):
        """Test frequency grouping method"""
        frequency = np.ones(6)
        intensity = np.array([1., 1., 1., 2., 3., 3])
        np.testing.assert_allclose(
            group_frequency(frequency, intensity), 
            ([3, 1, 2], [1, 2, 3])
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
