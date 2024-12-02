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

Test of interpolation module
"""

import unittest

import numpy as np

import climada.util.interpolation as u_interp


class TestFitMethods(unittest.TestCase):
    """Test different fit configurations"""

    def test_interpolate_ev_linear_interp(self):
        """Test linear interpolation"""
        x_train = np.array([1.0, 3.0, 5.0])
        y_train = np.array([8.0, 4.0, 2.0])
        x_test = np.array([0.0, 3.0, 4.0, 6.0])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 4.0, 3.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate_constant"
            ),
            np.array([8.0, 4.0, 3.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                extrapolation="extrapolate_constant",
                y_asymptotic=0,
            ),
            np.array([8.0, 4.0, 3.0, 0.0]),
        )

    def test_interpolate_ev_threshold_parameters(self):
        """Test input threshold parameters"""
        x_train = np.array([0.0, 3.0, 6.0])
        y_train = np.array([4.0, 1.0, 4.0])
        x_test = np.array([-1.0, 3.0, 4.0])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate_constant"
            ),
            np.array([4.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                x_threshold=1.0,
                extrapolation="extrapolate_constant",
            ),
            np.array([1.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                y_threshold=2.0,
                extrapolation="extrapolate_constant",
            ),
            np.array([4.0, 4.0, 4.0]),
        )

    def test_interpolate_ev_scale_parameters(self):
        """Test log scale parameters"""
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1.0, 3.0])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, logx=True, extrapolation="extrapolate"
            ),
            np.array([0.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                logx=True,
                extrapolation="extrapolate_constant",
            ),
            np.array([1.0, 2.0]),
        )
        x_train = np.array([1.0, 3.0])
        y_train = np.array([1e1, 1e3])
        x_test = np.array([0.0, 2.0])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, logy=True, extrapolation="extrapolate"
            ),
            np.array([1e0, 1e2]),
        )
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1e1, 1e5])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                logx=True,
                logy=True,
                extrapolation="extrapolate",
            ),
            np.array([1e-1, 1e3]),
        )

    def test_interpolate_ev_degenerate_input(self):
        """Test interp to constant zeros"""
        x_train = np.array([1.0, 3.0, 5.0])
        x_test = np.array([0.0, 2.0, 4.0])
        y_train = np.zeros(3)
        np.testing.assert_allclose(
            u_interp.interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 0.0, 0.0]),
        )

    def test_interpolate_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.0])
        y_train = np.array([2.0])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate"
            ),
            np.array([2.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate", y_asymptotic=0
            ),
            np.array([2.0, 2.0, 0.0]),
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )

        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp.interpolate_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            u_interp.interpolate_ev(
                x_test,
                x_train,
                y_train,
                extrapolation="extrapolate_constant",
                y_asymptotic=0,
            ),
            np.zeros(3),
        )

    def test_stepfunction_ev(self):
        """Test stepfunction method"""
        x_train = np.array([1.0, 3.0, 5.0])
        y_train = np.array([8.0, 4.0, 2.0])
        x_test = np.array([0.0, 3.0, 4.0, 6.0])
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train),
            np.array([8.0, 4.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0.0),
            np.array([8.0, 4.0, 2.0, 0.0]),
        )

    def test_stepfunction_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.0])
        y_train = np.array([2.0])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train),
            np.array([2.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.array([2.0, 2.0, 0.0]),
        )
        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            u_interp.stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.zeros(3),
        )

    def test_frequency_group(self):
        """Test frequency grouping method"""
        frequency = np.ones(6)
        intensity = np.array([1.00001, 0.999, 1.0, 2.0, 3.0, 3])
        np.testing.assert_allclose(
            u_interp.group_frequency(frequency, intensity), ([3, 1, 2], [1, 2, 3])
        )
        np.testing.assert_allclose(u_interp.group_frequency([], []), ([], []))
        with self.assertRaises(ValueError):
            u_interp.group_frequency(frequency, intensity[::-1])

    def test_round_to_sig_digits(self):
        array = [0.00111, 999.0, 55.5, 0.0, -1.001, -1.08]
        np.testing.assert_allclose(
            u_interp.round_to_sig_digits(array, n_sig_dig=2),
            [0.0011, 1000.0, 56, 0.0, -1.0, -1.1],
        )

    def test_preprocess_and_interpolate_ev(self):
        """Test wrapper function"""
        frequency = np.array([0.1, 0.9])
        values = np.array([100.0, 10.0])
        test_frequency = np.array([0.01, 0.55, 10.0])
        test_values = np.array([1.0, 55.0, 1000.0])

        # test interpolation
        np.testing.assert_allclose(
            [np.nan, 55.0, np.nan],
            u_interp.preprocess_and_interpolate_ev(
                test_frequency, None, frequency, values
            ),
        )
        np.testing.assert_allclose(
            [np.nan, 0.55, np.nan],
            u_interp.preprocess_and_interpolate_ev(
                None, test_values, frequency, values
            ),
        )

        # test extrapolation with constants
        np.testing.assert_allclose(
            [100.0, 55.0, 0.0],
            u_interp.preprocess_and_interpolate_ev(
                test_frequency,
                None,
                frequency,
                values,
                method="extrapolate_constant",
                y_asymptotic=0.0,
            ),
        )
        np.testing.assert_allclose(
            [1.0, 0.55, np.nan],
            u_interp.preprocess_and_interpolate_ev(
                None, test_values, frequency, values, method="extrapolate_constant"
            ),
        )

        # test error raising
        with self.assertRaises(ValueError):
            u_interp.preprocess_and_interpolate_ev(
                test_frequency, test_values, frequency, values
            )
        with self.assertRaises(ValueError):
            u_interp.preprocess_and_interpolate_ev(None, None, frequency, values)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFitMethods)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
