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
from scipy import sparse

import climada.util.interpolation as u_interp


class TestFitMethods(unittest.TestCase):
    """Test different fit configurations"""

    def test_interpolate_ev_linear_interp(self):
        """Test linear interpolation"""
        x_train = np.array([1.0, 3.0, 5.0])
        y_train = np.array([8.0, 4.0, 2.0])
        x_test = np.array([0.0, 3.0, 4.0, 6.0])
        np.testing.assert_allclose(
            u_interp._interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 4.0, 3.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate_constant"
            ),
            np.array([8.0, 4.0, 3.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
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
            u_interp._interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate_constant"
            ),
            np.array([4.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
                x_test,
                x_train,
                y_train,
                x_threshold=1.0,
                extrapolation="extrapolate_constant",
            ),
            np.array([1.0, 1.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
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
            u_interp._interpolate_ev(
                x_test, x_train, y_train, logx=True, extrapolation="extrapolate"
            ),
            np.array([0.0, 2.0]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
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
            u_interp._interpolate_ev(
                x_test, x_train, y_train, logy=True, extrapolation="extrapolate"
            ),
            np.array([1e0, 1e2]),
        )
        x_train = np.array([1e1, 1e3])
        y_train = np.array([1e1, 1e5])
        x_test = np.array([1e0, 1e2])
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
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
            u_interp._interpolate_ev(x_test, x_train, y_train),
            np.array([np.nan, 0.0, 0.0]),
        )

    def test_interpolate_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.0])
        y_train = np.array([2.0])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate"
            ),
            np.array([2.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
                x_test, x_train, y_train, extrapolation="extrapolate", y_asymptotic=0
            ),
            np.array([2.0, 2.0, 0.0]),
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )

        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp._interpolate_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            u_interp._interpolate_ev(
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
            u_interp._stepfunction_ev(x_test, x_train, y_train),
            np.array([8.0, 4.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp._stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0.0),
            np.array([8.0, 4.0, 2.0, 0.0]),
        )

    def test_stepfunction_ev_small_input(self):
        """Test small input"""
        x_train = np.array([1.0])
        y_train = np.array([2.0])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp._stepfunction_ev(x_test, x_train, y_train),
            np.array([2.0, 2.0, np.nan]),
        )
        np.testing.assert_allclose(
            u_interp._stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.array([2.0, 2.0, 0.0]),
        )
        x_train = np.array([])
        y_train = np.array([])
        x_test = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(
            u_interp._stepfunction_ev(x_test, x_train, y_train), np.full(3, np.nan)
        )
        np.testing.assert_allclose(
            u_interp._stepfunction_ev(x_test, x_train, y_train, y_asymptotic=0),
            np.zeros(3),
        )

    def test_frequency_group(self):
        """Test frequency grouping method"""
        frequency = np.ones(6)
        intensity = np.array([1.00001, 0.9998, 1.0, 2.0, 3.0, 3])
        np.testing.assert_allclose(
            u_interp._group_frequency(frequency, intensity, bin_decimals=6),
            (frequency, intensity),
        )
        np.testing.assert_allclose(
            u_interp._group_frequency(frequency, intensity, bin_decimals=3),
            ([3, 1, 2], [1, 2, 3]),
        )
        np.testing.assert_allclose(
            u_interp._group_frequency([], [], bin_decimals=3), ([], [])
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


class TestSparseLocalExceedance(unittest.TestCase):
    """The sparse fast path must reproduce preprocess_and_interpolate_ev exactly.

    Equality is asserted with assert_array_equal rather than assert_allclose: the fast
    path is only worth having if it is a drop-in, so "close" would hide a divergence.
    """

    def setUp(self):
        self.frequency = np.array([0.1, 0.1, 0.1, 0.1])
        self.test_frequency = 1 / np.array([5.0, 10.0, 20.0])
        self.test_values = np.array([2.0, 4.0])

    def reference_exceedance(self, values):
        """Per-column preprocess_and_interpolate_ev, the path sparse_local_exceedance replaces."""
        return np.array(
            [
                u_interp.preprocess_and_interpolate_ev(
                    self.test_frequency,
                    None,
                    self.frequency,
                    values[:, i_column],
                    log_frequency=True,
                    log_values=True,
                    value_threshold=0,
                    method="interpolate",
                    y_asymptotic=0.0,
                    bin_decimals=None,
                )
                for i_column in range(values.shape[1])
            ]
        )

    def reference_frequency(self, values):
        """Per-column preprocess_and_interpolate_ev, the path sparse_local_frequency replaces."""
        return np.array(
            [
                u_interp.preprocess_and_interpolate_ev(
                    None,
                    self.test_values,
                    self.frequency,
                    values[:, i_column],
                    log_frequency=True,
                    log_values=True,
                    value_threshold=0,
                    method="interpolate",
                    y_asymptotic=np.nan,
                    bin_decimals=None,
                )
                for i_column in range(values.shape[1])
            ]
        )

    def assert_matches_reference(self, dense, dtype=float):
        matrix = sparse.csr_matrix(dense.astype(dtype))
        np.testing.assert_array_equal(
            u_interp.sparse_local_exceedance(
                self.test_frequency, self.frequency, matrix
            ),
            self.reference_exceedance(dense.astype(dtype)),
        )
        np.testing.assert_array_equal(
            u_interp.sparse_local_frequency(self.test_values, self.frequency, matrix),
            self.reference_frequency(dense.astype(dtype)),
        )

    def test_typical_column(self):
        """A column with several distinct positive values."""
        self.assert_matches_reference(
            np.array([[5.0, 1.0], [4.0, 2.0], [3.0, 3.0], [1.0, 4.0]])
        )

    def test_duplicate_values(self):
        """Repeated values make the interpolation abscissa non-strictly-increasing."""
        self.assert_matches_reference(
            np.array([[5.0, 1.0], [5.0, 2.0], [3.0, 3.0], [0.0, 4.0]])
        )

    def test_zero_and_negative_values_are_dropped(self):
        """Only values above the threshold of 0 take part, as in the reference."""
        self.assert_matches_reference(
            np.array([[-2.0, 1.0], [5.0, 2.0], [3.0, 3.0], [0.0, 4.0]])
        )

    def test_all_zero_column(self):
        """A column with nothing above the threshold yields NaN, not an exception."""
        dense = np.zeros((4, 2))
        dense[:, 1] = [1.0, 2.0, 3.0, 4.0]
        self.assert_matches_reference(dense)

    def test_single_positive_value(self):
        """Interpolation needs two points; one is not enough."""
        dense = np.zeros((4, 1))
        dense[0, 0] = 7.0
        self.assert_matches_reference(dense)

    def test_explicitly_stored_zeros(self):
        """Stored zeros must be ignored the same way implicit ones are.

        Impact matrices come out of arithmetic, which can leave explicit zeros behind.
        """
        dense = np.array([[5.0], [0.0], [3.0], [0.0]])
        matrix = sparse.csr_matrix(dense)
        matrix.data = np.array([5.0, 0.0, 3.0, 0.0])
        matrix.indices = np.zeros(4, dtype=int)
        matrix.indptr = np.array([0, 1, 2, 3, 4])
        self.assertEqual(matrix.nnz, 4)  # the zeros really are stored
        np.testing.assert_array_equal(
            u_interp.sparse_local_exceedance(
                self.test_frequency, self.frequency, matrix
            ),
            self.reference_exceedance(dense),
        )

    def test_float32_matrix(self):
        """A float32 matrix must not be interpolated at float32 precision.

        Hazard.from_raster produces float32 intensity, so this is reachable. Taking
        logarithms before upcasting gives results that differ from the reference in
        the last decimals.
        """
        dense = np.array([[5.0, 1.0], [4.0, 2.0], [3.0, 3.0], [1.0, 4.0]])
        self.assert_matches_reference(dense, dtype=np.float32)

    def test_frequency_as_sequence(self):
        """The replaced function accepted any array_like, so this one must too."""
        matrix = sparse.csr_matrix(np.array([[5.0], [4.0], [3.0], [1.0]]))
        np.testing.assert_array_equal(
            u_interp.sparse_local_exceedance(
                self.test_frequency, list(self.frequency), matrix
            ),
            self.reference_exceedance(matrix.toarray()),
        )

    def test_supports_sparse_fast_path(self):
        """The guard must admit only the configuration the helpers implement."""
        self.assertTrue(
            u_interp.supports_sparse_fast_path("interpolate", True, True, 0, None)
        )
        for args in (
            ("extrapolate", True, True, 0, None),
            ("stepfunction", True, True, 0, None),
            ("interpolate", False, True, 0, None),
            ("interpolate", True, False, 0, None),
            ("interpolate", True, True, 1, None),
            ("interpolate", True, True, 0, 2),
        ):
            with self.subTest(args=args):
                self.assertFalse(u_interp.supports_sparse_fast_path(*args))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFitMethods)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestSparseLocalExceedance)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
