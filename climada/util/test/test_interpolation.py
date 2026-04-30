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
from scipy.stats import genextreme, genpareto

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

    def test_fit_tail_distribution_gpd(self):
        """Test GPD fitting with synthetic data"""
        rng = np.random.default_rng(42)
        xi_true = 0.15
        beta_true = 5
        n = 100000
        threshold_percentile = 90
        values = genpareto.rvs(c=xi_true, scale=beta_true, size=n, random_state=rng)
        frequency = np.ones(n) / n
        test_freq = np.array([0.01, 0.001])
        freq_out, val_out, fit_result = u_interp.fit_tail_distribution(
            test_frequency=test_freq,
            frequency=frequency,
            values=values,
            dist="GPD",
            threshold_percentile=threshold_percentile,
        )
        # test shapes
        np.testing.assert_equal(len(freq_out), len(test_freq))
        np.testing.assert_equal(len(val_out), len(test_freq))

        # test fitted parameters
        # changing the threshold does not change xi but changes beta, see Ch. 4 Eq. 4.16 in
        # (Coles, 2001, Chapters 4–5, https://doi.org/10.1007/978-1-4471-3675-0)
        threshold = np.percentile(values, threshold_percentile)
        expected_beta = beta_true + xi_true * threshold
        np.testing.assert_allclose(fit_result["xi"], xi_true, rtol=0.1)
        np.testing.assert_allclose(fit_result["beta"], expected_beta, rtol=0.05)

        # Test predicted tail
        # Ch. 4 Eq. 4.13 in (Coles, 2001, Chapters 4–5, https://doi.org/10.1007/978-1-4471-3675-0)
        # frequency of exceeding threhold
        lambda_u = np.sum(frequency[values >= threshold])
        expected_vals = threshold + (expected_beta / xi_true) * (
            (test_freq / lambda_u) ** (-xi_true) - 1.0
        )
        np.testing.assert_allclose(val_out, expected_vals, rtol=0.15)

    # def test_fit_tail_distribution_gev(self):
    #     """Test GEV fitting"""
    #     rng = np.random.default_rng(42)
    #     xi_true = 0.2
    #     mu_true = 20
    #     sigma_true = 5
    #     n = 100000
    #     values = genextreme.rvs(
    #         c=-xi_true, loc=mu_true, scale=sigma_true, size=n, random_state=rng
    #     )
    #     frequency = np.ones(n) / n
    #     test_freq = np.array([0.01, 0.001])
    #     freq_out, val_out, fit_result = u_interp.fit_tail_distribution(
    #         test_frequency=test_freq,
    #         frequency=frequency,
    #         values=values,
    #         dist="GEV",
    #         threshold_percentile=0,
    #         min_sample_size=10,
    #     )
    #     self.assertEqual(len(freq_out), len(test_freq))
    #     self.assertEqual(len(val_out), len(test_freq))
    #     # np.testing.assert_allclose(
    #     #     [fit_result["xi"], fit_result["mu"], fit_result["sigma"]],
    #     #     [xi_true, mu_true, sigma_true],
    #     #     rtol=0.1,
    #     # )

    # def test_fit_tail_distribution_test_values(self):
    #     """Test with test_values"""
    #     rng = np.random.default_rng(42)
    #     xi_true = 0.1
    #     beta_true = 5
    #     n = 1000
    #     values = genpareto.rvs(c=xi_true, scale=beta_true, size=n, random_state=rng)
    #     frequency = np.ones(n) / n
    #     test_vals = np.array([10, 11])
    #     freq_out, val_out, fit_result = u_interp.fit_tail_distribution(
    #         test_values=test_vals,
    #         frequency=frequency,
    #         values=values,
    #         dist="GPD",
    #         threshold_percentile=90,
    #         min_sample_size=10,
    #     )
    #     np.testing.assert_equal(len(freq_out), len(test_vals))
    #     np.testing.assert_array_equal(val_out, test_vals)

    # def test_fit_tail_distribution_errors(self):
    #     """Test error cases"""
    #     values = np.array([1, 2, 3])
    #     frequency = np.array([0.3, 0.3, 0.4])
    #     # Invalid dist
    #     with self.assertRaises(ValueError):
    #         u_interp.fit_tail_distribution(
    #             test_frequency=np.array([0.01]),
    #             frequency=frequency,
    #             values=values,
    #             dist="INVALID",
    #         )
    #     # Both test_freq and test_values
    #     with self.assertRaises(ValueError):
    #         u_interp.fit_tail_distribution(
    #             test_frequency=np.array([0.01]),
    #             test_values=np.array([5]),
    #             frequency=frequency,
    #             values=values,
    #         )
    #     # Not enough data
    #     with self.assertRaises(ValueError):
    #         u_interp.fit_tail_distribution(
    #             test_frequency=np.array([0.01]),
    #             frequency=frequency,
    #             values=values,
    #             min_sample_size=50,
    #         )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFitMethods)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
