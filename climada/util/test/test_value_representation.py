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

Test of util.math module
"""

import math
import unittest

import numpy as np

from climada.util.value_representation import (
    ABBREV,
    safe_divide,
    sig_dig,
    sig_dig_list,
    value_to_monetary_unit,
)


class TestDigits(unittest.TestCase):
    """Test date functions"""

    def test_sig_dig_pass(self):
        """Test sig_dig function"""
        n_sig_dig = 3
        nbs_in = [1.2345, 12345, -12345, -12.345]
        nbs_out = [1.23, 12300, -12300, -12.3]
        for nb_in, nb_out in zip(nbs_in, nbs_out):
            self.assertEqual(sig_dig(nb_in, n_sig_dig), nb_out)
        self.assertTrue(np.array_equal(sig_dig_list(nbs_in, n_sig_dig), nbs_out))

    def test_sig_dig_fail(self):
        """Test sig_dig function"""
        n_sig_dig_wrong = 4
        nbs_in = [1.2345, 12345, -12345, -12.345]
        nbs_out = [1.23, 12300, -12300, -12.3]
        for nb_in, nb_out in zip(nbs_in, nbs_out):
            self.assertNotEqual(sig_dig(nb_in, n_sig_dig_wrong), nb_out)
        self.assertFalse(np.array_equal(sig_dig_list(nbs_in, n_sig_dig_wrong), nbs_out))

    def test_value_to_monetary_unit_pass(self):
        """Test money_unit function"""
        nbs_in = [-1e10, -1e6, -1e2, 0, 1e3, 1e7, 1e11]
        nbs_out = [-10, -1, -100, 0, 1, 10, 100]
        names_out = [
            ABBREV[1e9],
            ABBREV[1e6],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1e3],
            ABBREV[1e6],
            ABBREV[1e9],
        ]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_0inf_pass(self):
        """Test money_unit function"""
        nbs_in = [-math.inf, 0, 1e-10, 1e-5, math.inf]
        nbs_out = [-math.inf, 0, 1e-10, 1e-5, math.inf]
        names_out = [
            ABBREV[1],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1],
        ]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_nan_pass(self):
        """Test money_unit function"""
        nb_in = math.nan
        money, name = value_to_monetary_unit(nb_in)
        self.assertTrue(math.isnan(money[0]))
        self.assertEqual(name, "")

    def test_value_to_monetary_unit_sigdig_pass(self):
        """Test money_unit function with significant digits"""
        nbs_in = [
            -1e10 * 1.2345,
            -1e6 * 1.2345,
            -1e2 * 1.2345,
            0,
            1e3 * 1.2345,
            1e7 * 1.2345,
            1e11 * 1.2345,
        ]
        nbs_out = [-12.3, -1.23, -123, 0, 1.23, 12.3, 123]
        names_out = [
            ABBREV[1e9],
            ABBREV[1e6],
            ABBREV[1],
            ABBREV[1],
            ABBREV[1e3],
            ABBREV[1e6],
            ABBREV[1e9],
        ]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in, n_sig_dig=3)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_list_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10 * 1.2345, -1e9 * 1.2345]
        nbs_out = [-12.3, -1.23]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)
        nbs_in = [
            1e4 * 1.2345,
            1e3 * 1.2345,
            1e2 * 1.2345,
        ]
        nbs_out = [12.3, 1.23, 0.123]
        name_out = ABBREV[1e3]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)

    def test_value_to_monetary_unit_list_0inf_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10 * 1.2345, -1e9 * 1.2345, 0, math.inf]
        nbs_out = [-12.3, -1.23, 0, math.inf]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)
        nbs_in = [1e4 * 1.2345, 1e3 * 1.2345, 1e2 * 1.2345, 0, math.inf]
        nbs_out = [12.3, 1.23, 0.123, 0, math.inf]
        name_out = ABBREV[1e3]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)

    def test_value_to_monetary_unit_list_nan_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10 * 1.2345, -1e9 * 1.2345, math.nan]
        nbs_out = [-12.3, -1.23, math.nan]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(math.isnan(nbs_out[-1]))
        self.assertTrue(np.array_equal(money[:-1], nbs_out[:-1]))
        self.assertEqual(name, name_out)


class TestSafeDivide(unittest.TestCase):

    def test_scalar_division(self):
        self.assertEqual(safe_divide(10, 2), 5)
        self.assertEqual(safe_divide(-10, 5), -2)

    def test_scalar_division_by_zero(self):
        self.assertTrue(np.isnan(safe_divide(1, 0)))
        self.assertEqual(safe_divide(1, 0, replace_with=0), 0)

    def test_array_division(self):
        np.testing.assert_array_equal(
            safe_divide(np.array([10, 20, 30]), np.array([2, 5, 10])),
            np.array([5, 4, 3]),
        )

    def test_array_division_by_zero(self):
        np.testing.assert_array_equal(
            safe_divide(np.array([1, 0, 3]), np.array([0, 0, 1])),
            np.array([np.nan, np.nan, 3]),
        )
        np.testing.assert_array_equal(
            safe_divide(np.array([1, 0, 3]), np.array([0, 0, 1]), replace_with=0),
            np.array([0, 0, 3]),
        )

    def test_list_division_by_zero(self):
        list_num = [10, 0, 30]
        list_denom = [2, 0, 10]
        expected_result = [5.0, np.nan, 3.0]
        np.testing.assert_array_almost_equal(
            safe_divide(list_num, list_denom), expected_result
        )

    def test_list_division(self):
        list_num = [10, 20, 30]
        list_denom = [2, 5, 10]
        expected_result = [5.0, 4.0, 3.0]
        np.testing.assert_array_almost_equal(
            safe_divide(list_num, list_denom), expected_result
        )

    def test_nan_handling(self):
        self.assertTrue(np.isnan(safe_divide(np.nan, 1)))
        self.assertTrue(np.isnan(safe_divide(1, np.nan)))
        self.assertEqual(safe_divide(np.nan, 1, replace_with=0), 0)
        self.assertEqual(safe_divide(1, np.nan, replace_with=0), 0)

    def test_nan_handling_in_arrays(self):
        np.testing.assert_array_equal(
            safe_divide(np.array([1, np.nan, 3]), np.array([3, 2, 0])),
            np.array([1 / 3, np.nan, np.nan]),
        )

    def test_nan_handling_in_scalars(self):
        self.assertTrue(np.isnan(safe_divide(np.nan, 1)))
        self.assertTrue(np.isnan(safe_divide(1, np.nan)))
        self.assertEqual(safe_divide(np.nan, 1, replace_with=0), 0)
        self.assertEqual(safe_divide(1, np.nan, replace_with=0), 0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDigits)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSafeDivide))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
