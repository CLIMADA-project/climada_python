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


from climada.util.value_representation import sig_dig, sig_dig_list, ABBREV
from climada.util.value_representation import value_to_monetary_unit
import unittest
import numpy as np
import math


class TestDigits(unittest.TestCase):
    """Test date functions"""

    def test_sig_dig_pass(self):
        """Test sig_dig function"""
        n_sig_dig = 3
        nbs_in = [1.2345, 12345, -12345, -12.345]
        nbs_out = [1.23, 12300, -12300, -12.3]
        for nb_in, nb_out in zip(nbs_in, nbs_out):
            self.assertEqual(sig_dig(nb_in, n_sig_dig), nb_out)
        self.assertTrue(
            np.array_equal(sig_dig_list(nbs_in, n_sig_dig), nbs_out)
            )

    def test_sig_dig_fail(self):
        """Test sig_dig function"""
        n_sig_dig_wrong = 4
        nbs_in = [1.2345, 12345, -12345, -12.345]
        nbs_out = [1.23, 12300, -12300, -12.3]
        for nb_in, nb_out in zip(nbs_in, nbs_out):
            self.assertNotEqual(sig_dig(nb_in, n_sig_dig_wrong), nb_out)
        self.assertFalse(
            np.array_equal(sig_dig_list(nbs_in, n_sig_dig_wrong), nbs_out)
            )

    def test_value_to_monetary_unit_pass(self):
        """Test money_unit function"""
        nbs_in = [-1e10, -1e6, -1e2, 0, 1e3, 1e7, 1e11]
        nbs_out = [-10, -1, -100, 0, 1, 10, 100]
        names_out = [ABBREV[1e9], ABBREV[1e6], ABBREV[1], ABBREV[1],
                 ABBREV[1e3], ABBREV[1e6], ABBREV[1e9]]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_0inf_pass(self):
        """Test money_unit function"""
        nbs_in = [-math.inf, 0, 1e-10, 1e-5, math.inf]
        nbs_out = [-math.inf, 0, 1e-10, 1e-5, math.inf]
        names_out = [ABBREV[1], ABBREV[1], ABBREV[1], ABBREV[1],
                 ABBREV[1], ABBREV[1], ABBREV[1]]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_nan_pass(self):
        """Test money_unit function"""
        nb_in = math.nan
        money, name = value_to_monetary_unit(nb_in)
        self.assertTrue(math.isnan(money[0]))
        self.assertEqual(name, '')


    def test_value_to_monetary_unit_sigdig_pass(self):
        """Test money_unit function with significant digits"""
        nbs_in = [-1e10*1.2345, -1e6*1.2345, -1e2*1.2345, 0, 1e3*1.2345,
                  1e7*1.2345, 1e11*1.2345]
        nbs_out = [-12.3, -1.23, -123, 0, 1.23, 12.3, 123]
        names_out = [ABBREV[1e9], ABBREV[1e6], ABBREV[1], ABBREV[1],
                 ABBREV[1e3], ABBREV[1e6], ABBREV[1e9]]
        for j, (nb_in, nb_out) in enumerate(zip(nbs_in, nbs_out)):
            money, names = value_to_monetary_unit(nb_in, n_sig_dig=3)
            self.assertEqual(money[0], nb_out)
            self.assertEqual(names, names_out[j])

    def test_value_to_monetary_unit_list_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10*1.2345, -1e9*1.2345]
        nbs_out = [-12.3, -1.23]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)
        nbs_in = [1e4*1.2345, 1e3*1.2345, 1e2*1.2345,]
        nbs_out = [12.3, 1.23, 0.123]
        name_out = ABBREV[1e3]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)

    def test_value_to_monetary_unit_list_0inf_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10*1.2345, -1e9*1.2345, 0, math.inf]
        nbs_out = [-12.3, -1.23, 0, math.inf]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)
        nbs_in = [1e4*1.2345, 1e3*1.2345, 1e2*1.2345, 0, math.inf]
        nbs_out = [12.3, 1.23, 0.123, 0, math.inf]
        name_out = ABBREV[1e3]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(np.array_equal(money, nbs_out))
        self.assertEqual(name, name_out)

    def test_value_to_monetary_unit_list_nan_pass(self):
        """Test money_unit function with list of numbers"""
        nbs_in = [-1e10*1.2345, -1e9*1.2345, math.nan]
        nbs_out = [-12.3, -1.23, math.nan]
        name_out = ABBREV[1e9]
        money, name = value_to_monetary_unit(nbs_in, n_sig_dig=3)
        self.assertTrue(math.isnan(nbs_out[-1]))
        self.assertTrue(np.array_equal(money[:-1], nbs_out[:-1]))
        self.assertEqual(name, name_out)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDigits)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
