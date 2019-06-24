"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

Unit Tests on GDP2Asset exposures.

"""
import numpy as np
import unittest
from climada.entity.exposures import exp_people as ex


class TestExpPop(unittest.TestCase):
    """Unit tests for the LitPop exposure class"""
    def test_wrong_iso3_fail(self):
        """Wrong ISO3 code"""
        testExpPop = ex.ExpPop()

        with self.assertRaises(KeyError):
            testExpPop.set_countries(countries=['OYY'])
        with self.assertRaises(KeyError):
            testExpPop.set_countries(countries=['DEU'], ref_year=2600)
        with self.assertRaises(ValueError):
            testExpPop.set_countries()


class TestExpPopFunctions(unittest.TestCase):
    """Test ExpPop Class methods"""

    def test_set_one_country(self):
        exp_test = ex.ExpPop._set_one_country('LIE', 2000)

        self.assertAlmostEqual(exp_test.iloc[0, 2], 9.5206968)
        self.assertAlmostEqual(exp_test.iloc[1, 2], 9.5623634)
        self.assertAlmostEqual(exp_test.iloc[2, 2], 9.60403)
        self.assertAlmostEqual(exp_test.iloc[3, 2], 9.5206968)
        self.assertAlmostEqual(exp_test.iloc[4, 2], 9.5623634)
        self.assertAlmostEqual(exp_test.iloc[5, 2], 9.60403)
        self.assertAlmostEqual(exp_test.iloc[6, 2], 9.5206968)
        self.assertAlmostEqual(exp_test.iloc[7, 2], 9.5623634)
        self.assertAlmostEqual(exp_test.iloc[8, 2], 9.60403)
        self.assertAlmostEqual(exp_test.iloc[9, 2], 9.5206968)
        self.assertAlmostEqual(exp_test.iloc[10, 2], 9.5623634)
        self.assertAlmostEqual(exp_test.iloc[11, 2], 9.5206968)
        self.assertAlmostEqual(exp_test.iloc[12, 2], 9.5623634)

        self.assertAlmostEqual(exp_test.iloc[0, 1], 47.0622474)
        self.assertAlmostEqual(exp_test.iloc[1, 1], 47.0622474)
        self.assertAlmostEqual(exp_test.iloc[2, 1], 47.0622474)
        self.assertAlmostEqual(exp_test.iloc[3, 1], 47.103914)
        self.assertAlmostEqual(exp_test.iloc[4, 1], 47.103914)
        self.assertAlmostEqual(exp_test.iloc[5, 1], 47.103914)
        self.assertAlmostEqual(exp_test.iloc[6, 1], 47.1455806)
        self.assertAlmostEqual(exp_test.iloc[7, 1], 47.1455806)
        self.assertAlmostEqual(exp_test.iloc[8, 1], 47.1455806)
        self.assertAlmostEqual(exp_test.iloc[9, 1], 47.1872472)
        self.assertAlmostEqual(exp_test.iloc[10, 1], 47.1872472)
        self.assertAlmostEqual(exp_test.iloc[11, 1], 47.2289138)
        self.assertAlmostEqual(exp_test.iloc[12, 1], 47.2289138)

        self.assertAlmostEqual(exp_test.iloc[0, 0], 1610.171753, 4)
        self.assertAlmostEqual(exp_test.iloc[1, 0], 1593.029907, 4)
        self.assertAlmostEqual(exp_test.iloc[2, 0], 16.597307, 4)
        self.assertAlmostEqual(exp_test.iloc[3, 0], 3181.014160, 4)
        self.assertAlmostEqual(exp_test.iloc[4, 0], 3148.719482, 4)
        self.assertAlmostEqual(exp_test.iloc[5, 0], 178.773102, 4)
        self.assertAlmostEqual(exp_test.iloc[6, 0], 3195.352051, 4)
        self.assertAlmostEqual(exp_test.iloc[7, 0], 3163.309326, 4)
        self.assertAlmostEqual(exp_test.iloc[8, 0], 216.529587, 4)
        self.assertAlmostEqual(exp_test.iloc[9, 0], 5034.735840, 4)
        self.assertAlmostEqual(exp_test.iloc[10, 0], 5035.010742, 4)
        self.assertAlmostEqual(exp_test.iloc[11, 0], 5033.083008, 4)
        self.assertAlmostEqual(exp_test.iloc[12, 0], 5033.828613, 4)

        self.assertAlmostEqual(exp_test.iloc[0, 3], 7.0)
        self.assertAlmostEqual(exp_test.iloc[12, 3], 7.0)
        self.assertAlmostEqual(exp_test.iloc[0, 4], 11.0)
        self.assertAlmostEqual(exp_test.iloc[12, 4], 11.0)
        self.assertAlmostEqual(exp_test.iloc[0, 5], 118.0)
        self.assertAlmostEqual(exp_test.iloc[12, 5], 118.0)

    def test_read_people(self):

        exp = ex.ExpPop._set_one_country('LIE', 2000)
        coordinates = np.zeros((exp.shape[0], 2))
        coordinates[:, 0] = np.array(exp['latitude'])
        coordinates[:, 1] = np.array(exp['longitude'])

        with self.assertRaises(KeyError):
            ex._read_people(coordinates, ref_year=2600)

        exp = ex._read_people(coordinates, ref_year=2000)

        self.assertAlmostEqual(exp[0], 1610.171753, 4)
        self.assertAlmostEqual(exp[1], 1593.029907, 4)
        self.assertAlmostEqual(exp[2], 16.597307, 4)
        self.assertAlmostEqual(exp[3], 3181.014160, 4)
        self.assertAlmostEqual(exp[4], 3148.719482, 4)
        self.assertAlmostEqual(exp[5], 178.773102, 4)
        self.assertAlmostEqual(exp[6], 3195.352051, 4)
        self.assertAlmostEqual(exp[7], 3163.309326, 4)
        self.assertAlmostEqual(exp[8], 216.529587, 4)
        self.assertAlmostEqual(exp[9], 5034.735840, 4)
        self.assertAlmostEqual(exp[10], 5035.010742, 4)
        self.assertAlmostEqual(exp[11], 5033.083008, 4)
        self.assertAlmostEqual(exp[12], 5033.828613, 4)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestExpPop)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(
            TestExpPopFunctions))
    unittest.TextTestRunner(verbosity=2).run(TESTS)