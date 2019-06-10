
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

---

Tests on GDP2Asset.
"""
import unittest
from climada.entity.exposures import gdp_asset as ga


class TestGDP2AssetClassCountries(unittest.TestCase):
    """Unit tests for the GDP2Asset exposure class"""
    def test_wrong_iso3_fail(self):
        """Wrong ISO3 code"""
        testGDP2A = ga.GDP2Asset()

        with self.assertRaises(KeyError):
            testGDP2A.set_countries(countries=['OYY'])
        with self.assertRaises(KeyError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600)
    def test_set_countries(self):
        testGDP2A_DEU = ga.GDP2Asset()
        testGDP2A_DEU.set_countries(countries=['DEU'])
        testGDP2A_RUS = ga.GDP2Asset()
        testGDP2A_RUS.set_countries(countries=['RUS'])
        testGDP2A_DEU_BRA = ga.GDP2Asset()
        testGDP2A_DEU_BRA.set_countries(countries=['DEU', 'BRA'])
        self.assertEqual(testGDP2A_DEU.shape[0], 26878)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[0, 0], 706843.8067164791)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[0, 1], 47.312247)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[0, 2], 10.229028999999969)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[0, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[0, 4], 11.0)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[26877, 0], 1054654.9189984929)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[26877, 1], 55.0622346)
        self.assertAlmostEqual(testGDP2A_DEU.iloc[26877, 2], 8.437365199999988)

        self.assertEqual(testGDP2A_RUS.shape[0], 1698649)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[0, 0], 316894.73009594914)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[0, 1], 41.2289234)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[0, 2], 47.52063599999997)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[0, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[0, 4], 9.0)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[1698648, 0], 0.0)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[1698648, 1], 81.8538584)
        self.assertAlmostEqual(testGDP2A_RUS.iloc[1698648, 2], 59.353950399999974)

        self.assertEqual(testGDP2A_DEU_BRA.shape[0], 437227)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[0, 0], 706843.8067164791)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[0, 1], 47.312247)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[0, 2], 10.229028999999969)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[0, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[0, 4], 11.0)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[437226, 0], 835.6131181309585)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[437226, 1], 5.22898099999999)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[437226, 2], -60.14585840000002)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[437226, 3], 6.0)
        self.assertAlmostEqual(testGDP2A_DEU_BRA.iloc[437226, 4], 3.0)



if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGDP2AssetClassCountries)
    unittest.TextTestRunner(verbosity=2).run(TESTS)