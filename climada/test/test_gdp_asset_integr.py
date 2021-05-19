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

Tests on GDP2Asset.
"""
import unittest
from climada.entity.exposures import gdp_asset as ga
from climada.util.constants import DEMO_GDP2ASSET

class TestGDP2AssetClassCountries(unittest.TestCase):
    """Unit tests for the GDP2Asset exposure class"""
    def test_wrong_iso3_fail(self):
        """Wrong ISO3 code"""
        testGDP2A = ga.GDP2Asset()

        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['OYY'], path=DEMO_GDP2ASSET)
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600, path=DEMO_GDP2ASSET)
        with self.assertRaises(LookupError):
            testGDP2A.set_countries(countries=['DEU'], ref_year=2600, path=DEMO_GDP2ASSET)
        with self.assertRaises(ValueError):
            testGDP2A.set_countries(path=DEMO_GDP2ASSET)
        with self.assertRaises(IOError):
            testGDP2A.set_countries(countries=['MEX'], path=DEMO_GDP2ASSET)

    def test_one_set_countries(self):
        testGDP2A_LIE = ga.GDP2Asset()
        testGDP2A_LIE.set_countries(countries=['LIE'], path=DEMO_GDP2ASSET)

        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[0, 2], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[1, 2], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[2, 2], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[3, 2], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[4, 2], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[5, 2], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[6, 2], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[7, 2], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[8, 2], 9.60403)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[9, 2], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[10, 2], 9.5623634)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[11, 2], 9.5206968)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[12, 2], 9.5623634)

        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[0, 1], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[1, 1], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[2, 1], 47.0622474)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[3, 1], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[4, 1], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[5, 1], 47.103914)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[6, 1], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[7, 1], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[8, 1], 47.1455806)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[9, 1], 47.1872472)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[10, 1], 47.1872472)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[11, 1], 47.2289138)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[12, 1], 47.2289138)

        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[0, 0], 174032107.65846416)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[1, 0], 20386409.991937194)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[2, 0], 2465206.6989314994)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[3, 0], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[4, 0], 12003959.733058406)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[5, 0], 97119771.42771776)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[6, 0], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[7, 0], 4137081.3646739507)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[8, 0], 27411196.308422357)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[9, 0], 0.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[10, 0], 4125847.312198318)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[11, 0], 88557558.43543366)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[12, 0], 191881403.05181965)

        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[0, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[12, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[0, 4], 11.0)
        self.assertAlmostEqual(testGDP2A_LIE.gdf.iloc[12, 4], 11.0)

    def test_two_countries(self):
        testGDP2A_LIE_CHE = ga.GDP2Asset()
        testGDP2A_LIE_CHE.set_countries(countries=['LIE', 'CHE'],
                                        path=DEMO_GDP2ASSET)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[0, 2], 9.520696799999968,
                               4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[45, 2], 7.39570019999996,
                               4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[1000, 2],
                               9.604029999999966, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[2500, 2],
                               9.395696999999984, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[0, 1],
                               47.062247399999976, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[45, 1],
                               45.978915799999996, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[1000, 1], 46.6039148, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[2500, 1],
                               47.3955802, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[0, 0],
                               174032107.65846416, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[45, 0],
                               11682292.467251074, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[1000, 0],
                               508470546.39168245, 4)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[2500, 0],
                               949321115.5175464, 4)

        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[0, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[12, 3], 3.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[0, 4], 11.0)
        self.assertAlmostEqual(testGDP2A_LIE_CHE.gdf.iloc[2500, 4], 11.0)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
            TestGDP2AssetClassCountries)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
