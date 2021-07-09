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

Test flood module.
"""

import unittest
from climada.hazard.river_flood import RiverFlood
from climada.util.constants import DEMO_GDP2ASSET, HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC
from climada.entity.exposures.gdp_asset import GDP2Asset
from climada.entity.impact_funcs.river_flood import flood_imp_func_set
from climada.engine import Impact


class TestRiverFlood(unittest.TestCase):
    """Test for reading flood event from file"""

    def test_exact_area_selection(self):
        testCentroids, iso, natID = RiverFlood._select_exact_area(['LIE'])

        self.assertEqual(testCentroids.lon.shape[0], 13)
        self.assertAlmostEqual(testCentroids.lon[0], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[1], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[2], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[3], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[4], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[5], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[6], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[7], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[8], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[9], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[10], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[11], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[12], 9.5623634)

        self.assertAlmostEqual(testCentroids.lat[0], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[1], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[2], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[3], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[4], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[5], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[6], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[7], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[8], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[9], 47.1872472)
        self.assertAlmostEqual(testCentroids.lat[10], 47.1872472)
        self.assertAlmostEqual(testCentroids.lat[11], 47.2289138)
        self.assertAlmostEqual(testCentroids.lat[12], 47.2289138)
        self.assertEqual(iso[0], 'LIE')

    def test_full_impact(self):
        """test full flood impact"""
        testRF = RiverFlood()
        testRF.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                           countries=['CHE'])

        gdpa = GDP2Asset()
        gdpa.set_countries(countries=['CHE'], ref_year=2000, path=DEMO_GDP2ASSET)

        impf_set = flood_imp_func_set()
        imp = Impact()
        imp.calc(gdpa, impf_set, testRF)

        self.assertAlmostEqual(imp.at_event[0], 226839.72426476143)
        self.assertAlmostEqual(gdpa.gdf['impf_RF'].iloc[0], 3.0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiverFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
