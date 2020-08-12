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

Test HazRegion class and georegion functionalities
"""

import unittest

import numpy as np
from shapely.geometry import Point

from climada.hazard.emulator import geo


class TestGeo(unittest.TestCase):
    """Test georegion functionalities"""

    def test_init_hazregion(self):
        """Test initialization of HazRegion class with different parameters"""
        reg = geo.HazRegion(extent=[0, 1, 0, 1])
        self.assertTrue(reg.shape.contains(Point(0.1, 0.1)))
        self.assertTrue(reg.shape.contains(Point(0.9, 0.1)))
        self.assertTrue(reg.shape.contains(Point(0.1, 0.9)))
        self.assertTrue(reg.shape.contains(Point(0.9, 0.9)))
        self.assertFalse(reg.shape.contains(Point(-0.1, 0.1)))
        self.assertFalse(reg.shape.contains(Point(0.5, 1.1)))

        # restrict region to land area
        reg = geo.HazRegion(extent=[6, 11, 3, 6], country="all")
        # point in Cameroon
        self.assertTrue(reg.shape.contains(Point(10.7, 4)))
        # point on Bioko (island)
        self.assertTrue(reg.shape.contains(Point(8.7, 3.5)))
        # point outside extent, but on land
        self.assertFalse(reg.shape.contains(Point(10, 2.8)))
        # point within extent, but in ocean
        self.assertFalse(reg.shape.contains(Point(9.3, 3.2)))

        # Test for country Malta
        reg = geo.HazRegion(extent=[14.3, 14.6, 35.85, 36], country="MLT")
        # Test for capital Valetta
        self.assertTrue(reg.shape.contains(Point(14.5167, 35.9)))
        # Inside of extent, but outside of Malta
        self.assertFalse(reg.shape.contains(Point(14.5, 35.95)))
        # Inside Malta, but outside of extent
        self.assertFalse(reg.shape.contains(Point(14.5, 35.83)))

        reg = geo.HazRegion(geometry=reg.geometry)
        # same tests as for Malta
        self.assertTrue(reg.shape.contains(Point(14.5167, 35.9)))
        self.assertFalse(reg.shape.contains(Point(14.5, 35.95)))
        self.assertFalse(reg.shape.contains(Point(14.5, 35.83)))


    def test_hazregion_centroids(self):
        """Test HazRegion.centroids method"""
        reg = geo.HazRegion(country="MLT")
        cen = reg.centroids()
        self.assertEqual(cen.lat.size, 2)
        self.assertAlmostEqual(cen.lat[0], 35.9)
        self.assertAlmostEqual(cen.lat[1], 35.9)
        self.assertAlmostEqual(cen.lon[0], 14.4)
        self.assertAlmostEqual(cen.lon[1], 14.5)

        cen = reg.centroids(latlon=(np.array([35.9, 36.0]), np.array([14.4, 14.4])))
        self.assertEqual(cen.lat.size, 1)
        self.assertAlmostEqual(cen.lat[0], 35.9)
        self.assertAlmostEqual(cen.lon[0], 14.4)


    def test_get_tc_basin_geometry(self):
        """Test get_tc_basin_geometry"""
        bas = geo.get_tc_basin_geometry("NA")
        polygon = bas.geometry[0]
        self.assertTrue(polygon.contains(Point(0.0, 1.0)))
        self.assertTrue(polygon.contains(Point(-90.0, 16.0)))
        self.assertFalse(polygon.contains(Point(-80.0, 1.0)))
        self.assertFalse(polygon.contains(Point(0.0, -1.0)))
        self.assertFalse(polygon.contains(Point(0.0, 61.0)))


    def test_tc_region(self):
        """Test TCRegion class"""
        # automatically determine basin
        reg = geo.TCRegion(extent=[0, 1, 0, 1])
        self.assertEqual(reg.hemisphere, "N")
        self.assertEqual(reg.tc_basin, "NAS")
        self.assertEqual(reg.season, [6, 11])
        reg = geo.TCRegion(extent=[70, 80, -30, -20])
        self.assertEqual(reg.hemisphere, "S")
        self.assertEqual(reg.tc_basin, "SI")

        # init by given basin name
        reg = geo.TCRegion(tc_basin="EP", season=[6, 12])
        self.assertEqual(reg.hemisphere, "N")
        self.assertEqual(reg.tc_basin, "EP")
        self.assertEqual(reg.season, [6, 12])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGeo)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
