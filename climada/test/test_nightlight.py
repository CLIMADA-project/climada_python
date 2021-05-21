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

Tests on Black marble.
"""

import unittest
import numpy as np
from cartopy.io import shapereader
from shapely.geometry import Polygon

from climada.entity.exposures.litpop import nightlight as nl_utils

from climada.util.constants import SYSTEM_DIR

def init_test_shape():
    bounds = (14.18, 35.78, 14.58, 36.09) 
    # (min_lon, max_lon, min_lat, max_lat)

    return bounds, Polygon([
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[1])
        ])

class TestNightlight(unittest.TestCase):
    """Test litpop.nightlight"""

    def test_load_nasa_nl_shape_single_tile_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()

    def test_load_nasa_nl_2016_shape_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()
        nl_utils.load_nasa_nl_shape(shape, 2016, data_dir=SYSTEM_DIR, dtype=None)

        # data, meta = nl_utils.load_nasa_nl_shape_single_tile(shape, path, layer=0):
        """
        country_name = ['Spain']
        ent = BlackMarble()
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=1)
        self.assertIn('GDP ESP 2013: 1.355e+12.', cm.output[0])
        self.assertIn('Income group ESP 2013: 4.', cm.output[1])

        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=1)
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.",
                      cm.output[0])
        self.assertIn("Processing country Spain.", cm.output[1])
        self.assertIn("Generating resolution of approx 1 km.", cm.output[2])
        self.assertTrue(np.isclose(ent.gdf.value.sum(), 1.355e+12 * (4 + 1), 0.001))
        self.assertTrue(u_coord.equal_crs(ent.crs, 'epsg:4326'))
        self.assertEqual(ent.meta['width'], 2699)
        self.assertEqual(ent.meta['height'], 1938)
        self.assertTrue(u_coord.equal_crs(ent.meta['crs'], 'epsg:4326'))
        self.assertAlmostEqual(ent.meta['transform'][0], 0.008333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][1], 0)
        self.assertAlmostEqual(ent.meta['transform'][2], -18.1625000000000)
        self.assertAlmostEqual(ent.meta['transform'][3], 0)
        self.assertAlmostEqual(ent.meta['transform'][4], -0.008333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][5], 43.79583333333333)

    def test_sint_maarten_pass(self):
        country_name = ['Sint Maarten']
        ent = BlackMarble()
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertIn('GDP SXM 2013: 1.023e+09.', cm.output[0])
        self.assertIn('Income group SXM 2013: 4.', cm.output[1])
        self.assertTrue(u_coord.equal_crs(ent.crs, 'epsg:4326'))

        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.",
                      cm.output[0])
        self.assertIn("Processing country Sint Maarten.", cm.output[1])
        self.assertIn("Generating resolution of approx 0.2 km.", cm.output[2])
        self.assertTrue(np.isclose(ent.gdf.value.sum(), 1.023e+09 * (4 + 1), 0.001))
        self.assertTrue(u_coord.equal_crs(ent.crs, 'epsg:4326'))

    def test_anguilla_pass(self):
        country_name = ['Anguilla']
        ent = BlackMarble()
        ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertEqual(ent.ref_year, 2013)
        self.assertIn("Anguilla 2013 GDP: 1.754e+08 income group: 3", ent.tag.description)
        self.assertAlmostEqual(ent.gdf.value.sum(), 1.754e+08 * (3 + 1))
        self.assertTrue(u_coord.equal_crs(ent.crs, 'epsg:4326'))"""


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightlight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
