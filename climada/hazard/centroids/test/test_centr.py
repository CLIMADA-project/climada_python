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

Test CentroidsVector and CentroidsRaster classes.
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from climada import CONFIG
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT, HAZ_TEMPLATE_XLS
import climada.hazard.test as hazard_test
from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coord


HAZ_TEST_MAT = Path(hazard_test.__file__).parent / 'data' / 'atl_prob_no_name.mat'


class TestCentroidsReader(unittest.TestCase):
    """Test read functions Centroids"""

    def test_centroid_pass(self):
        """Read a centroid excel file correctly."""
        centroids = Centroids.from_excel(HAZ_TEMPLATE_XLS, crs=DEF_CRS)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 33.88)

    def test_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        crs = DEF_CRS
        lat = np.arange(170, 180)
        lon = np.arange(-50, -40)
        region_id = np.arange(1, 11)
        extra = np.repeat(str('a'), 10)

        gdf = gpd.GeoDataFrame({
            'geometry' : gpd.points_from_xy(lon, lat),
            'region_id' : region_id,
            'extra' : extra
        }, crs=crs)

        centroids = Centroids.from_geodataframe(gdf)

        for name, array in zip(['lat', 'lon', 'region_id'],
                                [lat, lon, region_id]):
            np.testing.assert_array_equal(array, getattr(centroids, name))
        np.testing.assert_array_equal(extra, centroids.gdf.extra.values)
        self.assertTrue(u_coord.equal_crs(centroids.crs, crs))


class TestCentroidsWriter(unittest.TestCase):

    def test_read_write_hdf5(self):
        tmpfile = Path('test_write_hdf5.out.hdf5')
        latitude = np.arange(0,10)
        longitude = np.arange(-10,0)
        crs = DEF_CRS
        centroids_w = Centroids(latitude=latitude, longitude=longitude, crs=crs)
        centroids_w.write_hdf5(tmpfile)
        centroids_r = Centroids.from_hdf5(tmpfile)
        self.assertTrue(centroids_w == centroids_r)
        Path(tmpfile).unlink()


class TestCentroidsMethods(unittest.TestCase):

    def test_union(self):
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(latitude=lat, longitude=lon, on_land=on_land)

        lat2, lon2 = np.array([2, 3]), np.array([-2, 3])
        on_land2 = np.array([False, False])
        cent2 = Centroids(latitude=lat2, longitude=lon2, on_land=on_land2)

        lat3, lon3 = np.array([-1, -2]), np.array([1, 2])
        cent3 = Centroids(latitude=lat3,longitude=lon3)

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = cent1.union(cent1, cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = Centroids.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = cent1.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = Centroids.union(cent1, cent2, cent3)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3, -1, -2])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3, 1, 2])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsMethods))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
