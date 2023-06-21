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

HAZ_TEST_MAT = Path(hazard_test.__file__).parent / 'data' / 'atl_prob_no_name.mat'


class TestCentroidsReader(unittest.TestCase):
    """Test read functions Centroids"""

    def test_mat_pass(self):
        """Read a centroid mat file correctly."""
        centroids = Centroids.from_mat(HAZ_TEST_MAT)

        n_centroids = 100
        self.assertEqual(centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(centroids.coord[0][0], 21)
        self.assertEqual(centroids.coord[0][1], -84)
        self.assertEqual(centroids.coord[n_centroids - 1][0], 30)
        self.assertEqual(centroids.coord[n_centroids - 1][1], -75)

    def test_centroid_pass(self):
        """Read a centroid excel file correctly."""
        centroids = Centroids.from_excel(HAZ_TEMPLATE_XLS)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 33.88)

    def test_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        gdf = gpd.GeoDataFrame(pd.read_excel(HAZ_TEMPLATE_XLS))
        gdf.geometry = gpd.points_from_xy(
                gdf['longitude'], gdf['latitude']
        )
        gdf['elevation'] = np.random.rand(gdf.geometry.size)
        gdf['region_id'] = np.zeros(gdf.geometry.size)
        gdf['region_id'][0] = np.NaN
        gdf['geom'] = gdf.geometry  # this should have no effect on centroids

        centroids = Centroids.from_geodataframe(gdf)

        self.assertEqual(centroids.geometry.size, 45)
        self.assertEqual(centroids.lon[0], 32.57)
        self.assertEqual(centroids.lat[0], -25.95)
        self.assertEqual(centroids.elevation.size, 45)
        self.assertIsInstance(centroids.geometry, gpd.GeoSeries)
        self.assertIsInstance(centroids.geometry.total_bounds, np.ndarray)


class TestCentroidsWriter(unittest.TestCase):

    def test_write_hdf5(self):
        tmpfile = 'test_write_hdf5.out.hdf5'
        xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon = 5, 6.5, -0.08, 0.12, 4, 5
        centr = Centroids.from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon)
        with self.assertLogs('climada.hazard.centroids.centr', level='INFO') as cm:
            centr.write_hdf5(tmpfile)
        self.assertEqual(1, len(cm.output))
        self.assertIn(f"Writing {tmpfile}", cm.output[0])
        centr.meta['nodata'] = None
        with self.assertLogs('climada.hazard.centroids.centr', level='INFO') as cm:
            centr.write_hdf5(tmpfile)
        self.assertEqual(2, len(cm.output))
        self.assertIn("Skip writing Centroids.meta['nodata'] for it is None.", cm.output[1])
        Path(tmpfile).unlink()


class TestCentroidsMethods(unittest.TestCase):

    def test_union(self):
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        lat2, lon2 = np.array([2, 3]), np.array([-2, 3])
        on_land2 = np.array([False, False])
        cent2 = Centroids(lat=lat2, lon=lon2, on_land=on_land2)

        lat3, lon3 = np.array([-1, -2]), np.array([1, 2])
        cent3 = Centroids(lat=lat3,lon=lon3)

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
