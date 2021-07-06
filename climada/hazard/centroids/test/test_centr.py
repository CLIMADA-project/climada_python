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

import numpy as np
import pandas as pd
import geopandas as gpd

from climada import CONFIG
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT, HAZ_TEMPLATE_XLS

HAZ_DIR = CONFIG.hazard.test_data.dir()
HAZ_TEST_MAT = HAZ_DIR.joinpath('atl_prob_no_name.mat')


class TestCentroidsReader(unittest.TestCase):
    """Test read functions Centroids"""

    def test_mat_pass(self):
        """Read a centroid mat file correctly."""
        centroids = Centroids()
        centroids.read_mat(HAZ_TEST_MAT)

        n_centroids = 100
        self.assertEqual(centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(centroids.coord[0][0], 21)
        self.assertEqual(centroids.coord[0][1], -84)
        self.assertEqual(centroids.coord[n_centroids - 1][0], 30)
        self.assertEqual(centroids.coord[n_centroids - 1][1], -75)

    def test_mat_global_pass(self):
        """Test read GLB_CENTROIDS_MAT"""
        centroids = Centroids()
        centroids.read_mat(GLB_CENTROIDS_MAT)

        self.assertEqual(centroids.region_id[1062443], 35)
        self.assertEqual(centroids.region_id[170825], 28)

    def test_centroid_pass(self):
        """Read a centroid excel file correctly."""
        centroids = Centroids()
        centroids.read_excel(HAZ_TEMPLATE_XLS)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 33.88)

    def test_base_grid(self):
        """Read new centroids using from_base_grid, then select by extent."""
        centroids = Centroids.from_base_grid(land=True, res_as=150)
        self.assertEqual(centroids.lat.size, 8858035)
        self.assertTrue(np.all(np.diff(centroids.lat) <= 0))

        count_sandwich = np.sum(centroids.region_id == 239)
        self.assertEqual(count_sandwich, 321)

        count_sgi = centroids.select(
            reg_id=239,
            extent=(-39, -34.7, -55.5, -53.6)  # south georgia island
        ).size
        self.assertEqual(count_sgi, 296)

        # test negative latitudinal orientation by testing that northern hemisphere (Russia)
        # is listed before southern hemisphere (South Africa)
        russia_max_idx = (centroids.region_id == 643).nonzero()[0].max()
        safrica_min_idx = (centroids.region_id == 710).nonzero()[0].min()
        self.assertTrue(russia_max_idx < safrica_min_idx)

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
        centroids.check()

        self.assertEqual(centroids.geometry.size, 45)
        self.assertEqual(centroids.lon[0], 32.57)
        self.assertEqual(centroids.lat[0], -25.95)
        self.assertEqual(centroids.elevation.size, 45)
        self.assertEqual(centroids.on_land.sum(), 44)
        self.assertIsInstance(centroids.geometry, gpd.GeoSeries)
        self.assertIsInstance(centroids.geometry.total_bounds, np.ndarray)

class TestCentroidsMethods(unittest.TestCase):

    def test_union(self):
        cent1 = Centroids()
        cent1.lat, cent1.lon = np.array([0, 1]), np.array([0, -1])
        cent1.on_land = np.array([True, True])

        cent2 = Centroids()
        cent2.lat, cent2.lon = np.array([2, 3]), np.array([-2, 3])
        cent2.on_land = np.array([False, False])

        cent3 = Centroids()
        cent3.lat, cent3.lon = np.array([-1, -2]), np.array([1, 2])

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = cent1.union(cent1, cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = Centroids().union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = cent1.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = Centroids().union(cent1, cent2, cent3)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3, -1, -2])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3, 1, 2])


    def test_union_meta(self):
        cent1 = Centroids()
        cent1.set_raster_from_pnt_bounds((-1, -1, 0, 0), res=1)

        cent2 = Centroids()
        cent2.set_raster_from_pnt_bounds((0, 0, 1, 1), res=1)

        cent3 = Centroids()
        cent3.lat, cent3.lon = np.array([1]), np.array([1])

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0,  0, -1, -1,  1,  1,  0])
        np.testing.assert_array_equal(cent.lon, [-1,  0, -1,  0,  0,  1,  1])

        cent = cent3.union(cent1)
        np.testing.assert_array_equal(cent.lat, [1,  0,  0, -1, -1])
        np.testing.assert_array_equal(cent.lon, [1, -1,  0, -1,  0])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsMethods))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
