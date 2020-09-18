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

Test CentroidsVector and CentroidsRaster classes.
"""
import os
import unittest

import numpy as np
import pandas as pd
import geopandas as gpd

from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT, HAZ_TEMPLATE_XLS

HAZ_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                       'test/data/')
HAZ_TEST_MAT = os.path.join(HAZ_DIR, 'atl_prob_no_name.mat')


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

        centroids = Centroids().from_base_grid(land=True, res_as=150)

        count_sandwich = np.sum(centroids.region_id == 239)

        self.assertEqual(centroids.lat.size, 8858035)
        self.assertEqual(count_sandwich, 321)

        count_sgi = centroids.select(
            reg_id=239,
            extent=(-39, -34.7, -55.5, -53.6)  # south georgia island
        ).size

        self.assertEqual(count_sgi, 296)

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


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReader)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
