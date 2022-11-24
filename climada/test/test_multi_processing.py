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

Unit tests in multiprocessing mode are time consuming due to the overhead of setting up
processes or threads. In order to keep the load on automated tests small that run whenever a commit
is made to any branch, this test module is meant to catch failures in multi processing mode in
a test file that is run nightly only.
"""
import unittest

import numpy as np
import geopandas as gpd

import climada.util.coordinates as u_coord


class TestCoordinates(unittest.TestCase):
    """Test data coordinates methods in multi processing mode."""

    def test_set_df_geometry_points_scheduled_pass(self):
        """Test set_df_geometry_points with a scheduler other than None

        The same test runs with scheduler None runs in
        climada.util.test.test_coordinates.TestFunc.test_set_df_geometry_points_pass
        """
        for scheduler in ['threads', 'synchronous', 'processes']:
            df_val = gpd.GeoDataFrame()
            df_val['latitude'] = np.ones(10) * 40.0
            df_val['longitude'] = np.ones(10) * 0.50

            # set_df_geometry_points with a scheduler other than None uses dask.dataframe
            # methods to execute calculations in parallel
            u_coord.set_df_geometry_points(df_val, scheduler=scheduler, crs='epsg:2202')
            np.testing.assert_allclose(df_val.geometry.x.values, np.ones(10) * 0.5)
            np.testing.assert_allclose(df_val.geometry.y.values, np.ones(10) * 40.)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCoordinates)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
