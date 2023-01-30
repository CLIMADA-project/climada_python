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

Test Hazard base class.
"""

import unittest
import numpy as np
from scipy import sparse

from climada import CONFIG
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.util.constants import HAZ_DEMO_FL

DATA_DIR = CONFIG.test_data.dir()

class TestCentroids(unittest.TestCase):
    """Test centroids functionalities"""

    def test_read_write_raster_pass(self):
        """Test write_raster: Hazard from raster data"""
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL])
        haz_fl.tag.haz_type = 'FL'
        haz_fl.check()

        self.assertEqual(haz_fl.intensity.shape, (1, 1032226))
        self.assertEqual(haz_fl.intensity.min(), -9999.0)
        self.assertAlmostEqual(haz_fl.intensity.max(), 4.662774085998535)

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'))

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')])
        haz_fl.tag.haz_type = 'FL'
        self.assertTrue(np.allclose(haz_fl.intensity.toarray(), haz_read.intensity.toarray()))
        self.assertEqual(np.unique(np.array(haz_fl.fraction.toarray())).size, 2)

    def test_read_raster_pool_pass(self):
        """Test from_raster constructor with pool"""
        from pathos.pools import ProcessPool as Pool
        pool = Pool()
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL', pool=pool)
        haz_fl.check()

        self.assertEqual(haz_fl.intensity.shape, (1, 1032226))
        self.assertEqual(haz_fl.intensity.min(), -9999.0)
        self.assertAlmostEqual(haz_fl.intensity.max(), 4.662774085998535)
        pool.close()
        pool.join()

    def test_read_write_vector_pass(self):
        """Test write_raster: Hazard from vector data"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'))

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')], haz_type='FL')
        self.assertEqual(haz_read.intensity.shape, (1, 9))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.intensity.toarray())),
                                    np.array([0.0, 0.1, 0.2, 0.5])))

    def test_write_fraction_pass(self):
        """Test write_raster with fraction"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'), intensity=False)

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')],
                                      files_fraction=[DATA_DIR.joinpath('test_write_hazard.tif')],
                                      haz_type='FL')
        self.assertEqual(haz_read.intensity.shape, (1, 9))
        self.assertEqual(haz_read.fraction.shape, (1, 9))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.fraction.toarray())),
                                    np.array([0.0, 0.05, 0.1, 0.25])))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.intensity.toarray())),
                                    np.array([0.0, 0.05, 0.1, 0.25])))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroids)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
