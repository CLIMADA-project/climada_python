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

Test tc_surge module.
"""
import os
import copy
import unittest
import numpy as np

from climada.hazard import TCSurge, TropCyclone, TCTracks
from climada.hazard.centroids import Centroids

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'hazard/test/data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")

CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(os.path.join(DATA_DIR, 'centr_brb_test.mat'))

class TestEnd2End(unittest.TestCase):
    """Test reading TC from IBTrACS files"""

    def test_track_to_surge_point_pass(self):
        """ Test set_from_winds with default points (same as as_pixel=False) """

        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.calc_random_walk()
        tc_track.equal_timestep()

        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.check()

        tc_surge = TCSurge()
        tc_surge.set_from_winds(tc_haz)
        tc_surge.check()

        self.assertTrue(tc_surge.centroids is tc_haz.centroids)
        self.assertEqual(tc_surge.size, tc_haz.size)
        self.assertAlmostEqual(tc_surge.intensity[0, 6], 1.8288)
        self.assertAlmostEqual(tc_surge.intensity[9, 293], 4.170343378391969)
        self.assertEqual(tc_surge.fraction.min(), 0)
        self.assertEqual(tc_surge.fraction.max(), 1)
        self.assertEqual(np.unique(tc_surge.fraction.data).size, 1)

    def test_track_to_surge_raster_pass(self):
        """ Test set_from_winds with default raster (same as as_pixel=True) """

        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.calc_random_walk()
        tc_track.equal_timestep()

        centr_ras = copy.copy(CENTR_TEST_BRB)
        centr_ras.set_lat_lon_to_meta(min_resol=1.0e-2)
        centr_clean = Centroids()
        centr_clean.meta = centr_ras.meta
        centr_clean.check()

        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, centroids=centr_clean)
        tc_haz.check()

        tc_surge = TCSurge()
        tc_surge.set_from_winds(tc_haz)
        tc_surge.check()

        self.assertTrue(tc_surge.centroids is tc_haz.centroids)
        self.assertEqual(tc_surge.size, tc_haz.size)
        self.assertEqual(tc_surge.fraction.min(), 0)
        self.assertEqual(tc_surge.fraction.max(), 1)
        self.assertTrue(np.unique(tc_surge.fraction.data).size > 2)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEnd2End)
    unittest.TextTestRunner(verbosity=2).run(TESTS)