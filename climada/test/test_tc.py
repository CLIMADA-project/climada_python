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

Test TropCyclone class with multiprocess
"""

import os
import unittest
# from pathos.pools import ProcessPool as Pool
# from scipy import sparse

# from climada.hazard.tc_tracks import TCTracks
# from climada.hazard.trop_cyclone import TropCyclone
from climada.hazard.centroids.centr import Centroids

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'hazard/test/data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")

CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(os.path.join(DATA_DIR, 'centr_brb_test.mat'))

class TestTCParallel(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""
#    def test_set_one_file_pass(self):
#        """Test set function set_from_tracks with one input."""
#
#        pool = Pool()
#
#        tc_track = TCTracks(pool)
#
#        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
#        tc_track.calc_random_walk()
#        tc_track.equal_timestep()
#
#        tc_haz = TropCyclone(pool)
#        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
#        tc_haz.check()
#
#        pool.close()
#        pool.join()
#
#        self.assertEqual(tc_haz.tag.haz_type, 'TC')
#        self.assertEqual(tc_haz.tag.description, '')
#        self.assertEqual(tc_haz.units, 'm/s')
#        self.assertEqual(tc_haz.centroids.size, 296)
#        self.assertEqual(tc_haz.event_id.size, 10)
#        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
#        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
#        self.assertEqual(tc_haz.intensity.shape, (10, 296))
#        self.assertEqual(tc_haz.fraction.shape, (10, 296))


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTCParallel)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
