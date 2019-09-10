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

Test TropCyclone class
"""

import os
import unittest
import numpy as np
from scipy import sparse
import datetime as dt

import climada.hazard.trop_cyclone as tc
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import TropCyclone
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HAZ_TEST_MAT = os.path.join(DATA_DIR, 'atl_prob_no_name.mat')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")

CENTR_DIR = os.path.join(os.path.dirname(__file__), 'data/')
CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(os.path.join(CENTR_DIR, 'centr_brb_test.mat'))

CENT_CLB = Centroids()
CENT_CLB.read_mat(GLB_CENTROIDS_MAT)
CENT_CLB.dist_coast *= 1000 # set dist_coast to m

class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _hazard_from_track function."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.equal_timestep()
        coastal_centr = tc.coastal_centr_idx(CENT_CLB)
        tc_haz = TropCyclone._tc_from_track(tc_track.data[0], CENT_CLB, coastal_centr)

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 1656093)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.date.size, 1)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).year, 1951)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).month, 8)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).day, 27)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 1656093))
        self.assertEqual(tc_haz.fraction.shape, (1, 1656093))

        self.assertAlmostEqual(tc_haz.intensity[0, 1630393],
                               18.511077471450232, 6)
        self.assertEqual(tc_haz.intensity[0, 1630394], 0)
        self.assertEqual(tc_haz.fraction[0, 1630393], 1)
        self.assertEqual(tc_haz.fraction[0, 1630394], 0)

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 7)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 7)

    def test_set_one_file_pass(self):
        """ Test set function set_from_tracks with one input."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """ Test set function set_from_tracks with two ibtracs."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, ['IBTrACS: 1951239N12334', 'IBTrACS: 1951239N12334'])
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

class TestModel(unittest.TestCase):
    """Test modelling of tropical cyclone"""

    def test_coastal_centroids_pass(self):
        """ Test selection of centroids close to coast. MATLAB reference. """
        coastal = tc.coastal_centr_idx(CENT_CLB)

        self.assertEqual(coastal.size, 1044882)
        self.assertEqual(CENT_CLB.lat[coastal[0]], -55.800000000000004)
        self.assertEqual(CENT_CLB.lon[coastal[0]],  -68.200000000000003)

        self.assertEqual(CENT_CLB.lat[coastal[5]], -55.700000000000003)
        self.assertEqual(CENT_CLB.lat[coastal[10]],  -55.700000000000003)
        self.assertEqual(CENT_CLB.lat[coastal[100]], -55.300000000000004)
        self.assertEqual(CENT_CLB.lat[coastal[1000]], -53.900000000000006)
        self.assertEqual(CENT_CLB.lat[coastal[10000]], -44.400000000000006)
        self.assertEqual(CENT_CLB.lat[coastal[100000]], -25)
        self.assertEqual(CENT_CLB.lat[coastal[1000000]], 60.900000000000006)

        self.assertEqual(CENT_CLB.lat[coastal[1044881]], 60.049999999999997)
        self.assertEqual(CENT_CLB.lon[coastal[1044881]],  180.0000000000000)

    def test_gust_from_track(self):
        """ Test gust_from_track function. Compare to MATLAB reference. """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.equal_timestep()
        intensity = tc.gust_from_track(tc_track.data[0], CENT_CLB, model='H08')

        self.assertTrue(isinstance(intensity, sparse.csr.csr_matrix))
        self.assertEqual(intensity.shape, (1, 1656093))
        self.assertEqual(np.nonzero(intensity)[0].size, 7)

        self.assertEqual(intensity[0, 1630273], 0)
        self.assertAlmostEqual(intensity[0, 1630272], 18.505998796740347, 5)
        self.assertTrue(np.isclose(18.505998796740347,
                                   intensity[0, 1630272]))

        self.assertAlmostEqual(intensity[0, 1630393], 18.511077471450232, 6)
        self.assertTrue(np.isclose(18.511077471450232,
                                   intensity[0, 1630393]))

        self.assertAlmostEqual(intensity[0, 1630514], 18.297250626663271, 5)
        self.assertTrue(np.isclose(18.297250626663271,
                                   intensity[0, 1630514]))

        self.assertAlmostEqual(intensity[0, 1630635], 17.733240401598668, 6)
        self.assertTrue(np.isclose(17.733240401598668,
                                   intensity[0, 1630635]))

        self.assertAlmostEqual(intensity[0, 1630877], 17.525880201507256, 6)
        self.assertTrue(np.isclose(17.525880201507256,
                                   intensity[0, 1630877]))

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
