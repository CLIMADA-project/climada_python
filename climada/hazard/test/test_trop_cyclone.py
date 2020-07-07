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

from climada.util import ureg
import climada.hazard.trop_cyclone as tc
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import TropCyclone
from climada.hazard.centroids.centr import Centroids

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HAZ_TEST_MAT = os.path.join(DATA_DIR, 'atl_prob_no_name.mat')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")

CENTR_DIR = os.path.join(os.path.dirname(__file__), 'data/')
CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(os.path.join(CENTR_DIR, 'centr_brb_test.mat'))


class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _tc_from_track function."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, centroids=CENTR_TEST_BRB, model='H08')

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.date.size, 1)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).year, 1951)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).month, 8)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).day, 27)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))
        self.assertEqual(tc_haz.fraction[0, 100], 1)
        self.assertEqual(tc_haz.fraction[0, 260], 0)
        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 263)

        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(np.nonzero(tc_haz.intensity)[0].size, 263)

        self.assertEqual(tc_haz.intensity[0, 260], 0)
        self.assertAlmostEqual(tc_haz.intensity[0, 1], 25.32172387220036)
        self.assertAlmostEqual(tc_haz.intensity[0, 2], 26.77093534001236)
        self.assertAlmostEqual(tc_haz.intensity[0, 3], 24.03850320851069)
        self.assertAlmostEqual(tc_haz.intensity[0, 100], 35.1414962679868)
        self.assertAlmostEqual(tc_haz.intensity[0, 250], 30.924750568841727)
        self.assertAlmostEqual(tc_haz.intensity[0, 295], 40.605613790812285)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude
        wind = tc_haz.intensity.toarray()[0,:]
        self.assertAlmostEqual(wind[0] * to_kn, 46.59290508202303)
        self.assertAlmostEqual(wind[80] * to_kn, 58.04014412198915)
        self.assertAlmostEqual(wind[120] * to_kn, 37.87443335725528)
        self.assertAlmostEqual(wind[200] * to_kn, 52.12407964134238)
        self.assertAlmostEqual(wind[220] * to_kn, 62.6252906598788)

    def test_set_one_file_pass(self):
        """Test set function set_from_tracks with one input."""
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
        self.assertEqual(tc_haz.category, tc_track.data[0].category)
        self.assertTrue(np.isnan(tc_haz.basin[0]))
        self.assertIsInstance(tc_haz.basin, list)
        self.assertIsInstance(tc_haz.category, np.ndarray)
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """Test set function set_from_tracks with two ibtracs."""
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

    def test_bs_hol08_pass(self):
        """Test _bs_hol08 function. Compare to MATLAB reference."""
        v_trans = 5.241999541820597
        penv = 1010
        pcen = 1005.263333333329
        prepcen = 1005.258500000000
        lat = 12.299999504631343
        xx = 0.586781395348824
        tint = 1
        _bs_res = tc._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.270856908796045)

        v_trans = 5.123882725120426
        penv = 1010
        pcen = 1005.268166666671
        prepcen = 1005.263333333329
        lat = 12.299999279463769
        xx = 0.586794883720942
        tint = 1
        _bs_res = tc._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.265551666104679)

    def test_stat_holland(self):
        """Test _stat_holland function. Compare to MATLAB reference."""
        d_centr = np.array([[293.6067129546862, 298.2652319413182]])
        r_max = np.array([75.547902916671745])
        hol_b = np.array([1.265551666104679])
        penv = np.array([1010.0])
        pcen = np.array([1005.268166666671])
        lat = np.array([12.299999279463769])
        mask = np.ones_like(d_centr, dtype=bool)

        _v_arr = tc._stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertAlmostEqual(_v_arr[0], 5.384115724400597)
        self.assertAlmostEqual(_v_arr[1], 5.281356766052531)

        d_centr = np.array([[]])
        mask = np.ones_like(d_centr, dtype=bool)
        _v_arr = tc._stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertTrue(np.array_equal(_v_arr, np.array([])))

        d_centr = np.array([
                [299.4501244109841, 291.0737897183741, 292.5441003235722]
        ])
        r_max = np.array([40.665454622610511])
        hol_b = np.array([1.486076257880692])
        penv = np.array([1010.0])
        pcen = np.array([970.8727666672957])
        lat = np.array([14.089110370469488])
        mask = np.ones_like(d_centr, dtype=bool)

        _v_arr = tc._stat_holland(d_centr, r_max, hol_b, penv, pcen, lat, mask)[0]
        self.assertAlmostEqual(_v_arr[0], 11.279764005440288)
        self.assertAlmostEqual(_v_arr[1], 11.682978583939310)
        self.assertAlmostEqual(_v_arr[2], 11.610940769149384)

    def test_vtrans_correct(self):
        """Test _vtrans_correct function."""
        # The correction does only depend on the relative positions of
        # centroids and track points + hemisphere
        v_centr = np.array([
            [[-1, 0], [0, 1], [0, 1.4], [0, -1.4], [1.7, 2.2], [0.3, 0.9],],
            [[-1, 0], [0, 1], [0, 1.4], [0, -1.4], [1.7, 2.2], [0.3, 0.9],],
        ])
        v_centr = (np.linalg.norm(v_centr, axis=-1), v_centr)
        v_trans = np.array([[1, 0], [1, 0]])
        v_trans = (np.linalg.norm(v_trans, axis=-1), v_trans)
        t_rad = np.array([1.3, 1.3])
        t_lat = np.array([34, -13])
        close_centr = np.array([
            [True, True, True, True, False, True],
        ])
        corr = tc._vtrans_correct(v_centr, v_trans, t_rad, t_lat, close_centr)
        self.assertEqual(corr.shape, (2, 6))

        # -> factor is 0% in front/behind the track
        self.assertEqual(corr[0,0], 0)

        # -> factor is 100% to the right of the track,
        self.assertEqual(corr[0,1], 1.0)

        # -> factor decreases with distance from eye
        self.assertEqual(corr[0,2], 0.9285714285714287)

        # -> sign changes from right to left (and only that)
        self.assertEqual(corr[0,3], -0.9285714285714287)

        # -> centroids that are not close (according to mask) get value 0
        self.assertEqual(corr[0,4], 0)

        # -> factor decreases when rotating around the eye from right to front
        self.assertEqual(corr[0,5], 0.9486832980505139)

        # -> switching hemisphere changes sign (and only that)
        self.assertTrue(np.allclose(corr[0], -corr[1]))

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        v_trans, _ = tc._vtrans(
            tc_track.data[0].lat.values, tc_track.data[0].lon.values,
            tc_track.data[0].time_step.values)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(v_trans.size, tc_track.data[0].time.size)
        self.assertEqual(v_trans[-1], 0)
        self.assertAlmostEqual(v_trans[i_node-1]*to_kn, 10.191466078221902)


class TestClimateSce(unittest.TestCase):

    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        criterion = list()
        tmp_chg = {'criteria': {'basin': ['NA'], 'category':[1, 2, 3, 4, 5]},
                   'year': 2100, 'change': 1.045, 'variable': 'intensity', 'function': np.multiply}
        criterion.append(tmp_chg)
        scale = 0.75

        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'WP'
        tc.category = np.array([2, 0, 4, 1])
        tc.event_id = np.arange(4)

        tc_cc = tc._apply_criterion(criterion, scale)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[0, :].toarray()*1.03375, tc_cc.intensity[0, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[2, :].toarray()*1.03375, tc_cc.intensity[2, :].toarray()))

    def test_two_criterion_track(self):
        """Test _apply_criterion function with two criteria"""
        criterion = list()
        tmp_chg = {'criteria': {'basin': ['NA'], 'category':[1, 2, 3, 4, 5]},
                   'year': 2100, 'change': 1.045, 'variable': 'intensity', 'function': np.multiply}
        criterion.append(tmp_chg)
        tmp_chg = {'criteria': {'basin': ['WP'], 'category':[1, 2, 3, 4, 5]},
                   'year': 2100, 'change': 1.025, 'variable': 'intensity', 'function': np.multiply}
        criterion.append(tmp_chg)
        tmp_chg = {'criteria': {'basin': ['WP'], 'category':[1, 2, 3, 4, 5]},
                   'year': 2100, 'change': 1.025, 'variable': 'frequency', 'function': np.multiply}
        criterion.append(tmp_chg)
        scale = 0.75

        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.frequency = np.ones(4)*0.5
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'WP'
        tc.category = np.array([2, 0, 4, 1])
        tc.event_id = np.arange(4)

        tc_cc = tc._apply_criterion(criterion, scale)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertFalse(np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[0, :].toarray()*1.03375, tc_cc.intensity[0, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[2, :].toarray()*1.03375, tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[3, :].toarray()*1.01875, tc_cc.intensity[3, :].toarray()))
        res_frequency = np.ones(4)*0.5
        res_frequency[3] = 0.5*1.01875
        self.assertTrue(np.allclose(tc_cc.frequency, res_frequency))

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModel))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestClimateSce))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
