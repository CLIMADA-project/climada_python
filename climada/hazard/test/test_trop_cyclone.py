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
        coastal_centr = tc.coastal_centr_idx(CENTR_TEST_BRB)
        tc_haz = TropCyclone._tc_from_track(tc_track.data[0], CENTR_TEST_BRB, coastal_centr)

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
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertAlmostEqual(tc_haz.intensity[0, 100],
                               38.84863159321016, 6)
        self.assertEqual(tc_haz.intensity[0, 260], 0)
        self.assertEqual(tc_haz.fraction[0, 100], 1)
        self.assertEqual(tc_haz.fraction[0, 260], 0)

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 280)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 280)

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

    def test_extra_rad_max_wind_pass(self):
        """Test _extra_rad_max_wind function. Compare to MATLAB reference."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rad_max_wind = tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values,
            tc_track.data[0].radius_max_wind.values)

        self.assertEqual(rad_max_wind[0], 75.536713749999905)
        self.assertAlmostEqual(rad_max_wind[10], 75.592659583328057)
        self.assertAlmostEqual(rad_max_wind[128], 46.686527832605236)
        self.assertAlmostEqual(rad_max_wind[129], 46.089211533333333)
        self.assertAlmostEqual(rad_max_wind[130], 45.672274889277276)
        self.assertAlmostEqual(rad_max_wind[189], 45.132715266666666)
        self.assertAlmostEqual(rad_max_wind[190], 45.979603999211285)
        self.assertAlmostEqual(rad_max_wind[191], 47.287173876478825)
        self.assertEqual(rad_max_wind[192], 48.875090249999985)
        self.assertAlmostEqual(rad_max_wind[200], 59.975901084074955)

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
        r_arr = np.array([293.6067129546862, 298.2652319413182])
        r_max = 75.547902916671745
        hol_b = 1.265551666104679
        penv = 1010
        pcen = 1005.268166666671
        ycoord = 12.299999279463769

        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 5.384115724400597)
        self.assertAlmostEqual(_v_arr[1], 5.281356766052531)

        r_arr = np.array([])
        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertTrue(np.array_equal(_v_arr, np.array([])))

        r_arr = np.array([299.4501244109841,
                          291.0737897183741,
                          292.5441003235722])
        r_max = 40.665454622610511
        hol_b = 1.486076257880692
        penv = 1010
        pcen = 970.8727666672957
        ycoord = 14.089110370469488

        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 11.279764005440288)
        self.assertAlmostEqual(_v_arr[1], 11.682978583939310)
        self.assertAlmostEqual(_v_arr[2], 11.610940769149384)

    def test_coastal_centroids_pass(self):
        """Test selection of centroids close to coast. MATLAB reference."""
        coastal = tc.coastal_centr_idx(CENTR_TEST_BRB)

        self.assertEqual(coastal.size, CENTR_TEST_BRB.size)

    def test_vtrans_correct(self):
        """Test _vtrans_correct function. Compare to MATLAB reference."""
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values,
            tc_track.data[0].radius_max_wind.values))
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])

        v_trans_corr = tc._vtrans_correct(
            tc_track.data[0].lat.values[i_node:i_node+2],
            tc_track.data[0].lon.values[i_node:i_node+2],
            tc_track.data[0].radius_max_wind.values[i_node],
            CENTR_TEST_BRB.coord[:6, :], r_arr)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        v_trans = 10.191466256012880 / to_kn
        v_trans_corr *= v_trans
        self.assertEqual(v_trans_corr.size, 6)
        self.assertAlmostEqual(v_trans_corr[0] * to_kn, 0.06547673730228235)
        self.assertAlmostEqual(v_trans_corr[1] * to_kn, 0.07106877437273672)
        self.assertAlmostEqual(v_trans_corr[2] * to_kn, 0.07641714650288109)
        self.assertAlmostEqual(v_trans_corr[3] * to_kn, 0.0627289214278824)
        self.assertAlmostEqual(v_trans_corr[4] * to_kn, 0.0697427233582331)
        self.assertAlmostEqual(v_trans_corr[5] * to_kn, 0.06855335593983322)

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        v_trans = tc._vtrans(tc_track.data[0].lat.values, tc_track.data[0].lon.values,
                tc_track.data[0].time_step.values)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(v_trans.size, tc_track.data[0].time.size-1)
        self.assertAlmostEqual(v_trans[i_node-1]*to_kn, 10.191466256012880)

    def test_vang_sym(self):
        """Test _vang_sym function. Compare to MATLAB reference."""
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values,
            tc_track.data[0].radius_max_wind.values))
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        v_trans = 5.2429431910897559
        v_ang = tc._vang_sym(tc_track.data[0].environmental_pressure.values[i_node],
            tc_track.data[0].central_pressure.values[i_node-1:i_node+1],
            tc_track.data[0].lat.values[i_node],
            tc_track.data[0].time_step.values[i_node],
            tc_track.data[0].radius_max_wind.values[i_node],
            r_arr, v_trans, model=0)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertEqual(v_ang.size, 6)
        self.assertAlmostEqual(v_ang[0] * to_kn, 10.774196807905097)
        self.assertAlmostEqual(v_ang[1] * to_kn, 10.591725180482094)
        self.assertAlmostEqual(v_ang[2] * to_kn, 10.398212766600055)
        self.assertAlmostEqual(v_ang[3] * to_kn, 10.195108683240084)
        self.assertAlmostEqual(v_ang[4] * to_kn, 10.319869893291429)
        self.assertAlmostEqual(v_ang[5] * to_kn, 11.305188714213809)

    def test_windfield(self):
        """Test _windfield function. Compare to MATLAB reference."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        coast_centr = tc.coastal_centr_idx(CENTR_TEST_BRB)

        wind = tc._windfield(tc_track.data[0], CENTR_TEST_BRB.coord, coast_centr, model=0)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertEqual(wind.shape, (CENTR_TEST_BRB.size,))

        wind = wind[coast_centr]
        self.assertEqual(np.nonzero(wind)[0].size, 280)
        self.assertAlmostEqual(wind[0] * to_kn, 51.16153933277889)
        self.assertAlmostEqual(wind[80] * to_kn, 64.15891933409763)
        self.assertAlmostEqual(wind[120] * to_kn, 41.43819201370903)
        self.assertAlmostEqual(wind[200] * to_kn, 57.28814245245439)
        self.assertAlmostEqual(wind[220] * to_kn, 69.62477194818004)

    def test_gust_from_track(self):
        """Test gust_from_track function. Compare to MATLAB reference."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        intensity = tc.gust_from_track(tc_track.data[0], CENTR_TEST_BRB, model='H08')

        self.assertTrue(isinstance(intensity, sparse.csr.csr_matrix))
        self.assertEqual(intensity.shape, (1, 296))
        self.assertEqual(np.nonzero(intensity)[0].size, 280)

        self.assertEqual(intensity[0, 260], 0)
        self.assertAlmostEqual(intensity[0, 1], 27.835686180065114)
        self.assertAlmostEqual(intensity[0, 2], 29.46862830056694)
        self.assertAlmostEqual(intensity[0, 3], 26.36829914594632)
        self.assertAlmostEqual(intensity[0, 100], 38.84863159321016)
        self.assertAlmostEqual(intensity[0, 250], 34.26311998266044)
        self.assertAlmostEqual(intensity[0, 295], 44.273964728810924)


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
