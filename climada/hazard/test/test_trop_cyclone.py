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

Test TropCyclone class
"""

import unittest
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy import sparse

from climada.util import ureg
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import (
    TropCyclone, _close_centroids, _vtrans, _B_holland_1980, _bs_holland_2008,
    _v_max_s_holland_2008, _x_holland_2010, _stat_holland_1980, _stat_holland_2010,
    _stat_er_2011,
)
from climada.hazard.centroids.centr import Centroids
import climada.hazard.test as hazard_test

DATA_DIR = Path(hazard_test.__file__).parent.joinpath('data')

TEST_TRACK = DATA_DIR.joinpath("trac_brb_test.csv")
TEST_TRACK_SHORT = DATA_DIR.joinpath("trac_short_test.csv")

CENTR_TEST_BRB = Centroids.from_mat(DATA_DIR.joinpath('centr_brb_test.mat'))


class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _tc_from_track function."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            "geosphere": [25.60794285, 26.90906280, 28.26649026, 25.54076797, 31.21986961,
                          36.17171808, 21.11408573, 28.01457948, 32.65349378, 31.34027741, 0,
                          40.27362679],
            "equirect": [25.60778909, 26.90887264, 28.26624642, 25.54092386, 31.21941738,
                         36.16596567, 21.11399856, 28.01452136, 32.65076804, 31.33884098, 0,
                         40.27002104]
        }
        # the values for the two metrics should agree up to first digit at least
        for i, val in enumerate(intensity_values["geosphere"]):
            self.assertAlmostEqual(intensity_values["equirect"][i], val, 1)

        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]

        for metric in ["equirect", "geosphere"]:
            tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB, model='H08',
                                             store_windfields=True, metric=metric)

            self.assertEqual(tc_haz.tag.haz_type, 'TC')
            self.assertEqual(tc_haz.tag.description, '')
            self.assertEqual(tc_haz.tag.file_name, 'Name: 1951239N12334')
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
            self.assertIsNone(tc_haz._get_fraction())

            self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
            self.assertEqual(tc_haz.intensity.shape, (1, 296))
            self.assertEqual(np.nonzero(tc_haz.intensity)[0].size, 280)

            np.testing.assert_array_almost_equal(
                tc_haz.intensity[0, intensity_idx].toarray()[0], intensity_values[metric])
            for idx, val in zip(intensity_idx, intensity_values[metric]):
                if val == 0:
                    self.assertEqual(tc_haz.intensity[0, idx], 0)

            windfields = tc_haz.windfields[0].toarray()
            windfields = windfields.reshape(windfields.shape[0], -1, 2)
            windfield_norms = np.linalg.norm(windfields, axis=-1).max(axis=0)
            intensity = tc_haz.intensity.toarray()[0, :]
            msk = (intensity > 0)
            np.testing.assert_array_equal(windfield_norms[msk], intensity[msk])

    def test_windfield_models(self):
        """Test _tc_from_track function with different wind field models."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            "H08": [25.60778909, 26.90887264, 28.26624642, 25.54092386, 31.21941738, 36.16596567,
                    21.11399856, 28.01452136, 32.65076804, 31.33884098, 0, 40.27002104],
            "H10": [27.604317, 28.720708, 29.894993, 27.52234 , 32.512395, 37.114355,
                    23.848917, 29.614752, 33.775593, 32.545347, 19.957627, 41.014578],
            # Holland 1980 and Emanuel & Rotunno 2011 use recorded wind speeds, while the above use
            # pressure values only. That's why the results are so different:
            "H1980": [21.376807, 21.957217, 22.569568, 21.284351, 24.254226, 26.971303,
                      19.220149, 21.984516, 24.196388, 23.449116,  0, 31.550207],
            "ER11": [23.565332, 24.931413, 26.360758, 23.490333, 29.601171, 34.522795,
                     18.996389, 26.102109, 30.780737, 29.498453,  0, 38.368805],
        }

        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]

        for model in ["H08", "H10", "H1980", "ER11"]:
            tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB, model=model)
            np.testing.assert_array_almost_equal(
                tc_haz.intensity[0, intensity_idx].toarray()[0], intensity_values[model])
            for idx, val in zip(intensity_idx, intensity_values[model]):
                if val == 0:
                    self.assertEqual(tc_haz.intensity[0, idx], 0)

    def test_set_one_file_pass(self):
        """Test from_tracks with one input."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'Name: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertEqual(tc_haz.category, tc_track.data[0].category)
        self.assertEqual(tc_haz.basin[0], 'NA')
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
        """Test from_tracks with two ibtracs."""
        tc_track = TCTracks.from_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, ['Name: 1951239N12334', 'Name: 1951239N12334'])
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

class TestWindfieldHelpers(unittest.TestCase):
    """Test helper functions of TC wind field model"""

    def test_close_centroids_pass(self):
        """Test _close_centroids function."""
        t_lat = np.array([0, -0.5, 0])
        t_lon = np.array([0.9, 2, 3.2])
        centroids = np.array([[0, -0.2], [0, 0.9], [-1.1, 1.2], [1, 2.1], [0, 4.3], [0.6, 3.8]])
        test_mask = np.array([False, True, True, False, False, True])
        mask = _close_centroids(t_lat, t_lon, centroids, 1)
        np.testing.assert_equal(mask, test_mask)

        # example where antimeridian is crossed
        t_lat = np.linspace(-10, 10, 11)
        t_lon = np.linspace(170, 200, 11)
        t_lon[t_lon > 180] -= 360
        centroids = np.array([[-11, 169], [-7, 176], [4, -170], [10, 170], [-10, -160]])
        test_mask = np.array([True, True, True, False, False])
        mask = _close_centroids(t_lat, t_lon, centroids, 5)
        np.testing.assert_equal(mask, test_mask)

    def test_B_holland_1980_pass(self):
        """Test _B_holland_1980 function."""
        gradient_winds = np.array([35, 40])
        penv = np.array([1010, 1010])
        pcen = np.array([995, 980])
        _B_res = _B_holland_1980(gradient_winds, penv, pcen)
        np.testing.assert_array_almost_equal(_B_res, [2.5, 1.667213])

    def test_bs_holland_2008_pass(self):
        """Test _bs_holland_2008 function. Compare to MATLAB reference."""
        v_trans = np.array([5.241999541820597, 5.123882725120426])
        penv = np.array([1010, 1010])
        pcen = np.array([1005.263333333329, 1005.268166666671])
        prepcen = np.array([1005.258500000000, 1005.263333333329])
        lat = np.array([12.299999504631343, 12.299999279463769])
        tint = np.array([1.0, 1.0])
        _bs_res = _bs_holland_2008(v_trans, penv, pcen, prepcen, lat, tint)
        np.testing.assert_array_almost_equal(_bs_res, [1.270856908796045, 1.265551666104679])

    def test_v_max_s_holland_2008_pass(self):
        """Test _v_max_s_holland_2008 function."""
        # Numbers analogous to test_B_holland_1980_pass
        penv = np.array([1010, 1010])
        pcen = np.array([995, 980])
        b_s = np.array([2.5, 1.67])
        v_max_s = _v_max_s_holland_2008(penv, pcen, b_s)
        np.testing.assert_array_almost_equal(v_max_s, [34.635341, 40.033421])

    def test_holland_2010_pass(self):
        """Test Holland et al. 2010 wind field model."""
        # test at centroids within and outside of radius of max wind
        d_centr = np.array([[35, 75, 220], [30, 1000, 300]], dtype=float)
        r_max = np.array([75, 40], dtype=float)
        v_max_s = np.array([35.0, 40.0])
        hol_b = np.array([1.80, 2.5])
        mask = np.array([[True, True, True], [True, False, True]], dtype=bool)
        hol_x = _x_holland_2010(d_centr, r_max, v_max_s, hol_b, mask)
        np.testing.assert_array_almost_equal(hol_x, [[0.5, 0.5, 0.47273], [0.5, 0, 0.211602]])

        # test exactly at radius of maximum wind (35 m/s) and at peripheral radius (17 m/s)
        v_ang_norm = _stat_holland_2010(d_centr, v_max_s, r_max, hol_b, mask, hol_x)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[15.957853, 35.0, 20.99411], [33.854826, 0, 17.0]])

    def test_stat_holland_1980(self):
        """Test _stat_holland_1980 function. Compare to MATLAB reference."""
        d_centr = np.array([
            [299.4501244109841, 291.0737897183741, 292.5441003235722, 40.665454622610511],
            [293.6067129546862, 1000.0, 298.2652319413182, 70.0],
        ])
        r_max = np.array([40.665454622610511, 75.547902916671745])
        hol_b = np.array([1.486076257880692, 1.265551666104679])
        penv = np.array([1010.0, 1010.0])
        pcen = np.array([970.8727666672957, 1005.268166666671])
        lat = np.array([-14.089110370469488, 12.299999279463769])
        mask = np.array([[True, True, True, True], [True, False, True, True]], dtype=bool)
        v_ang_norm = _stat_holland_1980(d_centr, r_max, hol_b, penv, pcen, lat, mask)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[11.279764005440288, 11.682978583939310, 11.610940769149384, 42.412845],
             [5.384115724400597, 0, 5.281356766052531, 12.763087]])

        # without Coriolis force, values are higher, esp. far away from the center:
        v_ang_norm = _stat_holland_1980(d_centr, r_max, hol_b, penv, pcen, lat, mask,
                                        cyclostrophic=True)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[15.719924, 16.037052, 15.980323, 43.128461],
             [8.836768,  0,  8.764678, 13.807452]])

        d_centr = np.array([[], []])
        mask = np.ones_like(d_centr, dtype=bool)
        v_ang_norm = _stat_holland_1980(d_centr, r_max, hol_b, penv, pcen, lat, mask)
        np.testing.assert_array_equal(v_ang_norm, np.array([[], []]))

    def test_er_2011_pass(self):
        """Test Emanuel and Rotunno 2011 wind field model."""
        # test at centroids within and outside of radius of max wind
        d_centr = np.array([[35, 75, 220], [30, 1000, 300]], dtype=float)
        r_max = np.array([75, 40], dtype=float)
        v_max = np.array([35.0, 40.0])
        lat = np.array([20, 27])
        v_ang_norm = _stat_er_2011(d_centr, v_max, r_max, lat)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[28.258025, 36.869995, 22.521237],
             [39.670883,  3.300626, 10.827206]])

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        v_trans, _ = _vtrans(
            tc_track.data[0].lat.values, tc_track.data[0].lon.values,
            tc_track.data[0].time_step.values)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(v_trans.size, tc_track.data[0].time.size)
        self.assertEqual(v_trans[0], 0)
        self.assertAlmostEqual(v_trans[1] * to_kn, 10.191466246)


class TestClimateSce(unittest.TestCase):
    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        intensity = np.zeros((4, 10))
        intensity[0, :] = np.arange(10)
        intensity[1, 5] = 10
        intensity[2, :] = np.arange(10, 20)
        intensity[3, 3] = 3
        tc = TropCyclone(
            intensity=sparse.csr_matrix(intensity),
            basin=['NA', 'NA', 'NA', 'NO'],
            category=np.array([2, 0, 4, 1]),
            event_id=np.arange(4),
            frequency=np.ones(4) * 0.5,
        )

        tc_cc = tc.apply_climate_scenario_knu(ref_year=2050, rcp_scenario=45)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.frequency, tc_cc.frequency))
        self.assertEqual(
            tc_cc.tag.description,
            'climate change scenario for year 2050 and RCP 45 from Knutson et al 2015.')

    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        criterion = [{'basin': 'NA', 'category': [1, 2, 3, 4, 5],
                   'year': 2100, 'change': 1.045, 'variable': 'intensity'}
                   ]
        scale = 0.75

        # artificially increase the size of the hazard by repeating (tiling) the data:
        ntiles = 8

        intensity = np.zeros((4, 10))
        intensity[0, :] = np.arange(10)
        intensity[1, 5] = 10
        intensity[2, :] = np.arange(10, 20)
        intensity[3, 3] = 3
        intensity = np.tile(intensity, (ntiles, 1))
        tc = TropCyclone(
            intensity=sparse.csr_matrix(intensity),
            basin=ntiles * ['NA', 'NA', 'NA', 'WP'],
            category=np.array(ntiles * [2, 0, 4, 1]),
            event_id=np.arange(intensity.shape[0]),
        )

        tc_cc = tc._apply_knutson_criterion(criterion, scale)
        for i_tile in range(ntiles):
            offset = i_tile * 4
            # no factor applied because of category 0
            np.testing.assert_array_equal(
                tc.intensity[offset + 1, :].toarray(), tc_cc.intensity[offset + 1, :].toarray())
            # no factor applied because of basin "WP"
            np.testing.assert_array_equal(
                tc.intensity[offset + 3, :].toarray(), tc_cc.intensity[offset + 3, :].toarray())
            # factor is applied to the remaining events
            np.testing.assert_array_almost_equal(
                tc.intensity[offset + 0, :].toarray() * 1.03375,
                tc_cc.intensity[offset + 0, :].toarray())
            np.testing.assert_array_almost_equal(
                tc.intensity[offset + 2, :].toarray() * 1.03375,
                tc_cc.intensity[offset + 2, :].toarray())

    def test_two_criterion_track(self):
        """Test _apply_criterion function with two criteria"""
        criterion = [
            {'basin': 'NA', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.045, 'variable': 'intensity'},
            {'basin': 'WP', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.025, 'variable': 'intensity'},
            {'basin': 'WP', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1.025, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [0, 1, 2, 3, 4, 5],
             'year': 2100, 'change': 0.7, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [1, 2, 3, 4, 5],
             'year': 2100, 'change': 1, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [3, 4, 5],
             'year': 2100, 'change': 1, 'variable': 'frequency'},
            {'basin': 'NA', 'category': [4, 5],
             'year': 2100, 'change': 2, 'variable': 'frequency'}
            ]
        scale = 0.75

        intensity = np.zeros((4, 10))
        intensity[0, :] = np.arange(10)
        intensity[1, 5] = 10
        intensity[2, :] = np.arange(10, 20)
        intensity[3, 3] = 3
        tc = TropCyclone(
            intensity=sparse.csr_matrix(intensity),
            frequency=np.ones(4) * 0.5,
            basin=['NA', 'NA', 'NA', 'WP'],
            category=np.array([2, 0, 4, 1]),
            event_id=np.arange(4),
        )

        tc_cc = tc._apply_knutson_criterion(criterion, scale)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[0, :].toarray() * 1.03375, tc_cc.intensity[0, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[2, :].toarray() * 1.03375, tc_cc.intensity[2, :].toarray()))
        self.assertTrue(
            np.allclose(tc.intensity[3, :].toarray() * 1.01875, tc_cc.intensity[3, :].toarray()))

        res_frequency = np.ones(4) * 0.5
        res_frequency[1] = 0.5 * (1 + (0.7 - 1) * scale)
        res_frequency[2] = 0.5 * (1 + (2 - 1) * scale)
        res_frequency[3] = 0.5 * (1 + (1.025 - 1) * scale)
        self.assertTrue(np.allclose(tc_cc.frequency, res_frequency))

    def test_negative_freq_error(self):
        """Test _apply_knutson_criterion with infeasible input."""
        criterion = [{'basin': 'SP', 'category': [0, 1],
                      'year': 2100, 'change': 0.5,
                      'variable': 'frequency'}
                     ]

        tc = TropCyclone(
            frequency=np.ones(2),
            basin=['SP', 'SP'],
            category=np.array([0, 1]),
        )

        with self.assertRaises(ValueError):
            tc._apply_knutson_criterion(criterion, 3)


class TestDumpReloadCycle(unittest.TestCase):
    def setUp(self):
        """Create a TropCyclone object and a temporary directory"""
        self.tempdir = TemporaryDirectory()
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        self.tc_hazard = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB)

    def test_dump_reload_hdf5(self):
        """Try to write TropCyclone to a hdf5 file and read it back in"""
        hdf5_dump = Path(self.tempdir.name, "tc_dump.h5")
        self.tc_hazard.write_hdf5(hdf5_dump)
        recycled = TropCyclone.from_hdf5(hdf5_dump)
        np.testing.assert_array_equal(recycled.category, self.tc_hazard.category)

    def tearDown(self):
        """Delete the temporary directory"""
        self.tempdir.cleanup()


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWindfieldHelpers))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestClimateSce))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDumpReloadCycle))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
