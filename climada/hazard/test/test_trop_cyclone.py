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

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy import sparse

import climada.hazard.test as hazard_test
from climada.util import ureg
from climada.test import get_test_file
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone.trop_cyclone import (
    TropCyclone, )
from climada.hazard.centroids.centr import Centroids

DATA_DIR = Path(hazard_test.__file__).parent.joinpath('data')

TEST_TRACK = DATA_DIR.joinpath("trac_brb_test.csv")
TEST_TRACK_SHORT = DATA_DIR.joinpath("trac_short_test.csv")


CENTR_TEST_BRB = Centroids.from_hdf5(get_test_file('centr_test_brb', file_format='hdf5'))


class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""
    def test_memory_limit(self):
        """Test from_tracks when memory is (very) limited"""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]
        # A very low memory constraint forces the algorithm to split the track into chunks.
        # This should not affect the results. In practice, chunking is not applied due to limited
        # memory, but due to very high spatial/temporal resolution of the centroids/tracks. We
        # simulate this situation by artificially reducing the available memory.
        tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB, max_memory_gb=0.001)
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = [
            22.74903,  23.784691, 24.82255,  22.67403,  27.218706, 30.593959,
            18.980878, 24.540069, 27.826407, 26.846293,  0.,       34.568898,
        ]

        np.testing.assert_array_almost_equal(
            tc_haz.intensity[0, intensity_idx].toarray()[0],
            intensity_values,
        )

    def test_set_one_pass(self):
        """Test _tc_from_track function."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            "geosphere": [
                22.74927,  23.78498,  24.822908, 22.674202, 27.220042, 30.602122,
                18.981022, 24.540138, 27.830925, 26.8489,    0.,       34.572391,
            ],
            "equirect": [
                22.74903,  23.784691, 24.82255,  22.67403,  27.218706, 30.593959,
                18.980878, 24.540069, 27.826407, 26.846293,  0.,       34.568898,
            ]
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

            self.assertEqual(tc_haz.haz_type, 'TC')
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
            self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
            self.assertEqual(tc_haz.fraction.shape, (1, 296))
            self.assertIsNone(tc_haz._get_fraction())

            self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
            self.assertEqual(tc_haz.intensity.shape, (1, 296))
            self.assertEqual(np.nonzero(tc_haz.intensity)[0].size, 255)

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

    def test_cross_antimeridian(self):
        # Two locations on the island Taveuni (Fiji), one west and one east of 180° longitude.
        # We list the second point twice, with different lon-normalization:
        cen = Centroids.from_lat_lon([-16.95, -16.8, -16.8], [179.9, 180.1, -179.9])
        cen.set_dist_coast(precomputed=True)

        # Cyclone YASA (2020) passed directly over Fiji
        tr = TCTracks.from_ibtracs_netcdf(storm_id=["2020346S13168"])

        inten = TropCyclone.from_tracks(tr, centroids=cen).intensity.toarray()[0, :]

        # Centroids 1 and 2 are identical, they just use a different normalization for lon. This
        # should not affect the result at all:
        self.assertEqual(inten[1], inten[2])

        # All locations should be clearly affected by strong winds of appx. 40 m/s. The exact
        # values are not so important for this test:
        np.testing.assert_allclose(inten, 40, atol=10)

    def test_windfield_models(self):
        """Test _tc_from_track function with different wind field models."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = [
             ("H08", None, [
                 22.74903, 23.784691, 24.82255, 22.67403, 27.218706, 30.593959,
                 18.980878, 24.540069, 27.826407, 26.846293, 0., 34.568898,
             ]),
             ("H10", None, [
                 24.745521, 25.596484, 26.475329, 24.690914, 28.650107, 31.584395,
                 21.723546, 26.140293, 28.94964, 28.051915, 18.49378, 35.312152,
             ]),
             # The following model configurations use recorded wind speeds, while the above use
             # pressure values only. That's why some of the values are so different.
             ("H10", dict(vmax_from_cen=False, rho_air_const=1.2), [
                 23.702232, 24.327615, 24.947161, 23.589233, 26.616085, 29.389295,
                 21.338178, 24.257067, 26.472543, 25.662313, 18.535842, 31.886041,
             ]),
             ("H10", dict(vmax_from_cen=False, rho_air_const=None), [
                 24.244162, 24.835561, 25.432454, 24.139294, 27.127457, 29.719196,
                 21.910658, 24.692637, 26.783575, 25.971516, 19.005555, 31.904048,
             ]),
             ("H10", dict(vmax_from_cen=False, rho_air_const=None, vmax_in_brackets=True), [
                 23.592924, 24.208169, 24.817104, 23.483053, 26.468975, 29.221715,
                 21.260867, 24.150879, 26.34288 , 25.543635, 18.487385, 31.904048
             ]),
             ("H1980", None, [
                 21.376807, 21.957217, 22.569568, 21.284351, 24.254226, 26.971303,
                 19.220149, 21.984516, 24.196388, 23.449116, 0, 31.550207,
             ]),
             ("ER11", None, [
                 23.565332, 24.931413, 26.360758, 23.490333, 29.601171, 34.522795,
                 18.996389, 26.102109, 30.780737, 29.498453, 0, 38.368805,
             ]),
        ]

        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]

        for model, model_kwargs, inten_ref in intensity_values:
            tc_haz = TropCyclone.from_tracks(
                tc_track, centroids=CENTR_TEST_BRB, model=model, model_kwargs=model_kwargs,
            )
            np.testing.assert_array_almost_equal(
                tc_haz.intensity[0, intensity_idx].toarray()[0], inten_ref,
            )
            for idx, val in zip(intensity_idx, inten_ref):
                if val == 0:
                    self.assertEqual(tc_haz.intensity[0, idx], 0)

    def test_windfield_models_different_windunits(self):
        """
        Test _tc_from_track function should calculate the same results or raise ValueError
         with different windspeed units.
         """
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            # Holland 1980 and Emanuel & Rotunno 2011 use recorded wind speeds, that is why checking them for different
            # windspeed units is so important:
            "H1980": [
                21.376807, 21.957217, 22.569568, 21.284351, 24.254226, 26.971303,
                19.220149, 21.984516, 24.196388, 23.449116,  0, 31.550207,
            ],
            "ER11": [
                23.565332, 24.931413, 26.360758, 23.490333, 29.601171, 34.522795,
                18.996389, 26.102109, 30.780737, 29.498453,  0, 38.368805,
            ],
        }

        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data = tc_track.data[:1]

        tc_track_kmph = TCTracks(data=[ds.copy(deep=True) for ds in tc_track.data])
        tc_track_kmph.data[0]['max_sustained_wind'] *= (
            (1.0 * ureg.knot).to(ureg.km / ureg.hour).magnitude
        )
        tc_track_kmph.data[0].attrs['max_sustained_wind_unit'] = 'km/h'

        tc_track_mps = TCTracks(data=[ds.copy(deep=True) for ds in tc_track.data])
        tc_track_mps.data[0]['max_sustained_wind'] *= (
            (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
        )
        tc_track_mps.data[0].attrs['max_sustained_wind_unit'] = 'm/s'

        for model in ["H1980", "ER11"]:
            for tc_track_i in [tc_track_kmph, tc_track_mps]:
                tc_haz = TropCyclone.from_tracks(tc_track_i, centroids=CENTR_TEST_BRB, model=model)
                np.testing.assert_array_almost_equal(
                    tc_haz.intensity[0, intensity_idx].toarray()[0], intensity_values[model])
                for idx, val in zip(intensity_idx, intensity_values[model]):
                    if val == 0:
                        self.assertEqual(tc_haz.intensity[0, idx], 0)

        tc_track.data[0].attrs['max_sustained_wind_unit'] = 'elbows/fortnight'
        with self.assertRaises(ValueError):
            TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB, model=model)

    def test_set_one_file_pass(self):
        """Test from_tracks with one input."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TropCyclone.from_tracks(tc_track, centroids=CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.haz_type, 'TC')
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
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
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

        self.assertEqual(tc_haz.haz_type, 'TC')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)


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

    def test_apply_criterion_track2(self):
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
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestClimateSce))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDumpReloadCycle))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
