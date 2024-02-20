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
import xarray as xr

from climada.util import ureg
from climada.test import get_test_file
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import (
    TropCyclone, get_close_centroids, _vtrans, _B_holland_1980, _bs_holland_2008,
    _v_max_s_holland_2008, _x_holland_2010, _stat_holland_1980, _stat_holland_2010,
    _stat_er_2011, tctrack_to_si, MBAR_TO_PA, KM_TO_M, H_TO_S,
)
from climada.hazard.centroids.centr import Centroids
import climada.hazard.test as hazard_test

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

    def test_windfield_models(self):
        """Test _tc_from_track function with different wind field models."""
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            "H08": [
                22.74903,  23.784691, 24.82255,  22.67403,  27.218706, 30.593959,
                18.980878, 24.540069, 27.826407, 26.846293,  0.,       34.568898,
            ],
            "H10": [
                24.745521, 25.596484, 26.475329, 24.690914, 28.650107, 31.584395,
                21.723546, 26.140293, 28.94964,  28.051915, 18.49378, 35.312152,
            ],
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

    def test_windfield_models_different_windunits(self):
        """
        Test _tc_from_track function should calculate the same results or raise ValueError
         with different windspeed units.
         """
        intensity_idx = [0, 1, 2,  3,  80, 100, 120, 200, 220, 250, 260, 295]
        intensity_values = {
            # Holland 1980 and Emanuel & Rotunno 2011 use recorded wind speeds, that is why checking them for different
            # windspeed units is so important:
            "H1980": [21.376807, 21.957217, 22.569568, 21.284351, 24.254226, 26.971303,
                      19.220149, 21.984516, 24.196388, 23.449116,  0, 31.550207],
            "ER11": [23.565332, 24.931413, 26.360758, 23.490333, 29.601171, 34.522795,
                     18.996389, 26.102109, 30.780737, 29.498453,  0, 38.368805],
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

class TestWindfieldHelpers(unittest.TestCase):
    """Test helper functions of TC wind field model"""

    def test_get_close_centroids_pass(self):
        """Test get_close_centroids function."""
        t_lat = np.array([0, -0.5, 0])
        t_lon = np.array([0.9, 2, 3.2])
        centroids = np.array([
            [0, -0.2], [0, 0.9], [-1.1, 1.2], [1, 2.1], [0, 4.3], [0.6, 3.8], [0.9, 4.1],
        ])
        test_mask = np.array([[False, True, False, False, False, False, False],
                              [False, False, True, False, False, False, False],
                              [False, False, False, False, False, True, False]])
        mask = get_close_centroids(t_lat, t_lon, centroids, 112.0)
        np.testing.assert_equal(mask, test_mask)

        # example where antimeridian is crossed
        t_lat = np.linspace(-10, 10, 11)
        t_lon = np.linspace(170, 200, 11)
        t_lon[t_lon > 180] -= 360
        centroids = np.array([[-11, 169], [-7, 176], [4, -170], [10, 170], [-10, -160]])
        test_mask = np.array([True, True, True, False, False])
        mask = get_close_centroids(t_lat, t_lon, centroids, 600.0)
        np.testing.assert_equal(mask.any(axis=0), test_mask)

    def test_B_holland_1980_pass(self):
        """Test _B_holland_1980 function."""
        si_track = xr.Dataset({
            "env": ("time",  MBAR_TO_PA * np.array([1010, 1010])),
            "cen": ("time",  MBAR_TO_PA * np.array([995, 980])),
            "vgrad": ("time",  [35, 40]),
        })
        _B_holland_1980(si_track)
        np.testing.assert_array_almost_equal(si_track["hol_b"], [2.5, 1.667213])

    def test_bs_holland_2008_pass(self):
        """Test _bs_holland_2008 function. Compare to MATLAB reference."""
        si_track = xr.Dataset({
            "tstep": ("time", H_TO_S * np.array([1.0, 1.0, 1.0])),
            "lat": ("time", [12.299999504631234, 12.299999504631343, 12.299999279463769]),
            "env": ("time", MBAR_TO_PA * np.array([1010, 1010, 1010])),
            "cen": ("time", MBAR_TO_PA * np.array([1005.2585, 1005.2633, 1005.2682])),
            "vtrans_norm": ("time",  [np.nan, 5.241999541820597, 5.123882725120426]),
        })
        _bs_holland_2008(si_track)
        np.testing.assert_array_almost_equal(
            si_track["hol_b"], [np.nan, 1.27085617, 1.26555341])

    def test_v_max_s_holland_2008_pass(self):
        """Test _v_max_s_holland_2008 function."""
        # Numbers analogous to test_B_holland_1980_pass
        si_track = xr.Dataset({
            "env": ("time", MBAR_TO_PA * np.array([1010, 1010])),
            "cen": ("time", MBAR_TO_PA * np.array([995, 980])),
            "hol_b": ("time", [2.5, 1.67]),
        })
        _v_max_s_holland_2008(si_track)
        np.testing.assert_array_almost_equal(si_track["vmax"], [34.635341, 40.033421])

    def test_holland_2010_pass(self):
        """Test Holland et al. 2010 wind field model."""
        # The parameter "x" is designed to be exactly 0.5 inside the radius of max wind (RMW) and
        # to increase or decrease linearly outside of it in radial direction.
        #
        # An increase (decrease) of "x" outside of the RMW is for cases where the max wind is very
        # high (low), but the RMW is still comparably large (small). This means, wind speeds need
        # to decay very sharply (only moderately) outside of the RMW to reach the low prescribed
        # peripheral wind speeds.
        #
        # The "hol_b" parameter tunes the meaning of a "comparably" large or small RMW.
        si_track = xr.Dataset({
            # four test cases:
            # - low vmax, moderate RMW: x decreases moderately
            # - large hol_b: x decreases sharply
            # - very low vmax: x decreases so much, it needs to be clipped at 0
            # - large vmax, large RMW: x increases
            "rad": ("time", KM_TO_M * np.array([75, 75, 75, 90])),
            "vmax": ("time", [35.0, 35.0, 16.0, 90.0]),
            "hol_b": ("time", [1.75, 2.5, 1.9, 1.6]),
        })
        d_centr = KM_TO_M * np.array([
            # first column is for locations within the storm eye
            # second column is for locations at or close to the radius of max wind
            # third column is for locations outside the storm eye
            # fourth column is for locations exactly at the peripheral radius
            # fifth column is for locations outside the peripheral radius
            [0., 75, 220, 300, 490],
            [30, 74, 170, 300, 501],
            [21, 76, 230, 300, 431],
            [32, 91, 270, 300, 452],
        ], dtype=float)
        close_centr = np.array([
            # note that we set one of these to "False" for testing
            [True, True, True, True, True],
            [True, True, True, True, False],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ], dtype=bool)
        hol_x = _x_holland_2010(si_track, d_centr, close_centr)
        np.testing.assert_array_almost_equal(hol_x, [
            [0.5, 0.500000, 0.485077, 0.476844, 0.457291],
            [0.5, 0.500000, 0.410997, 0.289203, 0.000000],
            [0.5, 0.497620, 0.131072, 0.000000, 0.000000],
            [0.5, 0.505022, 1.403952, 1.554611, 2.317948],
        ])

        v_ang_norm = _stat_holland_2010(si_track, d_centr, close_centr, hol_x)
        np.testing.assert_array_almost_equal(v_ang_norm, [
            # first column: converge to 0 when approaching storm eye
            # second column: vmax at RMW
            # fourth column: peripheral speed (17 m/s) at peripheral radius (unless x is clipped!)
            [0.0000000, 35.000000, 21.181497, 17.00000, 12.103461],
            [1.2964800, 34.990037, 21.593755, 17.00000, 0.0000000],
            [0.3219518, 15.997500, 13.585498, 16.00000, 16.000000],
            [24.823469, 89.992938, 24.381965, 17.00000, 1.9292020],
        ])

    def test_stat_holland_1980(self):
        """Test _stat_holland_1980 function. Compare to MATLAB reference."""
        d_centr = KM_TO_M * np.array([
            [299.4501244109841, 291.0737897183741, 292.5441003235722, 40.665454622610511],
            [293.6067129546862, 1000.0, 298.2652319413182, 70.0],
        ])
        si_track = xr.Dataset({
            "rad": ("time", KM_TO_M * np.array([40.665454622610511, 75.547902916671745])),
            "hol_b": ("time", [1.486076257880692, 1.265551666104679]),
            "env": ("time", MBAR_TO_PA * np.array([1010.0, 1010.0])),
            "cen": ("time", MBAR_TO_PA * np.array([970.8727666672957, 1005.268166666671])),
            "lat": ("time", [-14.089110370469488, 12.299999279463769]),
            "cp": ("time", [3.54921922e-05, 3.10598285e-05]),
        })
        mask = np.array([[True, True, True, True], [True, False, True, True]], dtype=bool)
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[11.279764005440288, 11.682978583939310, 11.610940769149384, 42.412845],
             [5.384115724400597, 0, 5.281356766052531, 12.763087]])

        # without Coriolis force, values are higher, esp. far away from the center:
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask, cyclostrophic=True)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[15.719924, 16.037052, 15.980323, 43.128461],
             [8.836768,  0,  8.764678, 13.807452]])

        d_centr = np.array([[], []])
        mask = np.ones_like(d_centr, dtype=bool)
        v_ang_norm = _stat_holland_1980(si_track, d_centr, mask)
        np.testing.assert_array_equal(v_ang_norm, np.array([[], []]))

    def test_er_2011_pass(self):
        """Test Emanuel and Rotunno 2011 wind field model."""
        # test at centroids within and outside of radius of max wind
        d_centr = KM_TO_M * np.array([[35, 70, 75, 220], [30, 150, 1000, 300]], dtype=float)
        si_track = xr.Dataset({
            "rad": ("time", KM_TO_M * np.array([75.0, 40.0])),
            "vmax": ("time", [35.0, 40.0]),
            "lat": ("time", [20.0, 27.0]),
            "cp": ("time", [4.98665369e-05, 6.61918149e-05]),
        })
        mask = np.array([[True, True, True, True], [True, False, True, True]], dtype=bool)
        v_ang_norm = _stat_er_2011(si_track, d_centr, mask)
        np.testing.assert_array_almost_equal(v_ang_norm,
            [[28.258025, 36.782418, 36.869995, 22.521237],
             [39.670883, 0, 3.300626, 10.827206]])

    def test_vtrans_pass(self):
        """Test _vtrans function. Compare to MATLAB reference."""
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        track_ds = tc_track.data[0]

        si_track = tctrack_to_si(track_ds)
        _vtrans(si_track)

        to_kn = (1.0 * ureg.meter / ureg.second).to(ureg.knot).magnitude

        self.assertEqual(si_track["vtrans_norm"].size, track_ds["time"].size)
        self.assertEqual(si_track["vtrans_norm"].values[0], 0)
        self.assertAlmostEqual(si_track["vtrans_norm"].values[1] * to_kn, 10.191466246)

    def testtctrack_to_si(self):
        """ Test tctrack_to_si should create the same vmax output independent of the input unit """
        tc_track = TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT).data[0]

        tc_track_kmph = tc_track.copy(deep=True)
        tc_track_kmph['max_sustained_wind'] *= (
            (1.0 * ureg.knot).to(ureg.km / ureg.hour).magnitude
        )
        tc_track_kmph.attrs['max_sustained_wind_unit'] = 'km/h'

        si_track = tctrack_to_si(tc_track)
        si_track_from_kmph = tctrack_to_si(tc_track_kmph)

        np.testing.assert_array_almost_equal(si_track["vmax"], si_track_from_kmph["vmax"])

        tc_track.attrs['max_sustained_wind_unit'] = 'elbows/fortnight'
        with self.assertRaises(ValueError):
            tctrack_to_si(tc_track)


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
