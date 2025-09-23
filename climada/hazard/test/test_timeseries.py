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

Test HazardTimeSeries class
"""

import unittest

import numpy as np
import numpy.testing as npt
from scipy import sparse

from climada.hazard.test.test_base import dummy_hazard
from climada.hazard.timeseries import (
    HazardTimeSeries,
    _sample_independent_nevents_per_time_series_bin,
)

# DATA_DIR = CONFIG.hazard.test_data.dir()


class TestHazardTimeSeries(unittest.TestCase):
    def setUp(self):
        self.timesteps = np.array([737000, 737001, 737002])
        self.id_timeseries = np.array([0, 1, 1, 2])
        dummy = dummy_hazard()
        dummy.frequency_unit = "1/year"
        dummy.date = np.array([737000, 737001, 737002, 737000])
        self.hts = HazardTimeSeries(
            timesteps=self.timesteps, id_timeseries=self.id_timeseries, **dummy.__dict__
        )

    def test_init(self):
        self.assertTrue(hasattr(self.hts, "timesteps"))
        self.assertTrue(hasattr(self.hts, "id_timeseries"))
        npt.assert_array_equal(self.hts.timesteps, self.timesteps)
        npt.assert_array_equal(self.hts.id_timeseries, self.id_timeseries)

    def test_check_time_series_pass(self):
        # Should not raise
        self.hts.check_time_series()

    def test_check_time_series_fail_timesteps(self):
        self.hts.timesteps = np.array(
            [737000, 737000, 737002]
        )  # not strictly increasing
        with self.assertRaises(ValueError):
            self.hts.check_time_series()
        self.hts.timesteps = np.array([737001, 737002])  # event dates not in timesteps
        with self.assertRaises(ValueError):
            self.hts.check_time_series()

    def test_sample_from_hazard_set(self):
        hazard = dummy_hazard()
        hazard.frequency_unit = "1/year"
        timesteps = np.array([737000, 737001, 737002])
        hts = HazardTimeSeries.sample_from_hazard_set(
            hazard=hazard,
            n_timeseries=2,
            timesteps=timesteps,
            seasonality=None,
            intensity_increase=None,
            seed=42,
        )
        self.assertIsInstance(hts, HazardTimeSeries)
        self.assertEqual(hts.timesteps.shape[0], 3)
        self.assertEqual(hts.id_timeseries.shape[0], hts.date.shape[0])

    def test_sample_from_hazard_set_with_intensity_increase(self):
        hazard = dummy_hazard()
        hazard.frequency_unit = "1/year"
        timesteps = np.array([737000, 737001, 737002])
        intensity_increase = np.array([1.0, 2.0, 3.0])
        hts = HazardTimeSeries.sample_from_hazard_set(
            hazard=hazard,
            n_timeseries=2,
            timesteps=timesteps,
            seasonality=None,
            intensity_increase=intensity_increase,
            seed=42,
        )
        self.assertIsInstance(hts, HazardTimeSeries)
        self.assertEqual(hts.timesteps.shape[0], 3)


class TestSampleIndependentNEvents(unittest.TestCase):
    def test_shape_and_type(self):
        n_timeseries = 5
        n_timesteps = 10
        mean_frequency = 2.5
        arr = _sample_independent_nevents_per_time_series_bin(
            n_timeseries, n_timesteps, mean_frequency
        )
        self.assertEqual(arr.shape, (n_timeseries, n_timesteps))
        self.assertTrue(np.issubdtype(arr.dtype, np.integer))

    def test_weights(self):
        n_timeseries = 10
        n_timesteps = 4
        mean_frequency = 1.0
        weights = np.array([1, 2, 3, 4])
        arr = _sample_independent_nevents_per_time_series_bin(
            n_timeseries, n_timesteps, mean_frequency, weights=weights
        )
        self.assertEqual(arr.shape, (n_timeseries, n_timesteps))
        # Check that higher weights tend to produce higher event counts
        self.assertTrue(np.all(np.mean(arr, axis=0)[-1] >= np.mean(arr, axis=0)[0]))

    def test_weights_length_mismatch(self):
        with self.assertRaises(ValueError):
            _sample_independent_nevents_per_time_series_bin(
                2, 3, 1.0, weights=np.array([1, 2])
            )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestHazardTimeSeries)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestSampleIndependentNEvents)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
