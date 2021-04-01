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

Test statistical analysis functionalities
"""

import unittest

import numpy as np
import pandas as pd
import scipy.sparse as sp

from climada.hazard import Hazard, Centroids
from climada.hazard.emulator import stats


class TestStats(unittest.TestCase):
    """Test statistical analysis functionalities"""

    def test_seasonal_average(self):
        """Test seasonal_average function"""
        timed_data = pd.DataFrame({
            "year": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "month": [4, 6, 11, 3, 5, 8, 9, 10],
            "value": [6, 1, 3, 2, 4, 0, 5, 7],
        })
        data = stats.seasonal_average(timed_data, [4, 6])
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data['year'][0], 2000)
        self.assertEqual(data['year'][1], 2001)
        self.assertEqual(data['value'][0], 3.5)
        self.assertEqual(data['value'][1], 4)

        data = stats.seasonal_average(timed_data, [10, 4])
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data['year'][0], 2001)
        self.assertEqual(data['value'][0], 2.5)

        timed_data = pd.DataFrame({"year": [1990, 1991], "intensity": [0, 1]})
        data = stats.seasonal_average(timed_data, [5, 6])
        self.assertTrue(np.all(timed_data == data))


    def test_seasonal_statistics(self):
        """Test seasonal_statistics function"""
        events = pd.DataFrame({
            "year": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "month": [4, 6, 11, 3, 5, 8, 9, 10],
            "intensity": [6, 1, 3, 2, 4, 0, 5, 7],
        })
        data = stats.seasonal_statistics(events, [4, 6])
        self.assertEqual(data.shape[0], 2)
        self.assertEqual(data['year'][0], 2000)
        self.assertEqual(data['year'][1], 2001)
        self.assertEqual(data['intensity_mean'][0], 3.5)
        self.assertEqual(data['intensity_mean'][1], 4)
        self.assertEqual(data['eventcount'][0], 2)
        self.assertEqual(data['eventcount'][1], 1)

        data = stats.seasonal_statistics(events, [10, 4])
        self.assertEqual(data.shape[0], 1)
        self.assertEqual(data['year'][0], 2001)
        self.assertEqual(data['intensity_mean'][0], 2.5)
        self.assertEqual(data['eventcount'][0], 2)


    def test_norm_seas_stats(self):
        """Test normalize_seasonal_statistics function"""
        haz_stats = pd.DataFrame({
            "year": [1999, 2000, 2001, 2002],
            "eventcount": [27, 25, 26, 25],
            "intensity_mean": [4, 6, 5, 5],
            "intensity_std": [1, 1, 1, 1],
            "intensity_max": [12, 12, 13, 15],
        })
        haz_stats_obs = pd.DataFrame({
            "year": [2000, 2001],
            "eventcount": [12, 11],
            "intensity_mean": [10, 12],
            "intensity_std": [2, 1],
            "intensity_max": [11, 14],
        })
        freq = pd.DataFrame({
            "year": [1999, 2000, 2001, 2002],
            "freq": [0.5, 0.4, 0.5, 0.5],
        })

        data = stats.normalize_seasonal_statistics(haz_stats, haz_stats_obs, freq)
        self.assertSequenceEqual(data['year'].tolist(), haz_stats['year'].tolist())
        self.assertSequenceEqual(data['eventcount'].tolist(), [13.5, 10, 13, 12.5])
        self.assertSequenceEqual(data['intensity_mean'].tolist(), [8, 12, 10, 10])
        self.assertSequenceEqual(data['intensity_std'].tolist(), [1.5, 1.5, 1.5, 1.5])
        self.assertSequenceEqual(data['intensity_max'].tolist(), [12, 12, 13, 15])


    def test_haz_max_events(self):
        """Test haz_max_events function"""
        hazard = Hazard('TC')
        hazard.centroids = Centroids()
        hazard.centroids.set_lat_lon(np.array([1, 3, 5]), np.array([2, 4, 6]))
        hazard.event_id = np.array([1, 2, 3, 4])
        hazard.event_name = ['ev1', 'ev2', 'ev3', 'ev4']
        hazard.date = np.array([1, 3, 5, 7])
        hazard.intensity = sp.csr_matrix(
            [[0, 0, 4], [1, 0, 1], [43, 21, 0], [0, 53, 1]])
        data = stats.haz_max_events(hazard, min_thresh=18)
        self.assertSequenceEqual(data['id'].tolist(), [2, 3])
        self.assertSequenceEqual(data['name'].tolist(), ["ev3", "ev4"])
        self.assertSequenceEqual(data['year'].tolist(), [1, 1])
        self.assertSequenceEqual(data['month'].tolist(), [1, 1])
        self.assertSequenceEqual(data['day'].tolist(), [5, 7])
        self.assertSequenceEqual(data['lat'].tolist(), [1, 3])
        self.assertSequenceEqual(data['lon'].tolist(), [2, 4])
        self.assertSequenceEqual(data['intensity'].tolist(), [43, 53])


    def test_fit_data(self):
        """Test fit_data function"""
        haz_stats = pd.DataFrame({
            "year": [1999, 2000, 2001, 2002],
            "eventcount": [14, 12, 16, 11],
            "intensity_mean": [9, 13, 11, 11],
            "gmt": [4, 6, 5, 5],
            "esoi": [1, -1, 1, -1],
            "dummy": [0, 1, 2, 0],
        })

        sm_results = stats.fit_data(haz_stats, "intensity_mean", ["gmt", "esoi"])
        self.assertTrue("gmt" in sm_results[0].params)
        self.assertTrue("esoi" in sm_results[0].params)
        self.assertTrue("gmt" in sm_results[1].params)
        self.assertFalse("esoi" in sm_results[1].params)
        self.assertAlmostEqual(sm_results[1].params['gmt'], 2)
        self.assertAlmostEqual(sm_results[1].params['const'], 1)

        sm_results = stats.fit_data(haz_stats, "eventcount", ["gmt", "esoi"], poisson=True)
        self.assertTrue("gmt" in sm_results[0].params)
        self.assertTrue("esoi" in sm_results[0].params)
        self.assertTrue("gmt" in sm_results[1].params)
        self.assertTrue("esoi" in sm_results[1].params)
        self.assertAlmostEqual(sm_results[1].params['gmt'], 0.1, places=1)
        self.assertAlmostEqual(sm_results[1].params['esoi'], 0.2, places=1)
        self.assertAlmostEqual(sm_results[1].params['const'], 2, places=1)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestStats)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
