"""
Test TropCyclone class
"""

import unittest
import os
import numpy as np

from climada.hazard.trop_cyclone import TropCyclone, read_ibtracs, set_category, missing_pressure
from climada.util.config import CONFIG

TEST_TRACK = os.path.join(CONFIG['local_data']['repository'], "demo_trac.csv")

class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_read_one_pass(self):
        """Read and fill one tropical cyclone."""
        centroids = None
        trop = TropCyclone()
        with self.assertRaises(NotImplementedError): 
            trop._read_one(TEST_TRACK, 'TC', '', centroids)

    def test_read_fail(self):
        """Read a tropical cyclone from constructor."""
        with self.assertRaises(NotImplementedError): 
            TropCyclone(TEST_TRACK)

class TestIBTracs(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_category_pass(self):
        """Test category computation."""
        max_sus_wind = np.array([25, 30, 35, 40, 45, 45, 45, 45, 35, 25])
        max_sus_wind_unit = 'kn'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([25, 25, 25, 30, 30, 30, 30, 30, 25, 25, 20])
        max_sus_wind_unit = 'kn'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([80, 90, 100, 115, 120, 125, 130, 120, 110, 80, 75, 80, 65])
        max_sus_wind_unit = 'kn'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

        max_sus_wind = np.array([28.769475, 34.52337, 40.277265, 46.03116, 51.785055, \
        51.785055, 51.785055, 51.785055, 40.277265, 28.769475])
        max_sus_wind_unit = 'mph'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)
        
        max_sus_wind = np.array([12.86111437, 12.86111437, 12.86111437, 15.43333724, \
        15.43333724, 15.43333724, 15.43333724, 15.43333724, 12.86111437, 12.86111437, 10.2888915])
        max_sus_wind_unit = 'm/s'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([148.16, 166.68, 185.2, 212.98, 222.24, 231.5, 240.76, \
        222.24, 203.72, 148.16, 138.9, 148.16, 120.38])
        max_sus_wind_unit = 'km/h'
        cat = set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

    def test_missing_pres_pass(self):
        """Test central pressure function."""
        cen_pres = np.array([-999, -999, -999, -999, -999, -999, -999, -999, -999, 992, -999, -999, 993, -999, -999, 1004])
        v_max = np.array([45, 50, 50, 55, 60, 65, 70, 80, 75, 70, 70, 70, 70, 65, 55, 45]) 
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, 14.1, 14.1, 14.2, 14.2, 14.3, 14.4, 14.6, 14.8, 15, 15.1, 15.1])
        lon = np.array([-51.1, -52.8, -54.4, -56, -57.3, -58.4, -59.7, -61.1, -62.7, -64.3, -65.8, -67.4, -69.4, -71.4, -73, -74.2])
        out_pres = missing_pressure(cen_pres, v_max, lat, lon)

        ref_res = np.array([989.7085, 985.6725, 985.7236, 981.6847, 977.6324, 973.5743, 969.522, 961.3873, 965.5237, 969.6648, 969.713, 969.7688, 969.8362, 973.9936, 982.2247, 990.4395])
        np.testing.assert_array_almost_equal(ref_res, out_pres)

    def test_read_pass(self):
        """Read a tropical cyclone."""
        tc_track = read_ibtracs(TEST_TRACK)
        
        self.assertEqual(tc_track['lon'][11], -39.60)
        self.assertEqual(tc_track['lat'][23], 14.10)
        self.assertEqual(tc_track['time_step'][7], 6)
        self.assertEqual(np.max(tc_track['radius_max_wind']), 0)
        self.assertEqual(np.min(tc_track['radius_max_wind']), 0)
        self.assertEqual(tc_track['max_sustained_wind'][21], 55)
        self.assertEqual(tc_track['central_pressure'][29], 969.76880)
        self.assertEqual(np.max(tc_track['environmental_pressure']), 1010)
        self.assertEqual(np.min(tc_track['environmental_pressure']), 1010)
        self.assertEqual(tc_track['time.year'][13], 1951)
        self.assertEqual(tc_track['time.month'][26], 9)
        self.assertEqual(tc_track['time.day'][7], 29)
        self.assertEqual(tc_track.attrs['max_sustained_wind_unit'], 'kn')
        self.assertEqual(tc_track.attrs['central_pressure_unit'], 'kn')
        self.assertEqual(tc_track.attrs['orig_event_flag'], 1)
        self.assertEqual(tc_track.attrs['name'], '1951239N12334')
        self.assertEqual(tc_track.attrs['data_provider'], 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.attrs['basin']))
        self.assertEqual(tc_track.attrs['id_no'], 1951239012334)
        self.assertEqual(tc_track.attrs['category'], 1)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIBTracs))
unittest.TextTestRunner(verbosity=2).run(TESTS)
