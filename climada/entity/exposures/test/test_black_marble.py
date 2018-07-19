"""
Test BlackMarble base class.
"""
import unittest
import numpy as np

from climada.entity.exposures.black_marble import country_iso, cut_nightlight_country, load_nightlight
from climada.entity.exposures.black_marble import MIN_LAT, MAX_LAT, MIN_LON, MAX_LON, NOAA_RESOLUTION_DEG

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""

    def test_switzerland_pass(self):
        """CHE"""
        country_name = 'Switzerland'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'CHE')

    def test_haiti_pass(self):
        """HTI"""
        country_name = 'HaITi'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'HTI')

    def test_kosovo_pass(self):
        """KOS"""
        country_name = 'Kosovo'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'KOS')

    def test_wrong_fail(self):
        """Wrong name"""
        country_name = 'Kasovo'
        with self.assertRaises(ValueError):
            country_iso(country_name)

    def test_bolivia_pass(self):
        """BOL"""
        country_name = 'Bolivia'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'BOL')

    def test_korea_pass(self):
        """PRK"""
        country_name = 'Korea'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'PRK')

class TestNightLight(unittest.TestCase):
    """Test nightlight functions."""
        
    def test_cut_nightlight_country_pass(self):
        """Test cut_nightlight_country for three countries."""
        country_isos = ['HTI', 'ZMB', 'ESP']    
        nl_lat = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT, 16801)
        nl_lon = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON, 43201)
        in_lat, in_lon = cut_nightlight_country(country_isos, nl_lat, nl_lon)
        
        bounds = dict()
        bounds['ESP'] = (-18.16722571499986, 27.64223867400007, 4.337087436000104, 43.793443100999994)
        bounds['HTI'] = (-74.48916581899991, 18.02594635600012, -71.63911088099988, 20.08978913)
        bounds['ZMB'] = (21.97987756300006, -18.069231871999932, 33.67420251500005, -8.194124042999903)
        for cntry_key, cntry_val in in_lat.items():
            self.assertTrue(np.all(nl_lat[in_lat[cntry_key]]<=bounds[cntry_key][3]))
            self.assertTrue(np.all(nl_lat[in_lat[cntry_key]]>=bounds[cntry_key][1]))
            self.assertTrue(np.all(nl_lon[in_lon[cntry_key]]<=bounds[cntry_key][2]))
            self.assertTrue(np.all(nl_lon[in_lon[cntry_key]]>=bounds[cntry_key][0]))
    
    
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightLight)
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCountryIso)
unittest.TextTestRunner(verbosity=2).run(TESTS)
