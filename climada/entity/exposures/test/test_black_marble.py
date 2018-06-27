"""
Test BlackMarble base class.
"""
import unittest

from climada.entity.exposures.black_marble import country_iso

class TestCountryIso(unittest.TestCase):
    """Test country_iso function."""
    def test_germany_pass(self):
        """DEU """
        country_name = 'Germany'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'DEU')

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

    def test_barbados_pass(self):
        """BRB"""
        country_name = 'barbados'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'BRB')

    def test_zambia_pass(self):
        """ZMB"""
        country_name = 'ZAMBIA'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'ZMB')

    def test_kosovo_fail(self):
        """ZMB"""
        country_name = 'Kosovo'
        with self.assertRaises(ValueError):
            country_iso(country_name)

    def test_bolivia_pass(self):
        """ZMB"""
        country_name = 'Bolivia'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'BOL')

    def test_korea_pass(self):
        """ZMB"""
        country_name = 'Korea'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'PRK')
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCountryIso)
unittest.TextTestRunner(verbosity=2).run(TESTS)
