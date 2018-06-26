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
        country_name = 'Haiti'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'HTI')

    def test_barbados_pass(self):
        """BRB"""
        country_name = 'Barbados'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'BRB')

    def test_zambia_pass(self):
        """ZMB"""
        country_name = 'Zambia'
        iso_name = country_iso(country_name)
        self.assertEqual(iso_name, 'ZMB')        
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCountryIso)
unittest.TextTestRunner(verbosity=2).run(TESTS)
