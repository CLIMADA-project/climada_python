"""
Tests on Black marble.
"""

import unittest

from climada.entity.exposures.black_marble import BlackMarble

class Test2013(unittest.TestCase):
    """Test plot functions."""

    def test_spain_pass(self):
        country_name = ['Spain']
        ent = BlackMarble()
        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=1, sea_res=(200, 50))
        self.assertIn('GDP ESP 2013: 1.362e+12.', cm.output[0])
        self.assertIn('Income group ESP 2013: 4.', cm.output[1])
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.", cm.output[2])
        self.assertIn("Processing country Spain.", cm.output[3])
        self.assertIn("Generating resolution of approx 1 km.", cm.output[4])

    def test_sint_maarten_pass(self):
        country_name = ['Sint Maarten']
        ent = BlackMarble()
        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=0.2, sea_res=(200, 50))
        self.assertIn('GDP SXM 2014: 3.658e+08.', cm.output[0])
        self.assertIn('Income group SXM 2013: 4.', cm.output[1])
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.", cm.output[2])
        self.assertIn("Processing country Sint Maarten.", cm.output[3])
        self.assertIn("Generating resolution of approx 0.2 km.", cm.output[4])

    def test_anguilla_pass(self):
        country_name = ['Anguilla']
        ent = BlackMarble()
        ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertEqual(ent.ref_year, 2013)
        self.assertIn("Anguilla 2013 GDP: 1.754e+08 income group: 3", ent.tag.description)

class Test1968(unittest.TestCase):
    """Test plot functions."""
    def test_switzerland_pass(self):
        country_name = ['Switzerland']
        ent = BlackMarble()
        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 1968, res_km=0.5)
        self.assertIn('GDP CHE 1968: 1.894e+10.', cm.output[0])
        self.assertIn('Income group CHE 1987: 4.', cm.output[1])
        self.assertIn("Nightlights from NOAA's earth observation group for year 1992.", cm.output[2])
        self.assertIn("Processing country Switzerland.", cm.output[3])
        self.assertIn("Generating resolution of approx 0.5 km.", cm.output[4])    

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(Test2013)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(Test1968))
unittest.TextTestRunner(verbosity=2).run(TESTS)
