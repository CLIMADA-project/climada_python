"""
Tests on SPAM agriculture exposures.
"""

import unittest
# import numpy as np

from climada.entity.exposures.spam_agrar import SpamAgrar


class Test_V_agg(unittest.TestCase):
    """Test SPAM aggregated value (V_agg) exposure. (default)"""

    def test_suriname_pass(self):
        country_name = 'Suriname'
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name)
        self.assertIn('Lat. range: +1.875 to +5.958.', cm.output[0])
        self.assertIn('Lon. range: -58.042 to -54.042.', cm.output[1])
        self.assertIn("Total V_agg TA  Suriname: 78879225.2 USD.", cm.output[2])

    def test_zurich_pass(self):
        country_name = 'CHE'
        adm1 = 'Zurich'
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name, name_adm1=adm1)
        self.assertIn('Lat. range: +47.208 to +47.625.', cm.output[0])
        self.assertIn('Lon. range: +8.375 to +8.875.', cm.output[1])
        self.assertIn("Total V_agg TA  CHE Zurich: 56644555.1 USD.", cm.output[2])

class Test_OtherVar(unittest.TestCase):
    """Test SPAM exposures based on other variables. (default)"""
    def test_switzerland_pass(self):
        country_name = 'CHE'
        tech = 'TI' # irrigated
        var = 'H' # harvest area
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name, spam_variable=var,\
                                spam_technology=tech)
        self.assertIn('Lat. range: +45.875 to +47.792.', cm.output[0])
        self.assertIn('Lon. range: +6.042 to +10.375.', cm.output[1])
        self.assertIn("Total H TI  CHE: 28120.5 Ha.", cm.output[2])


# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(Test_V_agg)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(Test_OtherVar))
unittest.TextTestRunner(verbosity=2).run(TESTS)
