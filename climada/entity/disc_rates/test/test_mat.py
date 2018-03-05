"""
Test DiscRates from MATLAB file.
"""

import unittest
import numpy

from climada.entity.disc_rates.source_mat import DEF_VAR_NAME
from climada.entity.disc_rates.base import DiscRates
from climada.util.constants import ENT_DEMO_MAT

class TestReader(unittest.TestCase):
    """Test mat reader for discount rates"""
    def test_demo_file_pass(self):
        """ Read demo mat file"""
        # Read demo excel file
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read(ENT_DEMO_MAT, description)

        # Check results
        n_rates = 51

        self.assertEqual(type(disc_rate.years[0]), numpy.int64)
        self.assertEqual(len(disc_rate.years), n_rates)
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2050)

        self.assertEqual(type(disc_rate.rates[0]), numpy.float64)
        self.assertEqual(len(disc_rate.rates), n_rates)
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(disc_rate.tag.description, description)

    def test_wrong_file_fail(self):
        """ Read file without year column, fail."""
        new_var_name = DEF_VAR_NAME
        new_var_name['var_name']['year'] = 'wrong col'
        disc_rate = DiscRates()
        with self.assertRaises(KeyError):
            disc_rate.read(ENT_DEMO_MAT, var_names=new_var_name)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
