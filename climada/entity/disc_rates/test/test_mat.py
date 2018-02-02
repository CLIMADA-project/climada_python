"""
Test DiscRates from MATLAB file.
"""

import unittest
import numpy

from climada.entity.disc_rates import source_mat as mat
from climada.entity.disc_rates.base import DiscRates
from climada.util.constants import ENT_DEMO_MAT

class TestReader(unittest.TestCase):
    """Test reader functionality of the DiscRatesMat class"""

    def tearDown(self):
        mat.VAR_NAMES = {'year' : 'year',
                         'disc' : 'discount_rate'
                        }

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
        # Read demo excel file
        disc_rate = DiscRates()
        mat.VAR_NAMES['year'] = 'wrong col'
        with self.assertRaises(KeyError):
            disc_rate.read(ENT_DEMO_MAT)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
