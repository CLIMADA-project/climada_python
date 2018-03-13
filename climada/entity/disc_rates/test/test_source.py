"""
Test DiscRates from Excel.
"""

import unittest
import numpy

from climada.entity.disc_rates.base import DiscRates
from climada.entity.disc_rates import source as source
from climada.util.constants import ENT_DEMO_XLS, ENT_TEMPLATE_XLS, ENT_DEMO_MAT

class TestReaderExcel(unittest.TestCase):
    """Test excel reader for discount rates"""

    def tearDown(self):
        source.DEF_VAR_EXCEL = {'sheet_name': 'discount',
                                'col_name': {'year' : 'year',
                                             'disc' : 'discount_rate'
                                            }
                               }
                             
    def test_demo_file_pass(self):
        """ Read demo excel file."""      
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read(ENT_DEMO_XLS, description)

        # Check results
        n_rates = 51

        self.assertEqual(type(disc_rate.years[0]), numpy.int64)
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2050)

        self.assertEqual(type(disc_rate.rates[0]), numpy.float64)
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_DEMO_XLS)
        self.assertEqual(disc_rate.tag.description, description)

    def test_template_file_pass(self):
        """ Read demo excel file."""
        disc_rate = DiscRates(ENT_TEMPLATE_XLS)

        # Check results
        n_rates = 102

        self.assertEqual(type(disc_rate.years[0]), numpy.int64)
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2101)

        self.assertEqual(type(disc_rate.rates[0]), numpy.float64)
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(disc_rate.tag.description, '')

    def test_wrong_file_fail(self):
        """ Read file without year column, fail."""
        new_var_name = source.DEF_VAR_EXCEL
        new_var_name['col_name']['year'] = 'wrong col'
        disc_rate = DiscRates()
        with self.assertRaises(KeyError):
            disc_rate.read(ENT_DEMO_XLS, var_names=new_var_name)

class TestReaderMat(unittest.TestCase):
    """Test mat reader for discount rates"""
    def tearDown(self):
        source.DEF_VAR_MAT = {'sup_field_name': 'entity',
                              'field_name': 'discount',
                              'var_name': {'year' : 'year',
                                           'disc' : 'discount_rate'
                                          }
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
        new_var_name = source.DEF_VAR_MAT
        new_var_name['var_name']['year'] = 'wrong col'
        disc_rate = DiscRates()
        with self.assertRaises(KeyError):
            disc_rate.read(ENT_DEMO_MAT, var_names=new_var_name)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
unittest.TextTestRunner(verbosity=2).run(TESTS)
