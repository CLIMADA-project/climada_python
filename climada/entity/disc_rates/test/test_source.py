"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test DiscRates from Excel.
"""
import os
import unittest

from climada.entity.disc_rates.base import DiscRates
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

class TestReaderExcel(unittest.TestCase):
    """Test excel reader for discount rates"""
                             
    def test_demo_file_pass(self):
        """ Read demo excel file."""      
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read(ENT_TEST_XLS, description)

        # Check results
        n_rates = 51

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2050)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_TEST_XLS)
        self.assertEqual(disc_rate.tag.description, description)

    def test_template_file_pass(self):
        """ Read demo excel file."""
        disc_rate = DiscRates(ENT_TEMPLATE_XLS)

        # Check results
        n_rates = 102

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2101)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(disc_rate.tag.description, '')

class TestReaderMat(unittest.TestCase):
    """Test mat reader for discount rates"""
    
    def test_demo_file_pass(self):
        """ Read demo mat file"""
        # Read demo excel file
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read(ENT_DEMO_MAT, description)

        # Check results
        n_rates = 51

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(len(disc_rate.years), n_rates)
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2050)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(len(disc_rate.rates), n_rates)
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(disc_rate.tag.description, description)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
unittest.TextTestRunner(verbosity=2).run(TESTS)
