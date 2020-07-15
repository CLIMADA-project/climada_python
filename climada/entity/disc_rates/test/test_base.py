"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test DiscRates class.
"""
import os
import unittest
import numpy as np

from climada.entity.disc_rates.base import DiscRates
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_TODAY

CURR_DIR = os.path.dirname(__file__)
ENT_TEST_MAT = os.path.join(CURR_DIR,
                            '../../exposures/test/data/demo_today.mat')

class TestChecker(unittest.TestCase):
    """Test discount rates attributes checker"""

    def test_check_wrongRates_fail(self):
        """Wrong discount rates definition"""
        disc_rate = DiscRates()
        disc_rate.rates = np.array([3, 4])
        disc_rate.years = np.array([1])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                disc_rate.check()
        self.assertIn('Invalid DiscRates.rates size: 1 != 2.', cm.output[0])

class TestConstructor(unittest.TestCase):
    """Test discount rates attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        disc_rate = DiscRates()
        self.assertTrue(hasattr(disc_rate, 'years'))
        self.assertTrue(hasattr(disc_rate, 'rates'))

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_append_to_empty_same(self):
        """Append DiscRates to empty one."""
        disc_rate = DiscRates()
        disc_rate_add = DiscRates()
        disc_rate_add.tag.file_name = 'file1.txt'
        disc_rate_add.tag.description = 'descr1'
        disc_rate_add.years = np.array([2000, 2001, 2002])
        disc_rate_add.rates = np.array([0.1, 0.2, 0.3])

        disc_rate.append(disc_rate_add)
        disc_rate.check()

        self.assertTrue(np.array_equal(disc_rate.years, disc_rate_add.years))
        self.assertTrue(np.array_equal(disc_rate.rates, disc_rate_add.rates))
        self.assertTrue(np.array_equal(disc_rate.tag.file_name,
                                       disc_rate_add.tag.file_name))
        self.assertTrue(np.array_equal(disc_rate.tag.description,
                                       disc_rate_add.tag.description))

    def test_append_equal_same(self):
        """Append the same DiscRates. The inital DiscRates is obtained."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.array([2000, 2001, 2002])
        disc_rate.rates = np.array([0.1, 0.2, 0.3])

        disc_rate_add = DiscRates()
        disc_rate_add.tag.file_name = 'file1.txt'
        disc_rate_add.tag.description = 'descr1'
        disc_rate_add.years = np.array([2000, 2001, 2002])
        disc_rate_add.rates = np.array([0.1, 0.2, 0.3])

        disc_rate.append(disc_rate_add)
        disc_rate.check()

        self.assertTrue(np.array_equal(disc_rate.years, disc_rate_add.years))
        self.assertTrue(np.array_equal(disc_rate.rates, disc_rate_add.rates))
        self.assertTrue(np.array_equal(disc_rate.tag.file_name, disc_rate_add.tag.file_name))
        self.assertEqual(disc_rate.tag.description, disc_rate_add.tag.description)

    def test_append_different_append(self):
        """Append DiscRates with same and new values. The rates with repeated
        years are overwritten."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.array([2000, 2001, 2002])
        disc_rate.rates = np.array([0.1, 0.2, 0.3])

        disc_rate_add = DiscRates()
        disc_rate_add.tag.file_name = 'file2.txt'
        disc_rate_add.tag.description = 'descr2'
        disc_rate_add.years = np.array([2000, 2001, 2003])
        disc_rate_add.rates = np.array([0.11, 0.22, 0.33])

        disc_rate.append(disc_rate_add)
        disc_rate.check()

        self.assertTrue(np.array_equal(disc_rate.years,
                                       np.array([2000, 2001, 2002, 2003])))
        self.assertTrue(np.array_equal(disc_rate.rates,
                                       np.array([0.11, 0.22, 0.3, 0.33])))
        self.assertTrue(np.array_equal(disc_rate.tag.file_name, 'file1.txt + file2.txt'))
        self.assertTrue(np.array_equal(disc_rate.tag.description, 'descr1 + descr2'))

class TestSelect(unittest.TestCase):
    """Test select method"""
    def test_select_pass(self):
        """Test select right time range."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.arange(2000, 2050)
        disc_rate.rates = np.arange(disc_rate.years.size)

        year_range = np.arange(2010, 2020)
        sel_disc = disc_rate.select(year_range)

        self.assertTrue(np.array_equal(sel_disc.years, year_range))
        self.assertTrue(np.array_equal(sel_disc.rates, disc_rate.rates[10:20]))

    def test_select_wrong_pass(self):
        """Test select wrong time range."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.arange(2000, 2050)
        disc_rate.rates = np.arange(disc_rate.years.size)

        year_range = np.arange(2050, 2060)
        self.assertEqual(None, disc_rate.select(year_range))

class TestNetPresValue(unittest.TestCase):
    """Test select method"""
    def test_net_present_value_pass(self):
        """Test net_present_value right time range."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.arange(2000, 2050)
        disc_rate.rates = np.ones(disc_rate.years.size) * 0.02

        val_years = np.ones(23) * 6.512201157564418e9
        res = disc_rate.net_present_value(2018, 2040, val_years)
        self.assertEqual(res, 1.215049630691397e+11)

    def test_net_present_value_wrong_pass(self):
        """Test net_present_value wrong time range."""
        disc_rate = DiscRates()
        disc_rate.tag.file_name = 'file1.txt'
        disc_rate.tag.description = 'descr1'
        disc_rate.years = np.arange(2000, 2050)
        disc_rate.rates = np.arange(disc_rate.years.size)
        val_years = np.ones(11) * 6.512201157564418e9
        with self.assertRaises(ValueError):
            disc_rate.net_present_value(2050, 2060, val_years)

class TestReaderExcel(unittest.TestCase):
    """Test excel reader for discount rates"""

    def test_demo_file_pass(self):
        """Read demo excel file."""
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read_excel(ENT_DEMO_TODAY, description)

        # Check results
        n_rates = 51

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates - 1], 2050)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_DEMO_TODAY)
        self.assertEqual(disc_rate.tag.description, description)

    def test_template_file_pass(self):
        """Read demo excel file."""
        disc_rate = DiscRates()
        disc_rate.read_excel(ENT_TEMPLATE_XLS)

        # Check results
        n_rates = 102

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(disc_rate.years.shape, (n_rates,))
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates - 1], 2101)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(disc_rate.rates.shape, (n_rates,))
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(disc_rate.tag.description, '')

class TestReaderMat(unittest.TestCase):
    """Test mat reader for discount rates"""

    def test_demo_file_pass(self):
        """Read demo mat file"""
        # Read demo excel file
        disc_rate = DiscRates()
        description = 'One single file.'
        disc_rate.read_mat(ENT_TEST_MAT, description)

        # Check results
        n_rates = 51

        self.assertIn('int', str(disc_rate.years.dtype))
        self.assertEqual(len(disc_rate.years), n_rates)
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates - 1], 2050)

        self.assertIn('float', str(disc_rate.rates.dtype))
        self.assertEqual(len(disc_rate.rates), n_rates)
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(disc_rate.tag.description, description)


class TestWriter(unittest.TestCase):
    """Test excel reader for discount rates"""

    def test_write_read_pass(self):
        """Read demo excel file."""
        disc_rate = DiscRates()
        disc_rate.years = np.arange(1950, 2150)
        disc_rate.rates = np.ones(disc_rate.years.size) * 0.03

        file_name = os.path.join(os.path.join(CURR_DIR, 'data'), 'test_disc.xlsx')
        disc_rate.write_excel(file_name)

        disc_read = DiscRates()
        disc_read.read_excel(file_name)

        self.assertTrue(np.array_equal(disc_read.years, disc_rate.years))
        self.assertTrue(np.array_equal(disc_read.rates, disc_rate.rates))

        self.assertEqual(disc_read.tag.file_name, file_name)
        self.assertEqual(disc_read.tag.description, '')

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecker)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelect))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNetPresValue))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWriter))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
