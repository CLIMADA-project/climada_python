"""
Test DiscRates class.
"""

import unittest
import numpy as np

from climada.entity.disc_rates.base import DiscRates

class TestLoader(unittest.TestCase):
    """Test loading funcions from the DiscRates class"""

    def test_check_wrongRates_fail(self):
        """Wrong discount rates definition"""
        disc_rate = DiscRates()
        disc_rate.rates = np.array([3,4])
        disc_rate.years = np.array([1])

        with self.assertRaises(ValueError) as error:
            disc_rate.check()
        self.assertEqual('Invalid DiscRates.rates size: 1 != 2.', \
                         str(error.exception))

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
        self.assertTrue(np.array_equal(disc_rate.tag.file_name, disc_rate_add.tag.file_name))
        self.assertTrue(np.array_equal(disc_rate.tag.description, disc_rate_add.tag.description))

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
        self.assertTrue(np.array_equal(disc_rate.tag.file_name, \
               [disc_rate_add.tag.file_name, disc_rate_add.tag.file_name]))
        self.assertTrue(np.array_equal(disc_rate.tag.description, \
               [disc_rate_add.tag.description, disc_rate_add.tag.description]))

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

        self.assertTrue(np.array_equal(disc_rate.years, np.array([2000, 2001, 2002, 2003])))
        self.assertTrue(np.array_equal(disc_rate.rates, np.array([0.11, 0.22, 0.3, 0.33])))
        self.assertTrue(np.array_equal(disc_rate.tag.file_name, \
               ['file1.txt', 'file2.txt']))
        self.assertTrue(np.array_equal(disc_rate.tag.description, \
               ['descr1', 'descr2']))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
unittest.TextTestRunner(verbosity=2).run(TESTS)