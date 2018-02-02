"""
Test DiscRates class.
"""

import unittest
import numpy

from climada.entity.disc_rates.base import DiscRates

class TestLoader(unittest.TestCase):
    """Test loading funcions from the DiscRates class"""

    def test_check_wrongRates_fail(self):
        """Wrong discount rates definition"""
        disc_rate = DiscRates()
        disc_rate.rates = numpy.array([3,4])
        disc_rate.years = numpy.array([1])

        with self.assertRaises(ValueError) as error:
            disc_rate.check()
        self.assertEqual('Invalid DiscRates.rates size: 1 != 2', \
                         str(error.exception))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(TESTS)