"""
Test Discounts class.
"""

import unittest
import numpy

from climada.entity.discounts.base import Discounts
from climada.util.constants import ENT_DEMO_XLS

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Discounts class"""

    def test_check_wrongRates_fail(self):
        """Wrong discount rates definition"""
        disc_rate = Discounts()
        disc_rate.rates = numpy.array([3,4])
        disc_rate.years = numpy.array([1])

        with self.assertRaises(ValueError) as error:
            disc_rate.check()
        self.assertEqual('Invalid discount rates size: 1 != 2', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        disc_rate = Discounts()
        with self.assertRaises(NotImplementedError):
            disc_rate.load(ENT_DEMO_XLS)

    def test_read_notimplemented(self):
        """Read function not implemented"""
        disc_rate = Discounts()
        with self.assertRaises(NotImplementedError):
            disc_rate.read(ENT_DEMO_XLS)

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Discounts(ENT_DEMO_XLS)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
