"""
Test TropCyclone class
"""

import unittest

from climada.util.constants import HAZ_TEST_MAT
from climada.hazard.trop_cyclone import TropCyclone

class TestLoader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_read_fail(self):
        """Read a tropical cyclone."""
        with self.assertRaises(NotImplementedError): 
            TropCyclone(HAZ_TEST_MAT)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
