"""
Test Exposure base class.
"""

import unittest
import numpy as np

from climada.hazard.centroids.base import Centroids

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Centroids class"""

    @staticmethod
    def good_centroids():
        """Define well a Centroids"""
        cen = Centroids()
        cen.coord = np.array([[1, 2], [3, 4], [5, 6]])
        cen.id = np.array([1, 2, 3])
        cen.region_id = np.array([1, 2, 3])

        return cen

    def test_check_wrongCoord_fail(self):
        """Wrong centroids definition"""
        cen = self.good_centroids()
        cen.coord = np.array([[1, 2],[3, 4]])

        with self.assertRaises(ValueError) as error:
            cen.check()
        self.assertEqual('Invalid Centroids.coord size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongRegion_fail(self):
        """Wrong centroids definition"""
        cen = self.good_centroids()
        cen.region_id = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            cen.check()
        self.assertEqual('Invalid Centroids.region_id size: 3 != 2', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        cen = Centroids()
        with self.assertRaises(NotImplementedError):
            cen.load('filename')

    def test_read_notimplemented(self):
        """Read function not implemented"""
        cen = Centroids()
        with self.assertRaises(NotImplementedError):
            cen.read('filename')

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Centroids('filename')

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
