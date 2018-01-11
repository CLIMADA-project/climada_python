"""
Test Exposure base class.
"""

import unittest
import numpy as np
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Exposures class"""

    @staticmethod
    def good_hazard():
        """Define well a hazard"""
        haz = Hazard()
        haz.centroids = Centroids()
        haz.centroids.region_id = np.array([1, 2])
        haz.centroids.id = np.array([1, 2])
        haz.centroids.coord = np.array([[1, 2], [1, 2]])
        haz.event_id = np.array([1, 2, 3])
        haz.frequency = np.array([1, 2, 3])
        # events x centroids
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])
        haz.fraction = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])

        return haz

    def test_check_wrongCentroids_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.centroids.region_id = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid centroids regions size: 2 != 4', \
                         str(error.exception))

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.frequency = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid hazard frequency size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid hazard intensity row size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongFrac_fail(self):
        """Wrong exposures definition"""
        haz = self.good_hazard()
        haz.fraction = sparse.csr_matrix([[1], [1], [1]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid hazard fraction column size: 2 != 1', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        haz = Hazard()
        with self.assertRaises(NotImplementedError):
            haz.load('filename')

    def test_read_notimplemented(self):
        """Read function not implemented"""
        haz = Hazard()
        with self.assertRaises(NotImplementedError):
            haz.read('filename')

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Hazard('filename')

# Execute TestAssign
suite = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(suite)
