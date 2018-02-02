"""
Test Exposure base class.
"""

import pickle
import unittest
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_DEMO_XLS

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
    
    def test_centroid_save_pass(self):
        """ Save the centroids object after being read correctly"""

        # Read demo excel file
        centroids = Centroids()
        description = 'One single file.'
        out_file_name = 'centroid_excel.pkl'
        centroids.load(HAZ_DEMO_XLS, description, out_file_name)

        # Getting back the objects:
        with open(out_file_name, 'rb') as file:
            centroids_read = pickle.load(file)

        # Check the loaded hazard have all variables
        self.assertEqual(hasattr(centroids_read, 'tag'), True)
        self.assertEqual(hasattr(centroids_read, 'id'), True)
        self.assertEqual(hasattr(centroids_read, 'coord'), True)
        self.assertEqual(hasattr(centroids_read, 'region_id'), True)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
