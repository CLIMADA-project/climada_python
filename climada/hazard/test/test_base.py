"""
Test Exposure base class.
"""

import pickle
import unittest
import numpy as np
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_DEMO_XLS

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    @staticmethod
    def good_hazard():
        """Define well a hazard"""
        haz = Hazard()
        haz.centroids = Centroids()
        haz.centroids.region_id = np.array([1, 2])
        haz.centroids.id = np.array([1, 2])
        haz.centroids.coord = np.array([[1, 2], [1, 2]])
        haz.event_id = np.array([1, 2, 3])
        haz.event_name = ['A', 'B', 'C']
        haz.frequency = np.array([1, 2, 3])
        # events x centroids
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])
        haz.fraction = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])

        return haz

    def test_constructor_wrong_fail(self):
        """Fail if no hazard type provided in constructor when file name."""
        with self.assertRaises(ValueError) as error:
            Hazard(HAZ_DEMO_XLS)
        self.assertEqual('Provide hazard type acronym.', str(error.exception))

    def test_check_wrongCentroids_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.centroids.region_id = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Centroids.region_id size: 2 != 4', \
                         str(error.exception))

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.frequency = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.frequency size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.intensity row size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongFrac_fail(self):
        """Wrong exposures definition"""
        haz = self.good_hazard()
        haz.fraction = sparse.csr_matrix([[1], [1], [1]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.fraction column size: 2 != 1', \
                         str(error.exception))

    def test_check_wrongEvName_fail(self):
        """Wrong exposures definition"""
        haz = self.good_hazard()
        haz.event_name = ['M']

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.event_name size: 3 != 1', \
                         str(error.exception))

    def test_hazard_save_pass(self):
        """ Save the hazard object after being read correctly"""

        # Read demo excel file
        hazard = Hazard()
        description = 'One single file.'
        out_file = 'hazard_excel.pkl'
        hazard.load(HAZ_DEMO_XLS, description, out_file_name=out_file)

        # Getting back the objects:
        with open(out_file, 'rb') as file:
            hazard_read = pickle.load(file)

        # Check the loaded hazard have all variables
        self.assertEqual(hasattr(hazard_read, 'tag'), True)
        self.assertEqual(hasattr(hazard_read, 'id'), True)
        self.assertEqual(hasattr(hazard_read, 'units'), True)
        self.assertEqual(hasattr(hazard_read, 'centroids'), True)
        self.assertEqual(hasattr(hazard_read, 'event_id'), True)
        self.assertEqual(hasattr(hazard_read, 'frequency'), True)
        self.assertEqual(hasattr(hazard_read, 'intensity'), True)
        self.assertEqual(hasattr(hazard_read, 'fraction'), True)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
