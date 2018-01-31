"""
Test Exposure base class.
"""

import unittest
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.hazard.base import Hazard
from climada.util.constants import ENT_DEMO_XLS

class TestAssign(unittest.TestCase):
    """Check assign interface"""

    def test_assign_pass(self):
        """ Check that assigned attribute is correctly set."""
        # Fill with dummy values the coordinates
        expo = Exposures()
        num_coord = 4
        expo.coord = np.ones((num_coord, 2))
        # Fill with dummy values the centroids
        haz = Hazard()
        haz.centroids.coord = np.ones((num_coord+6, 2))
        # assign
        expo.assign(haz)

        # check assigned variable has been set with correct length
        self.assertEqual(num_coord, len(expo.assigned))

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Exposures class"""

    @staticmethod
    def good_exposures():
        expo = Exposures()
        # Followng values defined for each exposure
        expo.id = np.array([1, 2, 3])
        expo.coord = np.array([[1, 2], [2, 3], [3, 4]])
        expo.value = np.array([1, 2, 3])
        expo.deductible = np.array([1, 2, 3])
        expo.cover = np.array([])
        expo.impact_id = np.array([1, 2, 3])
        expo.category_id = np.array([1, 2, 3])
        expo.region_id = np.array([1, 2, 3])
        expo.assigned = np.array([1, 2, 3])

        return expo

    def test_check_wrongValue_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.value = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.value size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongCoord_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.coord = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Exposures.coord has wrong dimensions.', \
                         str(error.exception))

    def test_check_wrongDeduct_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.deductible = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.deductible size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongCover_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.cover = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.cover size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongImpact_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.impact_id = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.impact_id size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongCategory_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.category_id = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.category_id size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongRegion_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.region_id = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.region_id size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongAssigned_fail(self):
        """Wrong exposures definition"""
        expo = self.good_exposures()
        expo.assigned = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            expo.check()
        self.assertEqual('Invalid Exposures.assigned size: 3 != 2',\
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        expo = Exposures()
        with self.assertRaises(NotImplementedError):
            expo.load(ENT_DEMO_XLS)

    def test_read_notimplemented(self):
        """Read function not implemented"""
        expo = Exposures()
        with self.assertRaises(NotImplementedError):
            expo.read(ENT_DEMO_XLS)

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Exposures(ENT_DEMO_XLS)

if __name__ == '__main__':
    unittest.main()
