"""
Test coordinates module.
"""

import unittest
import numpy as np

from climada.util.coordinates import Coordinates, IrregularGrid, RegularGrid

class TestCoordinates(unittest.TestCase):
    ''' Test Coordinates class'''

    def test_shape_pass(self):
       """Check that shape returns expected value."""
       coord = Coordinates()
       self.assertEqual(coord.shape, (0,2))
    
class TestIrregularGrid(unittest.TestCase):
    ''' Test Coordinates class'''

    def test_shape_pass(self):
       """Check that shape returns expected value."""
       coord = IrregularGrid(np.array([[1, 2], [4, 5], [4, 5]]))
       self.assertEqual(coord.shape, (3,2))

    def test_resample_pass(self):
       """Check that resample works correctly."""
       coord_1 = IrregularGrid(np.array([[1, 2], [4, 5], [4, 5]]))
       coord_2 = coord_1
       result = coord_1.resample(coord_2)
       self.assertTrue(np.array_equal(result, np.array([ 0.,  1.,  1.])))

class TestRegularGrid(unittest.TestCase):
    ''' Test Coordinates class'''

    def test_resample_fail(self):
       """Check that resample is not implemented."""
       coord_1 = RegularGrid(1, 2, 3, 4)
       coord_2 = RegularGrid(1, 2, 3, 4)
       with self.assertRaises(NotImplementedError):
           coord_1.resample(coord_2)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCoordinates)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIrregularGrid))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRegularGrid))
unittest.TextTestRunner(verbosity=2).run(TESTS)
