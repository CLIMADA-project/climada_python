"""
Test coordinates module.
"""

import unittest
import numpy as np

from climada.util.coordinates import Coordinates
    
class TestCoordinates(unittest.TestCase):
    ''' Test Coordinates class'''

    def test_shape_pass(self):
        """Check that shape returns expected value."""
        coord = Coordinates(np.array([[1, 2], [4.5, 5.5], [4, 5]]))
        self.assertEqual(coord.shape, (3,2))

        Coordinates(np.array([[1, 2], [4, 5], [4, 5]]))
        self.assertEqual(coord.shape, (3,2))

        coord = Coordinates()
        self.assertEqual(coord.shape, (0,2))

    def test_wrong_value_fail(self):
        """Check good values in constructor."""
        with self.assertRaises(ValueError):
            Coordinates(np.array([[1, 2], [4.3, 5], [4, 5]]).transpose())

    def test_resample_pass(self):
        """Check that resample works correctly."""
        coord_1 = Coordinates(np.array([[1, 2], [4.1, 5.1], [4, 5]]))
        coord_2 = coord_1
        result = coord_1.resample(coord_2)
        self.assertTrue(np.array_equal(result, np.array([ 0.,  1.,  2.])))

    def test_is_regular_pass(self):
        """ Test is_regular function. """
        coord = Coordinates(np.array([[1, 2], [4.4, 5.4], [4, 5]]))
        self.assertFalse(coord.is_regular())

        coord = Coordinates(np.array([[1, 2], [4.4, 5], [4, 5]]))
        self.assertFalse(coord.is_regular())

        coord = Coordinates(np.array([[1, 2], [4, 5]]))
        self.assertFalse(coord.is_regular())
        
        coord = Coordinates(np.array([[1, 2], [4, 5], [1, 5], [4, 3]]))
        self.assertFalse(coord.is_regular())
        
        coord = Coordinates(np.array([[1, 2], [4, 5], [1, 5], [4, 2]]))
        self.assertTrue(coord.is_regular())
        
        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 5),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = Coordinates(np.array([grid_x, grid_y]).transpose())
        self.assertTrue(coord.is_regular())

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 4),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = Coordinates(np.array([grid_x, grid_y]).transpose())
        self.assertTrue(coord.is_regular())

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCoordinates)
unittest.TextTestRunner(verbosity=2).run(TESTS)
