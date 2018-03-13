"""
Test Interpolation class.
"""

import unittest
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.hazard.centroids.base import Centroids
import climada.util.interpolation as interp
from climada.util.constants import ENT_DEMO_XLS

def def_input_values():
    '''Default input coordinates and centroids values'''
    # Load exposures coordinates from demo entity file
    exposures = Exposures(ENT_DEMO_XLS)

    # Define centroids
    centroids = Centroids()
    centroids.coord = np.zeros((100, 2))
    inext = 0
    for ilon in range(10):
        for ilat in range(10):
            centroids.coord[inext][0] = 20 + ilat + 1
            centroids.coord[inext][1] = -85 + ilon + 1

            inext = inext + 1

    return exposures, centroids

def def_ref():
    '''Default output reference'''
    return np.array([46, 46, 36, 36, 36, 46, 46, 46, 46, 46, 46,\
                     36, 46, 46, 36, 46, 46, 46, 46, 46, 46, 46,\
                     46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,\
                     46, 46, 46, 45, 45, 45, 45, 45, 45, 45, 45,\
                     45, 45, 45, 45, 45, 45])

def def_ref_50():
    '''Default output reference for maximum distance threshold 50km'''
    return np.array([46, 46, 36, -1, 36, 46, 46, 46, 46, 46, 46, 36, 46, 46, \
                     36, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, \
                     46, 46, 46, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 45, \
                     45, 45, 45, 45, 45, 45, 45, 45])

class TestInterpIndex(unittest.TestCase):
    ''' Test interpol_index function's interface'''

    def test_wrong_method_fail(self):
        ''' Check exception is thrown when wrong method is given'''
        with self.assertLogs('climada.util.interpolation', level='ERROR') as cm:
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 2)), 'method')
        self.assertIn('Interpolation using method' + \
            ' with distance approx is not supported.', cm.output[0])

    def test_wrong_distance_fail(self):
        ''' Check exception is thrown when wrong distance is given'''
        with self.assertLogs('climada.util.interpolation', level='ERROR') as cm:
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 2)), \
                                  distance='distance')
        self.assertIn('Interpolation using NN' + \
            ' with distance distance is not supported.', cm.output[0])

    def test_wrong_centroid_fail(self):
        ''' Check exception is thrown when centroids missing one dimension'''
        with self.assertRaises(IndexError):
            interp.interpol_index(np.ones((10, 1)), np.ones((7, 2)))

    def test_wrong_coord_fail(self):
        ''' Check exception is thrown when coordinates missing one dimension'''
        with self.assertRaises(IndexError):
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 1)))

class TestNN(unittest.TestCase):
    '''Test interpolator neareast neighbor with approximate distance'''

    def tearDown(self):
        interp.THRESHOLD = 100

    def normal_pass(self, dist):
        '''Checking result against matlab climada_demo_step_by_step'''

        # Load input
        exposures, centroids = def_input_values()

        # Interpolate with default threshold
        neighbors = interp.interpol_index(centroids.coord, exposures.coord,
                                          'NN', dist)

        # Reference output
        ref_neighbors = def_ref()
        # Check results
        self.assertEqual(exposures.coord.shape[0], len(neighbors))
        self.assertTrue(np.array_equal(neighbors, ref_neighbors))

    def normal_warning(self, dist):
        '''Checking that a warning is raised when minimum distance greater
        than threshold'''

        # Load input
        exposures, centroids = def_input_values()

        # Interpolate with lower threshold to raise warnings
        interp.THRESHOLD = 50
        neighbors = interp.interpol_index(centroids.coord,
                                  exposures.coord, 'NN', dist)
        with self.assertLogs('climada.util.interpolation', level='INFO') as cm:
            neighbors = interp.interpol_index(centroids.coord,
                                              exposures.coord, 'NN', dist)
        self.assertIn("Distance to closest centroid", cm.output[0])

        ref_neighbors = def_ref_50()
        self.assertTrue(np.array_equal(neighbors, ref_neighbors))

    def repeat_coord_pass(self, dist):
        '''Check that exposures with the same coordinates have same
        neighbors'''

        # Load input
        exposures, centroids = def_input_values()

        # Repeat a coordinate
        exposures.coord[2, :] = exposures.coord[0, :]

        # Interpolate with default threshold
        neighbors = interp.interpol_index(centroids.coord, exposures.coord,
                                          'NN', dist)

        # Check output neighbors have same size as coordinates
        self.assertEqual(len(neighbors), exposures.coord.shape[0])
        # Check copied coordinates have same neighbors
        self.assertEqual(neighbors[2], neighbors[0])

    def test_approx_normal_pass(self):
        ''' Call normal_pass test for approxiamte distance'''
        self.normal_pass('approx')

    def test_approx_normal_warning(self):
        ''' Call normal_warning test for approxiamte distance'''
        self.normal_warning('approx')

    def test_approx_repeat_coord_pass(self):
        ''' Call repeat_coord_pass test for approxiamte distance'''
        self.repeat_coord_pass('approx')

    def test_haver_normal_pass(self):
        ''' Call normal_pass test for haversine distance'''
        self.normal_pass('haversine')

    def test_haver_normal_warning(self):
        ''' Call normal_warning test for haversine distance'''
        self.normal_warning('haversine')

    def test_haver_repeat_coord_pass(self):
        ''' Call repeat_coord_pass test for haversine distance'''
        self.repeat_coord_pass('haversine')

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNN)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInterpIndex))
unittest.TextTestRunner(verbosity=2).run(TESTS)
