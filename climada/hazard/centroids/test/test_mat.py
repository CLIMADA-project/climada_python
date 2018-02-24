"""
Test Centroids from MATLAB class.
"""

import warnings
import unittest
import numpy as np

import climada.hazard.centroids.source_mat as mat
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_TEST_MAT

class TestReader(unittest.TestCase):
    '''Test reader functionality of the CentroidsMat class'''

    def tearDown(self):
        """Set correct encapsulation variable name, as default."""
        mat.FIELD_NAMES = ['centroids', 'hazard']

    def test_centroid_pass(self):
        ''' Read a centroid mat file correctly.'''
        # Read demo mat file
        description = 'One single file.'
        centroids = Centroids()
        centroids.read(HAZ_TEST_MAT, description)

        n_centroids = 100
        self.assertEqual(centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(centroids.coord[0][0], 21)
        self.assertEqual(centroids.coord[0][1], -84)
        self.assertEqual(centroids.coord[n_centroids-1][0], 30)
        self.assertEqual(centroids.coord[n_centroids-1][1], -75)
        self.assertEqual(centroids.id.dtype, np.int64)
        self.assertEqual(centroids.id.shape, (n_centroids, ))
        self.assertEqual(centroids.id[0], 1)
        self.assertEqual(centroids.id[n_centroids-1], 100)

    def test_wrong_encapsulating_warning(self):
        """ Warning is raised when FIELD_NAME is not encapsulating."""
        mat.FIELD_NAMES = ['wrong']
        with warnings.catch_warnings(record=True) as w:
            try:
                Centroids(HAZ_TEST_MAT)
            except KeyError:
                pass
        self.assertIn("Variables are not under: ['wrong'].", \
              str(w[0].message))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
        