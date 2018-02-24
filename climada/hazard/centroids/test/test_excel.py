"""
Test CentroidsExcel class.
"""

import unittest
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_DEMO_XLS

class TestReader(unittest.TestCase):
    '''Test reader functionality of the CentroidsExcel class'''

    def test_centroid_pass(self):
        ''' Read a centroid excel file correctly.'''

        # Read demo excel file
        description = 'One single file.'
        centroids = Centroids(HAZ_DEMO_XLS, description)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(centroids.id.dtype, np.int64)
        self.assertEqual(len(centroids.id), n_centroids)
        self.assertEqual(centroids.id[0], 4001)
        self.assertEqual(centroids.id[n_centroids-1], 4045)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
        