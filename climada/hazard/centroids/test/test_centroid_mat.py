"""
Test CentroidsMat class.
"""

import unittest
import numpy

from climada.hazard.centroids.source_mat import CentroidsMat
from climada.util.constants import HAZ_DEMO_MAT

class TestReader(unittest.TestCase):
    '''Test reader functionality of the CentroidsMat class'''

    def test_centroid_pass(self):
        ''' Read a centroid mat file correctly.'''

        # Read demo excel file
        description = 'One single file.'
        centroids = CentroidsMat()
        centroids.field_name = 'hazard'
        centroids.read(HAZ_DEMO_MAT, description)

        n_centroids = 100
        self.assertEqual(centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(centroids.coord[0][0], 21)
        self.assertEqual(centroids.coord[0][1], -84)
        self.assertEqual(centroids.coord[n_centroids-1][0], 30)
        self.assertEqual(centroids.coord[n_centroids-1][1], -75)
        self.assertEqual(centroids.id.dtype, numpy.int64)
        self.assertEqual(centroids.id.shape, (n_centroids, ))
        self.assertEqual(centroids.id[0], 1)
        self.assertEqual(centroids.id[n_centroids-1], 100)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
        