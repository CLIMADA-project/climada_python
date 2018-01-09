"""
Test CentroidsExcel class.
"""

import unittest
import pickle
import numpy

from climada.hazard.centroids.source_excel import CentroidsExcel
from climada.util.constants import HAZ_DEMO_XLS

class TestReader(unittest.TestCase):
    '''Test reader functionality of the CentroidsExcel class'''

    def test_centroid_pass(self):
        ''' Read a centroid excel file correctly.'''

        # Read demo excel file
        description = 'One single file.'
        centroids = CentroidsExcel(HAZ_DEMO_XLS, description)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(centroids.id.dtype, numpy.int64)
        self.assertEqual(len(centroids.id), n_centroids)
        self.assertEqual(centroids.id[0], 4001)
        self.assertEqual(centroids.id[n_centroids-1], 4045)

    def test_centroid_save_pass(self):
        """ Save the centroids object after being read correctly"""

        # Read demo excel file
        centroids = CentroidsExcel()
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

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
        