"""
Test Centroids from MATLAB class.
"""

import unittest

from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_TEST_MAT, HAZ_TEST_XLS, GLB_CENTROIDS_MAT

class TestReaderMat(unittest.TestCase):
    '''Test reader functionality of MATLAB files'''

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
        self.assertEqual(centroids.id.dtype, int)
        self.assertEqual(centroids.id.shape, (n_centroids, ))
        self.assertEqual(centroids.id[0], 1)
        self.assertEqual(centroids.id[n_centroids-1], 100)
            
    def test_nat_global_pass(self):
        glb_cent = Centroids(GLB_CENTROIDS_MAT)
        self.assertEqual(glb_cent.region_id[1062443], 35)
        self.assertEqual(glb_cent.region_id[170825], 28)
        self.assertAlmostEqual(glb_cent.dist_coast[9], 21.366461094662913)
        self.assertAlmostEqual(glb_cent.dist_coast[1568370], 36.76908653021)
        self.assertEqual(glb_cent.admin0_name, '')
        self.assertEqual(glb_cent.admin0_iso3, '')

class TestReaderExcel(unittest.TestCase):
    '''Test reader functionality of Excel files'''

    def test_centroid_pass(self):
        ''' Read a centroid excel file correctly.'''
        description = 'One single file.'
        centroids = Centroids(HAZ_TEST_XLS, description)

        n_centroids = 45
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(centroids.id.dtype, int)
        self.assertEqual(len(centroids.id), n_centroids)
        self.assertEqual(centroids.id[0], 4001)
        self.assertEqual(centroids.id[n_centroids-1], 4045)

class TestReaderMix(unittest.TestCase):
    ''' Test reader several files. '''
    
    def test_xls_mat_pass(self):
        """Read xls and matlab files."""
        files = [HAZ_TEST_MAT, HAZ_TEST_XLS]
        centroids = Centroids(files)
        
        self.assertEqual(centroids.id.size, 145)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderMat)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMix))
unittest.TextTestRunner(verbosity=2).run(TESTS)
        