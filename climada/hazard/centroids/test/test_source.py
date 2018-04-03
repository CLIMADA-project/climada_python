"""
Test Centroids from MATLAB class.
"""

import unittest
import numpy as np

import climada.hazard.centroids.source as source
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_TEST_MAT, HAZ_DEMO_XLS, GLB_CENTROIDS

class TestReaderMat(unittest.TestCase):
    '''Test reader functionality of MATLAB files'''

    def tearDown(self):
        """Set correct encapsulation variable name, as default."""
        source.DEF_VAR_MAT = {'field_names': ['centroids', 'hazard'],
                              'var_name': {'cen_id' : 'centroid_ID',
                                           'lat' : 'lat',
                                           'lon' : 'lon',
                                           'dist_coast': 'distance2coast_km',
                                           'admin0_name': 'admin0_name',
                                           'admin0_iso3': 'admin0_ISO3',
                                           'comment': 'comment',
                                           'region_id': 'NatId'
                                          }
                             }

    def test_centroid_pass(self):
        ''' Read a centroid mat file correctly.'''
        # Read demo mat file
        description = 'One single file.'
        centroids = Centroids()
        centroids.read_one(HAZ_TEST_MAT, description)

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

    def test_wrong_var_name_fail(self):
        """ Error is raised when wrong variable name in source file."""
        new_names = source.DEF_VAR_MAT
        new_names['var_name']['lat'] = ['Latitude']
        cent = Centroids()
        with self.assertRaises(TypeError):
            cent.read_one(HAZ_TEST_MAT, var_names=new_names)

#    def test_wrong_encapsulating_warning(self):
#        """ Warning is raised when FIELD_NAME is not encapsulating."""
#        DEF_VAR_NAME['field_names'] = ['wrong']
#        with self.assertLogs('climada.entity.impact_funcs.base', level='WARNING') as cm:
#            Centroids(HAZ_TEST_MAT)
#        self.assertIn("Variables are not under: ['wrong'].", cm[0].output)
            
    def test_nat_global_pass(self):
        glb_cent = Centroids(GLB_CENTROIDS)
        self.assertEqual(glb_cent.region_id[1062443], 35)
        self.assertEqual(glb_cent.region_id[170825], 28)
        self.assertAlmostEqual(glb_cent.dist_coast[9], 21.366461094662913)
        self.assertAlmostEqual(glb_cent.dist_coast[1568370], 36.76908653021)

class TestReaderExcel(unittest.TestCase):
    '''Test reader functionality of Excel files'''
    def tearDown(self):
        source.DEF_VAR_EXCEL = {'sheet_name': 'centroids',
                         'col_name': {'cen_id' : 'centroid_ID',
                                      'lat' : 'Latitude',
                                      'lon' : 'Longitude'
                                     }
                        }

    def test_wrong_var_name_fail(self):
        """ Error is raised when wrong variable name in source file."""
        new_names = source.DEF_VAR_EXCEL
        new_names['col_name']['lat'] = ['lat']
        cent = Centroids()
        with self.assertRaises(TypeError):
            cent.read_one(HAZ_DEMO_XLS, var_names=new_names)

    def test_centroid_pass(self):
        ''' Read a centroid excel file correctly.'''
        description = 'One single file.'
        centroids = Centroids(HAZ_DEMO_XLS, description)

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

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderMat)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
unittest.TextTestRunner(verbosity=2).run(TESTS)
        