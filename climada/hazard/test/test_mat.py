"""
Test Hazard from MATLAB file.
"""

import unittest
import numpy as np

import climada.hazard.source_mat as mat
from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_TEST_MAT

class TestReader(unittest.TestCase):
    '''Test reader functionality of the ExposuresExcel class'''

    def tearDown(self):
        """Set correct encapsulation variable name, as default."""
        mat.FIELD_NAME = 'hazard'

    def test_hazard_pass(self):
        ''' Read a hazard mat file correctly.'''
        # Read demo excel file
        hazard = Hazard()
        hazard.read(HAZ_TEST_MAT, 'TC')

        # Check results
        n_events = 14450
        n_centroids = 100

        self.assertEqual(hazard.units, 'm/s')

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))

        self.assertEqual(hazard.event_id.dtype, np.int64)
        self.assertEqual(hazard.event_id.shape, (n_events,))

        self.assertEqual(hazard.frequency.dtype, np.float)
        self.assertEqual(hazard.frequency.shape, (n_events,))

        self.assertEqual(hazard.intensity.dtype, np.float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))
        self.assertEqual(hazard.intensity[12, 46], 12.071393519949979)
        self.assertEqual(hazard.intensity[13676, 49], 17.228323602220616)

        self.assertEqual(hazard.fraction.dtype, np.float)
        self.assertEqual(hazard.fraction.shape, (n_events, n_centroids))
        self.assertEqual(hazard.fraction[8454, 98], 1)
        self.assertEqual(hazard.fraction[85, 54], 0)

        self.assertEqual(len(hazard.event_name), n_events)
        self.assertEqual(hazard.event_name[124], 125)

        # tag hazard
        self.assertEqual(hazard.tag.file_name, HAZ_TEST_MAT)
        self.assertEqual(hazard.tag.description, '')
        self.assertEqual(hazard.tag.haz_type, 'TC')

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_TEST_MAT)
        self.assertEqual(hazard.centroids.tag.description, '')

    def test_wrong_centroid_error(self):
        """ Read centroid separately from the hazard. Wrong centroid data in
        size """
        # Read demo excel file
        read_cen = Centroids(HAZ_TEST_MAT)
        read_cen.id = np.ones(12)
        # Read demo excel file
        hazard = Hazard()

        # Expected exception because centroid size is smaller than the
        # one provided in the intensity matrix
        with self.assertRaises(ValueError):
            hazard.read(HAZ_TEST_MAT, 'TC', centroids=read_cen)

    def test_wrong_encapsulating_warning(self):
        """ Warning is raised when FIELD_NAME is not encapsulating."""
        mat.FIELD_NAME = 'wrong'
        #with warnings.catch_warnings(record=True) as w:
        try:
            Hazard(HAZ_TEST_MAT, 'TC')
        except KeyError:
            pass     

    def test_wrong_hazard_type_error(self):
        """ Error if provided hazard type different as contained"""
        hazard = Hazard()
        with self.assertRaises(ValueError):
            hazard.read(HAZ_TEST_MAT, 'WS')

    def test_centroid_hazard_pass(self):
        """ Read centroid separately from the hazard """
        # Read demo excel file
        description = 'One single file.'
        centroids = Centroids(HAZ_TEST_MAT, description)
        hazard = Hazard()
        hazard.read(HAZ_TEST_MAT, 'TC', description, centroids)

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

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_TEST_MAT)
        self.assertEqual(hazard.centroids.tag.description, description)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
