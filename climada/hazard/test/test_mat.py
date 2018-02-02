"""
Test HazardMat class.
"""

import unittest
import numpy

from climada.hazard.source_mat import HazardMat
from climada.hazard.centroids.source_excel import CentroidsExcel
from climada.util.constants import HAZ_DEMO_MAT, HAZ_DEMO_XLS

class TestReader(unittest.TestCase):
    '''Test reader functionality of the ExposuresExcel class'''

    def test_hazard_pass(self):
        ''' Read an hazard excel file correctly.'''
        # Read demo excel file
        hazard = HazardMat(HAZ_DEMO_MAT)

        # Check results
        n_events = 14450
        n_centroids = 100

        self.assertEqual(hazard.id, 0)
        self.assertEqual(hazard.units, 'm/s')

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))

        self.assertEqual(hazard.event_id.dtype, numpy.int64)
        self.assertEqual(hazard.event_id.shape, (n_events,))

        self.assertEqual(hazard.frequency.dtype, numpy.float)
        self.assertEqual(hazard.frequency.shape, (n_events,))

        self.assertEqual(hazard.intensity.dtype, numpy.float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))

        self.assertEqual(hazard.fraction.dtype, numpy.float)
        self.assertEqual(hazard.fraction.shape, (n_events, n_centroids))

        self.assertEqual(len(hazard.event_name), n_events)
        self.assertEqual(hazard.event_name[124], 'NNN_1185308_gen4')

        # tag hazard
        self.assertEqual(hazard.tag.file_name, HAZ_DEMO_MAT)
        self.assertEqual(hazard.tag.description, '')
        self.assertEqual(hazard.tag.type, 'TC')

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_DEMO_MAT)
        self.assertEqual(hazard.centroids.tag.description, '')

    def test_wrong_centroid_fail(self):
        """ Read centroid separately from the hazard. Wrong centroid data in
        size """
        # Read demo excel file
        read_cen = CentroidsExcel(HAZ_DEMO_XLS)
        # Read demo excel file
        hazard = HazardMat()

        # Expected exception because centroid size is smaller than the
        # one provided in the intensity matrix
        with self.assertRaises(ValueError):
            hazard.read(HAZ_DEMO_MAT, None, centroids=read_cen)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
