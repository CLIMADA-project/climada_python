"""
Test HazardExcel class.
"""

import unittest
import numpy as np

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_DEMO_XLS

class TestReader(unittest.TestCase):
    '''Test reader functionality of the ExposuresExcel class'''

    def test_hazard_pass(self):
        ''' Read an hazard excel file correctly.'''

        # Read demo excel file
        hazard = Hazard()
        description = 'One single file.'
        hazard.read(HAZ_DEMO_XLS, 'TC', description)

        # Check results
        n_events = 100
        n_centroids = 45

        self.assertEqual(hazard.id, 0)
        self.assertEqual(hazard.units, 'NA')

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(hazard.centroids.coord[0][0], -25.95)
        self.assertEqual(hazard.centroids.coord[0][1], 32.57)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(hazard.centroids.id.dtype, np.int64)
        self.assertEqual(hazard.centroids.id[0], 4001)
        self.assertEqual(hazard.centroids.id[n_centroids-1], 4045)

        self.assertEqual(len(hazard.event_name), 100)
        self.assertEqual(hazard.event_name[12], 'event013')

        self.assertEqual(hazard.event_id.dtype, np.int64)
        self.assertEqual(hazard.event_id.shape, (n_events,))
        self.assertEqual(hazard.event_id[0], 1)
        self.assertEqual(hazard.event_id[n_events-1], 100)

        self.assertEqual(hazard.frequency.dtype, np.float)
        self.assertEqual(hazard.frequency.shape, (n_events,))
        self.assertEqual(hazard.frequency[0], 0.01)
        self.assertEqual(hazard.frequency[n_events-2], 0.001)

        self.assertEqual(hazard.intensity.dtype, np.float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))

        self.assertEqual(hazard.fraction.dtype, np.float)
        self.assertEqual(hazard.fraction.shape, (n_events, n_centroids))
        self.assertEqual(hazard.fraction[0, 0], 1)
        self.assertEqual(hazard.fraction[10, 19], 1)
        self.assertEqual(hazard.fraction[n_events-1, n_centroids-1], 1)

        # tag hazard
        self.assertEqual(hazard.tag.file_name, HAZ_DEMO_XLS)
        self.assertEqual(hazard.tag.description, description)
        self.assertEqual(hazard.tag.type, 'TC')

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_DEMO_XLS)
        self.assertEqual(hazard.centroids.tag.description, description)

    def test_wrong_centroid_fail(self):
        """ Read centroid separately from the hazard. Wrong centroid data in
        size """
        # Read demo excel file
        read_cen = Centroids(HAZ_DEMO_XLS)
        read_cen.id = np.ones(12)
        # Read demo excel file
        hazard = Hazard()

        # Expected exception because centroid size is smaller than the
        # one provided in the intensity matrix
        with self.assertRaises(ValueError):
            hazard.read(HAZ_DEMO_XLS, 'TC', centroids=read_cen)

    def test_centroid_hazard_pass(self):
        """ Read centroid separately from the hazard """

        # Read demo excel file
        description = 'One single file.'
        centroids = Centroids(HAZ_DEMO_XLS, description)
        hazard = Hazard()
        hazard.read(HAZ_DEMO_XLS, 'TC', description, centroids)

        n_centroids = 45
        self.assertEqual(hazard.centroids.coord.shape[0], n_centroids)
        self.assertEqual(hazard.centroids.coord.shape[1], 2)
        self.assertEqual(hazard.centroids.coord[0][0], -25.95)
        self.assertEqual(hazard.centroids.coord[0][1], 32.57)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(hazard.centroids.id.dtype, np.int64)
        self.assertEqual(len(hazard.centroids.id), n_centroids)
        self.assertEqual(hazard.centroids.id[0], 4001)
        self.assertEqual(hazard.centroids.id[n_centroids-1], 4045)

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_DEMO_XLS)
        self.assertEqual(hazard.centroids.tag.description, description)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
        