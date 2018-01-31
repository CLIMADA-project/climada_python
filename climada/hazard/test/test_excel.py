"""
Test HazardExcel class.
"""

import unittest
import pickle
import numpy

from climada.hazard.source_excel import HazardExcel
from climada.hazard.centroids.source_excel import CentroidsExcel
from climada.util.constants import HAZ_DEMO_XLS

class TestReader(unittest.TestCase):
    '''Test reader functionality of the ExposuresExcel class'''

    def test_hazard_pass(self):
        ''' Read an hazard excel file correctly.'''

        # Read demo excel file
        hazard = HazardExcel()
        description = 'One single file.'
        hazard.read(HAZ_DEMO_XLS, description)

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
        self.assertEqual(hazard.centroids.id.dtype, numpy.int64)
        self.assertEqual(hazard.centroids.id[0], 4001)
        self.assertEqual(hazard.centroids.id[n_centroids-1], 4045)

        self.assertEqual(len(hazard.event_name), 100)
        self.assertEqual(hazard.event_name[12], 'event013')

        self.assertEqual(hazard.event_id.dtype, numpy.int64)
        self.assertEqual(hazard.event_id.shape, (n_events,))
        self.assertEqual(hazard.event_id[0], 1)
        self.assertEqual(hazard.event_id[n_events-1], 100)

        self.assertEqual(hazard.frequency.dtype, numpy.float)
        self.assertEqual(hazard.frequency.shape, (n_events,))
        self.assertEqual(hazard.frequency[0], 0.01)
        self.assertEqual(hazard.frequency[n_events-2], 0.001)

        self.assertEqual(hazard.intensity.dtype, numpy.float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))

        self.assertEqual(hazard.fraction.dtype, numpy.float)
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

    def test_hazard_save_pass(self):
        """ Save the hazard object after being read correctly"""

        # Read demo excel file
        hazard = HazardExcel()
        description = 'One single file.'
        out_file = 'hazard_excel.pkl'
        hazard.load(HAZ_DEMO_XLS, description, out_file_name=out_file)

        # Getting back the objects:
        with open(out_file, 'rb') as file:
            hazard_read = pickle.load(file)

        # Check the loaded hazard have all variables
        self.assertEqual(hasattr(hazard_read, 'tag'), True)
        self.assertEqual(hasattr(hazard_read, 'id'), True)
        self.assertEqual(hasattr(hazard_read, 'units'), True)
        self.assertEqual(hasattr(hazard_read, 'centroids'), True)
        self.assertEqual(hasattr(hazard_read, 'event_id'), True)
        self.assertEqual(hasattr(hazard_read, 'frequency'), True)
        self.assertEqual(hasattr(hazard_read, 'intensity'), True)
        self.assertEqual(hasattr(hazard_read, 'fraction'), True)

    def test_centroid_hazard_pass(self):
        """ Read centroid separately from the hazard """

        # Read demo excel file
        description = 'One single file.'
        centroids = CentroidsExcel(HAZ_DEMO_XLS, description)
        hazard = HazardExcel()
        hazard.read(HAZ_DEMO_XLS, description, centroids)

        n_centroids = 45
        self.assertEqual(hazard.centroids.coord.shape[0], n_centroids)
        self.assertEqual(hazard.centroids.coord.shape[1], 2)
        self.assertEqual(hazard.centroids.coord[0][0], -25.95)
        self.assertEqual(hazard.centroids.coord[0][1], 32.57)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][0], -24.7)
        self.assertEqual(hazard.centroids.coord[n_centroids-1][1], 33.88)
        self.assertEqual(hazard.centroids.id.dtype, numpy.int64)
        self.assertEqual(len(hazard.centroids.id), n_centroids)
        self.assertEqual(hazard.centroids.id[0], 4001)
        self.assertEqual(hazard.centroids.id[n_centroids-1], 4045)

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_DEMO_XLS)
        self.assertEqual(hazard.centroids.tag.description, description)

if __name__ == '__main__':
    unittest.main()
        