"""
=====================
test_mat module
=====================

Test HazardExcel class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 15:53:21 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import unittest
import numpy

from climada.hazard.source_mat import HazardMat
from climada.hazard.centroids import Centroids
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

        self.assertEqual(hazard.centroids.coord.shape[0], n_centroids)
        self.assertEqual(hazard.centroids.coord.shape[1], 2)

        self.assertEqual(hazard.event_id.dtype, numpy.int64)
        self.assertEqual(len(hazard.event_id), n_events)
        self.assertEqual(hazard.event_id.shape, (n_events,))

        self.assertEqual(hazard.frequency.dtype, numpy.float)
        self.assertEqual(len(hazard.frequency), n_events)
        self.assertEqual(hazard.frequency.shape, (n_events,))

        self.assertEqual(hazard.intensity.dtype, numpy.float)
        self.assertEqual(hazard.intensity.shape[0], n_events)
        self.assertEqual(hazard.intensity.shape[1], n_centroids)

        self.assertEqual(hazard.fraction.dtype, numpy.float)
        self.assertEqual(hazard.fraction.shape[0], n_events)
        self.assertEqual(hazard.fraction.shape[1], n_centroids)

        # tag hazard
        self.assertEqual(hazard.tag.file_name, HAZ_DEMO_MAT)
        self.assertEqual(hazard.tag.description, None)
        self.assertEqual(hazard.tag.type, 'TC')

        # tag centroids
        self.assertEqual(hazard.centroids.tag.file_name, HAZ_DEMO_MAT)
        self.assertEqual(hazard.centroids.tag.description, None)

    def test_wrong_centroid_fail(self):
        """ Read centroid separately from the hazard. Wrong centroid data in
        size """

        # Read demo excel file
        read_cen = Centroids()
        read_cen.read_excel(HAZ_DEMO_XLS)
        # Read demo excel file
        hazard = HazardMat()

        # Expected exception because centroid size is smaller than the
        # one provided in the intensity matrix
        with self.assertRaises(ValueError):
            hazard.read(HAZ_DEMO_MAT, None, centroids=read_cen)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
