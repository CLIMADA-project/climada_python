"""
=====================
test_base module
=====================

Test Exposure base class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Wed Dec  6 14:16:15 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import unittest
from unittest.mock import patch
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.hazard.base import Hazard

class TestAssign(unittest.TestCase):
    '''Check assign interface'''

    # ignore abstract methods
    p_exp = patch.multiple(Exposures, __abstractmethods__=set())
    p_haz = patch.multiple(Hazard, __abstractmethods__=set())

    def setUp(self):
        self.p_exp.start()
        self.p_haz.start()

    def tearDown(self):
        self.p_exp.stop()
        self.p_haz.stop()

    def test_assign_pass(self):
        ''' Check that assigned attribute is correctly set.'''

        # Fill with dummy values the coordinates
        expo = Exposures()
        num_coord = 4
        expo.coord = np.ones((num_coord, 2))
        # Fill with dummy values the centroids
        haz = Hazard()
        haz.centroids.coord = np.ones((num_coord+6, 2))
        # assign
        expo.assign(haz)

        # check assigned variable has been set with correct length
        self.assertEqual(num_coord, len(expo.assigned))

# Execute TestAssign
suite = unittest.TestLoader().loadTestsFromTestCase(TestAssign)
unittest.TextTestRunner(verbosity=2).run(suite)
