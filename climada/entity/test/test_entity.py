"""
=====================
test_entity module
=====================

Test Entity class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  8 14:52:28 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import unittest

from climada.entity.entity import Entity

class TestReader(unittest.TestCase):
    '''Test reader functionality of the Entity class'''

    def test_default(self):
        '''Test that by instantiating the Entity class the default entity
        file is loaded'''

        # Instance entity
        def_entity = Entity()

        # Check default demo excel file has been loaded
        self.assertEqual(len(def_entity.exposures.deductible), 50)
        self.assertEqual(def_entity.exposures.value[2], 12596064143.542929)

        self.assertEqual(len(def_entity.impact_funcs.data['TC'][1].mdd), 9)

        self.assertEqual(def_entity.measures.data[0].name, 'Mangroves')

        self.assertEqual(def_entity.discounts.years[5], 2005)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
