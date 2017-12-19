"""
=====================
test_excel module
=====================

Test MeasuresExcel class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  8 09:17:11 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import unittest
import numpy as np

from climada.entity.measures.source_excel import MeasuresExcel
from climada.util.config import entity_default
 
class TestReader(unittest.TestCase):

    def test_one_file(self):
        # Read demo excel file
        meas = MeasuresExcel()
        description = 'One single file.'
        meas.read(entity_default, description)

        # Check results        
        n_meas = 4
        
        self.assertEqual(len(meas.data), n_meas)
        self.assertEqual(list(meas.data.keys()), [0, 1, 2, 3])
        
        first = 0
        self.assertEqual(meas.data[first].name, 'Mangroves')
        self.assertEqual(type(meas.data[first].color_rgb), np.ndarray)
        self.assertEqual(len(meas.data[first].color_rgb), 3)
        self.assertEqual(meas.data[first].color_rgb[0], 0.1529)
        self.assertEqual(meas.data[first].color_rgb[1], 0.251)
        self.assertEqual(meas.data[first].color_rgb[2], 0.5451)
        self.assertEqual(meas.data[first].cost, 1311768360.8515418)
        self.assertEqual(meas.data[first].hazard_freq_cutoff, 0)
        self.assertEqual(meas.data[first].hazard_event_set, 'nil')
        self.assertEqual(meas.data[first].hazard_intensity, (1, -4))
        self.assertEqual(meas.data[first].mdd_impact, (1, 0))
        self.assertEqual(meas.data[first].paa_impact, (1, 0))
        self.assertEqual(meas.data[first].risk_transf_attach, 0)
        self.assertEqual(meas.data[first].risk_transf_cover, 0)
        
        self.assertEqual(meas.data[n_meas-1].name, 'Building code')
        self.assertEqual(type(meas.data[n_meas-1].color_rgb), np.ndarray)
        self.assertEqual(len(meas.data[n_meas-1].color_rgb), 3)
        self.assertEqual(meas.data[n_meas-1].color_rgb[0], 0.6980)
        self.assertEqual(meas.data[n_meas-1].color_rgb[1], 0.8745)
        self.assertEqual(meas.data[n_meas-1].color_rgb[2], 0.9333)
        self.assertEqual(meas.data[n_meas-1].cost, 9200000000.0000000)
        self.assertEqual(meas.data[n_meas-1].hazard_freq_cutoff, 0)
        self.assertEqual(meas.data[n_meas-1].hazard_event_set, 'nil')
        self.assertEqual(meas.data[n_meas-1].hazard_intensity, (1, 0))
        self.assertEqual(meas.data[n_meas-1].mdd_impact, (0.75, 0))
        self.assertEqual(meas.data[n_meas-1].paa_impact, (1, 0))
        self.assertEqual(meas.data[n_meas-1].risk_transf_attach, 0)
        self.assertEqual(meas.data[n_meas-1].risk_transf_cover, 0)

        self.assertEqual(meas.tag.file_name, entity_default)
        self.assertEqual(meas.tag.description, description)
        
# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)