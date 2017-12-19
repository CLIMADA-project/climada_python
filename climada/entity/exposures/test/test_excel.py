"""
=====================
test_excel module
=====================

Test ExposuresExcel class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 15:53:21 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import warnings
import unittest
import numpy as np

from climada.entity.exposures.source_excel import ExposuresExcel
from climada.util.config import entity_default

class TestReader(unittest.TestCase):
    '''Test reader functionality of the ExposuresExcel class'''

    def test_one_file(self):
        ''' Read one single excel file'''
        # Read demo excel file
        expo = ExposuresExcel()
        description = 'One single file.'
        expo.read(entity_default, description)

        # Check results
        n_expos = 50

        self.assertEqual(type(expo.id[0]), np.int64)
        self.assertEqual(len(expo.id), n_expos)
        self.assertEqual(expo.id[0], 0)
        self.assertEqual(expo.id[n_expos-1], n_expos-1)

        self.assertEqual(len(expo.value), n_expos)
        self.assertEqual(expo.value[0], 13927504367.680632)
        self.assertEqual(expo.value[n_expos-1], 12624818493.687229)

        self.assertEqual(len(expo.deductible), n_expos)
        self.assertEqual(expo.deductible[0], 0)
        self.assertEqual(expo.deductible[n_expos-1], 0)

        self.assertEqual(len(expo.cover), n_expos)
        self.assertEqual(expo.cover[0], 13927504367.680632)
        self.assertEqual(expo.cover[n_expos-1], 12624818493.687229)

        self.assertEqual(type(expo.impact_id[0]), np.int64)
        self.assertEqual(len(expo.impact_id), n_expos)
        self.assertEqual(expo.impact_id[0], 1)
        self.assertEqual(expo.impact_id[n_expos-1], 1)

        self.assertEqual(expo.coord.shape[0], n_expos)
        self.assertEqual(expo.coord.shape[1], 2)
        self.assertEqual(expo.coord[0][0], 26.93389900000)
        self.assertEqual(expo.coord[n_expos-1][0], 26.34795700000)
        self.assertEqual(expo.coord[0][1], -80.12879900000)
        self.assertEqual(expo.coord[n_expos-1][1], -80.15885500000)

        self.assertEqual(expo.ref_year, 2016)
        self.assertEqual(expo.value_unit, 'NA')
        self.assertEqual(expo.tag.file_name, entity_default)
        self.assertEqual(expo.tag.description, description)

    def test_not_cover_pass(self):
        ''' Read excel file with no covered value. Check that the coverage is
        set to its default value (the exposures value) and that a warning is
        raised.'''
        # Read demo excel file
        expo = ExposuresExcel()
        description = 'Default cover.'
        # Change cover column name to simulate no present column
        expo.col_names['cov'] = 'Dummy'

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            expo.read(entity_default, description)

            # Verify some things
            assert "not found. Cover set to exposures values." \
            in str(w[-1].message)

        # Check results
        self.assertEqual(True, np.array_equal(expo.value, expo.cover))
        
    def test_not_deductible_pass(self):
        ''' Read excel file with no deductible value. Check that the
        deductible is set to its default value (0) and that a warning is
        raised.'''
        # Read demo excel file
        expo = ExposuresExcel()
        description = 'Default deductible.'
        # Change deductible column name to simulate no present column
        expo.col_names['ded'] = 'Dummy'

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            expo.read(entity_default, description)

            # Verify some things
            assert "not found. Default zero values set for deductible."\
            in str(w[-1].message)

        # Check results
        self.assertEqual(True, np.array_equal(np.zeros(len(expo.value)), 
                                              expo.deductible))

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
