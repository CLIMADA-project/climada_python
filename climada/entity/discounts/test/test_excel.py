"""
Test DiscountsExcel class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Thu Dec  7 13:32:05 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import unittest
import numpy

from climada.entity.discounts.source_excel import DiscountsExcel
from climada.util.constants import ENT_DEMO_XLS

class TestReader(unittest.TestCase):
    """Test reader functionality of the DiscountsExcel class"""

    def test_one_file(self):
        """ Read one single excel file"""

        # Read demo excel file
        disc_rate = DiscountsExcel()
        description = 'One single file.'
        disc_rate._read(ENT_DEMO_XLS, description)

        # Check results
        n_rates = 51

        self.assertEqual(type(disc_rate.years[0]), numpy.int64)
        self.assertEqual(len(disc_rate.years), n_rates)
        self.assertEqual(disc_rate.years[0], 2000)
        self.assertEqual(disc_rate.years[n_rates-1], 2050)

        self.assertEqual(type(disc_rate.rates[0]), numpy.float64)
        self.assertEqual(len(disc_rate.rates), n_rates)
        self.assertEqual(disc_rate.rates.min(), 0.02)
        self.assertEqual(disc_rate.rates.max(), 0.02)

        self.assertEqual(disc_rate.tag.file_name, ENT_DEMO_XLS)
        self.assertEqual(disc_rate.tag.description, description)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
