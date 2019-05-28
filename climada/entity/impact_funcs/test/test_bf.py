"""
This file is part of CLIMADA.
Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.
CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.
CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.
---
Test IFBushfire class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.bushfire import IFBushfire

THRESH = 331

class TestIFBushfire(unittest.TestCase):
    """Impact function test"""

    def test_default_values_pass(self, threshold = THRESH):
        """Compute mdr interpolating values."""
        imp_fun = IFBushfire()
        imp_fun.set_default(THRESH)
        self.assertEqual(imp_fun.name, 'bushfire default')
        self.assertEqual(imp_fun.haz_type, 'BF')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'K')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.array([295, THRESH, THRESH, 367])))
        self.assertTrue(np.array_equal(imp_fun.paa, np.array([1, 1, 1, 1])))
        self.assertTrue(np.array_equal(imp_fun.mdd, np.array([0, 0, 1, 1])))
        
        
        
# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFBushfire)
    unittest.TextTestRunner(verbosity=2).run(TESTS)