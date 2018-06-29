"""
Test ImpactFunc class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

class TestInterpolation(unittest.TestCase):
    """Impact function interpolation test"""

    def test_calc_mdr_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpactFunc()
        imp_fun.intensity = np.arange(0, 100, 10)
        imp_fun.paa = np.arange(0, 1, 0.1)
        imp_fun.mdd = np.arange(0, 1, 0.1)
        new_inten = 17.2
        self.assertEqual(imp_fun.calc_mdr(new_inten), 0.029583999999999996)
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInterpolation)
unittest.TextTestRunner(verbosity=2).run(TESTS)
