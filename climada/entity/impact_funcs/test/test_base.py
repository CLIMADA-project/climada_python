"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test ImpactFunc class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc

class TestInterpolation(unittest.TestCase):
    """Impact function interpolation test"""

    def test_calc_mdr_pass(self):
        """Compute mdr interpolating values."""
        intensity = np.arange(0, 100, 10)
        paa = np.arange(0, 1, 0.1)
        mdd = np.arange(0, 1, 0.1)
        imp_fun = ImpactFunc(intensity=intensity, paa=paa, mdd=mdd)
        new_inten = 17.2
        self.assertEqual(imp_fun.calc_mdr(new_inten), 0.029583999999999996)

    def test_set_step(self):
        """Check default impact function: step function"""
        inten = (0, 5, 10)
        imp_fun = ImpactFunc.from_step_impf(inten)
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones(4)))
        self.assertTrue(np.array_equal(imp_fun.mdd, np.array([0, 0, 1, 1])))
        self.assertTrue(np.array_equal(imp_fun.intensity, np.array([0, 5, 5, 10])))

    def test_set_sigmoid(self):
        """Check default impact function: sigmoid function"""
        inten = (0, 100, 5)
        imp_fun = ImpactFunc.from_sigmoid_impf(inten, L=1.0, k=2., x0=50.)
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones(20)))
        self.assertEqual(imp_fun.mdd[10], 0.5)
        self.assertEqual(imp_fun.mdd[-1], 1.0)
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 100, 5)))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInterpolation)
    unittest.TextTestRunner(verbosity=2).run(TESTS)



