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
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInterpolation)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
