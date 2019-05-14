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

Test IF_TropCycl class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.trop_cyclone import IFTropCyclone

class TestEmanuelFormula(unittest.TestCase):
    """Impact function interpolation test"""

    def test_default_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = IFTropCyclone()
        imp_fun.set_emanuel_usa()
        self.assertEqual(imp_fun.name, 'Emanuel 2011')
        self.assertEqual(imp_fun.haz_type, 'TC')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 121, 5)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((25,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:6], np.zeros((6,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[6:10], 
                        np.array([0.0006753419543492556, 0.006790495604105169, 0.02425254393374475, 0.05758706257339458])))
        self.assertTrue(np.array_equal(imp_fun.mdd[10:15], 
                        np.array([0.10870556455111065, 0.1761433569521351, 0.2553983618763961, 0.34033822528795565, 0.4249447743109498])))
        self.assertTrue(np.array_equal(imp_fun.mdd[15:20], 
                        np.array([0.5045777092933046, 0.576424302849412, 0.6393091739184916, 0.6932203123193963, 0.7388256596555696])))
        self.assertTrue(np.array_equal(imp_fun.mdd[20:25], 
                        np.array([0.777104531116526, 0.8091124649261859, 0.8358522190681132, 0.8582150905529946, 0.8769633232141456])))

    def test_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = IFTropCyclone()
        imp_fun.set_emanuel_usa(if_id=5, intensity=np.arange(0,6,1), v_thresh=2,
                 v_half=5, scale=0.5)
        self.assertEqual(imp_fun.name, 'Emanuel 2011')
        self.assertEqual(imp_fun.haz_type, 'TC')
        self.assertEqual(imp_fun.id, 5)
        self.assertEqual(imp_fun.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 6, 1)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((6,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:3], np.zeros((3,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[3:], 
                        np.array([0.017857142857142853, 0.11428571428571425, 0.250000000000000])))

    def test_wrong_shape(self):
        """Set shape parameters."""
        imp_fun = IFTropCyclone()
        with self.assertRaises(ValueError):
            imp_fun.set_emanuel_usa(if_id=5, v_thresh=2, v_half=1, intensity=np.arange(0,6,1))

    def test_wrong_scale(self):
        """Set shape parameters."""
        imp_fun = IFTropCyclone()
        with self.assertRaises(ValueError):
            imp_fun.set_emanuel_usa(if_id=5, scale=2, intensity=np.arange(0,6,1))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmanuelFormula)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
