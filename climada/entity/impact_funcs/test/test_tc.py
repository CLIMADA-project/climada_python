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

Test impf_TropCycl class.
"""

import unittest
import numpy as np
import pandas as pd

from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone

class TestEmanuelFormula(unittest.TestCase):
    """Impact function interpolation test"""

    def test_default_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpfTropCyclone.from_emanuel_usa()
        self.assertEqual(imp_fun.name, 'Emanuel 2011')
        self.assertEqual(imp_fun.haz_type, 'TC')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 121, 5)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((25,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:6], np.zeros((6,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[6:10],
                        np.array([0.0006753419543492556, 0.006790495604105169,
                                  0.02425254393374475, 0.05758706257339458])))
        self.assertTrue(np.array_equal(imp_fun.mdd[10:15],
                        np.array([0.10870556455111065, 0.1761433569521351,
                                  0.2553983618763961, 0.34033822528795565,
                                  0.4249447743109498])))
        self.assertTrue(np.array_equal(imp_fun.mdd[15:20],
                        np.array([0.5045777092933046, 0.576424302849412,
                                  0.6393091739184916, 0.6932203123193963,
                                  0.7388256596555696])))
        self.assertTrue(np.array_equal(imp_fun.mdd[20:25],
                        np.array([0.777104531116526, 0.8091124649261859,
                                  0.8358522190681132, 0.8582150905529946,
                                  0.8769633232141456])))

    def test_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpfTropCyclone.from_emanuel_usa(impf_id=5,
                                                   intensity=np.arange(0, 6, 1),
                                                   v_thresh=2,
                                                   v_half=5,
                                                   scale=0.5)
        self.assertEqual(imp_fun.name, 'Emanuel 2011')
        self.assertEqual(imp_fun.haz_type, 'TC')
        self.assertEqual(imp_fun.id, 5)
        self.assertEqual(imp_fun.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 6, 1)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((6,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:3], np.zeros((3,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[3:],
                        np.array([0.017857142857142853, 0.11428571428571425,
                                  0.250000000000000])))

    def test_wrong_shape(self):
        """Set shape parameters."""
        with self.assertRaises(ValueError):
            imp_fun = ImpfTropCyclone.from_emanuel_usa(impf_id=5, v_thresh=2,
                                                       v_half=1,
                                                       intensity=np.arange(0, 6, 1))

    def test_wrong_scale(self):
        """Set shape parameters."""
        with self.assertRaises(ValueError):
            imp_fun = ImpfTropCyclone.from_emanuel_usa(impf_id=5, scale=2,
                                                       intensity=np.arange(0, 6, 1))

class TestCalibratedImpfSet(unittest.TestCase):
    """Test inititation of IFS with regional calibrated TC IFs
        based on Eberenz et al. (2020)"""

    def test_default_values_pass(self):
        """Test return TDR optimized IFs (TDR=1)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()
        v_halfs = ImpfSetTropCyclone.calibrated_regional_vhalf()
        # extract IF for region WP4
        impf_wp4 = impfs.get_func(fun_id=9)[0]
        self.assertIn('TC', impfs.get_ids().keys())
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs.get_ids()['TC'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(impf_wp4.intensity_unit, 'm/s')
        self.assertEqual(impf_wp4.name, 'North West Pacific (WP4)')
        self.assertAlmostEqual(v_halfs['WP2'], 188.4, places=7)
        self.assertAlmostEqual(v_halfs['ROW'], 110.1, places=7)
        self.assertListEqual(list(impf_wp4.intensity), list(np.arange(0, 121, 5)))
        self.assertEqual(impf_wp4.paa.min(), 1.)
        self.assertEqual(impf_wp4.mdd.min(), 0.0)
        self.assertAlmostEqual(impf_wp4.mdd.max(), 0.15779133833203, places=5)
        self.assertAlmostEqual(impf_wp4.calc_mdr(75), 0.02607326527808, places=5)

    def test_RMSF_pass(self):
        """Test return RMSF optimized impact function set (RMSF=minimum)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet('RMSF')
        v_halfs = ImpfSetTropCyclone.calibrated_regional_vhalf(calibration_approach='RMSF')
        # extract IF for region NA1
        impf_na1 = impfs.get_func(fun_id=1)[0]
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs.get_ids()['TC'], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(impf_na1.intensity_unit, 'm/s')
        self.assertEqual(impf_na1.name, 'Caribbean and Mexico (NA1)')
        self.assertAlmostEqual(v_halfs['NA1'], 59.6, places=7)
        self.assertAlmostEqual(v_halfs['ROW'], 73.4, places=7)
        self.assertListEqual(list(impf_na1.intensity), list(np.arange(0, 121, 5)))
        self.assertEqual(impf_na1.mdd.min(), 0.0)
        self.assertAlmostEqual(impf_na1.mdd.max(), 0.95560418241669, places=5)
        self.assertAlmostEqual(impf_na1.calc_mdr(75), 0.7546423895457, places=5)

    def test_quantile_pass(self):
        """Test return impact function set from quantile of inidividual event fitting (EDR=1)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet('EDR')
        impfs_p10 = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet('EDR', q=.1)
        # extract IF for region SI
        impf_si = impfs.get_func(fun_id=5)[0]
        impf_si_p10 = impfs_p10.get_func(fun_id=5)[0]
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs_p10.size(), 10)
        self.assertEqual(impf_si.intensity_unit, 'm/s')
        self.assertEqual(impf_si_p10.name, 'South Indian (SI)')
        self.assertAlmostEqual(impf_si_p10.mdd.max(), 0.99999999880, places=5)
        self.assertAlmostEqual(impf_si.calc_mdr(30), 0.01620503041, places=5)
        intensity = np.random.randint(26, impf_si.intensity.max())
        self.assertTrue(impf_si.calc_mdr(intensity) < impf_si_p10.calc_mdr(intensity))

    def test_get_countries_per_region(self):
        """Test static get_countries_per_region()"""
        ifs = ImpfSetTropCyclone()
        out = ifs.get_countries_per_region('NA2')
        self.assertEqual(out[0], 'USA and Canada')
        self.assertEqual(out[1], 2)
        self.assertListEqual(out[2], [124, 840])
        self.assertListEqual(out[3], ['CAN', 'USA'])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmanuelFormula)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalibratedImpfSet))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
