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

Test Impf_StormEurope class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.storm_europe import ImpfStormEurope

class TestStormEuropeDefault(unittest.TestCase):
    """Impact function interpolation test"""

    def test_default_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpfStormEurope.from_schwierz()
        self.assertEqual(imp_fun.name, 'Schwierz 2010')
        self.assertEqual(imp_fun.haz_type, 'WS')
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun.intensity, np.array([0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100])))
        self.assertTrue(np.array_equal(imp_fun.paa[4:8], np.array([0.03921, 0.10707, 0.25357, 0.48869])))
        self.assertTrue(np.array_equal(imp_fun.mdd[4:8], np.array([0.00367253, 0.00749977, 0.01263556, 0.01849639])))

        imp_fun2 = ImpfStormEurope.from_welker()
        self.assertEqual(imp_fun2.name, 'Welker 2021')
        self.assertEqual(imp_fun2.haz_type, 'WS')
        self.assertEqual(imp_fun2.id, 1)
        self.assertEqual(imp_fun2.intensity_unit, 'm/s')
        self.assertTrue(np.array_equal(imp_fun2.intensity[np.arange(0, 120, 13)],
                                                          np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 90.])))
        self.assertTrue(np.allclose(imp_fun2.paa[np.arange(0, 120, 13)],
                                                    np.array([0., 0., 0., 0.00900782, 0.1426727,
                                                              0.65118822, 1., 1., 1., 1.])))
        self.assertTrue(np.allclose(imp_fun2.mdd[np.arange(0, 120, 13)],
                                                    np.array([0., 0., 0., 0.00236542, 0.00999358,
                                                              0.02464677, 0.04964029, 0.04964029, 0.04964029, 0.04964029])))




# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestStormEuropeDefault)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
