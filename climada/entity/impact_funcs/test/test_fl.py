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
Test IFFlood class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs import flood as fl


class TestIFFlood(unittest.TestCase):
    """Impact function test"""
    def test_flood_imp_func_set(self):
        test_set = fl.flood_imp_func_set()
        self.assertTrue(np.array_equal(test_set.get_hazard_types(),
                        np.array(['RF'])))
        self.assertEqual(test_set.size(), 6)

    def test_flood_imp_func(self):

        test_set = fl.flood_imp_func_set()

        if_1 = test_set.get_func(haz_type='RF', fun_id=1)
        self.assertEqual(if_1.continent, 'Africa')
        self.assertEqual(if_1.name, 'Flood Africa JRC Residential noPAA')
        self.assertEqual(if_1.haz_type, 'RF')
        self.assertEqual(if_1.id, 1)
        self.assertEqual(if_1.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_1.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_1.mdd,
                        np.array([0., 0.2199254, 0.37822685,
                                  0.53058908, 0.63563673, 0.81693978,
                                  0.90343469, 0.95715217, 1., 1.])))
        self.assertTrue(np.allclose(if_1.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

        if_6 = test_set.get_func(haz_type='RF', fun_id=6)
        self.assertEqual(if_6.continent, 'SouthAmerica')
        self.assertEqual(if_6.name,
                         'Flood South America JRC Residential noPAA')
        self.assertEqual(if_6.haz_type, 'RF')
        self.assertEqual(if_6.id, 6)
        self.assertEqual(if_6.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_6.intensity,
                        np.array([0., 0.5, 1., 1.5, 2.,
                                  3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_6.mdd,
                        np.array([0., 0.49088595, 0.71129407, 0.84202601,
                                  0.9493691, 0.98363698, 1., 1., 1., 1.])))
        self.assertTrue(np.allclose(if_6.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
