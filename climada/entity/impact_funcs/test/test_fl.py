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

from climada.entity.impact_funcs import river_flood as fl


class TestIFRiverFlood(unittest.TestCase):
    """Impact function test"""
    def test_flood_imp_func_set(self):
        test_set = fl.flood_imp_func_set()
        self.assertTrue(np.array_equal(test_set.get_hazard_types(),
                        np.array(['RF'])))
        self.assertEqual(test_set.size(), 6)

    def test_set_RF_IF_Africa(self):

        if_1 = fl.IFRiverFlood()
        if_1.set_RF_IF_Africa()

        self.assertEqual(if_1.continent, 'Africa')
        self.assertEqual(if_1.name, 'Flood Africa JRC Residential noPAA')
        self.assertEqual(if_1.haz_type, 'RF')
        self.assertEqual(if_1.id, 1)
        self.assertEqual(if_1.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_1.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_1.mdd,
                        np.array([0., 0.2199, 0.3782,
                                  0.5306, 0.6356, 0.8169,
                                  0.9034, 0.9572, 1., 1.])))
        self.assertTrue(np.allclose(if_1.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_set_RF_IF_Asia(self):

        if_2 = fl.IFRiverFlood()
        if_2.set_RF_IF_Asia()
        self.assertEqual(if_2.continent, 'Asia')
        self.assertEqual(if_2.name, 'Flood Asia JRC Residential noPAA')
        self.assertEqual(if_2.haz_type, 'RF')
        self.assertEqual(if_2.id, 2)
        self.assertEqual(if_2.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_2.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_2.mdd,
                        np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207,
                                  0.8695, 0.9315, 0.9836, 1.0000, 1.0000])))
        self.assertTrue(np.allclose(if_2.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_set_RF_IF_Europe(self):

        if_3 = fl.IFRiverFlood()
        if_3.set_RF_IF_Europe()
        self.assertEqual(if_3.continent, 'Europe')
        self.assertEqual(if_3.name, 'Flood Europe JRC Residential noPAA')
        self.assertEqual(if_3.haz_type, 'RF')
        self.assertEqual(if_3.id, 3)
        self.assertEqual(if_3.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_3.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_3.mdd,
                        np.array([0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85,
                                  0.95, 1.00, 1.00])))
        self.assertTrue(np.allclose(if_3.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_set_RF_IF_NorthAmerica(self):

        if_4 = fl.IFRiverFlood()
        if_4.set_RF_IF_NorthAmerica()

        self.assertEqual(if_4.continent, 'NorthAmerica')
        self.assertEqual(if_4.name,
                         'Flood North America JRC Residential noPAA')
        self.assertEqual(if_4.haz_type, 'RF')
        self.assertEqual(if_4.id, 4)
        self.assertEqual(if_4.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_4.intensity,
                        np.array([0., 0.1, 0.5, 1., 1.5, 2., 3., 4., 5.,
                                  6., 12.])))
        self.assertTrue(np.allclose(if_4.mdd,
                        np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825,
                                  0.7840, 0.8543, 0.9237, 0.9585, 1.0000,
                                  1.0000])))
        self.assertTrue(np.allclose(if_4.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_set_RF_IF_Oceania(self):

        if_5 = fl.IFRiverFlood()
        if_5.set_RF_IF_Oceania()
        self.assertEqual(if_5.continent, 'Oceania')
        self.assertEqual(if_5.name, 'Flood Oceania JRC Residential noPAA')
        self.assertEqual(if_5.haz_type, 'RF')
        self.assertEqual(if_5.id, 5)
        self.assertEqual(if_5.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_5.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_5.mdd,
                        np.array([0.00, 0.48, 0.64, 0.71, 0.79, 0.93, 0.97,
                                  0.98, 1.00, 1.00])))
        self.assertTrue(np.allclose(if_5.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))

    def test_set_RF_IF_SouthAmerica(self):

        if_6 = fl.IFRiverFlood()
        if_6.set_RF_IF_SouthAmerica()
        self.assertEqual(if_6.continent, 'SouthAmerica')
        self.assertEqual(if_6.name,
                         'Flood South America JRC Residential noPAA')
        self.assertEqual(if_6.haz_type, 'RF')
        self.assertEqual(if_6.id, 6)
        self.assertEqual(if_6.intensity_unit, 'm')
        self.assertTrue(np.array_equal(if_6.intensity,
                        np.array([0., 0.5, 1., 1.5,
                                  2., 3., 4., 5., 6., 12.])))
        self.assertTrue(np.allclose(if_6.mdd,
                        np.array([0.0000, 0.4908, 0.7112, 0.8420, 0.9494,
                                  0.9836, 1.0000, 1.0000, 1.0000, 1.0000])))
        self.assertTrue(np.allclose(if_6.paa,
                        np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIFRiverFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
