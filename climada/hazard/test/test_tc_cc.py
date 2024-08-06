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

Test tc_clim_change module
"""

import unittest

import unittest
import pandas as pd
import numpy as np
import climada.hazard.tc_clim_change as tc_cc
from climada.hazard.tc_clim_change import MAP_BASINS_NAMES, MAP_VARS_NAMES, MAP_PERC_NAMES, YEAR_WINDOWS_PROPS

class TestKnutson(unittest.TestCase):

    def test_get_knutson_scaling_pass(self):
        """Test get_knutson_criterion function."""
        criterion = tc_cc.get_knutson_scaling_factor(
                    percentile='5/10',
                    baseline=(1950, 2018))
        self.assertEqual(criterion.shape, (21, 4))

        self.assertEqual(criterion.columns[0], '2.6')
        self.assertEqual(criterion.columns[1], '4.5')
        self.assertEqual(criterion.columns[2], '6.0')
        self.assertEqual(criterion.columns[3], '8.5')

        self.assertAlmostEqual(criterion.loc[2030, '2.6'], -16.13547, 4)
        self.assertAlmostEqual(criterion.loc[2050, '4.5'], -25.19448, 4)
        self.assertAlmostEqual(criterion.loc[2070, '6.0'], -31.06633, 4)
        self.assertAlmostEqual(criterion.loc[2100, '8.5'], -58.98637, 4)

    def test_get_gmst_pass(self):
        """Test get_gmst_info function."""
        gmst_info = tc_cc.get_gmst_info()

        self.assertAlmostEqual(gmst_info['gmst_start_year'], 1880)
        self.assertAlmostEqual(gmst_info['gmst_end_year'], 2100)
        self.assertAlmostEqual(len(gmst_info['rcps']), 4)

        self.assertAlmostEqual(gmst_info['gmst_data'].shape,
                               (len(gmst_info['rcps']),
                                gmst_info['gmst_end_year']-gmst_info['gmst_start_year']+1))
        self.assertAlmostEqual(gmst_info['gmst_data'][0,0], -0.16)
        self.assertAlmostEqual(gmst_info['gmst_data'][0,-1], 1.27641, 4)
        self.assertAlmostEqual(gmst_info['gmst_data'][-1,0], -0.16)
        self.assertAlmostEqual(gmst_info['gmst_data'][-1,-1], 4.477764, 4)

    def test_get_knutson_data_pass(self):
        """Test get_knutson_data function."""

        data_knutson = tc_cc.get_knutson_data()

        self.assertAlmostEqual(data_knutson.shape, (4,6,5))
        self.assertAlmostEqual(data_knutson[0,0,0], -34.49)
        self.assertAlmostEqual(data_knutson[-1,-1,-1], 15.419)
        self.assertAlmostEqual(data_knutson[0,-1,-1], 4.689)
        self.assertAlmostEqual(data_knutson[-1,0,0], 5.848)
        self.assertAlmostEqual(data_knutson[-1,0,-1], 22.803)
        self.assertAlmostEqual(data_knutson[2,3,2], 4.324)

    def test_valid_inputs(self):
        df = tc_cc.get_knutson_scaling_factor()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (21, 4))  # Default yearly steps produce 21 steps, 4 RCPs

    def test_invalid_baseline_start_year(self):
        with self.assertRaises(ValueError):
            tc_cc.get_knutson_scaling_factor(baseline=(1870, 2022))

    def test_invalid_baseline_end_year(self):
        with self.assertRaises(ValueError):
            tc_cc.get_knutson_scaling_factor(baseline=(1982, 2110))

    def test_no_scaling_factors(self):
        df = tc_cc.get_knutson_scaling_factor(basin='ZZZZZ')
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue((df.values == 1).all())  # Default value when no scaling factors found

    def test_variable_mapping_cat05(self):
        self.assertEqual(MAP_VARS_NAMES['cat05'], 0)

    def test_variable_mapping_cat45(self):
        self.assertEqual(MAP_VARS_NAMES['cat45'], 1)

    def test_variable_mapping_intensity(self):
        self.assertEqual(MAP_VARS_NAMES['intensity'], 2)

    def test_percentile_mapping_5_10(self):
        self.assertEqual(MAP_PERC_NAMES['5/10'], 0)

    def test_percentile_mapping_25(self):
        self.assertEqual(MAP_PERC_NAMES['25'], 1)

    def test_percentile_mapping_50(self):
        self.assertEqual(MAP_PERC_NAMES['50'], 2)

    def test_percentile_mapping_75(self):
        self.assertEqual(MAP_PERC_NAMES['75'], 3)

    def test_percentile_mapping_90_95(self):
        self.assertEqual(MAP_PERC_NAMES['90/95'], 4)

    def test_basin_mapping_NA(self):
        self.assertEqual(MAP_BASINS_NAMES['NA'], 0)

    def test_basin_mapping_WP(self):
        self.assertEqual(MAP_BASINS_NAMES['WP'], 1)

    def test_basin_mapping_EP(self):
        self.assertEqual(MAP_BASINS_NAMES['EP'], 2)

    def test_basin_mapping_NI(self):
        self.assertEqual(MAP_BASINS_NAMES['NI'], 3)

    def test_basin_mapping_SI(self):
        self.assertEqual(MAP_BASINS_NAMES['SI'], 4)

    def test_basin_mapping_SP(self):
        self.assertEqual(MAP_BASINS_NAMES['SP'], 5)
    
    def test_year_windows_props_start(self):
        self.assertEqual(YEAR_WINDOWS_PROPS['start'], 2000)
        
    def test_year_windows_props_end(self):
        self.assertEqual(YEAR_WINDOWS_PROPS['end'], 2100)
        
    def test_year_windows_props_smoothing(self):
        self.assertEqual(YEAR_WINDOWS_PROPS['smoothing'], 5)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestKnutson)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
