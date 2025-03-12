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
from math import log

import numpy as np
import pandas as pd

import climada.hazard.tc_clim_change as tc_cc


class TestKnutson(unittest.TestCase):

    def test_get_knutson_scaling_calculations(self):

        basin = "NA"
        variable = "cat05"
        percentile = "5/10"
        base_start, base_end = 1950, 2018
        yearly_steps = 5

        target_predicted_changes = tc_cc.get_knutson_scaling_factor(
            percentile=percentile,
            variable=variable,
            basin=basin,
            baseline=(base_start, base_end),
            yearly_steps=yearly_steps,
        )

        ## Test computations of future changes
        # Load data
        gmst_info = tc_cc.get_gmst_info()

        var_id, basin_id, perc_id = (
            tc_cc.MAP_VARS_NAMES[variable],
            tc_cc.MAP_BASINS_NAMES[basin],
            tc_cc.MAP_PERC_NAMES[percentile],
        )

        knutson_data = tc_cc.get_knutson_data()
        knutson_value = knutson_data[var_id, basin_id, perc_id]

        start_ind = base_start - gmst_info["gmst_start_year"]
        end_ind = base_end - gmst_info["gmst_start_year"]

        # Apply model
        beta = 0.5 * log(0.01 * knutson_value + 1)
        tc_properties = np.exp(beta * gmst_info["gmst_data"])

        # Assess baseline value
        baseline = np.mean(tc_properties[:, start_ind : end_ind + 1], 1)

        # Assess future value and test predicted change from baseline is
        # the same as given by function
        smoothing = 5

        for target_year in [2030, 2050, 2070, 2090]:
            target_year_ind = target_year - gmst_info["gmst_start_year"]
            ind1 = target_year_ind - smoothing
            ind2 = target_year_ind + smoothing + 1

            prediction = np.mean(tc_properties[:, ind1:ind2], 1)
            calculated_predicted_change = ((prediction - baseline) / baseline) * 100

            np.testing.assert_array_almost_equal(
                target_predicted_changes.loc[target_year, "2.6"],
                calculated_predicted_change[0],
            )
            np.testing.assert_array_almost_equal(
                target_predicted_changes.loc[target_year, "4.5"],
                calculated_predicted_change[1],
            )
            np.testing.assert_array_almost_equal(
                target_predicted_changes.loc[target_year, "6.0"],
                calculated_predicted_change[2],
            )
            np.testing.assert_array_almost_equal(
                target_predicted_changes.loc[target_year, "8.5"],
                calculated_predicted_change[3],
            )

    def test_get_knutson_scaling_structure(self):
        """Test get_knutson_criterion function."""

        yearly_steps = 8
        target_predicted_changes = tc_cc.get_knutson_scaling_factor(
            yearly_steps=yearly_steps
        )

        np.testing.assert_equal(
            target_predicted_changes.columns, np.array(["2.6", "4.5", "6.0", "8.5"])
        )

        simulated_years = np.arange(
            tc_cc.YEAR_WINDOWS_PROPS["start"],
            tc_cc.YEAR_WINDOWS_PROPS["end"] + 1,
            yearly_steps,
        )
        np.testing.assert_equal(target_predicted_changes.index, simulated_years)

    def test_get_knutson_scaling_valid_inputs(self):
        df = tc_cc.get_knutson_scaling_factor()
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.shape, (21, 4))

    def test_get_knutson_scaling_invalid_baseline_start_year(self):
        with self.assertRaises(ValueError):
            tc_cc.get_knutson_scaling_factor(baseline=(1870, 2022))

    def test_get_knutson_scaling_invalid_baseline_end_year(self):
        with self.assertRaises(ValueError):
            tc_cc.get_knutson_scaling_factor(baseline=(1982, 2110))

    def test_get_knutson_scaling_no_scaling_factors_for_unknonw_basin(self):
        df = tc_cc.get_knutson_scaling_factor(basin="ZZZZZ")
        self.assertIsInstance(df, pd.DataFrame)
        np.testing.assert_equal(df.values, np.ones_like(df.values))

    def test_get_gmst(self):
        """Test get_gmst_info function."""
        gmst_info = tc_cc.get_gmst_info()

        self.assertAlmostEqual(gmst_info["gmst_start_year"], 1880)
        self.assertAlmostEqual(gmst_info["gmst_end_year"], 2100)
        self.assertAlmostEqual(len(gmst_info["rcps"]), 4)

        self.assertAlmostEqual(
            gmst_info["gmst_data"].shape,
            (
                len(gmst_info["rcps"]),
                gmst_info["gmst_end_year"] - gmst_info["gmst_start_year"] + 1,
            ),
        )
        self.assertAlmostEqual(gmst_info["gmst_data"][0, 0], -0.16)
        self.assertAlmostEqual(gmst_info["gmst_data"][0, -1], 1.27641, 4)
        self.assertAlmostEqual(gmst_info["gmst_data"][-1, 0], -0.16)
        self.assertAlmostEqual(gmst_info["gmst_data"][-1, -1], 4.477764, 4)

    def test_get_knutson_data_pass(self):
        """Test get_knutson_data function."""

        data_knutson = tc_cc.get_knutson_data()

        self.assertAlmostEqual(data_knutson.shape, (4, 6, 5))
        self.assertAlmostEqual(data_knutson[0, 0, 0], -34.49)
        self.assertAlmostEqual(data_knutson[-1, -1, -1], 15.419)
        self.assertAlmostEqual(data_knutson[0, -1, -1], 4.689)
        self.assertAlmostEqual(data_knutson[-1, 0, 0], 5.848)
        self.assertAlmostEqual(data_knutson[-1, 0, -1], 22.803)
        self.assertAlmostEqual(data_knutson[2, 3, 2], 4.324)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestKnutson)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
