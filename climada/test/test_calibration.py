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

Test Calibration class.
"""

import unittest
from pathlib import Path

import pandas as pd

import climada.hazard.test as hazard_test
from climada import CONFIG
from climada.engine import ImpactCalc
from climada.engine.calibration_opt import calib_all, calib_instance
from climada.entity import ImpactFuncSet
from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.test import get_test_file
from climada.util.constants import ENT_DEMO_TODAY

HAZ_TEST_TC = get_test_file("test_tc_florida", file_format="hdf5")

DATA_FOLDER = CONFIG.test_data.dir()


class TestCalib(unittest.TestCase):
    """Test engine calibration method."""

    def test_calib_instance(self):
        """Test save calib instance"""
        # Read default entity values
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        # Read default hazard file
        hazard = Hazard.from_hdf5(HAZ_TEST_TC)

        # get impact function from set
        imp_func = ent.impact_funcs.get_func(
            hazard.haz_type, ent.exposures.gdf["impf_TC"].median()
        )

        # Assign centroids to exposures
        ent.exposures.assign_centroids(hazard)

        # create input frame
        df_in = pd.DataFrame.from_dict(
            {"v_threshold": [25.7], "other_param": [2], "hazard": [HAZ_TEST_TC]}
        )
        df_in_yearly = pd.DataFrame.from_dict(
            {"v_threshold": [25.7], "other_param": [2], "hazard": [HAZ_TEST_TC]}
        )

        # Compute the impact over the whole exposures
        df_out = calib_instance(hazard, ent.exposures, imp_func, df_in)
        df_out_yearly = calib_instance(
            hazard, ent.exposures, imp_func, df_in_yearly, yearly_impact=True
        )
        # calc Impact as comparison
        impact = ImpactCalc(ent.exposures, ent.impact_funcs, hazard).impact(
            assign_centroids=False
        )
        IYS = impact.impact_per_year(all_years=True)

        # do the tests
        self.assertTrue(isinstance(df_out, pd.DataFrame))
        self.assertTrue(isinstance(df_out_yearly, pd.DataFrame))
        self.assertEqual(df_out.shape[0], hazard.event_id.size)
        self.assertEqual(df_out_yearly.shape[0], 161)
        self.assertTrue(all(df_out["event_id"] == hazard.event_id))
        self.assertTrue(all(df_out[df_in.columns[0]].isin(df_in[df_in.columns[0]])))
        self.assertTrue(
            all(df_out_yearly[df_in.columns[1]].isin(df_in[df_in.columns[1]]))
        )
        self.assertTrue(
            all(df_out_yearly[df_in.columns[2]].isin(df_in[df_in.columns[2]]))
        )
        self.assertTrue(all(df_out["impact_CLIMADA"].values == impact.at_event))
        self.assertTrue(all(df_out_yearly["impact_CLIMADA"].values == [*IYS.values()]))


class TestCalibPandas2(unittest.TestCase):
    """Cover the two calibration_opt paths that used removed pandas APIs (#1319)."""

    @classmethod
    def setUpClass(cls):
        cls.hazard = Hazard.from_hdf5(HAZ_TEST_TC)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        cls.exposures = entity.exposures
        cls.exposures.assign_centroids(cls.hazard)
        cls.impf = entity.impact_funcs.get_func(
            cls.hazard.haz_type, entity.exposures.gdf["impf_TC"].median()
        )

    def test_calib_instance_yearly_multirow(self):
        """A multi-row df_out takes the `years_in_common` branch.

        test_calib_instance above passes a single-row frame, which takes the
        other branch, so this one was never covered.
        """
        impact = ImpactCalc(
            self.exposures, ImpactFuncSet([self.impf]), self.hazard
        ).impact(assign_centroids=False)
        iys = impact.impact_per_year(all_years=True)
        sel_years = sorted(iys.keys())[:3]

        df_out = calib_instance(
            self.hazard,
            self.exposures,
            self.impf,
            pd.DataFrame({"year": sel_years}),
            yearly_impact=True,
        )

        self.assertEqual(df_out.shape[0], len(sel_years))
        for year in sel_years:
            self.assertAlmostEqual(
                df_out.loc[df_out["year"] == year, "impact_CLIMADA"].iloc[0], iys[year]
            )

    def test_calib_all_multiple_params(self):
        """Several parameter combinations are concatenated into one frame.

        A single-row `impact_data_source` keeps `calib_instance` on its other
        branch, so this covers only the accumulation in `calib_all`.
        """
        df_result = calib_all(
            hazard=self.hazard,
            exposure=self.exposures,
            impf_name_or_instance="emanuel",
            param_full_dict={
                "v_thresh": [25.7, 20.0],
                "v_half": [70.0],
                "scale": [1.0],
            },
            impact_data_source=pd.DataFrame({"impact": [1.0e9], "region_id": [840]}),
            year_range=[2004, 2005],
            yearly_impact=True,
        )

        self.assertSetEqual(set(df_result["v_thresh"]), {25.7, 20.0})
        self.assertIn("impact_CLIMADA", df_result.columns)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCalib)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalibPandas2))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
