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
from climada.engine.calibration_opt import calib_instance
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


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCalib)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
