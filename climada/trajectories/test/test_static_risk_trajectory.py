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

unit tests for static_risk_trajectory

"""

import datetime
import types
import unittest
from itertools import product
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np  # For potential NaN/NA comparisons
import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.impact_calc_strat import ImpactCalcComputation
from climada.trajectories.riskperiod import (  # ImpactComputationStrategy, # If needed to mock its base class directly
    CalcRiskMetricsPoints,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.static_trajectory import (
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_RP,
    StaticRiskTrajectory,
)


class TestStaticRiskTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        self.dates1 = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
        self.dates2 = [pd.Timestamp("2026-01-01")]
        self.groups = ["GroupA", "GroupB", pd.NA]
        self.measures = ["MEAS1", "MEAS2"]
        self.metrics = ["aai"]
        self.aai_dates1 = pd.DataFrame(
            product(self.groups, self.dates1, self.measures, self.metrics),
            columns=["group", "date", "measure", "metric"],
        )
        self.aai_dates1["risk"] = np.arange(12) * 100
        self.aai_dates1["group"] = self.aai_dates1["group"].astype("category")

        self.aai_dates2 = pd.DataFrame(
            product(self.groups, self.dates2, self.measures, self.metrics),
            columns=["group", "date", "measure", "metric"],
        )
        self.aai_dates2["risk"] = np.arange(6) * 100 + 1200
        self.aai_dates2["group"] = self.aai_dates2["group"].astype("category")

        self.aai_alldates = pd.DataFrame(
            product(
                self.groups, self.dates1 + self.dates2, self.measures, self.metrics
            ),
            columns=["group", "date", "measure", "metric"],
        )
        self.aai_alldates["risk"] = np.arange(18) * 100
        self.aai_alldates["group"] = self.aai_alldates["group"].astype("category")
        self.aai_alldates["group"] = self.aai_alldates["group"].cat.add_categories(
            [DEFAULT_ALLGROUP_NAME]
        )
        self.aai_alldates["group"] = self.aai_alldates["group"].fillna(
            DEFAULT_ALLGROUP_NAME
        )
        self.expected_pre_npv_aai = self.aai_alldates
        self.expected_pre_npv_aai = self.expected_pre_npv_aai[
            ["group", "date", "measure", "metric", "risk"]
        ]

        self.expected_npv_aai = pd.DataFrame(
            product(
                self.groups, self.dates1 + self.dates2, self.measures, self.metrics
            ),
            columns=["group", "date", "measure", "metric"],
        )
        self.expected_npv_aai["risk"] = np.arange(18) * 90
        self.expected_npv_aai["group"] = self.expected_npv_aai["group"].astype(
            "category"
        )
        self.expected_npv_aai["group"] = self.expected_npv_aai[
            "group"
        ].cat.add_categories(["All"])
        self.expected_npv_aai["group"] = self.expected_npv_aai["group"].fillna(
            DEFAULT_ALLGROUP_NAME
        )
        expected_npv_df = self.expected_npv_aai
        expected_npv_df = expected_npv_df[
            ["group", "date", "measure", "metric", "risk"]
        ]

        self.mock_snapshot1 = MagicMock(spec=Snapshot)
        self.mock_snapshot1.date = datetime.date(2023, 1, 1)

        self.mock_snapshot2 = MagicMock(spec=Snapshot)
        self.mock_snapshot2.date = datetime.date(2024, 1, 1)

        self.mock_snapshot3 = MagicMock(spec=Snapshot)
        self.mock_snapshot3.date = datetime.date(2026, 1, 1)

        self.snapshots_list: list[Snapshot] = [
            self.mock_snapshot1,
            self.mock_snapshot2,
            self.mock_snapshot3,
        ]

        self.risk_disc_rates = MagicMock(spec=DiscRates)
        self.risk_disc_rates.years = [2023, 2024, 2025, 2026]
        self.risk_disc_rates.rates = [0.01, 0.02, 0.03, 0.04]  # Example rates

        self.mock_impact_computation_strategy = MagicMock(spec=ImpactCalcComputation)

        self.custom_all_groups_name = "custom"
        self.custom_return_periods = [10, 20]

        self.mock_static_traj = MagicMock(spec=StaticRiskTrajectory)
        self.mock_static_traj._all_groups_name = DEFAULT_ALLGROUP_NAME
        self.mock_static_traj._risk_disc_rates = None
        self.mock_static_traj._risk_metrics_calculators = MagicMock(
            spec=CalcRiskMetricsPoints
        )

    @patch(
        "climada.trajectories.static_trajectory.CalcRiskMetricsPoints",
        autospec=True,
    )
    def test_init_basic(self, MockCalcRiskPoints):
        mock_calculator = MagicMock(spec=CalcRiskMetricsPoints)
        mock_calculator.impact_computation_strategy = (
            self.mock_impact_computation_strategy
        )
        MockCalcRiskPoints.return_value = mock_calculator
        rt = StaticRiskTrajectory(
            self.snapshots_list,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        MockCalcRiskPoints.assert_has_calls(
            [
                call(
                    self.snapshots_list,
                    impact_computation_strategy=self.mock_impact_computation_strategy,
                ),
            ]
        )
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertIsNone(rt._risk_disc_rates)
        self.assertEqual(rt._all_groups_name, DEFAULT_ALLGROUP_NAME)
        self.assertEqual(rt._return_periods, DEFAULT_RP)
        self.assertEqual(
            rt.impact_computation_strategy, self.mock_impact_computation_strategy
        )
        # Check that metrics are reset (initially None)
        for metric in StaticRiskTrajectory.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))

    @patch(
        "climada.trajectories.static_trajectory.CalcRiskMetricsPoints",
        autospec=True,
    )
    def test_init_args(self, mock_calc_risk_metrics_points):
        rt = StaticRiskTrajectory(
            self.snapshots_list,
            return_periods=self.custom_return_periods,
            all_groups_name=self.custom_all_groups_name,
            risk_disc_rates=self.risk_disc_rates,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertEqual(rt._risk_disc_rates, self.risk_disc_rates)
        self.assertEqual(rt._all_groups_name, self.custom_all_groups_name)
        self.assertEqual(rt._return_periods, self.custom_return_periods)
        self.assertEqual(rt.return_periods, self.custom_return_periods)
        # Check that metrics are reset (initially None)
        for metric in StaticRiskTrajectory.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))
        self.assertIsInstance(rt._risk_metrics_calculators, CalcRiskMetricsPoints)
        mock_calc_risk_metrics_points.assert_called_with(
            self.snapshots_list,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )

    @patch.object(StaticRiskTrajectory, "_reset_metrics", new_callable=Mock)
    @patch(
        "climada.trajectories.static_trajectory.CalcRiskMetricsPoints",
        autospec=True,
    )
    def test_set_impact_computation_strategy(
        self, mock_calc_risk_metrics_points, mock_reset_metrics
    ):
        rt = StaticRiskTrajectory(
            self.snapshots_list,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        mock_reset_metrics.assert_called_once()  # Called during init
        with self.assertRaises(ValueError):
            rt.impact_computation_strategy = "A"

        # There is only one possibility at the moment so we just check against a new object
        new_impact_calc = ImpactCalcComputation()
        rt.impact_computation_strategy = new_impact_calc
        self.assertEqual(rt.impact_computation_strategy, new_impact_calc)
        mock_reset_metrics.assert_has_calls([call(), call()])

    def test_generic_metrics(self):
        self.mock_static_traj.POSSIBLE_METRICS = StaticRiskTrajectory.POSSIBLE_METRICS
        self.mock_static_traj._generic_metrics = types.MethodType(
            StaticRiskTrajectory._generic_metrics, self.mock_static_traj
        )
        self.mock_static_traj._risk_disc_rates = self.risk_disc_rates
        self.mock_static_traj._aai_metrics = None
        with self.assertRaises(ValueError):
            self.mock_static_traj._generic_metrics(None, "dummy_meth")

        with self.assertRaises(NotImplementedError):
            self.mock_static_traj._generic_metrics("dummy_name", "dummy_meth")

        self.mock_static_traj._risk_metrics_calculators.calc_aai_metric.return_value = (
            self.aai_alldates
        )
        self.mock_static_traj.npv_transform.return_value = self.expected_npv_aai
        result = self.mock_static_traj._generic_metrics("aai", "calc_aai_metric")

        self.mock_static_traj._risk_metrics_calculators.calc_aai_metric.assert_called_once_with()
        self.mock_static_traj.npv_transform.assert_called_once()
        pd.testing.assert_frame_equal(
            self.mock_static_traj.npv_transform.call_args[0][0].reset_index(drop=True),
            self.expected_pre_npv_aai.reset_index(drop=True),
        )
        self.assertEqual(
            self.mock_static_traj.npv_transform.call_args[0][1], self.risk_disc_rates
        )
        pd.testing.assert_frame_equal(
            result, self.expected_npv_aai
        )  # Final result is from NPV transform

        # Check internal storage
        stored_df = getattr(self.mock_static_traj, "_aai_metrics")
        # Assert that the stored DF is the one *before* NPV transformation
        pd.testing.assert_frame_equal(
            stored_df.reset_index(drop=True),
            self.expected_npv_aai.reset_index(drop=True),
        )

        result2 = self.mock_static_traj._generic_metrics("aai", "calc_aai_metric")
        # Check no new call
        self.mock_static_traj._risk_metrics_calculators.calc_aai_metric.assert_called_once_with()
        pd.testing.assert_frame_equal(
            result2,
            self.expected_npv_aai.reset_index(drop=True),
        )

    def test_eai_metrics(self):
        self.mock_static_traj.eai_metrics = types.MethodType(
            StaticRiskTrajectory.eai_metrics, self.mock_static_traj
        )
        self.mock_static_traj.eai_metrics(some_arg="test")
        self.mock_static_traj._compute_metrics.assert_called_once_with(
            metric_name="eai", metric_meth="calc_eai_gdf", some_arg="test"
        )

    def test_aai_metrics(self):
        self.mock_static_traj.aai_metrics = types.MethodType(
            StaticRiskTrajectory.aai_metrics, self.mock_static_traj
        )
        self.mock_static_traj.aai_metrics(some_arg="test")
        self.mock_static_traj._compute_metrics.assert_called_once_with(
            metric_name="aai", metric_meth="calc_aai_metric", some_arg="test"
        )

    def test_return_periods_metrics(self):
        self.mock_static_traj.return_periods = [1, 2]
        self.mock_static_traj.return_periods_metrics = types.MethodType(
            StaticRiskTrajectory.return_periods_metrics, self.mock_static_traj
        )
        self.mock_static_traj.return_periods_metrics(some_arg="test")
        self.mock_static_traj._compute_metrics.assert_called_once_with(
            metric_name="return_periods",
            metric_meth="calc_return_periods_metric",
            return_periods=[1, 2],
            some_arg="test",
        )

    def test_aai_per_group_metrics(self):
        self.mock_static_traj.aai_per_group_metrics = types.MethodType(
            StaticRiskTrajectory.aai_per_group_metrics, self.mock_static_traj
        )
        self.mock_static_traj.aai_per_group_metrics(some_arg="test")
        self.mock_static_traj._compute_metrics.assert_called_once_with(
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
            some_arg="test",
        )

    def test_per_date_risk_metrics_defaults(self):
        self.mock_static_traj.per_date_risk_metrics = types.MethodType(
            StaticRiskTrajectory.per_date_risk_metrics, self.mock_static_traj
        )
        # Set up mock return values for each method
        self.mock_static_traj.aai_metrics.return_value = pd.DataFrame(
            {"metric": ["aai"], "risk": [100]}
        )
        self.mock_static_traj.return_periods_metrics.return_value = pd.DataFrame(
            {"metric": ["rp"], "risk": [50]}
        )
        self.mock_static_traj.aai_per_group_metrics.return_value = pd.DataFrame(
            {"metric": ["aai_grp"], "risk": [10]}
        )
        result = self.mock_static_traj.per_date_risk_metrics()

        # Assert calls with default arguments
        self.mock_static_traj.aai_metrics.assert_called_once_with()
        self.mock_static_traj.return_periods_metrics.assert_called_once_with()
        self.mock_static_traj.aai_per_group_metrics.assert_called_once_with()

        # Assert concatenation
        expected_df = pd.concat(
            [
                self.mock_static_traj.aai_metrics.return_value,
                self.mock_static_traj.return_periods_metrics.return_value,
                self.mock_static_traj.aai_per_group_metrics.return_value,
            ]
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_df.reset_index(drop=True)
        )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestStaticRiskTrajectory)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
