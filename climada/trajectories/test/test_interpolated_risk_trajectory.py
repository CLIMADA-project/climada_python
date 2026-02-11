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

unit tests for interpolated_risk_trajectory

"""

import datetime
import unittest
from itertools import product
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np  # For potential NaN/NA comparisons
import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.calc_risk_metrics import (  # ImpactComputationStrategy, # If needed to mock its base class directly
    CalcRiskMetricsPeriod,
)
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    CONTRIBUTION_BASE_RISK_NAME,
    CONTRIBUTION_EXPOSURE_NAME,
    CONTRIBUTION_HAZARD_NAME,
    CONTRIBUTION_INTERACTION_TERM_NAME,
    CONTRIBUTION_TOTAL_RISK_NAME,
    CONTRIBUTION_VULNERABILITY_NAME,
    CONTRIBUTIONS_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    PERIOD_COL_NAME,
    RETURN_PERIOD_METRIC_NAME,
    RISK_COL_NAME,
    UNIT_COL_NAME,
)
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.interpolated_trajectory import (
    INDEXING_COLUMNS,
    InterpolatedRiskTrajectory,
)
from climada.trajectories.interpolation import (
    AllLinearStrategy,
    ExponentialExposureStrategy,
    InterpolationStrategy,
)
from climada.trajectories.snapshot import Snapshot


class TestInterpolatedRiskTrajectory(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.dates1 = [
            pd.Period("2023-01-01", freq="Y"),
            pd.Period("2024-01-01", freq="Y"),
        ]
        self.dates2 = [
            pd.Period("2025-01-01", freq="Y"),
            pd.Period("2026-01-01", freq="Y"),
        ]
        self.groups = ["GroupA", "GroupB", pd.NA]
        self.measures = ["MEAS1", "MEAS2"]
        self.metrics = [AAI_METRIC_NAME]
        self.aai_dates1 = pd.DataFrame(
            product(self.dates1, self.groups, self.measures, self.metrics),
            columns=INDEXING_COLUMNS,
        )
        self.aai_dates1[RISK_COL_NAME] = np.arange(12) * 100
        self.aai_dates1[GROUP_COL_NAME] = self.aai_dates1[GROUP_COL_NAME].astype(
            "category"
        )

        self.aai_dates2 = pd.DataFrame(
            product(self.dates2, self.groups, self.measures, self.metrics),
            columns=INDEXING_COLUMNS,
        )
        self.aai_dates2[RISK_COL_NAME] = np.arange(12) * 100 + 1200
        self.aai_dates2[GROUP_COL_NAME] = self.aai_dates2[GROUP_COL_NAME].astype(
            "category"
        )

        self.aai_alldates = pd.DataFrame(
            product(
                self.dates1 + self.dates2, self.groups, self.measures, self.metrics
            ),
            columns=INDEXING_COLUMNS,
        )
        self.aai_alldates[RISK_COL_NAME] = np.arange(24) * 100
        self.aai_alldates[GROUP_COL_NAME] = self.aai_alldates[GROUP_COL_NAME].astype(
            "category"
        )
        self.aai_alldates[GROUP_COL_NAME] = self.aai_alldates[
            GROUP_COL_NAME
        ].cat.add_categories(["All"])
        self.aai_alldates[GROUP_COL_NAME] = self.aai_alldates[GROUP_COL_NAME].fillna(
            "All"
        )
        self.expected_pre_npv_aai = self.aai_alldates
        self.expected_pre_npv_aai = self.expected_pre_npv_aai[
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                RISK_COL_NAME,
            ]
        ]

        self.expected_npv_aai = pd.DataFrame(
            product(
                self.dates1 + self.dates2, self.groups, self.measures, self.metrics
            ),
            columns=INDEXING_COLUMNS,
        )
        self.expected_npv_aai[RISK_COL_NAME] = np.arange(24) * 90
        self.expected_npv_aai[GROUP_COL_NAME] = self.expected_npv_aai[
            GROUP_COL_NAME
        ].astype("category")
        self.expected_npv_aai[GROUP_COL_NAME] = self.expected_npv_aai[
            GROUP_COL_NAME
        ].cat.add_categories(["All"])
        self.expected_npv_aai[GROUP_COL_NAME] = self.expected_npv_aai[
            GROUP_COL_NAME
        ].fillna("All")
        expected_npv_df = self.expected_npv_aai
        expected_npv_df = expected_npv_df[
            [
                GROUP_COL_NAME,
                DATE_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                RISK_COL_NAME,
            ]
        ]
        self.mock_snapshot1 = MagicMock(spec=Snapshot)
        self.mock_snapshot1.date = datetime.date(2023, 1, 1)

        self.mock_snapshot2 = MagicMock(spec=Snapshot)
        self.mock_snapshot2.date = datetime.date(2024, 1, 1)

        self.mock_snapshot3 = MagicMock(spec=Snapshot)
        self.mock_snapshot3.date = datetime.date(2025, 1, 1)

        self.snapshots_list: list[Snapshot] = [
            self.mock_snapshot1,
            self.mock_snapshot2,
            self.mock_snapshot3,
        ]
        # self.snapshots_list = cast(list[Snapshot], self.snapshots_list)

        # Mock interpolation strategy and impact computation strategy
        self.mock_interpolation_strategy = MagicMock(spec=AllLinearStrategy)
        self.mock_impact_computation_strategy = MagicMock(spec=ImpactCalcComputation)

        # Mock DiscRates if needed for NPV tests
        self.mock_disc_rates = MagicMock(spec=DiscRates)
        self.mock_disc_rates.years = [2023, 2024, 2025]
        self.mock_disc_rates.rates = [0.01, 0.02, 0.03]  # Example rates

        self.mock_risk_period_calc1 = MagicMock(spec=CalcRiskMetricsPeriod)
        self.mock_risk_period_calc2 = MagicMock(spec=CalcRiskMetricsPeriod)
        # Mock npv_transform return value
        self.mock_risk_period_calc1.calc_aai_metric.return_value = self.aai_dates1
        self.mock_risk_period_calc2.calc_aai_metric.return_value = self.aai_dates2
        self.mock_risk_metric_calculators = [
            self.mock_risk_period_calc1,
            self.mock_risk_period_calc2,
        ]

        self.mock_interpolated_risk_traj = MagicMock(spec=InterpolatedRiskTrajectory)
        self.mock_interpolated_risk_traj._risk_metrics_calcultators = (
            self.mock_risk_metric_calculators
        )
        self.mock_interpolated_risk_traj._risk_disc_rates = (
            self.mock_disc_rates
        )  # For NPV transform check

    # --- Test Initialization and Properties ---
    # These tests focus on the __init__ method and property getters/setters.

    ## Test `__init__` method
    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", return_value=1
    )
    def test_init_basic(self, mock_reset_metrics_calculators):
        # Test basic initialization with defaults
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertIsNone(rt._risk_disc_rates)
        mock_reset_metrics_calculators.assert_called_once_with(
            self.snapshots_list,
            "Y",
            self.mock_interpolation_strategy,
            self.mock_impact_computation_strategy,
        )
        self.assertEqual(rt._risk_metrics_calculators, 1)
        # Check that metrics are reset (initially None)
        for metric in InterpolatedRiskTrajectory.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))

    @patch.object(InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators")
    def test_init_with_custom_params(self, mock_reset_calculators):
        # Test initialization with custom parameters
        mock_disc = Mock(spec=DiscRates)
        mock_interp = Mock(spec=InterpolationStrategy)
        mock_impact_compute = Mock(spec=ImpactComputationStrategy)
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            time_resolution="MS",
            risk_disc_rates=mock_disc,
            interpolation_strategy=mock_interp,
            impact_computation_strategy=mock_impact_compute,
        )

        mock_reset_calculators.assert_has_calls(
            [call(self.snapshots_list, "MS", mock_interp, mock_impact_compute)]
        )
        self.assertEqual(rt._risk_disc_rates, mock_disc)

    @patch.object(InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators")
    @patch.object(InterpolatedRiskTrajectory, "_reset_metrics", new_callable=Mock)
    @patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    )
    def test_set_impact_computation_strategy(
        self,
        mock_calc_risk_metrics,
        mock_reset_metrics,
        mock_reset_risk_metrics_calculators,
    ):
        mock_reset_risk_metrics_calculators.return_value = (
            self.mock_risk_metric_calculators
        )
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
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
        for rp in self.mock_risk_metric_calculators:
            self.assertEqual(rp.impact_computation_strategy, new_impact_calc)

    @patch.object(InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators")
    @patch.object(InterpolatedRiskTrajectory, "_reset_metrics", new_callable=Mock)
    @patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    )
    def test_set_interpolation_strategy(
        self,
        mock_calc_risk_metrics,
        mock_reset_metrics,
        mock_reset_risk_metrics_calculators,
    ):
        mock_reset_risk_metrics_calculators.return_value = (
            self.mock_risk_metric_calculators
        )
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        mock_reset_metrics.assert_called_once()  # Called during init
        with self.assertRaises(ValueError):
            rt.interpolation_strategy = "A"

        # There is only one possibility at the moment so we just check against a new object
        new_interp = ExponentialExposureStrategy()
        rt.interpolation_strategy = new_interp
        self.assertEqual(rt.interpolation_strategy, new_interp)
        mock_reset_metrics.assert_has_calls([call(), call()])
        for rp in self.mock_risk_metric_calculators:
            self.assertEqual(rp.interpolation_strategy, new_interp)

    @patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    )
    def test_risk_periods_lazy_computation(self, MockCalcRiskPeriod):
        # Test that _calc_risk_periods is called only once, lazily
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )

        # First access should trigger calculation
        risk_periods = rt._risk_metrics_calculators
        MockCalcRiskPeriod.assert_has_calls(
            [
                call(
                    self.mock_snapshot1,
                    self.mock_snapshot2,
                    time_resolution="Y",
                    interpolation_strategy=self.mock_interpolation_strategy,
                    impact_computation_strategy=self.mock_impact_computation_strategy,
                ),
                call(
                    self.mock_snapshot2,
                    self.mock_snapshot3,
                    time_resolution="Y",
                    interpolation_strategy=self.mock_interpolation_strategy,
                    impact_computation_strategy=self.mock_impact_computation_strategy,
                ),
            ]
        )
        self.assertEqual(MockCalcRiskPeriod.call_count, 2)
        self.assertIsInstance(risk_periods, list)
        self.assertEqual(len(risk_periods), 2)  # N-1 periods for N snapshots

    @patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    )
    def test_calc_risk_periods_sorting(self, MockCalcRiskPeriod):
        # Test that snapshots are sorted by date before pairing
        unsorted_snapshots: list[Snapshot] = [
            self.mock_snapshot3,
            self.mock_snapshot1,
            self.mock_snapshot2,
        ]
        _ = InterpolatedRiskTrajectory(unsorted_snapshots)
        # Access the property to trigger calculation
        MockCalcRiskPeriod.assert_has_calls(
            [
                call(
                    self.mock_snapshot1,
                    self.mock_snapshot2,
                    **MockCalcRiskPeriod.call_args[1],
                ),
                call(
                    self.mock_snapshot2,
                    self.mock_snapshot3,
                    **MockCalcRiskPeriod.call_args[1],
                ),
            ]
        )
        self.assertEqual(MockCalcRiskPeriod.call_count, 2)

    @patch.object(InterpolatedRiskTrajectory, "_reset_metrics", new_callable=Mock)
    @patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    )
    def test_set_time_resolution(
        self, mock_calc_risk_metrics_points, mock_reset_metrics
    ):
        rt = InterpolatedRiskTrajectory(
            self.snapshots_list,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        mock_reset_metrics.assert_called_once()  # Called during init
        with self.assertRaises(ValueError):
            rt.time_resolution = 75

        # There is only one possibility at the moment so we just check against a new object
        rt.time_resolution = "5M"
        self.assertEqual(rt.time_resolution, "5M")
        mock_reset_metrics.assert_has_calls([call(), call()])

    # --- Test Generic Metric Computation (`_generic_metrics`) ---
    # This is a core internal method and deserves thorough testing.

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    @patch.object(InterpolatedRiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_basic_flow(
        self, mock_npv_transform, mock_risk_metrics_calculators
    ):
        mock_risk_metrics_calculators.return_value = self.mock_risk_metric_calculators
        mock_npv_transform.return_value = self.expected_npv_aai
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt._risk_disc_rates = self.mock_disc_rates
        result = rt._generic_metrics(
            metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric"
        )
        # Assertions
        self.mock_risk_period_calc1.calc_aai_metric.assert_called_once()
        self.mock_risk_period_calc2.calc_aai_metric.assert_called_once()

        # Check concatenated DataFrame before NPV
        # We need to manually recreate the expected intermediate DataFrame before NPV for assertion
        # npv_transform should be called with the correctly formatted (concatenated and ordered) DataFrame
        # and the risk_disc_rates attribute
        mock_npv_transform.assert_called_once()
        pd.testing.assert_frame_equal(
            mock_npv_transform.call_args[0][0].reset_index(drop=True),
            self.expected_pre_npv_aai.reset_index(drop=True),
        )
        self.assertEqual(mock_npv_transform.call_args[0][1], self.mock_disc_rates)

        pd.testing.assert_frame_equal(
            result, self.expected_npv_aai
        )  # Final result is from NPV transform

        # Check internal storage
        stored_df = getattr(rt, "_aai_metrics")
        # Assert that the stored DF is the one *before* NPV transformation
        pd.testing.assert_frame_equal(
            stored_df.reset_index(drop=True),
            self.expected_npv_aai.reset_index(drop=True),
        )

        result2 = rt._generic_metrics(
            metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric"
        )
        # Check no new calls
        self.mock_risk_period_calc1.calc_aai_metric.assert_called_once()
        self.mock_risk_period_calc2.calc_aai_metric.assert_called_once()
        pd.testing.assert_frame_equal(
            result2,
            self.expected_npv_aai.reset_index(drop=True),
        )

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    def test_generic_metrics_not_implemented_error(
        self, mock_reset_risk_metrics_calculators
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        with self.assertRaises(NotImplementedError):
            rt._generic_metrics(metric_name="non_existent", metric_meth="some_method")

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    def test_generic_metrics_value_error_no_name_or_method(
        self, mock_reset_risk_metrics_calculators
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        with self.assertRaises(ValueError):
            rt._generic_metrics(metric_name=None, metric_meth="some_method")
        with self.assertRaises(ValueError):
            rt._generic_metrics(metric_name=AAI_METRIC_NAME, metric_meth=None)

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    # @patch.object(InterpolatedRiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_None_concat_returns_empty(
        self, mock_reset_risk_metrics_calculators
    ):
        self.mock_risk_period_calc1.calc_aai_per_group_metric.return_value = None
        self.mock_risk_period_calc2.calc_aai_per_group_metric.return_value = None
        mock_reset_risk_metrics_calculators.return_value = (
            self.mock_risk_metric_calculators
        )
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        # rt = self.mock_interpolated_risk_traj
        # Mock CalcRiskPeriod instances return None, mimicking `calc_aai_per_group_metric` possibly

        result = rt._generic_metrics(
            metric_name=AAI_PER_GROUP_METRIC_NAME,
            metric_meth="calc_aai_per_group_metric",
        )
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    # @patch.object(InterpolatedRiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_empty_df_concat_returns_empty(
        self, mock_reset_risk_metrics_calculators
    ):
        self.mock_risk_period_calc1.calc_aai_per_group_metric.return_value = (
            pd.DataFrame()
        )
        self.mock_risk_period_calc2.calc_aai_per_group_metric.return_value = (
            pd.DataFrame()
        )
        mock_reset_risk_metrics_calculators.return_value = (
            self.mock_risk_metric_calculators
        )
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        # rt = self.mock_interpolated_risk_traj
        # Mock CalcRiskPeriod instances return None, mimicking `calc_aai_per_group_metric` possibly

        result = rt._generic_metrics(
            metric_name=AAI_PER_GROUP_METRIC_NAME,
            metric_meth="calc_aai_per_group_metric",
        )
        pd.testing.assert_frame_equal(result, pd.DataFrame())

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    @patch.object(
        InterpolatedRiskTrajectory,
        "_risk_contributions_post_treatment",
        new_callable=Mock,
    )
    def test_generic_metrics_risk_contribution_treatment(
        self,
        mock_risk_contributions_post_treatment,
        mock_reset_risk_metrics_calculators,
    ):
        mock_risk_contributions_post_treatment.return_value = pd.DataFrame([42])
        self.mock_risk_period_calc1.calc_risk_contributions_metric.return_value = (
            self.aai_dates1
        )
        self.mock_risk_period_calc2.calc_risk_contributions_metric.return_value = (
            self.aai_dates2
        )
        mock_reset_risk_metrics_calculators.return_value = (
            self.mock_risk_metric_calculators
        )
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        # rt = self.mock_interpolated_risk_traj
        # Mock CalcRiskPeriod instances return None, mimicking `calc_aai_per_group_metric` possibly
        result = rt._generic_metrics(
            metric_name=CONTRIBUTIONS_METRIC_NAME,
            metric_meth="calc_risk_contributions_metric",
        )
        mock_risk_contributions_post_treatment.assert_called_once()
        pd.testing.assert_frame_equal(result, pd.DataFrame([42]))

    @patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", new_callable=Mock
    )
    @patch.object(InterpolatedRiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_coord_id_handling(
        self, mock_npv_transform, mock_risk_metric_calc
    ):
        mock_risk_metric_calc.return_value = self.mock_risk_metric_calculators
        self.mock_risk_period_calc1.calc_eai_gdf.return_value = pd.DataFrame(
            {
                DATE_COL_NAME: [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
                GROUP_COL_NAME: pd.Categorical([pd.NA, pd.NA]),
                MEASURE_COL_NAME: ["MEAS1", "MEAS1"],
                METRIC_COL_NAME: [EAI_METRIC_NAME, EAI_METRIC_NAME],
                COORD_ID_COL_NAME: [1, 2],
                RISK_COL_NAME: [10.0, 20.0],
            }
        )
        self.mock_risk_period_calc2.calc_eai_gdf.return_value = pd.DataFrame()
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        result = rt._generic_metrics(
            metric_name=EAI_METRIC_NAME, metric_meth="calc_eai_gdf"
        )

        expected_df = pd.DataFrame(
            {
                GROUP_COL_NAME: pd.Categorical(["All", "All"]),
                DATE_COL_NAME: [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
                MEASURE_COL_NAME: ["MEAS1", "MEAS1"],
                METRIC_COL_NAME: [EAI_METRIC_NAME, EAI_METRIC_NAME],
                RISK_COL_NAME: [10.0, 20.0],
                COORD_ID_COL_NAME: [
                    1,
                    2,
                ],  # This column should remain and be placed at the end before risk if not in front_columns
            }
        )
        # The internal logic reorders columns, ensure it matches
        cols_order = [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            MEASURE_COL_NAME,
            METRIC_COL_NAME,
            COORD_ID_COL_NAME,
            RISK_COL_NAME,
        ]
        pd.testing.assert_frame_equal(result[cols_order], expected_df[cols_order])

    # --- Test Specific Metric Methods (e.g., `eai_metrics`, `aai_metrics`) ---
    # These are mostly thin wrappers around _compute_metrics/_generic_metrics.
    # Focus on ensuring they call _compute_metrics with the correct arguments.

    @patch.object(InterpolatedRiskTrajectory, "_compute_metrics")
    def test_eai_metrics(self, mock_compute_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.eai_metrics(npv=True, some_arg="test")
        mock_compute_metrics.assert_called_once_with(
            npv=True,
            metric_name=EAI_METRIC_NAME,
            metric_meth="calc_eai_gdf",
            some_arg="test",
        )

    @patch.object(InterpolatedRiskTrajectory, "_compute_metrics")
    def test_aai_metrics(self, mock_compute_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.aai_metrics(other_arg=123)
        mock_compute_metrics.assert_called_once_with(
            metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric", other_arg=123
        )

    @patch.object(InterpolatedRiskTrajectory, "_compute_metrics")
    def test_return_periods_metrics(self, mock_compute_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.return_periods_metrics(npv=True, rp_arg="xyz")
        mock_compute_metrics.assert_called_once_with(
            npv=True,
            metric_name=RETURN_PERIOD_METRIC_NAME,
            metric_meth="calc_return_periods_metric",
            return_periods=rt.return_periods,
            rp_arg="xyz",
        )

    @patch.object(InterpolatedRiskTrajectory, "_compute_metrics")
    def test_aai_per_group_metrics(self, mock_compute_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.aai_per_group_metrics()
        mock_compute_metrics.assert_called_once_with(
            metric_name=AAI_PER_GROUP_METRIC_NAME,
            metric_meth="calc_aai_per_group_metric",
        )

    @patch.object(InterpolatedRiskTrajectory, "_compute_metrics")
    def test_risk_components_metrics(self, mock_compute_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.risk_contributions_metrics()
        mock_compute_metrics.assert_called_once_with(
            metric_name=CONTRIBUTIONS_METRIC_NAME,
            metric_meth="calc_risk_contributions_metric",
        )

    ## Test `npv_transform` (class method)
    def test_npv_transform_no_group_col(self):
        df_input = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
                MEASURE_COL_NAME: ["m1", "m1", "m2", "m2"],
                METRIC_COL_NAME: [
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                ],
                RISK_COL_NAME: [100.0, 200.0, 80.0, 180.0],
            }
        )
        # Mock the internal calc_npv_cash_flows
        with patch(
            "climada.trajectories.trajectory.RiskTrajectory._calc_npv_cash_flows"
        ) as mock_calc_npv:
            # For each group, it will be called
            mock_calc_npv.side_effect = [
                pd.Series(
                    [100.0 * (1 / (1 + 0.01)) ** 0, 200.0 * (1 / (1 + 0.02)) ** 1],
                    index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
                ),
                pd.Series(
                    [80.0 * (1 / (1 + 0.01)) ** 0, 180.0 * (1 / (1 + 0.02)) ** 1],
                    index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
                ),
            ]
            result_df = InterpolatedRiskTrajectory.npv_transform(
                df_input.copy(), self.mock_disc_rates
            )
            # Assertions for mock calls
            # Grouping by 'measure', 'metric' (default _grouper)
            pd.testing.assert_series_equal(
                mock_calc_npv.mock_calls[0].args[0],
                pd.Series(
                    [100.0, 200.0],
                    index=pd.Index(
                        [
                            pd.Timestamp("2023-01-01"),
                            pd.Timestamp("2024-01-01"),
                        ],
                        name=DATE_COL_NAME,
                    ),
                    name=("m1", AAI_METRIC_NAME),
                ),
            )
            assert mock_calc_npv.mock_calls[0].args[1] == pd.Timestamp("2023-01-01")
            assert mock_calc_npv.mock_calls[0].args[2] == self.mock_disc_rates
            pd.testing.assert_series_equal(
                mock_calc_npv.mock_calls[1].args[0],
                pd.Series(
                    [80.0, 180.0],
                    index=pd.Index(
                        [
                            pd.Timestamp("2023-01-01"),
                            pd.Timestamp("2024-01-01"),
                        ],
                        name=DATE_COL_NAME,
                    ),
                    name=("m2", AAI_METRIC_NAME),
                ),
            )
            assert mock_calc_npv.mock_calls[1].args[1] == pd.Timestamp("2023-01-01")
            assert mock_calc_npv.mock_calls[1].args[2] == self.mock_disc_rates

            expected_df = pd.DataFrame(
                {
                    DATE_COL_NAME: pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
                    MEASURE_COL_NAME: ["m1", "m1", "m2", "m2"],
                    METRIC_COL_NAME: [
                        AAI_METRIC_NAME,
                        AAI_METRIC_NAME,
                        AAI_METRIC_NAME,
                        AAI_METRIC_NAME,
                    ],
                    RISK_COL_NAME: [
                        100.0 * (1 / (1 + 0.01)) ** 0,
                        200.0 * (1 / (1 + 0.02)) ** 1,
                        80.0 * (1 / (1 + 0.01)) ** 0,
                        180.0 * (1 / (1 + 0.02)) ** 1,
                    ],
                }
            )
            pd.testing.assert_frame_equal(
                result_df.sort_values(DATE_COL_NAME).reset_index(drop=True),
                expected_df.sort_values(DATE_COL_NAME).reset_index(drop=True),
                rtol=1e-6,
            )

    def test_npv_transform_with_group_col(self):
        df_input = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(
                    ["2023-01-01", "2024-01-01", "2023-01-01"]
                ),
                GROUP_COL_NAME: ["G1", "G1", "G2"],
                MEASURE_COL_NAME: ["m1", "m1", "m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME, AAI_METRIC_NAME, AAI_METRIC_NAME],
                RISK_COL_NAME: [100.0, 200.0, 150.0],
            }
        )
        with patch(
            "climada.trajectories.trajectory.RiskTrajectory._calc_npv_cash_flows"
        ) as mock_calc_npv:
            mock_calc_npv.side_effect = [
                # First group G1, m1, aai
                pd.Series(
                    [100.0 * (1 / (1 + 0.01)) ** 0, 200.0 * (1 / (1 + 0.02)) ** 1],
                    index=[pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
                ),
                # Second group G2, m1, aai
                pd.Series(
                    [150.0 * (1 / (1 + 0.01)) ** 0], index=[pd.Timestamp("2023-01-01")]
                ),
            ]
            result_df = InterpolatedRiskTrajectory.npv_transform(
                df_input.copy(), self.mock_disc_rates
            )

            expected_df = pd.DataFrame(
                {
                    DATE_COL_NAME: pd.to_datetime(
                        ["2023-01-01", "2024-01-01", "2023-01-01"]
                    ),
                    GROUP_COL_NAME: ["G1", "G1", "G2"],
                    MEASURE_COL_NAME: ["m1", "m1", "m1"],
                    METRIC_COL_NAME: [
                        AAI_METRIC_NAME,
                        AAI_METRIC_NAME,
                        AAI_METRIC_NAME,
                    ],
                    RISK_COL_NAME: [
                        100.0 * (1 / (1 + 0.01)) ** 0,
                        200.0 * (1 / (1 + 0.02)) ** 1,
                        150.0 * (1 / (1 + 0.01)) ** 0,
                    ],
                }
            )
            pd.testing.assert_frame_equal(
                result_df.sort_values([GROUP_COL_NAME, DATE_COL_NAME]).reset_index(
                    drop=True
                ),
                expected_df.sort_values([GROUP_COL_NAME, DATE_COL_NAME]).reset_index(
                    drop=True
                ),
                rtol=1e-6,
            )

    @patch.object(InterpolatedRiskTrajectory, "_generic_metrics")
    @patch.object(InterpolatedRiskTrajectory, "_date_to_period_agg")
    def test_compute_period_metrics(self, mock_date_to_period, mock_generic_metrics):
        mock_date_to_period.return_value = 42
        mock_generic_metrics.return_value = 46
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        result = rt._compute_period_metrics("name", "method", other_args=5)
        mock_generic_metrics.assert_called_once_with(
            metric_name="name", metric_meth="method", other_args=5
        )
        mock_date_to_period.assert_called_once_with(46, grouper=rt._grouper)
        self.assertEqual(result, 42)

    def test_risk_contributions_post_treatment(self):
        # Create a sample DataFrame
        data = {
            GROUP_COL_NAME: ["All"] * 15,
            DATE_COL_NAME: [
                pd.Period("2023-01-01", freq="Y"),
                pd.Period("2024-01-02", freq="Y"),
                pd.Period("2025-01-02", freq="Y"),
            ]
            * 5,
            MEASURE_COL_NAME: ["measure1"] * 15,
            METRIC_COL_NAME: [
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
            ],
            RISK_COL_NAME: [100, 100, 195, 0, 50, 100, 0, 10, 20, 0, 5, 10, 0, 30, 60],
        }
        df = pd.DataFrame(data)

        # Call the method
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        result_df = rt._risk_contributions_post_treatment(df)

        # Expected output
        expected_data = {
            DATE_COL_NAME: [
                pd.Period("2023-01-01", freq="Y"),
                pd.Period("2024-01-02", freq="Y"),
                pd.Period("2025-01-02", freq="Y"),
            ]
            * 5,
            GROUP_COL_NAME: ["All"] * 15,
            MEASURE_COL_NAME: ["measure1"] * 15,
            METRIC_COL_NAME: [
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_BASE_RISK_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_EXPOSURE_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_HAZARD_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_VULNERABILITY_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
                CONTRIBUTION_INTERACTION_TERM_NAME,
            ],
            RISK_COL_NAME: [100, 100, 100, 0, 50, 150, 0, 10, 30, 0, 5, 15, 0, 30, 90],
        }
        expected_df = pd.DataFrame(expected_data)

        # Assert the result
        pd.testing.assert_frame_equal(
            result_df.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    # --- Test Per Period Risk Aggregation (`_per_period_risk`) ---
    def test_per_period_risk_basic(self):
        df_input = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(
                    ["2023-01-01", "2024-01-01", "2025-01-01", "2023-01-01"]
                ),
                GROUP_COL_NAME: ["All", "All", "All", "GroupB"],
                MEASURE_COL_NAME: ["m1", "m1", "m1", "m1"],
                METRIC_COL_NAME: [
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                    AAI_METRIC_NAME,
                ],
                RISK_COL_NAME: [100.0, 200.0, 300.0, 50.0],
            }
        )
        result_df = InterpolatedRiskTrajectory._date_to_period_agg(
            df_input, grouper=InterpolatedRiskTrajectory._grouper
        )

        expected_df = pd.DataFrame(
            {
                PERIOD_COL_NAME: [
                    "2023-01-01 to 2025-01-01",
                    "2023-01-01 to 2023-01-01",
                ],
                GROUP_COL_NAME: ["All", "GroupB"],
                MEASURE_COL_NAME: ["m1", "m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME, AAI_METRIC_NAME],
                RISK_COL_NAME: [200.0, 50.0],  # 100+200+300 for 'All', 50 for 'GroupB'
            }
        )
        # Sorting for comparison consistency
        pd.testing.assert_frame_equal(
            result_df.sort_values([GROUP_COL_NAME, PERIOD_COL_NAME]).reset_index(
                drop=True
            ),
            expected_df.sort_values([GROUP_COL_NAME, PERIOD_COL_NAME]).reset_index(
                drop=True
            ),
        )

    def test_per_period_risk_multiple_risk_cols(self):
        df_input = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(["2023-01-01", "2024-01-01"]),
                GROUP_COL_NAME: ["All", "All"],
                MEASURE_COL_NAME: ["m1", "m1"],
                METRIC_COL_NAME: ["risk_components", "risk_components"],
                CONTRIBUTION_BASE_RISK_NAME: [10.0, 20.0],
                CONTRIBUTION_EXPOSURE_NAME: [5.0, 8.0],
            }
        )
        result_df = InterpolatedRiskTrajectory._date_to_period_agg(
            df_input,
            grouper=InterpolatedRiskTrajectory._grouper,
            colname=[CONTRIBUTION_BASE_RISK_NAME, CONTRIBUTION_EXPOSURE_NAME],
        )

        expected_df = pd.DataFrame(
            {
                PERIOD_COL_NAME: ["2023-01-01 to 2024-01-01"],
                GROUP_COL_NAME: ["All"],
                MEASURE_COL_NAME: ["m1"],
                METRIC_COL_NAME: ["risk_components"],
                CONTRIBUTION_BASE_RISK_NAME: [15.0],
                CONTRIBUTION_EXPOSURE_NAME: [6.5],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_per_period_risk_non_yearly_intervals(self):
        df_input = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(
                    ["2023-01-01", "2023-02-01", "2023-03-01"]
                ),
                GROUP_COL_NAME: ["All", "All", "All"],
                MEASURE_COL_NAME: ["m1", "m1", "m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME, AAI_METRIC_NAME, AAI_METRIC_NAME],
                RISK_COL_NAME: [10.0, 20.0, 30.0],
            }
        )
        # Test with 'month' time_unit
        result_df_month = InterpolatedRiskTrajectory._date_to_period_agg(
            df_input, grouper=InterpolatedRiskTrajectory._grouper, time_unit="month"
        )
        expected_df_month = pd.DataFrame(
            {
                PERIOD_COL_NAME: ["2023-01-01 to 2023-03-01"],
                GROUP_COL_NAME: ["All"],
                MEASURE_COL_NAME: ["m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME],
                RISK_COL_NAME: [20.0],
            }
        )
        pd.testing.assert_frame_equal(result_df_month, expected_df_month)

        # Introduce a gap for 'month' time_unit
        df_gap = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(
                    ["2023-01-01", "2023-02-01", "2023-04-01"]
                ),  # Gap in March
                GROUP_COL_NAME: ["All", "All", "All"],
                MEASURE_COL_NAME: ["m1", "m1", "m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME, AAI_METRIC_NAME, AAI_METRIC_NAME],
                RISK_COL_NAME: [10.0, 20.0, 40.0],
            }
        )
        result_df_gap = InterpolatedRiskTrajectory._date_to_period_agg(
            df_gap, grouper=InterpolatedRiskTrajectory._grouper, time_unit="month"
        )
        expected_df_gap = pd.DataFrame(
            {
                PERIOD_COL_NAME: [
                    "2023-01-01 to 2023-02-01",
                    "2023-04-01 to 2023-04-01",
                ],
                GROUP_COL_NAME: ["All", "All"],
                MEASURE_COL_NAME: ["m1", "m1"],
                METRIC_COL_NAME: [AAI_METRIC_NAME, AAI_METRIC_NAME],
                RISK_COL_NAME: [15.0, 40.0],
            }
        )
        pd.testing.assert_frame_equal(
            result_df_gap.sort_values(PERIOD_COL_NAME).reset_index(drop=True),
            expected_df_gap.sort_values(PERIOD_COL_NAME).reset_index(drop=True),
        )

    # --- Test Combined Metrics (`per_date_risk_metrics`, `per_period_risk_metrics`) ---

    @patch.object(InterpolatedRiskTrajectory, "aai_metrics")
    @patch.object(InterpolatedRiskTrajectory, "return_periods_metrics")
    @patch.object(InterpolatedRiskTrajectory, "aai_per_group_metrics")
    def test_per_date_risk_metrics_defaults(
        self, mock_aai_per_group, mock_return_periods, mock_aai
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        # Set up mock return values for each method
        mock_aai.return_value = pd.DataFrame(
            {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
        )
        mock_return_periods.return_value = pd.DataFrame(
            {METRIC_COL_NAME: ["rp"], RISK_COL_NAME: [50]}
        )
        mock_aai_per_group.return_value = pd.DataFrame(
            {METRIC_COL_NAME: ["aai_grp"], RISK_COL_NAME: [10]}
        )

        result = rt.per_date_risk_metrics()

        # Assert calls with default arguments
        mock_aai.assert_called_once_with()
        mock_return_periods.assert_called_once_with()
        mock_aai_per_group.assert_called_once_with()

        # Assert concatenation
        expected_df = pd.concat(
            [
                mock_aai.return_value,
                mock_return_periods.return_value,
                mock_aai_per_group.return_value,
            ]
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    @patch.object(InterpolatedRiskTrajectory, "aai_metrics")
    @patch.object(InterpolatedRiskTrajectory, "return_periods_metrics")
    @patch.object(InterpolatedRiskTrajectory, "aai_per_group_metrics")
    def test_per_date_risk_metrics_custom_metrics_and_rps(
        self, mock_aai_per_group, mock_return_periods, mock_aai
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        mock_aai.return_value = pd.DataFrame(
            {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
        )
        mock_return_periods.return_value = pd.DataFrame(
            {METRIC_COL_NAME: ["rp"], RISK_COL_NAME: [50]}
        )

        custom_metrics = [AAI_METRIC_NAME, RETURN_PERIOD_METRIC_NAME]
        result = rt.per_date_risk_metrics(metrics=custom_metrics)

        mock_aai.assert_called_once_with()
        mock_return_periods.assert_called_once_with()
        mock_aai_per_group.assert_not_called()  # Not in custom_metrics

        expected_df = pd.concat(
            [mock_aai.return_value, mock_return_periods.return_value]
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    @patch.object(InterpolatedRiskTrajectory, "per_date_risk_metrics")
    @patch.object(InterpolatedRiskTrajectory, "_date_to_period_agg")
    def test_per_period_risk_metrics(
        self, mock_per_period_risk, mock_per_date_risk_metrics
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        mock_date_df = pd.DataFrame(
            {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
        )
        mock_per_date_risk_metrics.return_value = mock_date_df
        mock_per_period_risk.return_value = pd.DataFrame(
            {PERIOD_COL_NAME: ["P1"], RISK_COL_NAME: [200]}
        )

        test_metrics = [AAI_METRIC_NAME]
        result = rt.per_period_risk_metrics(metrics=test_metrics, time_unit="month")

        mock_per_date_risk_metrics.assert_called_once_with(
            metrics=test_metrics, time_unit="month"
        )
        mock_per_period_risk.assert_called_once_with(
            mock_date_df, grouper=rt._grouper + [UNIT_COL_NAME], time_unit="month"
        )
        pd.testing.assert_frame_equal(result, mock_per_period_risk.return_value)

    # --- Test Plotting Related Methods ---
    # These methods primarily generate data for plotting or call plotting functions.
    # The actual plotting logic (matplotlib.pyplot calls) should be mocked.

    @patch.object(InterpolatedRiskTrajectory, "risk_contributions_metrics")
    def test_calc_waterfall_plot_data(self, mock_risk_contributions_metrics):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.start_date = datetime.date(2023, 1, 1)
        rt.end_date = datetime.date(2025, 1, 1)

        # Mock the return of risk_components_metrics
        mock_risk_contributions_metrics.return_value = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(
                    ["2023-01-01"] * 5
                    + ["2024-01-01"] * 5
                    + ["2025-01-01"] * 5
                    + ["2026-01-01"] * 5
                ),
                METRIC_COL_NAME: [
                    CONTRIBUTION_BASE_RISK_NAME,
                    CONTRIBUTION_EXPOSURE_NAME,
                    CONTRIBUTION_HAZARD_NAME,
                    CONTRIBUTION_VULNERABILITY_NAME,
                    CONTRIBUTION_INTERACTION_TERM_NAME,
                ]
                * 4,
                RISK_COL_NAME: np.arange(20)
                * 1.0,  # Dummy data for different components and dates
            }
        )  # .pivot_table(index=DATE_COL_NAME, columns=METRIC_COL_NAME, values=RISK_COL_NAME)
        # Flattened for simplicity, in reality it's more structured

        result = rt._calc_waterfall_plot_data(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2025, 1, 1),
        )

        mock_risk_contributions_metrics.assert_called_once_with()

        # Expected output should be filtered by date and unstacked
        expected_df = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime(["2024-01-01"] * 5 + ["2025-01-01"] * 5),
                METRIC_COL_NAME: [
                    CONTRIBUTION_BASE_RISK_NAME,
                    CONTRIBUTION_EXPOSURE_NAME,
                    CONTRIBUTION_HAZARD_NAME,
                    CONTRIBUTION_VULNERABILITY_NAME,
                    CONTRIBUTION_INTERACTION_TERM_NAME,
                ]
                * 2,
                RISK_COL_NAME: np.array([5.0, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            }
        ).pivot_table(
            index=DATE_COL_NAME, columns=METRIC_COL_NAME, values=RISK_COL_NAME
        )
        pd.testing.assert_frame_equal(
            result.sort_index(axis=1), expected_df.sort_index(axis=1)
        )  # Sort columns for stable comparison

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.dates.AutoDateLocator")
    @patch("matplotlib.dates.ConciseDateFormatter")
    @patch.object(InterpolatedRiskTrajectory, "_calc_waterfall_plot_data")
    def test_plot_per_date_waterfall(
        self, mock_calc_data, mock_formatter, mock_locator, mock_subplots
    ):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.start_date = datetime.date(2023, 1, 1)
        rt.end_date = datetime.date(2023, 1, 2)

        # Mock matplotlib objects
        mock_ax = Mock()
        mock_fig = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.get_ylim.return_value = (0, 100)  # For ylim scaling

        # Mock data returned by _calc_waterfall_plot_data
        mock_df_data = pd.DataFrame(
            {
                CONTRIBUTION_BASE_RISK_NAME: [10, 10],
                CONTRIBUTION_EXPOSURE_NAME: [2, 3],
                CONTRIBUTION_HAZARD_NAME: [5, 6],
                CONTRIBUTION_VULNERABILITY_NAME: [1, 2],
                CONTRIBUTION_INTERACTION_TERM_NAME: [0.5, 0.7],
            },
            index=pd.period_range(start="2023-01-01", end="2023-01-02", freq="D"),
        )
        mock_calc_data.return_value = mock_df_data

        # Call the method
        fig, ax = rt.plot_time_waterfall()

        # Assertions
        mock_calc_data.assert_called_once_with(
            start_date=datetime.date(2023, 1, 1),
            end_date=datetime.date(2023, 1, 2),
        )
        mock_ax.stackplot.assert_called_once()
        self.assertEqual(
            mock_ax.stackplot.call_args[0][0].tolist(),
            mock_df_data.index.to_timestamp().tolist(),  # type: ignore
        )  # Check x-axis data
        self.assertEqual(
            mock_ax.stackplot.call_args[0][1][0].tolist(),
            mock_df_data[CONTRIBUTION_EXPOSURE_NAME].tolist(),
        )  # Check first stacked data
        mock_ax.set_title.assert_called_once_with(
            "Contributions to change in risk between 2023-01-01 and 2023-01-02 (Average)"
        )
        mock_ax.set_ylabel.assert_called_once_with("Deviation from base risk")
        mock_ax.set_ylim.assert_called_once()  # Check ylim was set
        mock_ax.xaxis.set_major_locator.assert_called_once()
        mock_ax.xaxis.set_major_formatter.assert_called_once()
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)

    @patch("matplotlib.pyplot.subplots")
    @patch.object(InterpolatedRiskTrajectory, "_calc_waterfall_plot_data")
    def test_plot_waterfall(self, mock_calc_data, mock_subplots):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        rt.start_date = datetime.date(2023, 1, 1)
        rt.end_date = datetime.date(2024, 1, 1)

        mock_ax = Mock()
        mock_fig = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.get_ylim.return_value = (0, 100)

        # Mock _calc_waterfall_plot_data to return a DataFrame for two dates,
        # where the second date (end_date) is relevant for plot_waterfall
        start_date = "2023-01-01"
        end_date = "2024-01-01"
        mock_data = pd.DataFrame(
            {
                DATE_COL_NAME: pd.to_datetime([start_date] * 5 + [end_date] * 5),
                METRIC_COL_NAME: [
                    CONTRIBUTION_BASE_RISK_NAME,
                    CONTRIBUTION_EXPOSURE_NAME,
                    CONTRIBUTION_HAZARD_NAME,
                    CONTRIBUTION_VULNERABILITY_NAME,
                    CONTRIBUTION_INTERACTION_TERM_NAME,
                ]
                * 2,
                RISK_COL_NAME: [
                    10,
                    2,
                    5,
                    1,
                    0.5,
                    15,
                    3,
                    7,
                    2,
                    1,
                ],  # values for 2023-01-01 and 2024-01-01
            }
        ).pivot_table(
            index=DATE_COL_NAME, columns=METRIC_COL_NAME, values=RISK_COL_NAME
        )
        mock_calc_data.return_value = mock_data
        # Call the method
        ax = rt.plot_waterfall()

        # Assertions
        mock_calc_data.assert_called_once_with(
            start_date=datetime.date.fromisoformat(start_date),
            end_date=datetime.date.fromisoformat(end_date),
        )
        mock_ax.bar.assert_called_once()
        # Verify the bar arguments are correct for the end_date data
        end_date_data = mock_data.loc[pd.Timestamp(end_date)]
        expected_values = [
            end_date_data[CONTRIBUTION_BASE_RISK_NAME],
            end_date_data[CONTRIBUTION_EXPOSURE_NAME],
            end_date_data[CONTRIBUTION_HAZARD_NAME],
            end_date_data[CONTRIBUTION_VULNERABILITY_NAME],
            end_date_data[CONTRIBUTION_INTERACTION_TERM_NAME],
            end_date_data.sum(),
        ]
        # Compare values passed to bar
        np.testing.assert_allclose(mock_ax.bar.call_args[0][1], expected_values)
        start_date_p = pd.to_datetime(start_date).to_period(rt.time_resolution)
        end_date_p = pd.to_datetime(end_date).to_period(rt.time_resolution)
        mock_ax.set_title.assert_called_once_with(
            f"Evolution of the contributions of risk between {start_date_p} and {end_date_p} (Average impact)"
        )
        mock_ax.set_ylabel.assert_called_once_with("USD")
        mock_ax.set_ylim.assert_called_once()
        mock_ax.tick_params.assert_called_once_with(axis="x", labelrotation=90)
        self.assertEqual(ax, mock_ax)

    # --- Test Private Helper Methods (`_reset_metrics`, `_get_risk_periods`) ---

    def test_reset_metrics(self):
        rt = InterpolatedRiskTrajectory(self.snapshots_list)
        # Set some metrics to non-None values
        rt._eai_metrics = "dummy_eai"  # type:ignore
        rt._aai_metrics = "dummy_aai"  # type:ignore
        rt._reset_metrics()

        for metric in rt.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))

    def test_get_risk_periods(self):
        # Create dummy CalcRiskPeriod mocks with specific dates
        mock_rp1 = Mock()
        mock_rp1.snapshot_start.date = datetime.date(2020, 1, 1)
        mock_rp1.snapshot_end.date = datetime.date(2021, 1, 1)

        mock_rp2 = Mock()
        mock_rp2.snapshot_start.date = datetime.date(2021, 1, 1)
        mock_rp2.snapshot_end.date = datetime.date(2022, 1, 1)

        mock_rp3 = Mock()
        mock_rp3.snapshot_start.date = datetime.date(2022, 1, 1)
        mock_rp3.snapshot_end.date = datetime.date(2023, 1, 1)

        all_risk_periods: list[CalcRiskMetricsPeriod] = [mock_rp1, mock_rp2, mock_rp3]

        # Strict case

        # Test case 1: Full range, all periods included
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2020, 1, 1), datetime.date(2023, 1, 1)
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 1b: More than full range, all periods included
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2018, 1, 1), datetime.date(2024, 1, 1)
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 2: Range including some period
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2021, 1, 1), datetime.date(2023, 1, 1)
        )
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, all_risk_periods[1:])

        # Test case 2: Range including no period
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2021, 6, 1), datetime.date(2022, 6, 1)
        )
        self.assertEqual(len(result), 0)
        self.assertListEqual(result, [])

        # Overlap case

        # Test case 1: Full range, all periods included (should still work)
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods,
            datetime.date(2020, 1, 1),
            datetime.date(2023, 1, 1),
            strict=False,
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 1b: More than full range, all periods included
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods,
            datetime.date(2018, 1, 1),
            datetime.date(2024, 1, 1),
            strict=False,
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 2: Range including some period
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods,
            datetime.date(2021, 1, 1),
            datetime.date(2023, 1, 1),
            strict=False,
        )
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, all_risk_periods[1:])

        # Test case 2: Range including no period but overlap
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods,
            datetime.date(2021, 6, 1),
            datetime.date(2022, 6, 1),
            strict=False,
        )
        self.assertEqual(len(result), 2)
        self.assertListEqual(result, all_risk_periods[1:])

        # Test case 2: Range including no period at all
        result = InterpolatedRiskTrajectory._get_risk_periods(
            all_risk_periods,
            datetime.date(2024, 6, 1),
            datetime.date(2026, 6, 1),
            strict=False,
        )
        self.assertEqual(len(result), 0)
        self.assertListEqual(result, [])


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInterpolatedRiskTrajectory)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
