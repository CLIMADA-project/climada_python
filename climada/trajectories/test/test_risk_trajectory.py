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

unit tests for risk_trajectory

"""

import datetime
import unittest
from itertools import product
from unittest.mock import Mock, PropertyMock, call, patch

import numpy as np  # For potential NaN/NA comparisons
import pandas as pd

from climada.entity.disc_rates.base import DiscRates

# Assuming your RiskTrajectory class is in a file named 'climada.trajectories.risk_trajectory'
# and the auxiliary classes are in 'climada.trajectories.riskperiod' etc.
# Adjust imports based on your actual file structure.
from climada.trajectories.risk_trajectory import (
    calc_npv_cash_flows,  # standalone function
)
from climada.trajectories.risk_trajectory import (
    DEFAULT_RP,
    POSSIBLE_METRICS,
    RiskTrajectory,
)
from climada.trajectories.riskperiod import (  # ImpactComputationStrategy, # If needed to mock its base class directly
    AllLinearStrategy,
    ImpactCalcComputation,
)
from climada.trajectories.snapshot import Snapshot


class TestRiskTrajectory(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.mock_snapshot1 = Mock(spec=Snapshot)
        self.mock_snapshot1.date = datetime.date(2023, 1, 1)

        self.mock_snapshot2 = Mock(spec=Snapshot)
        self.mock_snapshot2.date = datetime.date(2024, 1, 1)

        self.mock_snapshot3 = Mock(spec=Snapshot)
        self.mock_snapshot3.date = datetime.date(2025, 1, 1)

        self.snapshots_list = [
            self.mock_snapshot1,
            self.mock_snapshot2,
            self.mock_snapshot3,
        ]

        # Mock interpolation strategy and impact computation strategy
        self.mock_interpolation_strategy = Mock(spec=AllLinearStrategy)
        self.mock_impact_computation_strategy = Mock(spec=ImpactCalcComputation)

        # Mock DiscRates if needed for NPV tests
        self.mock_disc_rates = Mock(spec=DiscRates)
        self.mock_disc_rates.years = [2023, 2024, 2025]
        self.mock_disc_rates.rates = [0.01, 0.02, 0.03]  # Example rates

    # --- Test Initialization and Properties ---
    # These tests focus on the __init__ method and property getters/setters.

    ## Test `__init__` method
    def test_init_basic(self):
        # Test basic initialization with defaults
        rt = RiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertIsNone(rt._risk_disc)
        self.assertEqual(rt._interpolation_strategy, self.mock_interpolation_strategy)
        self.assertEqual(
            rt._impact_computation_strategy, self.mock_impact_computation_strategy
        )
        self.assertFalse(rt._risk_period_up_to_date)
        # Check that metrics are reset (initially None)
        for metric in POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))
        self.assertIsNone(rt._all_risk_metrics)

    def test_init_with_custom_params(self):
        # Test initialization with custom parameters
        mock_disc = Mock(spec=DiscRates)
        rt = RiskTrajectory(
            self.snapshots_list,
            interval_freq="MS",
            all_groups_name="CustomAll",
            risk_disc=mock_disc,
            interpolation_strategy=Mock(),
            impact_computation_strategy=Mock(),
        )
        self.assertEqual(rt._interval_freq, "MS")
        self.assertEqual(rt._all_groups_name, "CustomAll")
        self.assertEqual(rt._risk_disc, mock_disc)
        self.assertEqual(rt._risk_transf_cover, 0.5)
        self.assertEqual(rt._risk_transf_attach, 0.1)

    ## Test Properties (`@property` and `@setter`)
    def test_default_rp_getter_setter(self):
        rt = RiskTrajectory(self.snapshots_list)
        self.assertEqual(rt.default_rp, DEFAULT_RP)
        rt.default_rp = [10, 20]
        self.assertEqual(rt.default_rp, [10, 20])
        # Check that setting resets metrics
        rt._return_periods_metrics = "some_data"  # Simulate old data
        rt._all_risk_metrics = "some_data"
        rt.default_rp = [10, 20, 30]
        self.assertIsNone(rt._return_periods_metrics)
        self.assertIsNone(rt._all_risk_metrics)

    def test_default_rp_setter_validation(self):
        rt = RiskTrajectory(self.snapshots_list)
        with self.assertRaises(ValueError):
            rt.default_rp = "not a list"
        with self.assertRaises(ValueError):
            rt.default_rp = [10, "not an int"]

    # --- Test Core Risk Period Calculation (`risk_periods` property and `_calc_risk_periods`) ---
    # This is critical as many other methods depend on it.

    @patch("climada.trajectories.risk_trajectory.CalcRiskPeriod", autospec=True)
    def test_risk_periods_lazy_computation(self, MockCalcRiskPeriod):
        # Test that _calc_risk_periods is called only once, lazily
        rt = RiskTrajectory(
            self.snapshots_list,
            interpolation_strategy=self.mock_interpolation_strategy,
            impact_computation_strategy=self.mock_impact_computation_strategy,
        )
        self.assertFalse(rt._risk_period_up_to_date)
        self.assertIsNone(rt._risk_periods_calculators)

        # First access should trigger calculation
        risk_periods = rt.risk_periods
        MockCalcRiskPeriod.assert_has_calls(
            [
                call(
                    self.mock_snapshot1,
                    self.mock_snapshot2,
                    interval_freq="YS",
                    interpolation_strategy=self.mock_interpolation_strategy,
                    impact_computation_strategy=self.mock_impact_computation_strategy,
                    risk_transf_cover=None,
                    risk_transf_attach=None,
                    calc_residual=True,
                ),
                call(
                    self.mock_snapshot2,
                    self.mock_snapshot3,
                    interval_freq="YS",
                    interpolation_strategy=self.mock_interpolation_strategy,
                    impact_computation_strategy=self.mock_impact_computation_strategy,
                    risk_transf_cover=None,
                    risk_transf_attach=None,
                    calc_residual=True,
                ),
            ]
        )
        self.assertEqual(MockCalcRiskPeriod.call_count, 2)
        self.assertTrue(rt._risk_period_up_to_date)
        self.assertIsInstance(risk_periods, list)
        self.assertEqual(len(risk_periods), 2)  # N-1 periods for N snapshots

        # Second access should not trigger recalculation
        rt.risk_periods
        self.assertEqual(MockCalcRiskPeriod.call_count, 2)  # Still 2 calls

    @patch("climada.trajectories.risk_trajectory.CalcRiskPeriod", autospec=True)
    def test_calc_risk_periods_sorting(self, MockCalcRiskPeriod):
        # Test that snapshots are sorted by date before pairing
        unsorted_snapshots = [
            self.mock_snapshot3,
            self.mock_snapshot1,
            self.mock_snapshot2,
        ]
        rt = RiskTrajectory(unsorted_snapshots)
        # Access the property to trigger calculation
        _ = rt.risk_periods
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

    # --- Test Generic Metric Computation (`_generic_metrics`) ---
    # This is a core internal method and deserves thorough testing.

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    @patch.object(RiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_basic_flow(self, mock_npv_transform, mock_risk_periods):
        rt = RiskTrajectory(self.snapshots_list)
        rt._all_groups_name = "All"  # Ensure default
        rt._risk_disc = self.mock_disc_rates  # For NPV transform check

        # Mock CalcRiskPeriod instances returned by risk_periods property
        mock_calc_period1 = Mock()
        mock_calc_period2 = Mock()
        mock_risk_periods.return_value = [mock_calc_period1, mock_calc_period2]

        # Mock the metric method on CalcRiskPeriod instances
        dates1 = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
        dates2 = [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]
        groups = ["GroupA", "GroupB", pd.NA]
        measures = ["MEAS1", "MEAS2"]
        metrics = ["aai"]
        df1 = pd.DataFrame(
            product(dates1, groups, measures, metrics),
            columns=["date", "group", "measure", "metric"],
        )
        df1["risk"] = np.arange(12) * 100
        df1["group"] = df1["group"].astype("category")
        df2 = pd.DataFrame(
            product(dates2, groups, measures, metrics),
            columns=["date", "group", "measure", "metric"],
        )
        df2["risk"] = np.arange(12) * 100 + 1200
        df2["group"] = df2["group"].astype("category")
        mock_calc_period1.calc_aai_metric.return_value = df1
        mock_calc_period2.calc_aai_metric.return_value = df2

        # Mock npv_transform return value
        mock_npv_transform.return_value = "discounted_df"

        result = rt._generic_metrics(
            npv=True, metric_name="aai", metric_meth="calc_aai_metric"
        )

        # Assertions
        mock_risk_periods.assert_called_once()  # Ensure risk_periods was accessed
        mock_calc_period1.calc_aai_metric.assert_called_once()
        mock_calc_period2.calc_aai_metric.assert_called_once()

        # Check concatenated DataFrame before NPV
        # We need to manually recreate the expected intermediate DataFrame before NPV for assertion
        df3 = pd.DataFrame(
            product(dates1 + dates2, groups, measures, metrics),
            columns=["date", "group", "measure", "metric"],
        )
        df3["risk"] = np.arange(24) * 100
        df3["group"] = df3["group"].astype("category")
        df3["group"] = df3["group"].cat.add_categories(["All"])
        df3["group"] = df3["group"].fillna("All")
        expected_pre_npv_df = df3
        expected_pre_npv_df = expected_pre_npv_df[
            ["group", "date", "measure", "metric", "risk"]
        ]
        # npv_transform should be called with the correctly formatted (concatenated and ordered) DataFrame
        # and the risk_disc attribute
        mock_npv_transform.assert_called_once()
        pd.testing.assert_frame_equal(
            mock_npv_transform.call_args[0][0].reset_index(drop=True),
            expected_pre_npv_df.reset_index(drop=True),
        )
        self.assertEqual(mock_npv_transform.call_args[0][1], self.mock_disc_rates)

        self.assertEqual(result, "discounted_df")  # Final result is from NPV transform

        # Check internal storage
        stored_df = getattr(rt, "_aai_metrics")
        # Assert that the stored DF is the one *before* NPV transformation
        pd.testing.assert_frame_equal(
            stored_df.reset_index(drop=True), expected_pre_npv_df.reset_index(drop=True)
        )

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    @patch.object(RiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_no_npv(self, mock_npv_transform, mock_risk_periods):
        rt = RiskTrajectory(self.snapshots_list)
        # Mock CalcRiskPeriod instances
        mock_calc_period1 = Mock()
        mock_risk_periods.return_value = [mock_calc_period1]
        dates1 = [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")]
        groups = ["GroupA", "GroupB", pd.NA]
        measures = ["MEAS1", "MEAS2"]
        metrics = ["aai"]
        df1 = pd.DataFrame(
            product(groups, dates1, measures, metrics),
            columns=["group", "date", "measure", "metric"],
        )
        df1["risk"] = np.arange(12) * 100
        df1["group"] = df1["group"].astype("category")
        mock_calc_period1.calc_aai_metric.return_value = df1

        result = rt._generic_metrics(
            npv=False, metric_name="aai", metric_meth="calc_aai_metric"
        )

        # Assertions
        mock_npv_transform.assert_not_called()
        expected_df = df1.copy()
        expected_df["group"] = expected_df["group"].cat.add_categories(["All"])
        expected_df["group"] = expected_df["group"].fillna("All")
        pd.testing.assert_frame_equal(result, expected_df)
        pd.testing.assert_frame_equal(getattr(rt, "_aai_metrics"), expected_df)

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    def test_generic_metrics_not_implemented_error(self, mock_risk_periods):
        rt = RiskTrajectory(self.snapshots_list)
        with self.assertRaises(NotImplementedError):
            rt._generic_metrics(metric_name="non_existent", metric_meth="some_method")

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    def test_generic_metrics_value_error_no_name_or_method(self, mock_risk_periods):
        rt = RiskTrajectory(self.snapshots_list)
        with self.assertRaises(ValueError):
            rt._generic_metrics(metric_name=None, metric_meth="some_method")
        with self.assertRaises(ValueError):
            rt._generic_metrics(metric_name="aai", metric_meth=None)

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    @patch.object(RiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_empty_concat_returns_None(
        self, mock_npv_transform, mock_risk_periods
    ):
        rt = RiskTrajectory(self.snapshots_list)
        # Mock CalcRiskPeriod instances return None, mimicking `calc_aai_per_group_metric` possibly
        mock_calc_period1 = Mock()
        mock_calc_period2 = Mock()
        mock_risk_periods.return_value = [mock_calc_period1, mock_calc_period2]
        mock_calc_period1.calc_aai_per_group_metric.return_value = None
        mock_calc_period2.calc_aai_per_group_metric.return_value = None

        result = rt._generic_metrics(
            npv=False,
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
        )
        self.assertIsNone(result)
        self.assertIsNone(getattr(rt, "_aai_per_group_metrics"))  # Should also be None

    @patch.object(RiskTrajectory, "risk_periods", new_callable=PropertyMock)
    @patch.object(RiskTrajectory, "npv_transform", new_callable=Mock)
    def test_generic_metrics_coord_id_handling(
        self, mock_npv_transform, mock_risk_periods
    ):
        rt = RiskTrajectory(self.snapshots_list)
        mock_calc_period = Mock()
        mock_risk_periods.return_value = [mock_calc_period]
        mock_calc_period.calc_eai_gdf.return_value = pd.DataFrame(
            {
                "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
                "group": pd.Categorical([pd.NA, pd.NA]),
                "measure": ["MEAS1", "MEAS1"],
                "metric": ["eai", "eai"],
                "coord_id": [1, 2],
                "risk": [10.0, 20.0],
            }
        )

        result = rt._generic_metrics(
            npv=False, metric_name="eai", metric_meth="calc_eai_gdf"
        )

        expected_df = pd.DataFrame(
            {
                "group": pd.Categorical(["All", "All"]),
                "date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-01")],
                "measure": ["MEAS1", "MEAS1"],
                "metric": ["eai", "eai"],
                "risk": [10.0, 20.0],
                "coord_id": [
                    1,
                    2,
                ],  # This column should remain and be placed at the end before risk if not in front_columns
            }
        )
        # The internal logic reorders columns, ensure it matches
        cols_order = ["group", "date", "measure", "metric", "coord_id", "risk"]
        pd.testing.assert_frame_equal(result[cols_order], expected_df[cols_order])

    # --- Test Specific Metric Methods (e.g., `eai_metrics`, `aai_metrics`) ---
    # These are mostly thin wrappers around _compute_metrics/_generic_metrics.
    # Focus on ensuring they call _compute_metrics with the correct arguments.

    @patch.object(RiskTrajectory, "_compute_metrics")
    def test_eai_metrics(self, mock_compute_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        rt.eai_metrics(npv=True, some_arg="test")
        mock_compute_metrics.assert_called_once_with(
            npv=True, metric_name="eai", metric_meth="calc_eai_gdf", some_arg="test"
        )

    @patch.object(RiskTrajectory, "_compute_metrics")
    def test_aai_metrics(self, mock_compute_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        rt.aai_metrics(npv=False, other_arg=123)
        mock_compute_metrics.assert_called_once_with(
            npv=False, metric_name="aai", metric_meth="calc_aai_metric", other_arg=123
        )

    @patch.object(RiskTrajectory, "_compute_metrics")
    def test_return_periods_metrics(self, mock_compute_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        test_rps = [10, 20]
        rt.return_periods_metrics(test_rps, npv=True, rp_arg="xyz")
        mock_compute_metrics.assert_called_once_with(
            npv=True,
            metric_name="return_periods",
            metric_meth="calc_return_periods_metric",
            return_periods=test_rps,
            rp_arg="xyz",
        )

    @patch.object(RiskTrajectory, "_compute_metrics")
    def test_aai_per_group_metrics(self, mock_compute_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        rt.aai_per_group_metrics(npv=False)
        mock_compute_metrics.assert_called_once_with(
            npv=False,
            metric_name="aai_per_group",
            metric_meth="calc_aai_per_group_metric",
        )

    @patch.object(RiskTrajectory, "_compute_metrics")
    def test_risk_components_metrics(self, mock_compute_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        rt.risk_components_metrics(npv=True)
        mock_compute_metrics.assert_called_once_with(
            npv=True,
            metric_name="risk_components",
            metric_meth="calc_risk_components_metric",
        )

    # --- Test NPV Transformation (`npv_transform` and `calc_npv_cash_flows`) ---

    ## Test `calc_npv_cash_flows` (standalone function)
    def test_calc_npv_cash_flows_no_disc(self):
        cash_flows = pd.Series(
            [100, 200, 300],
            index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
        )
        start_date = datetime.date(2023, 1, 1)
        result = calc_npv_cash_flows(cash_flows, start_date, disc=None)
        # If no disc, it should return the original cash_flows Series
        pd.testing.assert_series_equal(result, cash_flows)

    def test_calc_npv_cash_flows_with_disc(self):
        cash_flows = pd.Series(
            [100, 200, 300],
            index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
        )
        start_date = datetime.date(2023, 1, 1)
        # Using the mock_disc_rates from setUp
        # year 2023: (2023-01-01 - 2023-01-01) days // 365 = 0, factor = (1/(1+0.01))^0 = 1
        # year 2024: (2024-01-01 - 2023-01-01) days // 365 = 1, factor = (1/(1+0.02))^1 = 0.98039215...
        # year 2025: (2025-01-01 - 2023-01-01) days // 365 = 2, factor = (1/(1+0.03))^2 = 0.9425959...
        expected_cash_flows = pd.Series(
            [
                100 * (1 / (1 + 0.01)) ** 0,
                200 * (1 / (1 + 0.02)) ** 1,
                300 * (1 / (1 + 0.03)) ** 2,
            ],
            index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
            name="npv_cash_flow",
        )

        result = calc_npv_cash_flows(cash_flows, start_date, disc=self.mock_disc_rates)
        pd.testing.assert_series_equal(
            result, expected_cash_flows, check_dtype=False, rtol=1e-6
        )

    def test_calc_npv_cash_flows_invalid_index(self):
        cash_flows = pd.Series([100, 200, 300])  # No datetime index
        start_date = datetime.date(2023, 1, 1)
        with self.assertRaises(
            ValueError, msg="cash_flows must be a pandas Series with a datetime index"
        ):
            calc_npv_cash_flows(cash_flows, start_date, disc=self.mock_disc_rates)

    ## Test `npv_transform` (class method)
    def test_npv_transform_no_group_col(self):
        df_input = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
                "measure": ["m1", "m1", "m2", "m2"],
                "metric": ["aai", "aai", "aai", "aai"],
                "risk": [100.0, 200.0, 80.0, 180.0],
            }
        )
        # Mock the internal calc_npv_cash_flows
        with patch(
            "climada.trajectories.risk_trajectory.calc_npv_cash_flows"
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
            result_df = RiskTrajectory.npv_transform(
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
                        name="date",
                    ),
                    name=("m1", "aai"),
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
                        name="date",
                    ),
                    name=("m2", "aai"),
                ),
            )
            assert mock_calc_npv.mock_calls[1].args[1] == pd.Timestamp("2023-01-01")
            assert mock_calc_npv.mock_calls[1].args[2] == self.mock_disc_rates

            expected_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
                    "measure": ["m1", "m1", "m2", "m2"],
                    "metric": ["aai", "aai", "aai", "aai"],
                    "risk": [
                        100.0 * (1 / (1 + 0.01)) ** 0,
                        200.0 * (1 / (1 + 0.02)) ** 1,
                        80.0 * (1 / (1 + 0.01)) ** 0,
                        180.0 * (1 / (1 + 0.02)) ** 1,
                    ],
                }
            )
            pd.testing.assert_frame_equal(
                result_df.sort_values("date").reset_index(drop=True),
                expected_df.sort_values("date").reset_index(drop=True),
                rtol=1e-6,
            )

    def test_npv_transform_with_group_col(self):
        df_input = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2023-01-01"]),
                "group": ["G1", "G1", "G2"],
                "measure": ["m1", "m1", "m1"],
                "metric": ["aai", "aai", "aai"],
                "risk": [100.0, 200.0, 150.0],
            }
        )
        with patch(
            "climada.trajectories.risk_trajectory.calc_npv_cash_flows"
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
            result_df = RiskTrajectory.npv_transform(
                df_input.copy(), self.mock_disc_rates
            )

            expected_df = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2023-01-01", "2024-01-01", "2023-01-01"]),
                    "group": ["G1", "G1", "G2"],
                    "measure": ["m1", "m1", "m1"],
                    "metric": ["aai", "aai", "aai"],
                    "risk": [
                        100.0 * (1 / (1 + 0.01)) ** 0,
                        200.0 * (1 / (1 + 0.02)) ** 1,
                        150.0 * (1 / (1 + 0.01)) ** 0,
                    ],
                }
            )
            pd.testing.assert_frame_equal(
                result_df.sort_values(["group", "date"]).reset_index(drop=True),
                expected_df.sort_values(["group", "date"]).reset_index(drop=True),
                rtol=1e-6,
            )

    # --- Test Per Period Risk Aggregation (`_per_period_risk`) ---
    def test_per_period_risk_basic(self):
        df_input = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-01", "2024-01-01", "2025-01-01", "2023-01-01"]
                ),
                "group": ["All", "All", "All", "GroupB"],
                "measure": ["m1", "m1", "m1", "m1"],
                "metric": ["aai", "aai", "aai", "aai"],
                "risk": [100.0, 200.0, 300.0, 50.0],
            }
        )
        result_df = RiskTrajectory._per_period_risk(df_input)

        expected_df = pd.DataFrame(
            {
                "period": ["2023-01-01 to 2025-01-01", "2023-01-01 to 2023-01-01"],
                "group": ["All", "GroupB"],
                "measure": ["m1", "m1"],
                "metric": ["aai", "aai"],
                "risk": [600.0, 50.0],  # 100+200+300 for 'All', 50 for 'GroupB'
            }
        )
        # Sorting for comparison consistency
        pd.testing.assert_frame_equal(
            result_df.sort_values(["group", "period"]).reset_index(drop=True),
            expected_df.sort_values(["group", "period"]).reset_index(drop=True),
        )

    def test_per_period_risk_multiple_risk_cols(self):
        df_input = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2024-01-01"]),
                "group": ["All", "All"],
                "measure": ["m1", "m1"],
                "metric": ["risk_components", "risk_components"],
                "base risk": [10.0, 20.0],
                "exposure contribution": [5.0, 8.0],
            }
        )
        result_df = RiskTrajectory._per_period_risk(
            df_input, colname=["base risk", "exposure contribution"]
        )

        expected_df = pd.DataFrame(
            {
                "period": ["2023-01-01 to 2024-01-01"],
                "group": ["All"],
                "measure": ["m1"],
                "metric": ["risk_components"],
                "base risk": [30.0],
                "exposure contribution": [13.0],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_per_period_risk_non_yearly_intervals(self):
        df_input = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
                "group": ["All", "All", "All"],
                "measure": ["m1", "m1", "m1"],
                "metric": ["aai", "aai", "aai"],
                "risk": [10.0, 20.0, 30.0],
            }
        )
        # Test with 'month' time_unit
        result_df_month = RiskTrajectory._per_period_risk(df_input, time_unit="month")
        expected_df_month = pd.DataFrame(
            {
                "period": ["2023-01-01 to 2023-03-01"],
                "group": ["All"],
                "measure": ["m1"],
                "metric": ["aai"],
                "risk": [60.0],
            }
        )
        pd.testing.assert_frame_equal(result_df_month, expected_df_month)

        # Introduce a gap for 'month' time_unit
        df_gap = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-01", "2023-02-01", "2023-04-01"]
                ),  # Gap in March
                "group": ["All", "All", "All"],
                "measure": ["m1", "m1", "m1"],
                "metric": ["aai", "aai", "aai"],
                "risk": [10.0, 20.0, 40.0],
            }
        )
        result_df_gap = RiskTrajectory._per_period_risk(df_gap, time_unit="month")
        expected_df_gap = pd.DataFrame(
            {
                "period": ["2023-01-01 to 2023-02-01", "2023-04-01 to 2023-04-01"],
                "group": ["All", "All"],
                "measure": ["m1", "m1"],
                "metric": ["aai", "aai"],
                "risk": [30.0, 40.0],
            }
        )
        pd.testing.assert_frame_equal(
            result_df_gap.sort_values("period").reset_index(drop=True),
            expected_df_gap.sort_values("period").reset_index(drop=True),
        )

    # --- Test Combined Metrics (`per_date_risk_metrics`, `per_period_risk_metrics`) ---

    @patch.object(RiskTrajectory, "aai_metrics")
    @patch.object(RiskTrajectory, "return_periods_metrics")
    @patch.object(RiskTrajectory, "aai_per_group_metrics")
    def test_per_date_risk_metrics_defaults(
        self, mock_aai_per_group, mock_return_periods, mock_aai
    ):
        rt = RiskTrajectory(self.snapshots_list)
        # Set up mock return values for each method
        mock_aai.return_value = pd.DataFrame({"metric": ["aai"], "risk": [100]})
        mock_return_periods.return_value = pd.DataFrame(
            {"metric": ["rp"], "risk": [50]}
        )
        mock_aai_per_group.return_value = pd.DataFrame(
            {"metric": ["aai_grp"], "risk": [10]}
        )

        result = rt.per_date_risk_metrics(npv=False)

        # Assert calls with default arguments
        mock_aai.assert_called_once_with(False)
        mock_return_periods.assert_called_once_with(rt.default_rp, False)
        mock_aai_per_group.assert_called_once_with(False)

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

    @patch.object(RiskTrajectory, "aai_metrics")
    @patch.object(RiskTrajectory, "return_periods_metrics")
    @patch.object(RiskTrajectory, "aai_per_group_metrics")
    def test_per_date_risk_metrics_custom_metrics_and_rps(
        self, mock_aai_per_group, mock_return_periods, mock_aai
    ):
        rt = RiskTrajectory(self.snapshots_list)
        mock_aai.return_value = pd.DataFrame({"metric": ["aai"], "risk": [100]})
        mock_return_periods.return_value = pd.DataFrame(
            {"metric": ["rp"], "risk": [50]}
        )

        custom_metrics = ["aai", "return_periods"]
        custom_rps = [1, 2]
        result = rt.per_date_risk_metrics(
            metrics=custom_metrics, return_periods=custom_rps, npv=True
        )

        mock_aai.assert_called_once_with(True)
        mock_return_periods.assert_called_once_with(custom_rps, True)
        mock_aai_per_group.assert_not_called()  # Not in custom_metrics

        expected_df = pd.concat(
            [mock_aai.return_value, mock_return_periods.return_value]
        )
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected_df.reset_index(drop=True)
        )

    @patch.object(RiskTrajectory, "per_date_risk_metrics")
    @patch.object(RiskTrajectory, "_per_period_risk")
    def test_per_period_risk_metrics(
        self, mock_per_period_risk, mock_per_date_risk_metrics
    ):
        rt = RiskTrajectory(self.snapshots_list)
        mock_date_df = pd.DataFrame({"metric": ["aai"], "risk": [100]})
        mock_per_date_risk_metrics.return_value = mock_date_df
        mock_per_period_risk.return_value = pd.DataFrame(
            {"period": ["P1"], "risk": [200]}
        )

        test_metrics = ["aai"]
        result = rt.per_period_risk_metrics(metrics=test_metrics, time_unit="month")

        mock_per_date_risk_metrics.assert_called_once_with(
            metrics=test_metrics, time_unit="month"
        )
        mock_per_period_risk.assert_called_once_with(mock_date_df, time_unit="month")
        pd.testing.assert_frame_equal(result, mock_per_period_risk.return_value)

    # --- Test Plotting Related Methods ---
    # These methods primarily generate data for plotting or call plotting functions.
    # The actual plotting logic (matplotlib.pyplot calls) should be mocked.

    @patch.object(RiskTrajectory, "risk_components_metrics")
    def test_calc_waterfall_plot_data(self, mock_risk_components_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        rt.start_date = datetime.date(2023, 1, 1)
        rt.end_date = datetime.date(2025, 1, 1)

        # Mock the return of risk_components_metrics
        mock_risk_components_metrics.return_value = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2023-01-01"] * 5
                    + ["2024-01-01"] * 5
                    + ["2025-01-01"] * 5
                    + ["2026-01-01"] * 5
                ),
                "metric": [
                    "base risk",
                    "exposure contribution",
                    "hazard contribution",
                    "vulnerability contribution",
                    "interaction contribution",
                ]
                * 4,
                "risk": np.arange(20)
                * 1.0,  # Dummy data for different components and dates
            }
        )  # .pivot_table(index="date", columns="metric", values="risk")
        # Flattened for simplicity, in reality it's more structured

        result = rt._calc_waterfall_plot_data(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2025, 1, 1),
            npv=False,
        )

        mock_risk_components_metrics.assert_called_once_with(False)

        # Expected output should be filtered by date and unstacked
        expected_df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 5 + ["2025-01-01"] * 5),
                "metric": [
                    "base risk",
                    "exposure contribution",
                    "hazard contribution",
                    "vulnerability contribution",
                    "interaction contribution",
                ]
                * 2,
                "risk": np.array([5.0, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
            }
        ).pivot_table(index="date", columns="metric", values="risk")
        pd.testing.assert_frame_equal(
            result.sort_index(axis=1), expected_df.sort_index(axis=1)
        )  # Sort columns for stable comparison

    @patch("matplotlib.pyplot.subplots")
    @patch("matplotlib.dates.AutoDateLocator")
    @patch("matplotlib.dates.ConciseDateFormatter")
    @patch.object(RiskTrajectory, "_calc_waterfall_plot_data")
    def test_plot_per_date_waterfall(
        self, mock_calc_data, mock_formatter, mock_locator, mock_subplots
    ):
        rt = RiskTrajectory(self.snapshots_list)
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
                "base risk": [10, 12],
                "exposure contribution": [2, 3],
                "hazard contribution": [5, 6],
                "vulnerability contribution": [1, 2],
                "interaction contribution": [0.5, 0.7],
            },
            index=pd.to_datetime(["2023-01-01", "2023-01-02"]),
        )
        mock_calc_data.return_value = mock_df_data

        # Call the method
        fig, ax = rt.plot_per_date_waterfall(
            start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2023, 1, 2)
        )

        # Assertions
        mock_calc_data.assert_called_once_with(
            start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2023, 1, 2)
        )
        mock_ax.stackplot.assert_called_once()
        self.assertEqual(
            mock_ax.stackplot.call_args[0][0].tolist(), mock_df_data.index.tolist()
        )  # Check x-axis data
        self.assertEqual(
            mock_ax.stackplot.call_args[0][1][0].tolist(),
            mock_df_data["base risk"].tolist(),
        )  # Check first stacked data
        mock_ax.set_title.assert_called_once_with(
            "Risk between 2023-01-01 and 2023-01-02 (Average impact)"
        )
        mock_ax.set_ylabel.assert_called_once_with("USD")
        mock_ax.set_ylim.assert_called_once()  # Check ylim was set
        mock_ax.xaxis.set_major_locator.assert_called_once()
        mock_ax.xaxis.set_major_formatter.assert_called_once()
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)

    @patch("matplotlib.pyplot.subplots")
    @patch.object(RiskTrajectory, "_calc_waterfall_plot_data")
    def test_plot_waterfall(self, mock_calc_data, mock_subplots):
        rt = RiskTrajectory(self.snapshots_list)
        rt.start_date = datetime.date(2023, 1, 1)
        rt.end_date = datetime.date(2024, 1, 1)

        mock_ax = Mock()
        mock_fig = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_ax.get_ylim.return_value = (0, 100)

        # Mock _calc_waterfall_plot_data to return a DataFrame for two dates,
        # where the second date (end_date) is relevant for plot_waterfall
        mock_data = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01"] * 5 + ["2024-01-01"] * 5),
                "metric": [
                    "base risk",
                    "exposure contribution",
                    "hazard contribution",
                    "vulnerability contribution",
                    "interaction contribution",
                ]
                * 2,
                "risk": [
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
        ).pivot_table(index="date", columns="metric", values="risk")
        mock_calc_data.return_value = mock_data

        # Call the method
        ax = rt.plot_waterfall(
            start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2024, 1, 1)
        )

        # Assertions
        mock_calc_data.assert_called_once_with(
            start_date=datetime.date(2023, 1, 1), end_date=datetime.date(2024, 1, 1)
        )
        mock_ax.bar.assert_called_once()
        # Verify the bar arguments are correct for the end_date data
        end_date_data = mock_data.loc[pd.Timestamp("2024-01-01")]
        expected_values = [
            end_date_data["base risk"],
            end_date_data["exposure contribution"],
            end_date_data["hazard contribution"],
            end_date_data["vulnerability contribution"],
            end_date_data["interaction contribution"],
            end_date_data.sum(),
        ]
        # Compare values passed to bar
        np.testing.assert_allclose(mock_ax.bar.call_args[0][1], expected_values)

        mock_ax.set_title.assert_called_once_with(
            "Risk at 2023-01-01 and 2024-01-01 (Average impact)"
        )
        mock_ax.set_ylabel.assert_called_once_with("USD")
        mock_ax.set_ylim.assert_called_once()
        mock_ax.tick_params.assert_called_once_with(axis="x", labelrotation=90)
        self.assertEqual(ax, mock_ax)

    # --- Test Private Helper Methods (`_reset_metrics`, `_get_risk_periods`) ---

    def test_reset_metrics(self):
        rt = RiskTrajectory(self.snapshots_list)
        # Set some metrics to non-None values
        rt._eai_metrics = "dummy_eai"
        rt._aai_metrics = "dummy_aai"
        rt._all_risk_metrics = "dummy_all"

        rt._reset_metrics()

        for metric in POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))
        self.assertIsNone(rt._all_risk_metrics)

    def test_get_risk_periods(self):
        # Create dummy CalcRiskPeriod mocks with specific dates
        mock_rp1 = Mock()
        mock_rp1.snapshot0.date = datetime.date(2020, 1, 1)
        mock_rp1.snapshot1.date = datetime.date(2021, 1, 1)

        mock_rp2 = Mock()
        mock_rp2.snapshot0.date = datetime.date(2021, 1, 1)
        mock_rp2.snapshot1.date = datetime.date(2022, 1, 1)

        mock_rp3 = Mock()
        mock_rp3.snapshot0.date = datetime.date(2022, 1, 1)
        mock_rp3.snapshot1.date = datetime.date(2023, 1, 1)

        all_risk_periods = [mock_rp1, mock_rp2, mock_rp3]

        # Test case 1: Full range, all periods included
        result = RiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2020, 1, 1), datetime.date(2023, 1, 1)
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 2: Subset range
        result = RiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2021, 6, 1), datetime.date(2022, 6, 1)
        )
        result = RiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2021, 6, 1), datetime.date(2022, 6, 1)
        )
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)

        # Test case 3: Dates completely outside the periods
        result = RiskTrajectory._get_risk_periods(
            all_risk_periods, datetime.date(2025, 1, 1), datetime.date(2026, 1, 1)
        )
        # rp1: (2025 >= 2020) OR (2026 <= 2021) -> T OR F -> T
        # rp2: (2025 >= 2021) OR (2026 <= 2022) -> T OR F -> T
        # rp3: (2025 >= 2022) OR (2026 <= 2023) -> T OR F -> T
        self.assertEqual(len(result), 3)
        self.assertListEqual(result, all_risk_periods)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
