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
from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.interpolated_trajectory import DEFAULT_RP
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import DEFAULT_ALLGROUP_NAME, RiskTrajectory


class TestRiskTrajectory(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_snapshot1 = MagicMock(spec=Snapshot)
        self.mock_snapshot1.date = datetime.date(2023, 1, 1)

        self.mock_snapshot2 = MagicMock(spec=Snapshot)
        self.mock_snapshot2.date = datetime.date(2024, 1, 1)

        self.mock_snapshot3 = MagicMock(spec=Snapshot)
        self.mock_snapshot3.date = datetime.date(2025, 1, 1)

        self.risk_disc_rates = MagicMock(spec=DiscRates)
        self.risk_disc_rates.years = [2023, 2024, 2025]
        self.risk_disc_rates.rates = [0.01, 0.02, 0.03]  # Example rates

        self.snapshots_list: list[Snapshot] = [
            self.mock_snapshot1,
            self.mock_snapshot2,
            self.mock_snapshot3,
        ]

        self.custom_all_groups_name = "custom"
        self.custom_return_periods = [10, 20]

    def test_init_basic(self):
        rt = RiskTrajectory(self.snapshots_list)
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertIsNone(rt._risk_disc_rates)
        self.assertEqual(rt._all_groups_name, DEFAULT_ALLGROUP_NAME)
        self.assertEqual(rt._return_periods, DEFAULT_RP)
        # Check that metrics are reset (initially None)
        for metric in RiskTrajectory.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))

    def test_init_args(self):
        rt = RiskTrajectory(
            self.snapshots_list,
            return_periods=self.custom_return_periods,
            all_groups_name=self.custom_all_groups_name,
            risk_disc_rates=self.risk_disc_rates,
        )
        self.assertEqual(rt.start_date, self.mock_snapshot1.date)
        self.assertEqual(rt.end_date, self.mock_snapshot3.date)
        self.assertEqual(rt._risk_disc_rates, self.risk_disc_rates)
        self.assertEqual(rt._all_groups_name, self.custom_all_groups_name)
        self.assertEqual(rt._return_periods, self.custom_return_periods)
        self.assertEqual(rt.return_periods, self.custom_return_periods)
        # Check that metrics are reset (initially None)
        for metric in RiskTrajectory.POSSIBLE_METRICS:
            self.assertIsNone(getattr(rt, "_" + metric + "_metrics"))

    @patch.object(RiskTrajectory, "_generic_metrics", new_callable=Mock)
    def test_compute_metrics(self, mock_generic_metrics):
        mock_generic_metrics.return_value = "42"
        rt = RiskTrajectory(self.snapshots_list)
        result = rt._compute_metrics(
            metric_name="dummy_name",
            metric_meth="dummy_meth",
            dummy_kwarg1="A",
            dummy_kwarg2=12,
        )
        mock_generic_metrics.assert_called_once_with(
            metric_name="dummy_name",
            metric_meth="dummy_meth",
            dummy_kwarg1="A",
            dummy_kwarg2=12,
        )
        self.assertEqual(result, "42")

    def test_set_return_periods(self):
        rt = RiskTrajectory(self.snapshots_list)
        with self.assertRaises(ValueError):
            rt.return_periods = "A"
        with self.assertRaises(ValueError):
            rt.return_periods = ["A"]

        rt.return_periods = [1, 2]
        self.assertEqual(rt._return_periods, [1, 2])
        self.assertEqual(rt.return_periods, [1, 2])

    @patch.object(RiskTrajectory, "_reset_metrics", new_callable=Mock)
    def test_set_disc_rates(self, mock_reset_metrics):
        rt = RiskTrajectory(self.snapshots_list)
        mock_reset_metrics.assert_called_once()  # Called during init
        with self.assertRaises(ValueError):
            rt.risk_disc_rates = "A"

        rt.risk_disc_rates = self.risk_disc_rates
        mock_reset_metrics.assert_has_calls([call(), call()])
        self.assertEqual(rt._risk_disc_rates, self.risk_disc_rates)
        self.assertEqual(rt.risk_disc_rates, self.risk_disc_rates)

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
            result_df = RiskTrajectory.npv_transform(
                df_input.copy(), self.risk_disc_rates
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
            assert mock_calc_npv.mock_calls[0].args[2] == self.risk_disc_rates
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
            assert mock_calc_npv.mock_calls[1].args[2] == self.risk_disc_rates

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
            result_df = RiskTrajectory.npv_transform(
                df_input.copy(), self.risk_disc_rates
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

    # --- Test NPV Transformation (`npv_transform` and `calc_npv_cash_flows`) ---

    ## Test `calc_npv_cash_flows` (standalone function)
    def test_calc_npv_cash_flows_no_disc(self):
        cash_flows = pd.Series(
            [100, 200, 300],
            index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
        )
        start_date = datetime.date(2023, 1, 1)
        result = RiskTrajectory._calc_npv_cash_flows(
            cash_flows, start_date, disc_rates=None
        )
        # If no disc, it should return the original cash_flows Series
        pd.testing.assert_series_equal(result, cash_flows)

    def test_calc_npv_cash_flows_with_disc(self):
        cash_flows = pd.Series(
            [100, 200, 300],
            index=pd.period_range(start="2023-01-01", end="2025-01-01", freq="Y"),
        )
        start_date = datetime.date(2023, 1, 1)
        # Using the risk_disc_rates from SetUp

        # year 2023: (2023-01-01 - 2023-01-01) days // 365 = 0, factor = (1/(1+0.01))^0 = 1
        # year 2024: (2024-01-01 - 2023-01-01) days // 365 = 1, factor = (1/(1+0.02))^1 = 0.98039215...
        # year 2025: (2025-01-01 - 2023-01-01) days // 365 = 2, factor = (1/(1+0.03))^2 = 0.9425959...
        expected_cash_flows = pd.Series(
            [
                100 * (1 / (1 + 0.01)) ** 0,
                200 * (1 / (1 + 0.02)) ** 1,
                300 * (1 / (1 + 0.03)) ** 2,
            ],
            index=pd.period_range(start="2023-01-01", end="2025-01-01", freq="Y"),
            name="npv_cash_flow",
        )

        result = RiskTrajectory._calc_npv_cash_flows(
            cash_flows, start_date, disc_rates=self.risk_disc_rates
        )
        pd.testing.assert_series_equal(
            result, expected_cash_flows, check_dtype=False, rtol=1e-6
        )

    def test_calc_npv_cash_flows_invalid_index(self):
        cash_flows = pd.Series([100, 200, 300])  # No datetime index
        start_date = datetime.date(2023, 1, 1)
        with self.assertRaises(
            ValueError, msg="cash_flows must be a pandas Series with a datetime index"
        ):
            RiskTrajectory._calc_npv_cash_flows(
                cash_flows, start_date, disc_rates=self.risk_disc_rates
            )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiskTrajectory)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
