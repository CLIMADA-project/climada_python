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

This modules implements different sparce matrices interpolation approaches.

"""

import types
import unittest
from unittest.mock import MagicMock, call, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from shapely import Point

# Assuming these are the necessary imports from climada
from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.entity.measures.base import Measure
from climada.hazard import Hazard
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    CONTRIBUTION_BASE_RISK_NAME,
    CONTRIBUTION_EXPOSURE_NAME,
    CONTRIBUTION_HAZARD_NAME,
    CONTRIBUTION_INTERACTION_TERM_NAME,
    CONTRIBUTION_VULNERABILITY_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    GROUP_ID_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RISK_COL_NAME,
    UNIT_COL_NAME,
)

# Import the CalcRiskPeriod class and other necessary classes/functions
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.interpolation import (
    AllLinearStrategy,
    InterpolationStrategyBase,
)
from climada.trajectories.riskperiod import (
    CalcRiskMetricsPeriod,
    CalcRiskMetricsPoints,
    calc_freq_curve,
    calc_per_date_aais,
    calc_per_date_eais,
    calc_per_date_rps,
)
from climada.trajectories.snapshot import Snapshot
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5


class TestCalcRiskMetricsPoints(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.present_date = 2020
        self.future_date = 2025
        self.exposure_present = Exposures.from_hdf5(EXP_DEMO_H5)
        self.exposure_present.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
        self.exposure_present.gdf["impf_TC"] = 1
        self.exposure_present.gdf[GROUP_ID_COL_NAME] = (
            self.exposure_present.gdf["value"]
            > self.exposure_present.gdf["value"].mean()
        ) * 1
        self.hazard_present = Hazard.from_hdf5(HAZ_DEMO_H5)
        self.exposure_present.assign_centroids(self.hazard_present, distance="approx")
        self.impfset_present = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])

        self.exposure_future = Exposures.from_hdf5(EXP_DEMO_H5)
        n_years = self.future_date - self.present_date + 1
        growth_rate = 1.02
        growth = growth_rate**n_years
        self.exposure_future.gdf["value"] = self.exposure_future.gdf["value"] * growth
        self.exposure_future.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
        self.exposure_future.gdf["impf_TC"] = 1
        self.exposure_future.gdf[GROUP_ID_COL_NAME] = (
            self.exposure_future.gdf["value"] > self.exposure_future.gdf["value"].mean()
        ) * 1
        self.hazard_future = Hazard.from_hdf5(HAZ_DEMO_H5)
        self.hazard_future.intensity *= 1.1
        self.exposure_future.assign_centroids(self.hazard_future, distance="approx")
        self.impfset_future = ImpactFuncSet(
            [
                ImpfTropCyclone.from_emanuel_usa(impf_id=1, v_half=60.0),
            ]
        )

        self.measure = MagicMock(spec=Measure)
        self.measure.name = "Test Measure"

        # Setup mock return values for measure.apply
        self.measure_exposure = MagicMock(spec=Exposures)
        self.measure_hazard = MagicMock(spec=Hazard)
        self.measure_impfset = MagicMock(spec=ImpactFuncSet)
        self.measure.apply.return_value = (
            self.measure_exposure,
            self.measure_impfset,
            self.measure_hazard,
        )

        # Create mock snapshots
        self.mock_snapshot_start = Snapshot(
            exposure=self.exposure_present,
            hazard=self.hazard_present,
            impfset=self.impfset_present,
            date=self.present_date,
        )
        self.mock_snapshot_end = Snapshot(
            exposure=self.exposure_future,
            hazard=self.hazard_future,
            impfset=self.impfset_future,
            date=self.future_date,
        )

        # Create an instance of CalcRiskPeriod
        self.calc_risk_metrics_points = CalcRiskMetricsPoints(
            [self.mock_snapshot_start, self.mock_snapshot_end],
            impact_computation_strategy=ImpactCalcComputation(),
        )

        self.expected_eai = np.array(
            [
                [
                    8702904.63375606,
                    7870925.19290905,
                    1805021.12653289,
                    3827196.02428828,
                    5815346.97427834,
                    7870925.19290905,
                    7871847.53906951,
                    7870925.19290905,
                    7886487.76136572,
                    7870925.19290905,
                    7876058.84500811,
                    3858228.67061225,
                    8401461.85304853,
                    9210350.19520265,
                    1806363.23553602,
                    6922250.59852326,
                    6711006.70101515,
                    6886568.00391817,
                    6703749.80009753,
                    6704689.17531993,
                    6703401.93516038,
                    6818839.81873556,
                    6716262.5286998,
                    6703369.87656195,
                    6703952.06070945,
                    5678897.05935781,
                    4984034.77073219,
                    6708908.84462217,
                    6702586.9472999,
                    4961843.43826371,
                    5139913.92380089,
                    5255310.96072403,
                    4981705.85074492,
                    4926529.74583162,
                    4973726.6063121,
                    4926015.68274236,
                    4937618.79350358,
                    4926144.19851468,
                    4926015.68274236,
                    9575288.06765627,
                    5100904.22956578,
                    3501325.10900064,
                    5093920.89144773,
                    3505527.05928994,
                    4002552.92232482,
                    3512012.80001039,
                    3514993.26161994,
                    3562009.79687436,
                    3869298.39771648,
                    3509317.94922485,
                ],
                [
                    46651387.10647343,
                    42191612.28496882,
                    14767621.68800634,
                    24849532.38841432,
                    32260334.11128166,
                    42191612.28496882,
                    42196556.46505447,
                    42191612.28496882,
                    42275034.47974126,
                    42191612.28496882,
                    42219130.91253302,
                    24227735.90988531,
                    45035521.54835925,
                    49371517.94999501,
                    14778602.03484606,
                    39909758.65668079,
                    38691846.52720026,
                    39834520.43061425,
                    38650007.36519716,
                    38655423.2682883,
                    38648001.77388126,
                    39313550.93419428,
                    38722148.63941796,
                    38647816.9422419,
                    38651173.48481285,
                    33700748.42359267,
                    30195870.8789255,
                    38679751.48077733,
                    38643303.01755095,
                    30061424.26274527,
                    31140267.73715352,
                    31839402.91317674,
                    30181761.07222111,
                    29847475.57538872,
                    30133418.66577969,
                    29844361.11423809,
                    29914658.78479145,
                    29845139.72952577,
                    29844361.11423809,
                    58012067.61585025,
                    30903926.75151934,
                    23061159.87895984,
                    33550647.3781805,
                    23088835.64296583,
                    26362451.35547444,
                    23131553.38525813,
                    23151183.92499699,
                    23460854.06493051,
                    24271571.95828693,
                    23113803.99527559,
                ],
            ]
        )

        self.expected_aai = np.array([2.88895461e08, 1.69310367e09])
        self.expected_aai_per_group = np.array(
            [2.33513758e08, 5.53817034e07, 1.37114041e09, 3.21963264e08]
        )
        self.expected_return_period_metric = np.array(
            [
                0.00000000e00,
                0.00000000e00,
                7.10925472e09,
                4.53975437e10,
                1.36547014e10,
                7.69981714e10,
            ]
        )

    def test_reset_impact_data(self):
        self.calc_risk_metrics_points._impacts = "A"  # type:ignore
        self.calc_risk_metrics_points._eai_gdf = "B"  # type:ignore
        self.calc_risk_metrics_points._per_date_eai = "C"  # type:ignore
        self.calc_risk_metrics_points._per_date_aai = "D"  # type:ignore
        self.calc_risk_metrics_points._reset_impact_data()
        self.assertIsNone(self.calc_risk_metrics_points._impacts)
        self.assertIsNone(self.calc_risk_metrics_points._eai_gdf)
        self.assertIsNone(self.calc_risk_metrics_points._per_date_aai)
        self.assertIsNone(self.calc_risk_metrics_points._per_date_eai)

    def test_set_impact_computation_strategy(self):
        new_impact_computation_strategy = MagicMock(spec=ImpactComputationStrategy)
        self.calc_risk_metrics_points.impact_computation_strategy = (
            new_impact_computation_strategy
        )
        self.assertEqual(
            self.calc_risk_metrics_points.impact_computation_strategy,
            new_impact_computation_strategy,
        )

    def test_set_impact_computation_strategy_wtype(self):
        with self.assertRaises(ValueError):
            self.calc_risk_metrics_points.impact_computation_strategy = "A"

    @patch.object(CalcRiskMetricsPoints, "impact_computation_strategy")
    def test_impacts_arrays(self, mock_impact_compute):
        mock_impact_compute.compute_impacts.side_effect = ["A", "B"]
        results = self.calc_risk_metrics_points.impacts
        mock_impact_compute.compute_impacts.assert_has_calls(
            [
                call(
                    self.mock_snapshot_start.exposure,
                    self.mock_snapshot_start.hazard,
                    self.mock_snapshot_start.impfset,
                ),
                call(
                    self.mock_snapshot_end.exposure,
                    self.mock_snapshot_end.hazard,
                    self.mock_snapshot_end.impfset,
                ),
            ]
        )
        self.assertEqual(results, ["A", "B"])

    def test_per_date_eai(self):
        np.testing.assert_allclose(
            self.calc_risk_metrics_points.per_date_eai, self.expected_eai
        )

    def test_per_date_aai(self):
        np.testing.assert_allclose(
            self.calc_risk_metrics_points.per_date_aai,
            self.expected_aai,
        )

    def test_eai_gdf(self):
        result_gdf = self.calc_risk_metrics_points.calc_eai_gdf()
        self.assertIsInstance(result_gdf, pd.DataFrame)
        self.assertEqual(
            result_gdf.shape[0],
            len(self.mock_snapshot_start.exposure.gdf)
            + len(self.mock_snapshot_end.exposure.gdf),
        )
        expected_columns = [
            DATE_COL_NAME,
            COORD_ID_COL_NAME,
            GROUP_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
            UNIT_COL_NAME,
        ]
        self.assertTrue(
            all(col in list(result_gdf.columns) for col in expected_columns)
        )
        np.testing.assert_allclose(
            np.array(result_gdf[RISK_COL_NAME].values), self.expected_eai.flatten()
        )
        # Check constants and column transformations
        self.assertEqual(result_gdf[METRIC_COL_NAME].unique(), EAI_METRIC_NAME)
        self.assertEqual(result_gdf[MEASURE_COL_NAME].iloc[0], NO_MEASURE_VALUE)
        self.assertEqual(
            result_gdf[UNIT_COL_NAME].iloc[0],
            self.mock_snapshot_start.exposure.value_unit,
        )
        self.assertEqual(result_gdf[GROUP_COL_NAME].dtype.name, "category")
        self.assertListEqual(
            list(result_gdf[GROUP_COL_NAME].cat.categories),
            list(self.calc_risk_metrics_points._group_id),
        )

    def test_calc_aai_metric(self):
        result_df = self.calc_risk_metrics_points.calc_aai_metric()
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(
            result_df.shape[0], len(self.calc_risk_metrics_points.snapshots)
        )
        expected_columns = [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
            UNIT_COL_NAME,
        ]
        self.assertTrue(all(col in result_df.columns for col in expected_columns))
        np.testing.assert_allclose(
            np.array(result_df[RISK_COL_NAME].values), self.expected_aai
        )
        # Check constants and column transformations
        self.assertEqual(result_df[METRIC_COL_NAME].unique(), AAI_METRIC_NAME)
        self.assertEqual(result_df[MEASURE_COL_NAME].iloc[0], NO_MEASURE_VALUE)
        self.assertEqual(
            result_df[UNIT_COL_NAME].iloc[0],
            self.mock_snapshot_start.exposure.value_unit,
        )
        self.assertEqual(result_df[GROUP_COL_NAME].dtype.name, "category")

    def test_calc_aai_per_group_metric(self):
        result_df = self.calc_risk_metrics_points.calc_aai_per_group_metric()
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(
            result_df.shape[0],
            len(self.calc_risk_metrics_points.snapshots)
            * len(self.calc_risk_metrics_points._group_id),
        )
        expected_columns = [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
            UNIT_COL_NAME,
        ]
        self.assertTrue(all(col in result_df.columns for col in expected_columns))
        np.testing.assert_allclose(
            np.array(result_df[RISK_COL_NAME].values), self.expected_aai_per_group
        )
        # Check constants and column transformations
        self.assertEqual(result_df[METRIC_COL_NAME].unique(), AAI_METRIC_NAME)
        self.assertEqual(result_df[MEASURE_COL_NAME].iloc[0], NO_MEASURE_VALUE)
        self.assertEqual(
            result_df[UNIT_COL_NAME].iloc[0],
            self.mock_snapshot_start.exposure.value_unit,
        )
        self.assertEqual(result_df[GROUP_COL_NAME].dtype.name, "category")
        self.assertListEqual(list(result_df[GROUP_COL_NAME].unique()), [0, 1])

    def test_calc_return_periods_metric(self):
        result_df = self.calc_risk_metrics_points.calc_return_periods_metric(
            [20, 50, 100]
        )
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertEqual(
            result_df.shape[0], len(self.calc_risk_metrics_points.snapshots) * 3
        )
        expected_columns = [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
            UNIT_COL_NAME,
        ]
        self.assertTrue(all(col in result_df.columns for col in expected_columns))
        np.testing.assert_allclose(
            np.array(result_df[RISK_COL_NAME].values),
            self.expected_return_period_metric,
        )
        # Check constants and column transformations
        self.assertListEqual(
            list(result_df[METRIC_COL_NAME].unique()), ["rp_20", "rp_50", "rp_100"]
        )
        self.assertEqual(result_df[MEASURE_COL_NAME].iloc[0], NO_MEASURE_VALUE)
        self.assertEqual(
            result_df[UNIT_COL_NAME].iloc[0],
            self.mock_snapshot_start.exposure.value_unit,
        )
        self.assertEqual(result_df[GROUP_COL_NAME].dtype.name, "category")

    @patch.object(Snapshot, "apply_measure")
    @patch("climada.trajectories.riskperiod.CalcRiskMetricsPoints")
    def test_apply_measure(self, mock_CalcRiskMetricPoints, mock_snap_apply_measure):
        mock_CalcRiskMetricPoints.return_value = MagicMock(spec=CalcRiskMetricsPeriod)
        mock_snap_apply_measure.return_value = 42
        result = self.calc_risk_metrics_points.apply_measure(self.measure)
        mock_snap_apply_measure.assert_called_with(self.measure)
        mock_CalcRiskMetricPoints.assert_called_with(
            [42, 42],
            self.calc_risk_metrics_points.impact_computation_strategy,
        )
        self.assertEqual(result.measure, self.measure)


class TestCalcRiskMetricsPeriod_TopLevel(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.present_date = 2020
        self.future_date = 2025
        self.exposure_present = Exposures.from_hdf5(EXP_DEMO_H5)
        self.exposure_present.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
        self.exposure_present.gdf["impf_TC"] = 1
        self.exposure_present.gdf[GROUP_ID_COL_NAME] = (
            self.exposure_present.gdf["value"] > 500000
        ) * 1
        self.hazard_present = Hazard.from_hdf5(HAZ_DEMO_H5)
        self.exposure_present.assign_centroids(self.hazard_present, distance="approx")
        self.impfset_present = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])

        self.exposure_future = Exposures.from_hdf5(EXP_DEMO_H5)
        n_years = self.future_date - self.present_date + 1
        growth_rate = 1.02
        growth = growth_rate**n_years
        self.exposure_future.gdf["value"] = self.exposure_future.gdf["value"] * growth
        self.exposure_future.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
        self.exposure_future.gdf["impf_TC"] = 1
        self.exposure_future.gdf[GROUP_ID_COL_NAME] = (
            self.exposure_future.gdf["value"] > 500000
        ) * 1
        self.hazard_future = Hazard.from_hdf5(HAZ_DEMO_H5)
        self.hazard_future.intensity *= 1.1
        self.exposure_future.assign_centroids(self.hazard_future, distance="approx")
        self.impfset_future = ImpactFuncSet(
            [
                ImpfTropCyclone.from_emanuel_usa(impf_id=1, v_half=60.0),
            ]
        )

        self.measure = MagicMock(spec=Measure)
        self.measure.name = "Test Measure"

        # Setup mock return values for measure.apply
        self.measure_exposure = MagicMock(spec=Exposures)
        self.measure_hazard = MagicMock(spec=Hazard)
        self.measure_impfset = MagicMock(spec=ImpactFuncSet)
        self.measure.apply.return_value = (
            self.measure_exposure,
            self.measure_impfset,
            self.measure_hazard,
        )

        # Create mock snapshots
        self.mock_snapshot_start = Snapshot(
            exposure=self.exposure_present,
            hazard=self.hazard_present,
            impfset=self.impfset_present,
            date=self.present_date,
        )
        self.mock_snapshot_end = Snapshot(
            exposure=self.exposure_future,
            hazard=self.hazard_future,
            impfset=self.impfset_future,
            date=self.future_date,
        )

        # Create an instance of CalcRiskPeriod
        self.calc_risk_period = CalcRiskMetricsPeriod(
            self.mock_snapshot_start,
            self.mock_snapshot_end,
            time_resolution="Y",
            interpolation_strategy=AllLinearStrategy(),
            impact_computation_strategy=ImpactCalcComputation(),
            # These will have to be tested when implemented
            # risk_transf_attach=0.1,
            # risk_transf_cover=0.9,
            # calc_residual=False
        )

    def test_init(self):
        self.assertEqual(self.calc_risk_period.snapshot_start, self.mock_snapshot_start)
        self.assertEqual(self.calc_risk_period.snapshot_end, self.mock_snapshot_end)
        self.assertEqual(self.calc_risk_period.time_resolution, "Y")
        self.assertEqual(
            self.calc_risk_period.time_points, self.future_date - self.present_date + 1
        )
        self.assertIsInstance(
            self.calc_risk_period.interpolation_strategy, AllLinearStrategy
        )
        self.assertIsInstance(
            self.calc_risk_period.impact_computation_strategy, ImpactCalcComputation
        )
        np.testing.assert_array_equal(
            self.calc_risk_period._group_id_E0,
            self.mock_snapshot_start.exposure.gdf[GROUP_ID_COL_NAME].values,
        )
        np.testing.assert_array_equal(
            self.calc_risk_period._group_id_E1,
            self.mock_snapshot_end.exposure.gdf[GROUP_ID_COL_NAME].values,
        )
        self.assertIsInstance(self.calc_risk_period.date_idx, pd.PeriodIndex)
        self.assertEqual(
            len(self.calc_risk_period.date_idx),
            self.future_date - self.present_date + 1,
        )

    def test_set_date_idx_wrong_type(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.date_idx = "A"

    def test_set_date_idx_periods(self):
        new_date_idx = pd.period_range("2023-01-01", periods=24)
        self.calc_risk_period.date_idx = new_date_idx
        self.assertEqual(len(self.calc_risk_period.date_idx), 24)

    def test_set_date_idx_freq(self):
        new_date_idx = pd.period_range("2023-01-01", "2023-12-01", freq="M")
        self.calc_risk_period.date_idx = new_date_idx
        self.assertEqual(len(self.calc_risk_period.date_idx), 12)
        pd.testing.assert_index_equal(
            self.calc_risk_period.date_idx,
            pd.period_range("2023-01-01", "2023-12-01", freq="M"),
        )

    def test_set_time_resolution(self):
        self.calc_risk_period.time_resolution = "M"
        self.assertEqual(self.calc_risk_period.time_resolution, "M")
        pd.testing.assert_index_equal(
            self.calc_risk_period.date_idx,
            pd.PeriodIndex(
                [
                    "2020-01-01",
                    "2020-02-01",
                    "2020-03-01",
                    "2020-04-01",
                    "2020-05-01",
                    "2020-06-01",
                    "2020-07-01",
                    "2020-08-01",
                    "2020-09-01",
                    "2020-10-01",
                    "2020-11-01",
                    "2020-12-01",
                    "2021-01-01",
                    "2021-02-01",
                    "2021-03-01",
                    "2021-04-01",
                    "2021-05-01",
                    "2021-06-01",
                    "2021-07-01",
                    "2021-08-01",
                    "2021-09-01",
                    "2021-10-01",
                    "2021-11-01",
                    "2021-12-01",
                    "2022-01-01",
                    "2022-02-01",
                    "2022-03-01",
                    "2022-04-01",
                    "2022-05-01",
                    "2022-06-01",
                    "2022-07-01",
                    "2022-08-01",
                    "2022-09-01",
                    "2022-10-01",
                    "2022-11-01",
                    "2022-12-01",
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01",
                    "2023-05-01",
                    "2023-06-01",
                    "2023-07-01",
                    "2023-08-01",
                    "2023-09-01",
                    "2023-10-01",
                    "2023-11-01",
                    "2023-12-01",
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                    "2024-07-01",
                    "2024-08-01",
                    "2024-09-01",
                    "2024-10-01",
                    "2024-11-01",
                    "2024-12-01",
                    "2025-01-01",
                ],
                name=DATE_COL_NAME,
                freq="M",
            ),
        )

    def test_set_interpolation_strategy(self):
        new_interpolation_strategy = MagicMock(spec=InterpolationStrategyBase)
        self.calc_risk_period.interpolation_strategy = new_interpolation_strategy
        self.assertEqual(
            self.calc_risk_period.interpolation_strategy, new_interpolation_strategy
        )

    def test_set_interpolation_strategy_wtype(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.interpolation_strategy = "A"

    def test_set_impact_computation_strategy(self):
        new_impact_computation_strategy = MagicMock(spec=ImpactComputationStrategy)
        self.calc_risk_period.impact_computation_strategy = (
            new_impact_computation_strategy
        )
        self.assertEqual(
            self.calc_risk_period.impact_computation_strategy,
            new_impact_computation_strategy,
        )

    def test_set_impact_computation_strategy_wtype(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.impact_computation_strategy = "A"

    # The computation are tested in the CalcImpactStrategy / InterpolationStrategyBase tests
    # Here we just make sure that the calling works
    @patch.object(CalcRiskMetricsPeriod, "impact_computation_strategy")
    def test_impacts_arrays(self, mock_impact_compute):
        mock_impact_compute.compute_impacts.side_effect = [1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(self.calc_risk_period.E0H0V0, 1)
        self.assertEqual(self.calc_risk_period.E1H0V0, 2)
        self.assertEqual(self.calc_risk_period.E0H1V0, 3)
        self.assertEqual(self.calc_risk_period.E1H1V0, 4)
        self.assertEqual(self.calc_risk_period.E0H0V1, 5)
        self.assertEqual(self.calc_risk_period.E1H0V1, 6)
        self.assertEqual(self.calc_risk_period.E0H1V1, 7)
        self.assertEqual(self.calc_risk_period.E1H1V1, 8)
        mock_impact_compute.compute_impacts.assert_has_calls(
            [
                call(
                    exp,
                    haz,
                    impf,
                )
                for exp, haz, impf in [
                    (
                        self.mock_snapshot_start.exposure,
                        self.mock_snapshot_start.hazard,
                        self.mock_snapshot_start.impfset,
                    ),
                    (
                        self.mock_snapshot_end.exposure,
                        self.mock_snapshot_start.hazard,
                        self.mock_snapshot_start.impfset,
                    ),
                    (
                        self.mock_snapshot_start.exposure,
                        self.mock_snapshot_end.hazard,
                        self.mock_snapshot_start.impfset,
                    ),
                    (
                        self.mock_snapshot_end.exposure,
                        self.mock_snapshot_end.hazard,
                        self.mock_snapshot_start.impfset,
                    ),
                    (
                        self.mock_snapshot_start.exposure,
                        self.mock_snapshot_start.hazard,
                        self.mock_snapshot_end.impfset,
                    ),
                    (
                        self.mock_snapshot_end.exposure,
                        self.mock_snapshot_start.hazard,
                        self.mock_snapshot_end.impfset,
                    ),
                    (
                        self.mock_snapshot_start.exposure,
                        self.mock_snapshot_end.hazard,
                        self.mock_snapshot_end.impfset,
                    ),
                    (
                        self.mock_snapshot_end.exposure,
                        self.mock_snapshot_end.hazard,
                        self.mock_snapshot_end.impfset,
                    ),
                ]
            ]
        )

    @patch.object(CalcRiskMetricsPeriod, "interpolation_strategy")
    def test_imp_mats_H0V0(self, mock_interpolate):
        mock_interpolate.interp_over_exposure_dim.return_value = 1
        result = self.calc_risk_period.imp_mats_H0V0
        self.assertEqual(result, 1)
        mock_interpolate.interp_over_exposure_dim.assert_called_with(
            self.calc_risk_period.E0H0V0.imp_mat,
            self.calc_risk_period.E1H0V0.imp_mat,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskMetricsPeriod, "interpolation_strategy")
    def test_imp_mats_H1V0(self, mock_interpolate):
        mock_interpolate.interp_over_exposure_dim.return_value = 1
        result = self.calc_risk_period.imp_mats_H1V0
        self.assertEqual(result, 1)
        mock_interpolate.interp_over_exposure_dim.assert_called_with(
            self.calc_risk_period.E0H1V0.imp_mat,
            self.calc_risk_period.E1H1V0.imp_mat,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskMetricsPeriod, "interpolation_strategy")
    def test_imp_mats_H0V1(self, mock_interpolate):
        mock_interpolate.interp_over_exposure_dim.return_value = 1
        result = self.calc_risk_period.imp_mats_H0V1
        self.assertEqual(result, 1)
        mock_interpolate.interp_over_exposure_dim.assert_called_with(
            self.calc_risk_period.E0H0V1.imp_mat,
            self.calc_risk_period.E1H0V1.imp_mat,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskMetricsPeriod, "interpolation_strategy")
    def test_imp_mats_H1V1(self, mock_interpolate):
        mock_interpolate.interp_over_exposure_dim.return_value = 1
        result = self.calc_risk_period.imp_mats_H1V1
        self.assertEqual(result, 1)
        mock_interpolate.interp_over_exposure_dim.assert_called_with(
            self.calc_risk_period.E0H1V1.imp_mat,
            self.calc_risk_period.E1H1V1.imp_mat,
            self.calc_risk_period.time_points,
        )

    @patch("climada.trajectories.riskperiod.calc_per_date_eais")
    def test_per_date_eai_H0V0(self, mock_calc_per_date_eais):
        mock_calc_per_date_eais.return_value = 1
        result = self.calc_risk_period.per_date_eai_H0V0

        actual_arg0 = mock_calc_per_date_eais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H0V0

        actual_arg1 = mock_calc_per_date_eais.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_eais")
    def test_per_date_eai_H1V0(self, mock_calc_per_date_eais):
        mock_calc_per_date_eais.return_value = 1
        result = self.calc_risk_period.per_date_eai_H1V0
        actual_arg0 = mock_calc_per_date_eais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H1V0

        actual_arg1 = mock_calc_per_date_eais.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_eais")
    def test_per_date_eai_H0V1(self, mock_calc_per_date_eais):
        mock_calc_per_date_eais.return_value = 1
        result = self.calc_risk_period.per_date_eai_H0V1

        actual_arg0 = mock_calc_per_date_eais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H0V1

        actual_arg1 = mock_calc_per_date_eais.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_eais")
    def test_per_date_eai_H1V1(self, mock_calc_per_date_eais):
        mock_calc_per_date_eais.return_value = 1
        result = self.calc_risk_period.per_date_eai_H1V1
        actual_arg0 = mock_calc_per_date_eais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H1V1

        actual_arg1 = mock_calc_per_date_eais.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_aais")
    def test_per_date_aai_H0V0(self, mock_calc_per_date_aais):
        mock_calc_per_date_aais.return_value = 1
        result = self.calc_risk_period.per_date_aai_H0V0

        actual_arg0 = mock_calc_per_date_aais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.per_date_eai_H0V0
        self.assertEqual(result, 1)
        np.testing.assert_array_equal(actual_arg0, expected_arg0)

    @patch("climada.trajectories.riskperiod.calc_per_date_aais")
    def test_per_date_aai_H1V0(self, mock_calc_per_date_aais):
        mock_calc_per_date_aais.return_value = 1
        result = self.calc_risk_period.per_date_aai_H1V0

        actual_arg0 = mock_calc_per_date_aais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.per_date_eai_H1V0
        self.assertEqual(result, 1)
        np.testing.assert_array_equal(actual_arg0, expected_arg0)

    @patch("climada.trajectories.riskperiod.calc_per_date_aais")
    def test_per_date_aai_H0V1(self, mock_calc_per_date_aais):
        mock_calc_per_date_aais.return_value = 1
        result = self.calc_risk_period.per_date_aai_H0V1

        actual_arg0 = mock_calc_per_date_aais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.per_date_eai_H0V1
        self.assertEqual(result, 1)
        np.testing.assert_array_equal(actual_arg0, expected_arg0)

    @patch("climada.trajectories.riskperiod.calc_per_date_aais")
    def test_per_date_aai_H1V1(self, mock_calc_per_date_aais):
        mock_calc_per_date_aais.return_value = 1
        result = self.calc_risk_period.per_date_aai_H1V1

        actual_arg0 = mock_calc_per_date_aais.call_args[0][0]
        expected_arg0 = self.calc_risk_period.per_date_eai_H1V1
        self.assertEqual(result, 1)
        np.testing.assert_array_equal(actual_arg0, expected_arg0)

    @patch("climada.trajectories.riskperiod.calc_per_date_rps")
    def test_per_date_return_periods_H0V0(self, mock_calc_per_date_rps):
        mock_calc_per_date_rps.return_value = 1
        result = self.calc_risk_period.per_date_return_periods_H0V0([10, 50])

        actual_arg0 = mock_calc_per_date_rps.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H0V0

        actual_arg1 = mock_calc_per_date_rps.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        actual_arg2 = mock_calc_per_date_rps.call_args[0][2]
        expected_arg2 = [10, 50]

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(actual_arg2, expected_arg2)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_rps")
    def test_per_date_return_periods_H1V0(self, mock_calc_per_date_rps):
        mock_calc_per_date_rps.return_value = 1
        result = self.calc_risk_period.per_date_return_periods_H1V0([10, 50])

        actual_arg0 = mock_calc_per_date_rps.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H1V0

        actual_arg1 = mock_calc_per_date_rps.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_end.hazard.frequency

        actual_arg2 = mock_calc_per_date_rps.call_args[0][2]
        expected_arg2 = [10, 50]

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(actual_arg2, expected_arg2)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_rps")
    def test_per_date_return_periods_H0V1(self, mock_calc_per_date_rps):
        mock_calc_per_date_rps.return_value = 1
        result = self.calc_risk_period.per_date_return_periods_H0V1([10, 50])

        actual_arg0 = mock_calc_per_date_rps.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H0V1

        actual_arg1 = mock_calc_per_date_rps.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_start.hazard.frequency

        actual_arg2 = mock_calc_per_date_rps.call_args[0][2]
        expected_arg2 = [10, 50]

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(actual_arg2, expected_arg2)
        self.assertEqual(result, 1)

    @patch("climada.trajectories.riskperiod.calc_per_date_rps")
    def test_per_date_return_periods_H1V1(self, mock_calc_per_date_rps):
        mock_calc_per_date_rps.return_value = 1
        result = self.calc_risk_period.per_date_return_periods_H1V1([10, 50])

        actual_arg0 = mock_calc_per_date_rps.call_args[0][0]
        expected_arg0 = self.calc_risk_period.imp_mats_H1V1

        actual_arg1 = mock_calc_per_date_rps.call_args[0][1]
        expected_arg1 = self.calc_risk_period.snapshot_end.hazard.frequency

        actual_arg2 = mock_calc_per_date_rps.call_args[0][2]
        expected_arg2 = [10, 50]

        assert_sparse_matrix_array_equal(actual_arg0, expected_arg0)
        np.testing.assert_array_equal(actual_arg1, expected_arg1)
        self.assertEqual(actual_arg2, expected_arg2)
        self.assertEqual(result, 1)

    @patch.object(CalcRiskMetricsPeriod, "calc_eai_gdf", return_value=1)
    def test_eai_gdf(self, mock_calc_eai_gdf):
        result = self.calc_risk_period.eai_gdf
        mock_calc_eai_gdf.assert_called_once()
        self.assertEqual(result, 1)

    # Here we mock the impact calc method just to make sure it is rightfully called
    def test_calc_per_date_eais(self):
        results = calc_per_date_eais(
            imp_mats=[
                csr_matrix(
                    [
                        [1, 1, 1],
                        [2, 2, 2],
                    ]
                ),
                csr_matrix(
                    [
                        [2, 0, 1],
                        [2, 0, 2],
                    ]
                ),
            ],
            frequency=np.array([1, 1]),
        )
        np.testing.assert_array_equal(results, np.array([[3, 3, 3], [4, 0, 3]]))

    def test_calc_per_date_aais(self):
        results = calc_per_date_aais(np.array([[3, 3, 3], [4, 0, 3]]))
        np.testing.assert_array_equal(results, np.array([9, 7]))

    def test_calc_freq_curve(self):
        results = calc_freq_curve(
            imp_mat_intrpl=csr_matrix(
                [
                    [0.1, 0, 0],
                    [1, 0, 0],
                    [10, 0, 0],
                ]
            ),
            frequency=np.array([0.5, 0.05, 0.005]),
            return_per=[10, 50, 100],
        )
        np.testing.assert_array_equal(results, np.array([0.55045, 2.575, 5.05]))

    def test_calc_per_date_rps(self):
        base_imp = csr_matrix(
            [
                [0.1, 0, 0],
                [1, 0, 0],
                [10, 0, 0],
            ]
        )
        results = calc_per_date_rps(
            [base_imp, base_imp * 2, base_imp * 4],
            frequency=np.array([0.5, 0.05, 0.005]),
            return_periods=[10, 50, 100],
        )
        np.testing.assert_array_equal(
            results,
            np.array(
                [[0.55045, 2.575, 5.05], [1.1009, 5.15, 10.1], [2.2018, 10.3, 20.2]]
            ),
        )


class TestCalcRiskPeriod_LowLevel(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.calc_risk_period = MagicMock(spec=CalcRiskMetricsPeriod)

        # Little trick to bind the mocked object method to the real one
        self.calc_risk_period.calc_eai = types.MethodType(
            CalcRiskMetricsPeriod.calc_eai, self.calc_risk_period
        )

        self.calc_risk_period.calc_eai_gdf = types.MethodType(
            CalcRiskMetricsPeriod.calc_eai_gdf, self.calc_risk_period
        )
        self.calc_risk_period.calc_aai_metric = types.MethodType(
            CalcRiskMetricsPeriod.calc_aai_metric, self.calc_risk_period
        )

        self.calc_risk_period.calc_aai_per_group_metric = types.MethodType(
            CalcRiskMetricsPeriod.calc_aai_per_group_metric, self.calc_risk_period
        )
        self.calc_risk_period.calc_return_periods_metric = types.MethodType(
            CalcRiskMetricsPeriod.calc_return_periods_metric, self.calc_risk_period
        )
        self.calc_risk_period.calc_risk_components_metric = types.MethodType(
            CalcRiskMetricsPeriod.calc_risk_contributions_metric, self.calc_risk_period
        )
        self.calc_risk_period.apply_measure = types.MethodType(
            CalcRiskMetricsPeriod.apply_measure, self.calc_risk_period
        )

        self.calc_risk_period.per_date_eai_H0V0 = np.array(
            [[1, 0, 1], [1, 2, 0], [3, 3, 3]]
        )
        self.calc_risk_period.per_date_eai_H1V0 = np.array(
            [[2, 0, 2], [2, 4, 0], [12, 6, 6]]
        )
        self.calc_risk_period.per_date_aai_H0V0 = np.array([2, 3, 9])
        self.calc_risk_period.per_date_aai_H1V0 = np.array([4, 6, 24])

        self.calc_risk_period.per_date_eai_H0V1 = np.array(
            [[1, 0, 1], [1, 2, 0], [3, 3, 3]]
        )
        self.calc_risk_period.per_date_eai_H1V1 = np.array(
            [[2, 0, 2], [2, 4, 0], [12, 6, 6]]
        )
        self.calc_risk_period.per_date_aai_H0V1 = np.array([2, 3, 9])
        self.calc_risk_period.per_date_aai_H1V1 = np.array([4, 6, 24])

        self.calc_risk_period.date_idx = pd.PeriodIndex(
            ["2020-01-01", "2025-01-01", "2030-01-01"], name=DATE_COL_NAME, freq="5Y"
        )
        self.calc_risk_period.snapshot_start.exposure.gdf = gpd.GeoDataFrame(
            {
                GROUP_ID_COL_NAME: [1, 2, 2],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "value": [10, 10, 20],
            }
        )
        self.calc_risk_period.snapshot_end.exposure.gdf = gpd.GeoDataFrame(
            {
                GROUP_ID_COL_NAME: [1, 2, 2],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "value": [10, 10, 20],
            }
        )
        self.calc_risk_period.measure = MagicMock(spec=Measure)
        self.calc_risk_period.measure.name = "dummy_measure"

    def test_calc_eai(self):
        # Mock the return values of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.side_effect = [
            "V0_interpolated_data",  # First call (for per_date_eai_V0)
            "V1_interpolated_data",  # Second call (for per_date_eai_V1)
        ]
        # Mock the return value of interp_over_vulnerability_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.return_value = (
            "final_eai_result"
        )

        result = self.calc_risk_period.calc_eai()

        # Assert that interp_over_hazard_dim was called with the correct arguments
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.assert_has_calls(
            [
                call(
                    self.calc_risk_period.per_date_eai_H0V0,
                    self.calc_risk_period.per_date_eai_H1V0,
                ),
                call(
                    self.calc_risk_period.per_date_eai_H0V1,
                    self.calc_risk_period.per_date_eai_H1V1,
                ),
            ]
        )

        # Assert that interp_over_vulnerability_dim was called with the results of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.assert_called_once_with(
            "V0_interpolated_data", "V1_interpolated_data"
        )

        # Assert the final returned value
        self.assertEqual(result, "final_eai_result")

    def test_calc_eai_gdf(self):
        self.calc_risk_period._groups_id = np.array([0])
        expected_risk = np.array([[1.0, 1.5, 12], [0, 3, 6], [1, 0, 6]])
        self.calc_risk_period.per_date_eai = expected_risk
        result = self.calc_risk_period.calc_eai_gdf()
        expected_columns = {
            GROUP_COL_NAME,
            COORD_ID_COL_NAME,
            DATE_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result[METRIC_COL_NAME] == EAI_METRIC_NAME).all())
        self.assertTrue((result[MEASURE_COL_NAME] == "dummy_measure").all())
        # Check calculated risk values by coord_id, date
        actual_risk = result[RISK_COL_NAME].values
        np.testing.assert_allclose(expected_risk.T.flatten(), actual_risk)

    def test_calc_aai_metric(self):
        expected_aai = np.array([2, 4.5, 24])
        self.calc_risk_period.per_date_aai = expected_aai
        self.calc_risk_period._groups_id = np.array([0])
        result = self.calc_risk_period.calc_aai_metric()
        expected_columns = {
            GROUP_COL_NAME,
            DATE_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result[METRIC_COL_NAME] == AAI_METRIC_NAME).all())
        self.assertTrue((result[MEASURE_COL_NAME] == "dummy_measure").all())

        # Check calculated risk values by coord_id, date
        actual_risk = result[RISK_COL_NAME].values
        np.testing.assert_allclose(expected_aai, actual_risk)

    def test_calc_aai_per_group_metric(self):
        self.calc_risk_period._group_id_E0 = np.array([1, 1, 2])
        self.calc_risk_period._group_id_E1 = np.array([2, 2, 2])
        self.calc_risk_period._groups_id = np.array([1, 2])
        self.calc_risk_period.eai_gdf = pd.DataFrame(
            {
                DATE_COL_NAME: pd.PeriodIndex(
                    ["2020-01-01"] * 3 + ["2025-01-01"] * 3 + ["2030-01-01"] * 3,
                    name=DATE_COL_NAME,
                    freq="5Y",
                ),
                COORD_ID_COL_NAME: [0, 1, 2, 0, 1, 2, 0, 1, 2],
                GROUP_COL_NAME: [1, 1, 2, 1, 1, 2, 1, 1, 2],
                RISK_COL_NAME: [2, 3, 4, 5, 6, 7, 8, 9, 10],
                METRIC_COL_NAME: [EAI_METRIC_NAME, EAI_METRIC_NAME, EAI_METRIC_NAME]
                * 3,
                MEASURE_COL_NAME: ["dummy_measure", "dummy_measure", "dummy_measure"]
                * 3,
            }
        )
        self.calc_risk_period.eai_gdf[GROUP_COL_NAME] = self.calc_risk_period.eai_gdf[
            GROUP_COL_NAME
        ].astype("category")
        result = self.calc_risk_period.calc_aai_per_group_metric()
        expected_columns = {
            GROUP_COL_NAME,
            DATE_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result[METRIC_COL_NAME] == AAI_METRIC_NAME).all())
        self.assertTrue((result[MEASURE_COL_NAME] == "dummy_measure").all())
        # Check calculated risk values by coord_id, date
        expected_risk = np.array([5, 5, 6.6, 13.6, 3.4, 27])
        actual_risk = result[RISK_COL_NAME].values
        np.testing.assert_allclose(expected_risk, actual_risk)

    def test_calc_return_periods_metric(self):
        self.calc_risk_period._groups_id = np.array([0])
        self.calc_risk_period.per_date_return_periods_H0V0.return_value = "H0V0"
        self.calc_risk_period.per_date_return_periods_H1V0.return_value = "H1V0"
        self.calc_risk_period.per_date_return_periods_H0V1.return_value = "H0V1"
        self.calc_risk_period.per_date_return_periods_H1V1.return_value = "H1V1"
        # Mock the return values of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.side_effect = [
            "V0_interpolated_data",  # First call (for per_date_rp_V0)
            "V1_interpolated_data",  # Second call (for per_date_rp_V1)
        ]
        # Mock the return value of interp_over_vulnerability_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.return_value = np.array(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        )

        result = self.calc_risk_period.calc_return_periods_metric([10, 20, 30])

        # Assert that interp_over_hazard_dim was called with the correct arguments
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.assert_has_calls(
            [call("H0V0", "H1V0"), call("H0V1", "H1V1")]
        )

        # Assert that interp_over_vulnerability_dim was called with the results of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.assert_called_once_with(
            "V0_interpolated_data", "V1_interpolated_data"
        )

        # Assert the final returned value

        expected_columns = {
            GROUP_COL_NAME,
            DATE_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue(
            all(result[METRIC_COL_NAME].unique() == ["rp_10", "rp_20", "rp_30"])
        )
        self.assertTrue((result[MEASURE_COL_NAME] == "dummy_measure").all())

        # Check calculated risk values by rp, date
        np.testing.assert_allclose(
            result[RISK_COL_NAME].values, np.array([1, 4, 7, 2, 5, 8, 3, 6, 9])
        )

    def test_calc_risk_components_metric(self):
        self.calc_risk_period._groups_id = np.array([0])
        self.calc_risk_period.per_date_aai_H0V0 = np.array([0, 0, 0])
        self.calc_risk_period.per_date_aai_H1V0 = np.array([1, 1, 1])
        self.calc_risk_period.per_date_aai_H0V1 = np.array([2, 2, 2])
        self.calc_risk_period.per_date_aai_H1V1 = np.array([3, 3, 3])
        self.calc_risk_period.per_date_aai = np.array([0, 6 / 4, 3])

        # Mock the return values of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.return_value = np.array(
            [0, 0.5, 1]
        )

        # Mock the return value of interp_over_vulnerability_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.return_value = np.array(
            [0, 1, 2]
        )

        result = self.calc_risk_period.calc_risk_components_metric()

        # Assert that interp_over_hazard_dim was called with the correct arguments
        self.calc_risk_period.interpolation_strategy.interp_over_hazard_dim.assert_called_once_with(
            self.calc_risk_period.per_date_aai_H0V0,
            self.calc_risk_period.per_date_aai_H1V0,
        )

        # Assert that interp_over_vulnerability_dim was called with the results of interp_over_hazard_dim
        self.calc_risk_period.interpolation_strategy.interp_over_vulnerability_dim.assert_called_once_with(
            self.calc_risk_period.per_date_aai_H0V0,
            self.calc_risk_period.per_date_aai_H0V1,
        )

        # Assert the final returned value
        expected_columns = {
            GROUP_COL_NAME,
            DATE_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue(
            all(
                result[METRIC_COL_NAME].unique()
                == [
                    CONTRIBUTION_BASE_RISK_NAME,
                    CONTRIBUTION_EXPOSURE_NAME,
                    CONTRIBUTION_HAZARD_NAME,
                    CONTRIBUTION_VULNERABILITY_NAME,
                    CONTRIBUTION_INTERACTION_TERM_NAME,
                ]
            )
        )
        self.assertTrue((result[MEASURE_COL_NAME] == "dummy_measure").all())

        # Check calculated risk values by rp, date
        np.testing.assert_allclose(
            result[RISK_COL_NAME].values,
            np.array([0, 0, 0, 0, 0, 0, 0, 0.5, 1.0, 0, 1, 2, 0, 0, 0]),
        )

    @patch("climada.trajectories.riskperiod.CalcRiskMetricsPeriod")
    def test_apply_measure(self, mock_CalcRiskPeriod):
        mock_CalcRiskPeriod.return_value = MagicMock(spec=CalcRiskMetricsPeriod)
        self.calc_risk_period.snapshot_start.apply_measure.return_value = 2
        self.calc_risk_period.snapshot_end.apply_measure.return_value = 3
        result = self.calc_risk_period.apply_measure(self.calc_risk_period.measure)
        self.assertEqual(result.measure, self.calc_risk_period.measure)
        mock_CalcRiskPeriod.assert_called_with(
            2,
            3,
            self.calc_risk_period.time_resolution,
            self.calc_risk_period.interpolation_strategy,
            self.calc_risk_period.impact_computation_strategy,
        )


def assert_sparse_matrix_array_equal(expected_array, actual_array):
    """
    Compares two numpy arrays where elements are sparse matrices.
    Uses numpy testing for robust comparison of the sparse matrix internals.
    """
    if len(expected_array) != len(actual_array):
        raise AssertionError(
            f"Expected array length {len(expected_array)} but got {len(actual_array)}"
        )

    for i, (expected_mat, actual_mat) in enumerate(zip(expected_array, actual_array)):
        if not (issparse(expected_mat) and issparse(actual_mat)):
            raise TypeError(f"Element at index {i} is not a sparse matrix.")

        # Robustly compare the underlying data
        np.testing.assert_array_equal(
            expected_mat.data,
            actual_mat.data,
            err_msg=f"Data differs at matrix index {i}",
        )
        np.testing.assert_array_equal(
            expected_mat.indices,
            actual_mat.indices,
            err_msg=f"Indices differ at matrix index {i}",
        )
        np.testing.assert_array_equal(
            expected_mat.indptr,
            actual_mat.indptr,
            err_msg=f"Indptr differs at matrix index {i}",
        )
        # You may also want to assert equal shapes:
        assert (
            expected_mat.shape == actual_mat.shape
        ), f"Shape differs at matrix index {i}"


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(
        TestCalcRiskMetricsPeriod_TopLevel
    )
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestCalcRiskMetricsPoints)
    )
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestCalcRiskPeriod_LowLevel)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
