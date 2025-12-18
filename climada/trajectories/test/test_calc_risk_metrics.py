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

import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd

# Assuming these are the necessary imports from climada
from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.entity.measures.base import Measure
from climada.hazard import Hazard
from climada.trajectories.calc_risk_metrics import CalcRiskMetricsPoints
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
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
        mock_CalcRiskMetricPoints.return_value = MagicMock(spec=CalcRiskMetricsPoints)
        mock_snap_apply_measure.return_value = 42
        result = self.calc_risk_metrics_points.apply_measure(self.measure)
        mock_snap_apply_measure.assert_called_with(self.measure)
        mock_CalcRiskMetricPoints.assert_called_with(
            [42, 42],
            self.calc_risk_metrics_points.impact_computation_strategy,
        )
        self.assertEqual(result.measure, self.measure)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCalcRiskMetricsPoints)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
