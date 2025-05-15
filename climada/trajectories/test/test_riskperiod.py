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
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely import Point

# Assuming these are the necessary imports from climada
from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.entity.measures.base import Measure
from climada.hazard import Hazard

# Import the CalcRiskPeriod class and other necessary classes/functions
from climada.trajectories.riskperiod import (
    CalcRiskPeriod,
    ImpactCalcComputation,
    ImpactComputationStrategy,
    InterpolationStrategy,
    LinearInterpolation,
    Snapshot,
)
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5


class TestCalcRiskPeriod_TopLevel(unittest.TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.present_date = 2020
        self.future_date = 2025
        self.exposure_present = Exposures.from_hdf5(EXP_DEMO_H5)
        self.exposure_present.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
        self.exposure_present.gdf["impf_TC"] = 1
        self.exposure_present.gdf["group_id"] = (
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
        self.exposure_future.gdf["group_id"] = (
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
        self.mock_snapshot0 = Snapshot(
            self.exposure_present,
            self.hazard_present,
            self.impfset_present,
            self.present_date,
        )
        self.mock_snapshot1 = Snapshot(
            self.exposure_future,
            self.hazard_future,
            self.impfset_future,
            self.future_date,
        )

        # Create an instance of CalcRiskPeriod
        self.calc_risk_period = CalcRiskPeriod(
            self.mock_snapshot0,
            self.mock_snapshot1,
            interval_freq="AS-JAN",
            interpolation_strategy=LinearInterpolation(),
            impact_computation_strategy=ImpactCalcComputation(),
            # These will have to be tested when implemented
            # risk_transf_attach=0.1,
            # risk_transf_cover=0.9,
            # calc_residual=False
        )

    def test_init(self):
        self.assertEqual(self.calc_risk_period.snapshot0, self.mock_snapshot0)
        self.assertEqual(self.calc_risk_period.snapshot1, self.mock_snapshot1)
        self.assertEqual(self.calc_risk_period.interval_freq, "AS-JAN")
        self.assertEqual(
            self.calc_risk_period.time_points, self.future_date - self.present_date + 1
        )
        self.assertIsInstance(
            self.calc_risk_period.interpolation_strategy, LinearInterpolation
        )
        self.assertIsInstance(
            self.calc_risk_period.impact_computation_strategy, ImpactCalcComputation
        )
        np.testing.assert_array_equal(
            self.calc_risk_period._group_id_E0,
            self.mock_snapshot0.exposure.gdf["group_id"].values,
        )
        np.testing.assert_array_equal(
            self.calc_risk_period._group_id_E1,
            self.mock_snapshot1.exposure.gdf["group_id"].values,
        )
        self.assertIsInstance(self.calc_risk_period.date_idx, pd.DatetimeIndex)
        self.assertEqual(
            len(self.calc_risk_period.date_idx),
            self.future_date - self.present_date + 1,
        )

    def test_set_date_idx_wrong_type(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.date_idx = "A"

    def test_set_date_idx_periods(self):
        new_date_idx = pd.date_range("2023-01-01", "2023-12-01", periods=24)
        self.calc_risk_period.date_idx = new_date_idx
        self.assertEqual(len(self.calc_risk_period.date_idx), 24)

    def test_set_date_idx_freq(self):
        new_date_idx = pd.date_range("2023-01-01", "2023-12-01", freq="MS")
        self.calc_risk_period.date_idx = new_date_idx
        self.assertEqual(len(self.calc_risk_period.date_idx), 12)
        pd.testing.assert_index_equal(
            self.calc_risk_period.date_idx,
            pd.date_range("2023-01-01", "2023-12-01", freq="MS", normalize=True),
        )

    def test_set_time_points(self):
        self.calc_risk_period.time_points = 10
        self.assertEqual(self.calc_risk_period.time_points, 10)
        self.assertEqual(len(self.calc_risk_period.date_idx), 10)
        pd.testing.assert_index_equal(
            self.calc_risk_period.date_idx,
            pd.DatetimeIndex(
                pd.DatetimeIndex(
                    [
                        "2020-01-01",
                        "2020-07-22",
                        "2021-02-10",
                        "2021-09-01",
                        "2022-03-23",
                        "2022-10-12",
                        "2023-05-03",
                        "2023-11-22",
                        "2024-06-12",
                        "2025-01-01",
                    ],
                    name="date",
                )
            ),
        )

    def test_set_time_points_wtype(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.time_points = "1"

    def test_set_interval_freq(self):
        self.calc_risk_period.interval_freq = "MS"
        self.assertEqual(self.calc_risk_period.interval_freq, "MS")
        pd.testing.assert_index_equal(
            self.calc_risk_period.date_idx,
            pd.DatetimeIndex(
                pd.DatetimeIndex(
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
                    name="date",
                )
            ),
        )

    def test_set_interpolation_strategy(self):
        new_interpolation_strategy = MagicMock(spec=InterpolationStrategy)
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

    def test_set_calc_residual_wtype(self):
        with self.assertRaises(ValueError):
            self.calc_risk_period.calc_residual = "A"

    # The computation are tested in the CalcImpactStrategy / InterpolationStrategy tests
    # Here we just make sure that the calling works
    @patch.object(CalcRiskPeriod, "impact_computation_strategy")
    def test_impacts_arrays(self, mock_impact_compute):
        mock_impact_compute.compute_impacts.return_value = 1
        self.assertEqual(self.calc_risk_period.impacts_arrays, 1)
        mock_impact_compute.compute_impacts.assert_called_with(
            self.calc_risk_period.snapshot0,
            self.calc_risk_period.snapshot1,
            self.calc_risk_period.risk_transf_attach,
            self.calc_risk_period.risk_transf_cover,
            self.calc_risk_period.calc_residual,
        )

    @patch.object(CalcRiskPeriod, "interpolation_strategy")
    def test_imp_mats_H0(self, mock_interpolate):
        mock_interpolate.interpolate.return_value = 1
        result = self.calc_risk_period.imp_mats_H0
        self.assertEqual(result, 1)
        mock_interpolate.interpolate.assert_called_with(
            self.calc_risk_period._E0H0,
            self.calc_risk_period._E1H0,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskPeriod, "interpolation_strategy")
    def test_imp_mats_H1(self, mock_interpolate):
        mock_interpolate.interpolate.return_value = 1
        result = self.calc_risk_period.imp_mats_H1
        self.assertEqual(result, 1)
        mock_interpolate.interpolate.assert_called_with(
            self.calc_risk_period._E0H1,
            self.calc_risk_period._E1H1,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskPeriod, "interpolation_strategy")
    def test_imp_mats_E0(self, mock_interpolate):
        mock_interpolate.interpolate.return_value = 1
        result = self.calc_risk_period.imp_mats_E0
        self.assertEqual(result, 1)
        mock_interpolate.interpolate.assert_called_with(
            self.calc_risk_period._E0H0,
            self.calc_risk_period._E0H1,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskPeriod, "interpolation_strategy")
    def test_imp_mats_E1(self, mock_interpolate):
        mock_interpolate.interpolate.return_value = 1
        result = self.calc_risk_period.imp_mats_E1
        self.assertEqual(result, 1)
        mock_interpolate.interpolate.assert_called_with(
            self.calc_risk_period._E1H0,
            self.calc_risk_period._E1H1,
            self.calc_risk_period.time_points,
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_eais", return_value=1)
    def test_per_date_eai_H0(self, mock_calc_per_date_eais):
        result = self.calc_risk_period.per_date_eai_H0
        self.assertEqual(result, 1)
        mock_calc_per_date_eais.assert_called_with(
            self.calc_risk_period.imp_mats_H0,
            self.calc_risk_period.snapshot0.hazard.frequency,
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_eais", return_value=1)
    def test_per_date_eai_H1(self, mock_calc_per_date_eais):
        result = self.calc_risk_period.per_date_eai_H1
        self.assertEqual(result, 1)
        mock_calc_per_date_eais.assert_called_with(
            self.calc_risk_period.imp_mats_H1,
            self.calc_risk_period.snapshot1.hazard.frequency,
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_aais", return_value=1)
    def test_per_date_aai_H0(self, mock_calc_per_date_aais):
        result = self.calc_risk_period.per_date_aai_H0
        self.assertEqual(result, 1)
        mock_calc_per_date_aais.assert_called_with(
            self.calc_risk_period.per_date_eai_H0
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_aais", return_value=1)
    def test_per_date_aai_H1(self, mock_calc_per_date_aais):
        result = self.calc_risk_period.per_date_aai_H1
        self.assertEqual(result, 1)
        mock_calc_per_date_aais.assert_called_with(
            self.calc_risk_period.per_date_eai_H1
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_rps", return_value=1)
    def test_per_date_return_periods_H0(self, mock_calc_per_date_rps):
        result = self.calc_risk_period.per_date_return_periods_H0([10, 50])
        self.assertEqual(result, 1)
        mock_calc_per_date_rps.assert_called_with(
            self.calc_risk_period.imp_mats_H0,
            self.calc_risk_period.snapshot0.hazard.frequency,
            [10, 50],
        )

    @patch.object(CalcRiskPeriod, "calc_per_date_rps", return_value=1)
    def test_per_date_return_periods_H1(self, mock_calc_per_date_rps):
        result = self.calc_risk_period.per_date_return_periods_H1([10, 50])
        self.assertEqual(result, 1)
        mock_calc_per_date_rps.assert_called_with(
            self.calc_risk_period.imp_mats_H1,
            self.calc_risk_period.snapshot1.hazard.frequency,
            [10, 50],
        )

    @patch.object(CalcRiskPeriod, "calc_eai_gdf", return_value=1)
    def test_eai_gdf(self, mock_calc_eai_gdf):
        result = self.calc_risk_period.eai_gdf
        mock_calc_eai_gdf.assert_called_once()
        self.assertEqual(result, 1)

    # Here we mock the impact calc method just to make sure it is rightfully called
    def test_calc_per_date_eais(self):
        results = self.calc_risk_period.calc_per_date_eais(
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
        results = self.calc_risk_period.calc_per_date_aais(
            np.array([[3, 3, 3], [4, 0, 3]])
        )
        np.testing.assert_array_equal(results, np.array([9, 7]))

    def test_calc_freq_curve(self):
        results = self.calc_risk_period.calc_freq_curve(
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
        results = self.calc_risk_period.calc_per_date_rps(
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
        self.calc_risk_period = MagicMock(spec=CalcRiskPeriod)

        # Little trick to bind the mocked object method to the real one
        self.calc_risk_period.calc_eai_gdf = types.MethodType(
            CalcRiskPeriod.calc_eai_gdf, self.calc_risk_period
        )
        self.calc_risk_period.calc_aai_metric = types.MethodType(
            CalcRiskPeriod.calc_aai_metric, self.calc_risk_period
        )
        self.calc_risk_period.calc_aai_per_group_metric = types.MethodType(
            CalcRiskPeriod.calc_aai_per_group_metric, self.calc_risk_period
        )
        self.calc_risk_period.calc_return_periods_metric = types.MethodType(
            CalcRiskPeriod.calc_return_periods_metric, self.calc_risk_period
        )
        self.calc_risk_period.calc_risk_components_metric = types.MethodType(
            CalcRiskPeriod.calc_risk_components_metric, self.calc_risk_period
        )
        self.calc_risk_period.apply_measure = types.MethodType(
            CalcRiskPeriod.apply_measure, self.calc_risk_period
        )

        self.calc_risk_period.per_date_eai_H0 = np.array(
            [[1, 0, 1], [1, 2, 0], [3, 3, 3]]
        )
        self.calc_risk_period.per_date_eai_H1 = np.array(
            [[2, 0, 2], [2, 4, 0], [12, 6, 6]]
        )
        self.calc_risk_period.per_date_aai_H0 = np.array([2, 3, 9])
        self.calc_risk_period.per_date_aai_H1 = np.array([4, 6, 24])
        self.calc_risk_period._prop_H0 = np.array([1, 0.5, 0])
        self.calc_risk_period._prop_H1 = 1.0 - self.calc_risk_period._prop_H0
        self.calc_risk_period.date_idx = pd.DatetimeIndex(
            ["2020-01-01", "2025-01-01", "2030-01-01"], name="date"
        )
        self.calc_risk_period.snapshot1.exposure.gdf = gpd.GeoDataFrame(
            {
                "group_id": [1, 2, 2],
                "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "value": [10, 10, 20],
            }
        )
        self.calc_risk_period._group_id_E0 = np.array([1, 1, 2])
        self.calc_risk_period._group_id_E1 = np.array([1, 2, 2])

        self.calc_risk_period.per_date_return_periods_H0 = MagicMock(
            return_value=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        )
        self.calc_risk_period.per_date_return_periods_H1 = MagicMock(
            return_value=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        )
        self.calc_risk_period.measure = MagicMock(spec=Measure)
        self.calc_risk_period.measure.name = "dummy_measure"

    def test_calc_eai_gdf(self):
        result = self.calc_risk_period.calc_eai_gdf()
        expected_columns = {
            "group",
            "geometry",
            "coord_id",
            "date",
            "risk",
            "metric",
            "measure",
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result["metric"] == "eai").all())
        self.assertTrue((result["measure"] == "dummy_measure").all())

        # Check calculated risk values by coord_id, date
        expected_risk = np.array([1.0, 1.5, 12, 0, 3, 6, 1, 0, 6])
        actual_risk = result["risk"].values
        np.testing.assert_array_almost_equal(expected_risk, actual_risk)

    def test_calc_aai_metric(self):
        result = self.calc_risk_period.calc_aai_metric()
        expected_columns = {
            "group",
            "date",
            "risk",
            "metric",
            "measure",
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result["metric"] == "aai").all())
        self.assertTrue((result["measure"] == "dummy_measure").all())

        # Check calculated risk values by coord_id, date
        expected_risk = np.array([2, 4.5, 24])
        actual_risk = result["risk"].values
        np.testing.assert_array_almost_equal(expected_risk, actual_risk)

    def test_calc_aai_per_group_metric(self):
        result = self.calc_risk_period.calc_aai_per_group_metric()
        expected_columns = {
            "group",
            "date",
            "risk",
            "metric",
            "measure",
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue((result["metric"] == "aai").all())
        self.assertTrue((result["measure"] == "dummy_measure").all())
        # Check calculated risk values by coord_id, date
        expected_risk = np.array([1.0, 2.5, 12.0, 1.0, 2.0, 12])
        actual_risk = result["risk"].values
        np.testing.assert_array_almost_equal(expected_risk, actual_risk)

    def test_calc_return_periods_metric(self):
        result = self.calc_risk_period.calc_return_periods_metric([10, 20, 30])
        expected_columns = {
            "group",
            "date",
            "risk",
            "metric",
            "measure",
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue(all(result["metric"].unique() == ["rp_10", "rp_20", "rp_30"]))
        self.assertTrue((result["measure"] == "dummy_measure").all())

        # Check calculated risk values by rp, date
        expected_risk = np.array([1.0, 22.0, 70.0, 2.0, 27.5, 80, 3, 33, 90])
        actual_risk = result["risk"].values
        np.testing.assert_array_almost_equal(expected_risk, actual_risk)

    def test_calc_risk_components_metric(self):
        result = self.calc_risk_period.calc_risk_components_metric()
        expected_columns = {
            "group",
            "date",
            "risk",
            "metric",
            "measure",
        }
        self.assertTrue(expected_columns.issubset(set(result.columns)))
        self.assertTrue(
            all(
                result["metric"].unique()
                == ["base risk", "delta from exposure", "delta from hazard"]
            )
        )
        self.assertTrue((result["measure"] == "dummy_measure").all())

        # Check calculated risk values by rp, date
        expected_risk = np.array([2.0, 2.0, 2.0, 0.0, 1.0, 7.0, 0, 1.5, 15])
        actual_risk = result["risk"].values
        np.testing.assert_array_almost_equal(expected_risk, actual_risk)

    @patch("climada.trajectories.riskperiod.CalcRiskPeriod")
    def test_apply_measure(self, mock_CalcRiskPeriod):
        mock_CalcRiskPeriod.return_value = MagicMock(spec=CalcRiskPeriod)
        self.calc_risk_period.snapshot0.apply_measure.return_value = 2
        self.calc_risk_period.snapshot1.apply_measure.return_value = 3
        result = self.calc_risk_period.apply_measure(self.calc_risk_period.measure)
        self.assertEqual(result.measure, self.calc_risk_period.measure)
        mock_CalcRiskPeriod.assert_called_with(
            2,
            3,
            self.calc_risk_period.interval_freq,
            self.calc_risk_period.time_points,
            self.calc_risk_period.interpolation_strategy,
            self.calc_risk_period.impact_computation_strategy,
            self.calc_risk_period.risk_transf_attach,
            self.calc_risk_period.risk_transf_cover,
            self.calc_risk_period.calc_residual,
        )


if __name__ == "__main__":
    unittest.main()
