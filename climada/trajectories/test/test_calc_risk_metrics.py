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

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

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
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.snapshot import Snapshot
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5


@pytest.fixture(scope="module")
def sample_data():
    """Fixture to manage expensive data loading and setup once for the module."""
    present_date = 2020
    future_date = 2025

    # Present Data Setup
    exp_present = Exposures.from_hdf5(EXP_DEMO_H5)
    exp_present.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
    exp_present.gdf["impf_TC"] = 1
    exp_present.gdf[GROUP_ID_COL_NAME] = (
        exp_present.gdf["value"] > exp_present.gdf["value"].mean()
    ) * 1
    haz_present = Hazard.from_hdf5(HAZ_DEMO_H5)
    exp_present.assign_centroids(haz_present, distance="approx")
    impfset_present = ImpactFuncSet([ImpfTropCyclone.from_emanuel_usa()])

    # Future Data Setup
    exp_future = Exposures.from_hdf5(EXP_DEMO_H5)
    n_years = future_date - present_date + 1
    growth = 1.02**n_years
    exp_future.gdf["value"] *= growth
    exp_future.gdf.rename(columns={"impf_": "impf_TC"}, inplace=True)
    exp_future.gdf["impf_TC"] = 1
    exp_future.gdf[GROUP_ID_COL_NAME] = (
        exp_future.gdf["value"] > exp_future.gdf["value"].mean()
    ) * 1
    haz_future = Hazard.from_hdf5(HAZ_DEMO_H5)
    haz_future.intensity *= 1.1
    exp_future.assign_centroids(haz_future, distance="approx")
    impfset_future = ImpactFuncSet(
        [ImpfTropCyclone.from_emanuel_usa(impf_id=1, v_half=60.0)]
    )

    return {
        "snapshots": [
            Snapshot(
                exposure=exp_present,
                hazard=haz_present,
                impfset=impfset_present,
                date=str(present_date),
            ),
            Snapshot(
                exposure=exp_future,
                hazard=haz_future,
                impfset=impfset_future,
                date=str(future_date),
            ),
        ],
        "expected_eai": np.array(
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
        ),
        "expected_aai": np.array([2.88895461e08, 1.69310367e09]),
        "expected_aai_per_group": np.array(
            [2.33513758e08, 5.53817034e07, 1.37114041e09, 3.21963264e08]
        ),
        "expected_rp": np.array(
            [0.0, 0.0, 7.10925472e09, 4.53975437e10, 1.36547014e10, 7.69981714e10]
        ),
    }


class TestCalcRiskMetricsPoints:

    @pytest.fixture(autouse=True)
    def setup_calc(self, sample_data):
        self.snapshots = sample_data["snapshots"]
        self.calc = CalcRiskMetricsPoints(
            self.snapshots,
            impact_computation_strategy=ImpactCalcComputation(),
        )
        self.expected = sample_data

    def test_reset_impact_data(self):
        self.calc._impacts = "A"
        self.calc._eai_gdf = "B"
        self.calc._per_date_eai = "C"
        self.calc._per_date_aai = "D"

        self.calc._reset_impact_data()

        assert self.calc._impacts is None
        assert self.calc._eai_gdf is None
        assert self.calc._per_date_aai is None
        assert self.calc._per_date_eai is None

    def test_set_impact_computation_strategy(self):
        mock_strat = MagicMock(spec=ImpactComputationStrategy)
        self.calc.impact_computation_strategy = mock_strat
        assert self.calc.impact_computation_strategy == mock_strat

    def test_set_impact_computation_strategy_wtype(self):
        with pytest.raises(
            ValueError,
            match="The provided value is not an ImpactComputationStrategy object",
        ):
            self.calc.impact_computation_strategy = "NotAStrategy"

    @patch.object(CalcRiskMetricsPoints, "impact_computation_strategy")
    def test_impacts_arrays(self, mock_impact_compute):
        mock_impact_compute.compute_impacts.side_effect = ["A", "B"]
        results = self.calc.impacts

        expected_calls = [call(s.exposure, s.hazard, s.impfset) for s in self.snapshots]
        mock_impact_compute.compute_impacts.assert_has_calls(expected_calls)
        assert results == ["A", "B"]

    def test_per_date_eai(self):
        np.testing.assert_allclose(
            self.calc.per_date_eai, self.expected["expected_eai"]
        )

    def test_per_date_aai(self):
        np.testing.assert_allclose(
            self.calc.per_date_aai, self.expected["expected_aai"]
        )

    def test_eai_gdf(self):
        result_gdf = self.calc.calc_eai_gdf()
        assert isinstance(result_gdf, pd.DataFrame)
        assert result_gdf.shape[0] == sum(len(s.exposure.gdf) for s in self.snapshots)

        expected_cols = {
            DATE_COL_NAME,
            COORD_ID_COL_NAME,
            GROUP_COL_NAME,
            RISK_COL_NAME,
            METRIC_COL_NAME,
            MEASURE_COL_NAME,
            UNIT_COL_NAME,
        }
        assert expected_cols.issubset(result_gdf.columns)

        np.testing.assert_allclose(
            result_gdf[RISK_COL_NAME].values, self.expected["expected_eai"].flatten()
        )
        assert (result_gdf[METRIC_COL_NAME] == EAI_METRIC_NAME).all()
        assert result_gdf[MEASURE_COL_NAME].iloc[0] == NO_MEASURE_VALUE
        assert (
            result_gdf[UNIT_COL_NAME].iloc[0] == self.snapshots[0].exposure.value_unit
        )
        assert result_gdf[GROUP_COL_NAME].dtype.name == "category"

    def test_calc_aai_metric(self):
        result_df = self.calc.calc_aai_metric()
        assert len(result_df) == len(self.snapshots)
        np.testing.assert_allclose(
            result_df[RISK_COL_NAME].values, self.expected["expected_aai"]
        )
        assert (result_df[METRIC_COL_NAME] == AAI_METRIC_NAME).all()

    def test_calc_aai_per_group_metric(self):
        result_df = self.calc.calc_aai_per_group_metric()
        assert len(result_df) == len(self.snapshots) * len(self.calc._group_id)
        np.testing.assert_allclose(
            result_df[RISK_COL_NAME].values, self.expected["expected_aai_per_group"]
        )

    def test_calc_return_periods_metric(self):
        rps = [20, 50, 100]
        result_df = self.calc.calc_return_periods_metric(rps)
        assert len(result_df) == len(self.snapshots) * len(rps)
        np.testing.assert_allclose(
            result_df[RISK_COL_NAME].values, self.expected["expected_rp"]
        )

        unique_metrics = result_df[METRIC_COL_NAME].unique()
        for rp in rps:
            assert f"rp_{rp}" in unique_metrics

    @patch.object(Snapshot, "apply_measure")
    @patch("climada.trajectories.calc_risk_metrics.CalcRiskMetricsPoints")
    def test_apply_measure(self, mock_calc_class, mock_snap_apply):
        mock_measure = MagicMock(spec=Measure)
        mock_snap_apply.return_value = "MockedSnapshot"

        # We need the class mock to return a mock instance that has a .measure attribute
        mock_instance = MagicMock(spec=CalcRiskMetricsPoints)
        mock_calc_class.return_value = mock_instance

        result = self.calc.apply_measure(mock_measure)

        assert mock_snap_apply.call_count == len(self.snapshots)
        mock_calc_class.assert_called_with(
            ["MockedSnapshot", "MockedSnapshot"], self.calc.impact_computation_strategy
        )
        # Note: In the original test, result.measure was checked.
        # Since we mocked the return of CalcRiskMetricsPoints, we check the mock instance.
        assert result == mock_instance
