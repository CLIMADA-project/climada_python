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

Unit tests for `calc_risk_metrics.py` .

"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from climada.entity._legacy_measures.base import Measure
from climada.trajectories.calc_risk_metrics import CalcRiskMetricsPoints
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
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
from climada.trajectories.test.conftest import CATEGORIES


@pytest.fixture(scope="module")
def sample_data(snapshot_factory):
    """Fixture to manage expensive data loading and setup once for the module."""
    snap1 = snapshot_factory(date=2020, group_id=CATEGORIES)
    snap2 = snapshot_factory(date=2022, hazard_intensity_factor=2, group_id=CATEGORIES)
    snap3 = snapshot_factory(
        date=2025,
        hazard_intensity_factor=2,
        exposure_value_factor=3,
        group_id=CATEGORIES,
    )
    return {
        "snapshots": [snap1, snap2, snap3],
        "expected_eai": np.array(
            [
                [0.0, 4.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 8.0, 4.0, 6.0, 8.0, 10.0],
                [0.0, 24.0, 12.0, 18.0, 24.0, 30.0],
            ]
        ),
        "expected_aai": np.array([18.0, 36.0, 108.0]),
        "expected_aai_per_group": np.array(
            [11.0, 2.0, 5.0, 22.0, 4.0, 10.0, 66.0, 12.0, 30.0]
        ),
        "expected_rp": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0, 1000.0, 3000.0]),
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
        mock_impact_compute.compute_impacts.side_effect = ["A", "B", "C"]
        results = self.calc.impacts

        expected_calls = [call(s.exposure, s.hazard, s.impfset) for s in self.snapshots]
        mock_impact_compute.compute_impacts.assert_has_calls(expected_calls)
        assert results == ["A", "B", "C"]

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
            ["MockedSnapshot", "MockedSnapshot", "MockedSnapshot"],
            self.calc.impact_computation_strategy,
        )
        # Note: In the original test, result.measure was checked.
        # Since we mocked the return of CalcRiskMetricsPoints, we check the mock instance.
        assert result == mock_instance
