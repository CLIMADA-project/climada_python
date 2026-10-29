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
from itertools import product
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    RISK_COL_NAME,
)
from climada.trajectories.impact_calc_strat import ImpactCalcComputation
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.static_trajectory import (
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_RP,
    StaticRiskTrajectory,
)
from climada.trajectories.trajectory import RiskTrajectory

# --- Fixtures ---


@pytest.fixture
def mock_snapshots():
    """Provides a list of mock Snapshot objects with sequential dates."""
    snaps = []
    for year in [2023, 2024, 2025]:
        m = MagicMock(spec=Snapshot)
        m.date = datetime.date(year, 1, 1)
        snaps.append(m)
    return snaps


@pytest.fixture
def mock_disc_rates():
    """Provides a mock DiscRates object."""
    dr = MagicMock(spec=DiscRates)
    dr.years = [2023, 2024, 2025]
    dr.rates = [0.01, 0.02, 0.03]
    return dr


@pytest.fixture
def rt_basic(mock_snapshots):
    """A basic StaticRiskTrajectory instance."""
    return StaticRiskTrajectory(mock_snapshots)


@pytest.fixture
def trajectory_metadata():
    """Common metadata for DataFrame generation."""
    return {
        "dates1": [pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")],
        "dates2": [pd.Timestamp("2026-01-01")],
        "groups": ["GroupA", "GroupB", pd.NA],
        "measures": ["MEAS1", "MEAS2"],
        "metrics": [AAI_METRIC_NAME],
    }


@pytest.fixture
def expected_aai_data(trajectory_metadata):
    """Generates the expected AAI DataFrames used for comparison."""
    meta = trajectory_metadata
    all_dates = meta["dates1"] + meta["dates2"]

    df = pd.DataFrame(
        product(meta["groups"], all_dates, meta["measures"], meta["metrics"]),
        columns=[GROUP_COL_NAME, DATE_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME],
    )
    df[RISK_COL_NAME] = np.arange(len(df)) * 100.0

    # Handle Categories and Nulls
    df[GROUP_COL_NAME] = df[GROUP_COL_NAME].astype("category")
    df[GROUP_COL_NAME] = df[GROUP_COL_NAME].cat.add_categories([DEFAULT_ALLGROUP_NAME])
    df[GROUP_COL_NAME] = df[GROUP_COL_NAME].fillna(DEFAULT_ALLGROUP_NAME)

    cols = [
        DATE_COL_NAME,
        GROUP_COL_NAME,
        MEASURE_COL_NAME,
        METRIC_COL_NAME,
        RISK_COL_NAME,
    ]
    return df[cols]


@pytest.fixture
def mock_components():
    """Provides standard CLIMADA mock objects."""
    snaps = [
        MagicMock(spec=Snapshot, date=datetime.date(2023 + i, 1, 1)) for i in range(3)
    ]
    strat = MagicMock(spec=ImpactCalcComputation)
    dr = MagicMock(
        spec=DiscRates, years=[2023, 2024, 2025, 2026], rates=[0.01, 0.02, 0.03, 0.04]
    )
    return {"snaps": snaps, "strat": strat, "disc_rates": dr}


# --- Pure RiskTrajectory Tests ---


def test_init_basic(rt_basic, mock_snapshots):
    assert rt_basic.start_date == mock_snapshots[0].date
    assert rt_basic.end_date == mock_snapshots[-1].date
    assert rt_basic._risk_disc_rates is None
    assert rt_basic._all_groups_name == DEFAULT_ALLGROUP_NAME
    assert rt_basic._return_periods == DEFAULT_RP

    for metric in StaticRiskTrajectory.POSSIBLE_METRICS:
        assert getattr(rt_basic, f"_{metric}_metrics") is None


def test_init_args(mock_snapshots, mock_disc_rates):
    custom_rp = [10, 20]
    rt = StaticRiskTrajectory(
        mock_snapshots,
        return_periods=custom_rp,
        risk_disc_rates=mock_disc_rates,
    )
    assert rt._risk_disc_rates == mock_disc_rates
    assert rt.return_periods == custom_rp


# --- Property & Setter Tests ---


def test_set_return_periods(rt_basic):
    with pytest.raises(ValueError):
        rt_basic.return_periods = "A"

    rt_basic.return_periods = [1, 2]
    assert rt_basic.return_periods == [1, 2]


def test_set_disc_rates(rt_basic, mock_disc_rates):
    # Mock the reset_metrics method on the instance
    with patch.object(rt_basic, "_reset_metrics", wraps=rt_basic._reset_metrics) as spy:
        with pytest.raises(ValueError):
            rt_basic.risk_disc_rates = "A"

        rt_basic.risk_disc_rates = mock_disc_rates
        # Once in __init__, once in setter
        assert spy.call_count == 1
        assert rt_basic.risk_disc_rates == mock_disc_rates


# --- NPV Transformation Tests ---


def test_npv_transform_no_group_col(mock_disc_rates):
    df_input = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
            "measure": ["m1", "m1", "m2", "m2"],
            "metric": [AAI_METRIC_NAME] * 4,
            "risk": [100.0, 200.0, 80.0, 180.0],
        }
    )

    with patch(
        "climada.trajectories.trajectory.RiskTrajectory._calc_npv_cash_flows"
    ) as mock_calc:
        # Side effects to simulate discounted values
        mock_calc.side_effect = [
            pd.Series(
                [99.0, 196.0], index=pd.to_datetime(["2023-01-01", "2024-01-01"])
            ),
            pd.Series(
                [79.2, 176.4], index=pd.to_datetime(["2023-01-01", "2024-01-01"])
            ),
        ]

        _ = RiskTrajectory.npv_transform(df_input.copy(), mock_disc_rates)

        # Check calls: Grouping should happen by (measure, metric)
        assert mock_calc.call_count == 2
        # Verify first group args
        args, _ = mock_calc.call_args_list[0]
        assert args[1] == pd.Timestamp("2023-01-01")
        assert args[2] == mock_disc_rates


def test_calc_npv_cash_flows_logic(mock_disc_rates):
    """Standalone test for the math inside _calc_npv_cash_flows."""
    cash_flows = pd.Series(
        [100, 200, 300],
        index=pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"]),
    )
    start_date = datetime.date(2023, 1, 1)

    # NPV Factor: Product[ (1 / (1 + rate_i))]
    # For a constant rate or 0.01
    # 2023: (1/1.01)^0 = 1.0 -> 100
    # 2024: (1/1.01)^1 = 0.99099... -> 198.019...
    # 2025: (1/1.01)^2 = 0.98029... -> 294.088...

    result = RiskTrajectory._calc_npv_cash_flows(
        cash_flows, start_date, mock_disc_rates
    )
    assert result.iloc[0] == pytest.approx(100.0)
    assert result.iloc[1] == pytest.approx(200 / (1.02))
    assert result.iloc[2] == pytest.approx(300 * (1 / 1.02) * (1 / 1.03))


def test_calc_npv_cash_flows_invalid_index(mock_disc_rates):
    cash_flows = pd.Series([100, 200])  # No datetime index
    with pytest.raises(ValueError, match="PeriodIndex or DatetimeIndex"):
        RiskTrajectory._calc_npv_cash_flows(
            cash_flows, datetime.date(2023, 1, 1), mock_disc_rates
        )


# ---- StaticRiskTrajectory tests ---

# ---  Metric Computation Tests   ---


def test_compute_metrics(rt_basic):
    with patch.object(
        StaticRiskTrajectory, "_generic_metrics", return_value="42"
    ) as mock_generic:
        result = rt_basic._compute_metrics(
            metric_name="dummy", metric_meth="meth", arg1="A", arg2=12
        )

        mock_generic.assert_called_once_with(
            metric_name="dummy", metric_meth="meth", arg1="A", arg2=12
        )
        assert result == "42"


def test_init_basic_static(mock_components):
    # Patch the calculator class used inside __init__
    with patch(
        "climada.trajectories.static_trajectory.CalcRiskMetricsPoints", autospec=True
    ) as mock_calc_cls:
        rt = StaticRiskTrajectory(
            mock_components["snaps"],
            impact_computation_strategy=mock_components["strat"],
        )

        mock_calc_cls.assert_called_once_with(
            mock_components["snaps"],
            impact_computation_strategy=mock_components["strat"],
        )
        assert rt.start_date == mock_components["snaps"][0].date


def test_set_impact_strategy_resets(mock_components):
    rt = StaticRiskTrajectory(mock_components["snaps"])
    with patch.object(rt, "_reset_metrics", wraps=rt._reset_metrics) as spy_reset:
        new_strat = ImpactCalcComputation()
        rt.impact_computation_strategy = new_strat

        assert rt.impact_computation_strategy == new_strat
        # Called once in init, once in setter
        assert spy_reset.call_count == 1


# --- Logic & Metric Tests ---


def test_generic_metrics_caching_and_npv(mock_components, expected_aai_data):
    """Tests the complex logic of _generic_metrics including NPV transform and caching."""
    rt = StaticRiskTrajectory(
        mock_components["snaps"], risk_disc_rates=mock_components["disc_rates"]
    )

    # Mock the internal calculator's method
    mock_calc = MagicMock()
    mock_calc.calc_aai_metric.return_value = expected_aai_data
    rt._risk_metrics_calculators = mock_calc

    # Mock NPV transform to return a modified version
    npv_data = expected_aai_data.copy()
    npv_data[RISK_COL_NAME] *= 0.9
    with patch.object(rt, "npv_transform", return_value=npv_data) as mock_npv:

        # First call
        result = rt._generic_metrics(AAI_METRIC_NAME, "calc_aai_metric")

        mock_calc.calc_aai_metric.assert_called_once()
        mock_npv.assert_called_once()
        pd.testing.assert_frame_equal(result, npv_data)

        # Verify Internal Cache
        assert rt._aai_metrics is not None  # type: ignore

        # Second call (should be cached)
        result2 = rt._generic_metrics(AAI_METRIC_NAME, "calc_aai_metric")
        assert mock_calc.calc_aai_metric.call_count == 1  # No new call
        pd.testing.assert_frame_equal(result2, npv_data)


@pytest.mark.parametrize(
    "metric_name, method_name, attr_name",
    [
        (EAI_METRIC_NAME, "calc_eai_gdf", "eai_metrics"),
        (AAI_METRIC_NAME, "calc_aai_metric", "aai_metrics"),
        (
            AAI_PER_GROUP_METRIC_NAME,
            "calc_aai_per_group_metric",
            "aai_per_group_metrics",
        ),
    ],
)
def test_metric_wrappers(mock_components, metric_name, method_name, attr_name):
    """Uses parametrization to test all simple metric wrapper methods at once."""
    rt = StaticRiskTrajectory(mock_components["snaps"])
    with patch.object(rt, "_compute_metrics") as mock_compute:
        wrapper_func = getattr(rt, attr_name)
        wrapper_func(test_arg="val")
        mock_compute.assert_called_once_with(
            metric_name=metric_name, metric_meth=method_name, test_arg="val"
        )


def test_per_date_risk_metrics_aggregation(mock_components):
    rt = StaticRiskTrajectory(mock_components["snaps"])

    # Setup mock returns for the constituent parts
    df_aai = pd.DataFrame({METRIC_COL_NAME: ["aai"], RISK_COL_NAME: [100]})
    df_rp = pd.DataFrame({METRIC_COL_NAME: ["rp"], RISK_COL_NAME: [50]})
    df_grp = pd.DataFrame({METRIC_COL_NAME: ["grp"], RISK_COL_NAME: [10]})

    with (
        patch.object(rt, "aai_metrics", return_value=df_aai) as m1,
        patch.object(rt, "return_periods_metrics", return_value=df_rp) as m2,
        patch.object(rt, "aai_per_group_metrics", return_value=df_grp) as m3,
    ):

        result = rt.per_date_risk_metrics()
        assert len(result) == 3
        assert list(result[METRIC_COL_NAME]) == ["aai", "rp", "grp"]
        # Verify it called all three internal methods
        m1.assert_called_once()
        m2.assert_called_once()
        m3.assert_called_once()
