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
from itertools import product
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.calc_risk_metrics import CalcRiskMetricsPeriod
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    CONTRIBUTION_BASE_RISK_NAME,
    CONTRIBUTION_EXPOSURE_NAME,
    CONTRIBUTION_HAZARD_NAME,
    CONTRIBUTION_INTERACTION_TERM_NAME,
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
    ImpactInterpolationStrategy,
)
from climada.trajectories.snapshot import Snapshot

# --- Fixtures ---


@pytest.fixture
def mock_snapshots():
    snaps = []
    for year in [2023, 2024, 2025]:
        m = MagicMock(spec=Snapshot)
        m.date = pd.Timestamp(year=year, month=1, day=1)
        snaps.append(m)
    return snaps


@pytest.fixture
def mock_disc_rates():
    dr = MagicMock(spec=DiscRates)
    dr.years = [2023, 2024, 2025]
    dr.rates = [0.01, 0.02, 0.03]
    return dr


@pytest.fixture
def mock_interpolation_strategy():
    return MagicMock(spec=AllLinearStrategy)


@pytest.fixture
def mock_impact_computation_strategy():
    return MagicMock(spec=ImpactCalcComputation)


@pytest.fixture
def dates():
    return {
        "dates1": [
            pd.Period("2023-01-01", freq="Y"),
            pd.Period("2024-01-01", freq="Y"),
        ],
        "dates2": [
            pd.Period("2025-01-01", freq="Y"),
            pd.Period("2026-01-01", freq="Y"),
        ],
    }


@pytest.fixture
def aai_data(dates):
    groups = ["GroupA", "GroupB", pd.NA]
    measures = ["MEAS1", "MEAS2"]
    metrics = [AAI_METRIC_NAME]

    df1 = pd.DataFrame(
        product(dates["dates1"], groups, measures, metrics),
        columns=INDEXING_COLUMNS,
    )
    df1[RISK_COL_NAME] = np.arange(12) * 100
    df1[GROUP_COL_NAME] = df1[GROUP_COL_NAME].astype("category")

    df2 = pd.DataFrame(
        product(dates["dates2"], groups, measures, metrics),
        columns=INDEXING_COLUMNS,
    )
    df2[RISK_COL_NAME] = np.arange(12) * 100 + 1200
    df2[GROUP_COL_NAME] = df2[GROUP_COL_NAME].astype("category")

    df_all = pd.DataFrame(
        product(dates["dates1"] + dates["dates2"], groups, measures, metrics),
        columns=INDEXING_COLUMNS,
    )
    df_all[RISK_COL_NAME] = np.arange(24) * 100
    df_all[GROUP_COL_NAME] = df_all[GROUP_COL_NAME].astype("category")
    df_all[GROUP_COL_NAME] = df_all[GROUP_COL_NAME].cat.add_categories(["All"])
    df_all[GROUP_COL_NAME] = df_all[GROUP_COL_NAME].fillna("All")

    return {"df1": df1, "df2": df2, "df_all": df_all}


@pytest.fixture
def mock_calculators(aai_data):
    calc1 = MagicMock(spec=CalcRiskMetricsPeriod)
    calc2 = MagicMock(spec=CalcRiskMetricsPeriod)
    calc1.calc_aai_metric.return_value = aai_data["df1"]
    calc2.calc_aai_metric.return_value = aai_data["df2"]
    return [calc1, calc2]


@pytest.fixture
def rt_basic(mock_snapshots):
    return InterpolatedRiskTrajectory(mock_snapshots)


@pytest.fixture
def period_agg_df():
    return pd.DataFrame(
        {
            DATE_COL_NAME: pd.PeriodIndex(
                ["2023", "2024", "2025", "2026", "2027"], freq="Y"
            ),
            GROUP_COL_NAME: ["All"] * 5,
            MEASURE_COL_NAME: ["m1"] * 5,
            METRIC_COL_NAME: [AAI_METRIC_NAME] * 5,
            RISK_COL_NAME: [100.0, 200.0, 300.0, 400.0, 500.0],
        }
    )


@pytest.fixture
def period_agg_bins():
    edges = pd.DatetimeIndex(
        [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2027-01-01"),
            pd.Timestamp("2029-01-01"),
        ]
    )
    labels = [
        "2023-01-01 to 2025-01-01",
        "2025-01-01 to 2027-01-01",
        "2027-01-01 to 2029-01-01",
    ]
    return edges, labels


@pytest.fixture
def waterfall_contributions():
    dates = pd.to_datetime(["2023-01-01", "2024-01-01", "2025-01-01"])
    metrics = [
        CONTRIBUTION_BASE_RISK_NAME,
        CONTRIBUTION_EXPOSURE_NAME,
        CONTRIBUTION_HAZARD_NAME,
        CONTRIBUTION_VULNERABILITY_NAME,
        CONTRIBUTION_INTERACTION_TERM_NAME,
    ]
    rows = list(product(dates, metrics))
    df = pd.DataFrame(rows, columns=[DATE_COL_NAME, METRIC_COL_NAME])
    df[RISK_COL_NAME] = np.arange(len(df), dtype=float)
    return df


# --- Init & Properties ---


def test_init_basic(
    mock_snapshots, mock_interpolation_strategy, mock_impact_computation_strategy
):
    with patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators", return_value=1
    ) as mock_reset:
        rt = InterpolatedRiskTrajectory(
            mock_snapshots,
            interpolation_strategy=mock_interpolation_strategy,
            impact_computation_strategy=mock_impact_computation_strategy,
        )
        assert rt.start_date == mock_snapshots[0].date
        assert rt.end_date == mock_snapshots[-1].date
        assert rt._risk_disc_rates is None
        mock_reset.assert_called_once_with(
            mock_snapshots,
            "Y",
            mock_interpolation_strategy,
            mock_impact_computation_strategy,
        )
        assert rt._risk_metrics_calculators == 1
        for metric in InterpolatedRiskTrajectory.POSSIBLE_METRICS:
            assert getattr(rt, f"_{metric}_metrics") is None


def test_init_with_custom_params(mock_snapshots, mock_disc_rates):
    mock_interp = Mock(spec=ImpactInterpolationStrategy)
    mock_impact = Mock(spec=ImpactComputationStrategy)
    with patch.object(
        InterpolatedRiskTrajectory, "_reset_risk_metrics_calculators"
    ) as mock_reset:
        rt = InterpolatedRiskTrajectory(
            mock_snapshots,
            time_resolution="MS",
            risk_disc_rates=mock_disc_rates,
            interpolation_strategy=mock_interp,
            impact_computation_strategy=mock_impact,
        )
        mock_reset.assert_called_once_with(
            mock_snapshots, "MS", mock_interp, mock_impact
        )
        assert rt._risk_disc_rates == mock_disc_rates


def test_set_time_resolution(mock_snapshots, mock_impact_computation_strategy):
    rt = InterpolatedRiskTrajectory(
        mock_snapshots, impact_computation_strategy=mock_impact_computation_strategy
    )
    with patch.object(rt, "_reset_metrics", wraps=rt._reset_metrics) as spy:
        with pytest.raises(ValueError):
            rt.time_resolution = 75
        rt.time_resolution = "5M"
        assert rt.time_resolution == "5M"
        assert spy.call_count == 1


@pytest.mark.parametrize(
    "strategy_attr,strategy_cls,new_strategy_cls",
    [
        ("impact_computation_strategy", ImpactCalcComputation, ImpactCalcComputation),
        ("interpolation_strategy", AllLinearStrategy, ExponentialExposureStrategy),
    ],
)
def test_set_strategies(
    mock_snapshots, mock_calculators, strategy_attr, strategy_cls, new_strategy_cls
):
    with patch.object(
        InterpolatedRiskTrajectory,
        "_reset_risk_metrics_calculators",
        return_value=mock_calculators,
    ):
        rt = InterpolatedRiskTrajectory(mock_snapshots)
        with patch.object(rt, "_reset_metrics", wraps=rt._reset_metrics) as spy:
            with pytest.raises(ValueError):
                setattr(rt, strategy_attr, "invalid")
            new_strategy = new_strategy_cls()
            setattr(rt, strategy_attr, new_strategy)
            assert getattr(rt, strategy_attr) == new_strategy
            assert spy.call_count == 1
            for calc in mock_calculators:
                assert getattr(calc, strategy_attr) == new_strategy


def test_risk_periods_lazy_computation(
    mock_snapshots, mock_interpolation_strategy, mock_impact_computation_strategy
):
    with patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    ) as MockCalc:
        rt = InterpolatedRiskTrajectory(
            mock_snapshots,
            interpolation_strategy=mock_interpolation_strategy,
            impact_computation_strategy=mock_impact_computation_strategy,
        )
        risk_periods = rt._risk_metrics_calculators
        MockCalc.assert_has_calls(
            [
                call(
                    mock_snapshots[0],
                    mock_snapshots[1],
                    time_resolution="Y",
                    interpolation_strategy=mock_interpolation_strategy,
                    impact_computation_strategy=mock_impact_computation_strategy,
                ),
                call(
                    mock_snapshots[1],
                    mock_snapshots[2],
                    time_resolution="Y",
                    interpolation_strategy=mock_interpolation_strategy,
                    impact_computation_strategy=mock_impact_computation_strategy,
                ),
            ]
        )
        assert MockCalc.call_count == 2
        assert isinstance(risk_periods, list)
        assert len(risk_periods) == 2


def test_calc_risk_periods_sorting(mock_snapshots):
    with patch(
        "climada.trajectories.interpolated_trajectory.CalcRiskMetricsPeriod",
        autospec=True,
    ) as MockCalc:
        unsorted = [mock_snapshots[2], mock_snapshots[0], mock_snapshots[1]]
        _ = InterpolatedRiskTrajectory(unsorted)
        MockCalc.assert_has_calls(
            [
                call(mock_snapshots[0], mock_snapshots[1], **MockCalc.call_args[1]),
                call(mock_snapshots[1], mock_snapshots[2], **MockCalc.call_args[1]),
            ]
        )
        assert MockCalc.call_count == 2


# --- Generic Metrics ---


def test_generic_metrics_basic_flow(
    mock_snapshots, mock_calculators, aai_data, mock_disc_rates
):
    expected_pre_npv = aai_data["df_all"][
        [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            MEASURE_COL_NAME,
            METRIC_COL_NAME,
            RISK_COL_NAME,
        ]
    ]
    expected_npv = expected_pre_npv.copy()
    expected_npv[RISK_COL_NAME] *= 0.9

    with patch.object(
        InterpolatedRiskTrajectory,
        "_reset_risk_metrics_calculators",
        return_value=mock_calculators,
    ):
        rt = InterpolatedRiskTrajectory(mock_snapshots)
        rt._risk_disc_rates = mock_disc_rates

        with patch.object(rt, "npv_transform", return_value=expected_npv) as mock_npv:
            result = rt._generic_metrics(
                metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric"
            )

            mock_calculators[0].calc_aai_metric.assert_called_once()
            mock_calculators[1].calc_aai_metric.assert_called_once()
            mock_npv.assert_called_once()
            pd.testing.assert_frame_equal(
                mock_npv.call_args[0][0].reset_index(drop=True),
                expected_pre_npv.reset_index(drop=True),
            )
            pd.testing.assert_frame_equal(result, expected_npv)
            pd.testing.assert_frame_equal(
                rt._aai_metrics.reset_index(drop=True),
                expected_npv.reset_index(drop=True),
            )

            # Second call should use cache
            result2 = rt._generic_metrics(
                metric_name=AAI_METRIC_NAME, metric_meth="calc_aai_metric"
            )
            assert mock_calculators[0].calc_aai_metric.call_count == 1
            pd.testing.assert_frame_equal(
                result2.reset_index(drop=True), expected_npv.reset_index(drop=True)
            )


def test_generic_metrics_not_implemented_error(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with pytest.raises(NotImplementedError):
        rt._generic_metrics(metric_name="non_existent", metric_meth="some_method")


def test_generic_metrics_missing_args(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with pytest.raises(ValueError):
        rt._generic_metrics(metric_name=None, metric_meth="some_method")
    with pytest.raises(ValueError):
        rt._generic_metrics(metric_name=AAI_METRIC_NAME, metric_meth=None)


@pytest.mark.parametrize("return_value", [None, pd.DataFrame()])
def test_generic_metrics_empty_returns_empty_df(
    mock_snapshots, mock_calculators, return_value
):
    mock_calculators[0].calc_aai_per_group_metric.return_value = return_value
    mock_calculators[1].calc_aai_per_group_metric.return_value = return_value
    with patch.object(
        InterpolatedRiskTrajectory,
        "_reset_risk_metrics_calculators",
        return_value=mock_calculators,
    ):
        rt = InterpolatedRiskTrajectory(mock_snapshots)
        result = rt._generic_metrics(
            metric_name=AAI_PER_GROUP_METRIC_NAME,
            metric_meth="calc_aai_per_group_metric",
        )
        pd.testing.assert_frame_equal(result, pd.DataFrame())


def test_generic_metrics_risk_contribution_treatment(
    mock_snapshots, mock_calculators, aai_data
):
    mock_calculators[0].calc_risk_contributions_metric.return_value = aai_data["df1"]
    mock_calculators[1].calc_risk_contributions_metric.return_value = aai_data["df2"]
    with patch.object(
        InterpolatedRiskTrajectory,
        "_reset_risk_metrics_calculators",
        return_value=mock_calculators,
    ):
        rt = InterpolatedRiskTrajectory(mock_snapshots)
        with patch.object(
            rt, "_risk_contributions_post_treatment", return_value=pd.DataFrame([42])
        ) as mock_post:
            result = rt._generic_metrics(
                metric_name=CONTRIBUTIONS_METRIC_NAME,
                metric_meth="calc_risk_contributions_metric",
            )
            mock_post.assert_called_once()
            pd.testing.assert_frame_equal(result, pd.DataFrame([42]))


def test_generic_metrics_coord_id_handling(mock_snapshots, mock_calculators):
    mock_calculators[0].calc_eai_gdf.return_value = pd.DataFrame(
        {
            DATE_COL_NAME: [pd.Timestamp("2023-01-01")] * 2,
            GROUP_COL_NAME: pd.Categorical([pd.NA, pd.NA]),
            MEASURE_COL_NAME: ["MEAS1", "MEAS1"],
            METRIC_COL_NAME: [EAI_METRIC_NAME, EAI_METRIC_NAME],
            COORD_ID_COL_NAME: [1, 2],
            RISK_COL_NAME: [10.0, 20.0],
        }
    )
    mock_calculators[1].calc_eai_gdf.return_value = pd.DataFrame()

    with patch.object(
        InterpolatedRiskTrajectory,
        "_reset_risk_metrics_calculators",
        return_value=mock_calculators,
    ):
        rt = InterpolatedRiskTrajectory(mock_snapshots)
        result = rt._generic_metrics(
            metric_name=EAI_METRIC_NAME, metric_meth="calc_eai_gdf"
        )

    cols = [
        DATE_COL_NAME,
        GROUP_COL_NAME,
        MEASURE_COL_NAME,
        METRIC_COL_NAME,
        COORD_ID_COL_NAME,
        RISK_COL_NAME,
    ]
    expected = pd.DataFrame(
        {
            GROUP_COL_NAME: pd.Categorical(["All", "All"]),
            DATE_COL_NAME: [pd.Timestamp("2023-01-01")] * 2,
            MEASURE_COL_NAME: ["MEAS1", "MEAS1"],
            METRIC_COL_NAME: [EAI_METRIC_NAME, EAI_METRIC_NAME],
            RISK_COL_NAME: [10.0, 20.0],
            COORD_ID_COL_NAME: [1, 2],
        }
    )
    pd.testing.assert_frame_equal(result[cols], expected[cols])


# --- Metric Wrapper Methods ---


@pytest.mark.parametrize(
    "method,metric_name,metric_meth,extra_kwargs",
    [
        ("aai_metrics", AAI_METRIC_NAME, "calc_aai_metric", {}),
        (
            "aai_per_group_metrics",
            AAI_PER_GROUP_METRIC_NAME,
            "calc_aai_per_group_metric",
            {},
        ),
        (
            "risk_contributions_metrics",
            CONTRIBUTIONS_METRIC_NAME,
            "calc_risk_contributions_metric",
            {},
        ),
    ],
)
def test_metric_wrappers(
    mock_snapshots, method, metric_name, metric_meth, extra_kwargs
):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(rt, "_compute_metrics") as mock_compute:
        getattr(rt, method)(**extra_kwargs)
        mock_compute.assert_called_once_with(
            metric_name=metric_name, metric_meth=metric_meth, **extra_kwargs
        )


def test_return_periods_metrics_wrapper(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(rt, "_compute_metrics") as mock_compute:
        rt.return_periods_metrics(npv=True, rp_arg="xyz")
        mock_compute.assert_called_once_with(
            npv=True,
            metric_name=RETURN_PERIOD_METRIC_NAME,
            metric_meth="calc_return_periods_metric",
            return_periods=rt.return_periods,
            rp_arg="xyz",
        )


def test_eai_metrics_wrapper(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(rt, "_compute_metrics") as mock_compute:
        rt.eai_metrics(npv=True, some_arg="test")
        mock_compute.assert_called_once_with(
            npv=True,
            metric_name=EAI_METRIC_NAME,
            metric_meth="calc_eai_gdf",
            some_arg="test",
        )


# --- NPV Transform ---


def test_npv_transform_no_group_col(mock_disc_rates):
    df_input = pd.DataFrame(
        {
            DATE_COL_NAME: pd.to_datetime(["2023-01-01", "2024-01-01"] * 2),
            MEASURE_COL_NAME: ["m1", "m1", "m2", "m2"],
            METRIC_COL_NAME: [AAI_METRIC_NAME] * 4,
            RISK_COL_NAME: [100.0, 200.0, 80.0, 180.0],
        }
    )
    with patch(
        "climada.trajectories.trajectory.RiskTrajectory._calc_npv_cash_flows"
    ) as mock_calc:
        mock_calc.side_effect = [
            pd.Series(
                [100.0, 196.0], index=pd.to_datetime(["2023-01-01", "2024-01-01"])
            ),
            pd.Series(
                [79.2, 176.4], index=pd.to_datetime(["2023-01-01", "2024-01-01"])
            ),
        ]
        InterpolatedRiskTrajectory.npv_transform(df_input.copy(), mock_disc_rates)
        assert mock_calc.call_count == 2
        assert mock_calc.call_args_list[0].args[1] == pd.Timestamp("2023-01-01")
        assert mock_calc.call_args_list[0].args[2] == mock_disc_rates


def test_npv_transform_with_group_col(mock_disc_rates):
    df_input = pd.DataFrame(
        {
            DATE_COL_NAME: pd.to_datetime(["2023-01-01", "2024-01-01", "2023-01-01"]),
            GROUP_COL_NAME: ["G1", "G1", "G2"],
            MEASURE_COL_NAME: ["m1", "m1", "m1"],
            METRIC_COL_NAME: [AAI_METRIC_NAME] * 3,
            RISK_COL_NAME: [100.0, 200.0, 150.0],
        }
    )
    with patch(
        "climada.trajectories.trajectory.RiskTrajectory._calc_npv_cash_flows"
    ) as mock_calc:
        mock_calc.side_effect = [
            pd.Series(
                [99.0, 196.0], index=pd.to_datetime(["2023-01-01", "2024-01-01"])
            ),
            pd.Series([148.5], index=pd.to_datetime(["2023-01-01"])),
        ]
        result = InterpolatedRiskTrajectory.npv_transform(
            df_input.copy(), mock_disc_rates
        )
        assert mock_calc.call_count == 2
        assert result[RISK_COL_NAME].notna().all()


# --- Period Aggregation ---


def test_compute_period_metrics(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with (
        patch.object(rt, "_generic_metrics", return_value=46) as mock_generic,
        patch.object(rt, "_date_to_period_agg", return_value=42) as mock_agg,
    ):
        result = rt._compute_period_metrics("name", "method", other_args=5)
        mock_generic.assert_called_once_with(
            metric_name="name", metric_meth="method", other_args=5
        )
        mock_agg.assert_called_once_with(46, grouper=rt._grouper)
        assert result == 42


def test_make_period_bins_no_freq(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    edges, labels = rt._make_period_bins()

    expected_edges = pd.DatetimeIndex([snap.date for snap in mock_snapshots])
    pd.testing.assert_index_equal(edges, expected_edges)
    assert labels == [
        "2023-01-01 to 2024-01-01",
        "2024-01-01 to 2025-01-01",
    ]


def test_make_period_bins_start_anchored_freq(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    edges, labels = rt._make_period_bins(freq="YS")

    assert edges[0] == pd.Timestamp("2023-01-01")
    assert edges[-1] == pd.Timestamp("2025-01-01")
    assert len(edges) == len(labels) + 1
    assert labels == [
        "2023-01-01 to 2024-01-01",
        "2024-01-01 to 2025-01-01",
    ]


def test_make_period_bins_end_anchored_freq_warns(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch("climada.trajectories.interpolated_trajectory.LOGGER") as mock_logger:
        rt._make_period_bins(freq="YE")
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "end-anchored" in warning_msg


def test_make_period_bins_labels_match_edges(mock_snapshots):
    """Labels must always be consistent with edges regardless of freq."""
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    for freq in [None, "YS", "6MS"]:
        edges, labels = rt._make_period_bins(freq=freq)
        assert len(labels) == len(edges) - 1
        for i, label in enumerate(labels):
            assert label == f"{edges[i].date()} to {edges[i + 1].date()}"


def test_make_period_bins_freq_extends_to_cover_end():
    """When end date is not on a frequency boundary, the range is extended by one period."""
    snaps = [MagicMock(spec=Snapshot) for _ in range(2)]
    snaps[0].date = pd.Timestamp(2023, 1, 1)
    snaps[1].date = pd.Timestamp(2025, 6, 1)  # Not on a 2YS boundary
    rt = InterpolatedRiskTrajectory(snaps)

    edges, labels = rt._make_period_bins(freq="2YS")

    assert edges[0] == pd.Timestamp("2023-01-01")
    assert edges[-1] >= pd.Timestamp("2025-06-01")
    assert len(edges) == len(labels) + 1


def test_date_to_period_agg_basic(period_agg_df, period_agg_bins):
    edges, labels = period_agg_bins
    result = InterpolatedRiskTrajectory._date_to_period_agg(
        period_agg_df,
        grouper=[MEASURE_COL_NAME, METRIC_COL_NAME],
        bin_edges=edges,
        labels=labels,
    )
    assert PERIOD_COL_NAME in result.columns
    assert set(result[PERIOD_COL_NAME].dropna()) == set(labels)
    assert result[RISK_COL_NAME].notna().any()


def test_date_to_period_agg_mean(period_agg_df, period_agg_bins):
    edges, labels = period_agg_bins
    result = InterpolatedRiskTrajectory._date_to_period_agg(
        period_agg_df,
        grouper=[MEASURE_COL_NAME, METRIC_COL_NAME],
        bin_edges=edges,
        labels=labels,
        aggfunc="mean",
    )
    risk_by_period = result.set_index(PERIOD_COL_NAME)[RISK_COL_NAME]
    assert risk_by_period[labels[0]] == pytest.approx(150.0)
    assert risk_by_period[labels[1]] == pytest.approx(350.0)
    assert risk_by_period[labels[2]] == pytest.approx(500.0)


def test_date_to_period_agg_custom_aggfunc(period_agg_df, period_agg_bins):
    edges, labels = period_agg_bins
    result = InterpolatedRiskTrajectory._date_to_period_agg(
        period_agg_df,
        grouper=[MEASURE_COL_NAME, METRIC_COL_NAME],
        bin_edges=edges,
        labels=labels,
        aggfunc="sum",
    )
    risk_by_period = result.set_index(PERIOD_COL_NAME)[RISK_COL_NAME]
    assert risk_by_period[labels[0]] == pytest.approx(300.0)
    assert risk_by_period[labels[1]] == pytest.approx(700.0)
    assert risk_by_period[labels[2]] == pytest.approx(500.0)


def test_date_to_period_agg_group_col_added_if_missing(period_agg_bins):
    """GROUP_COL_NAME should be prepended to grouper if present in df but not in grouper."""
    edges, labels = period_agg_bins
    df = pd.DataFrame(
        {
            DATE_COL_NAME: pd.PeriodIndex(["2023", "2023", "2024"], freq="Y"),
            GROUP_COL_NAME: ["G1", "G2", "G1"],
            MEASURE_COL_NAME: ["m1", "m1", "m1"],
            METRIC_COL_NAME: [AAI_METRIC_NAME] * 3,
            RISK_COL_NAME: [100.0, 200.0, 300.0],
        }
    )
    result = InterpolatedRiskTrajectory._date_to_period_agg(
        df,
        grouper=[MEASURE_COL_NAME, METRIC_COL_NAME],
        bin_edges=edges,
        labels=labels,
    )
    assert GROUP_COL_NAME in result.columns
    assert set(result[GROUP_COL_NAME]) == {"G1", "G2"}


def test_date_to_period_agg_multiple_colnames(period_agg_bins):
    edges, labels = period_agg_bins
    df = pd.DataFrame(
        {
            DATE_COL_NAME: pd.PeriodIndex(["2023", "2024"], freq="Y"),
            GROUP_COL_NAME: ["All", "All"],
            MEASURE_COL_NAME: ["m1", "m1"],
            METRIC_COL_NAME: ["components"] * 2,
            CONTRIBUTION_BASE_RISK_NAME: [10.0, 20.0],
            CONTRIBUTION_EXPOSURE_NAME: [5.0, 8.0],
        }
    )
    result = InterpolatedRiskTrajectory._date_to_period_agg(
        df,
        grouper=[MEASURE_COL_NAME, METRIC_COL_NAME],
        bin_edges=edges,
        labels=labels,
        colname=[CONTRIBUTION_BASE_RISK_NAME, CONTRIBUTION_EXPOSURE_NAME],
    )
    assert CONTRIBUTION_BASE_RISK_NAME in result.columns
    assert CONTRIBUTION_EXPOSURE_NAME in result.columns


def test_per_period_risk_metrics(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    mock_date_df = pd.DataFrame(
        {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
    )
    mock_period_df = pd.DataFrame({PERIOD_COL_NAME: ["P1"], RISK_COL_NAME: [200]})

    with (
        patch.object(
            rt, "per_date_risk_metrics", return_value=mock_date_df
        ) as mock_date,
        patch.object(
            rt, "_date_to_period_agg", return_value=mock_period_df
        ) as mock_agg,
    ):
        # Default: snapshot-based bins
        result = rt.per_period_risk_metrics(metrics=[AAI_METRIC_NAME])
        mock_date.assert_called_once_with(metrics=[AAI_METRIC_NAME])

        _, kwargs = mock_agg.call_args
        expected_edges = pd.DatetimeIndex(
            [snap.date for snap in sorted(mock_snapshots, key=lambda s: s.date)]
        )
        pd.testing.assert_index_equal(kwargs["bin_edges"], expected_edges)
        assert kwargs["labels"] == [
            f"{expected_edges[i].date()} to {expected_edges[i+1].date()}"
            for i in range(len(expected_edges) - 1)
        ]
        assert kwargs["grouper"] == rt._grouper + [UNIT_COL_NAME]
        pd.testing.assert_frame_equal(result, mock_period_df)


def test_per_period_risk_metrics_custom_freq(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    mock_date_df = pd.DataFrame(
        {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
    )
    mock_period_df = pd.DataFrame({PERIOD_COL_NAME: ["P1"], RISK_COL_NAME: [200]})

    with (
        patch.object(rt, "per_date_risk_metrics", return_value=mock_date_df),
        patch.object(
            rt, "_date_to_period_agg", return_value=mock_period_df
        ) as mock_agg,
    ):
        rt.per_period_risk_metrics(metrics=[AAI_METRIC_NAME], freq="YS")

        _, kwargs = mock_agg.call_args
        # With freq="YS", edges should be annual from start to end
        expected_edges = pd.date_range(
            start=mock_snapshots[0].date, end=mock_snapshots[-1].date, freq="YS"
        )
        pd.testing.assert_index_equal(kwargs["bin_edges"], expected_edges)


def test_per_period_risk_metrics_custom_aggfunc(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    mock_date_df = pd.DataFrame(
        {METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]}
    )
    mock_period_df = pd.DataFrame({PERIOD_COL_NAME: ["P1"], RISK_COL_NAME: [200]})

    with (
        patch.object(rt, "per_date_risk_metrics", return_value=mock_date_df),
        patch.object(
            rt, "_date_to_period_agg", return_value=mock_period_df
        ) as mock_agg,
    ):
        rt.per_period_risk_metrics(metrics=[AAI_METRIC_NAME], aggfunc="sum")
        _, kwargs = mock_agg.call_args
        assert kwargs["aggfunc"] == "sum"


# --- Per Date Risk Metrics ---


def test_per_date_risk_metrics_defaults(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    df_aai = pd.DataFrame({METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]})
    df_rp = pd.DataFrame({METRIC_COL_NAME: ["rp"], RISK_COL_NAME: [50]})
    df_grp = pd.DataFrame({METRIC_COL_NAME: ["aai_grp"], RISK_COL_NAME: [10]})

    with (
        patch.object(rt, "aai_metrics", return_value=df_aai) as m1,
        patch.object(rt, "return_periods_metrics", return_value=df_rp) as m2,
        patch.object(rt, "aai_per_group_metrics", return_value=df_grp) as m3,
    ):
        result = rt.per_date_risk_metrics()
        m1.assert_called_once_with()
        m2.assert_called_once_with()
        m3.assert_called_once_with()
        expected = pd.concat([df_aai, df_rp, df_grp])
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )


def test_per_date_risk_metrics_custom(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    df_aai = pd.DataFrame({METRIC_COL_NAME: [AAI_METRIC_NAME], RISK_COL_NAME: [100]})
    df_rp = pd.DataFrame({METRIC_COL_NAME: ["rp"], RISK_COL_NAME: [50]})

    with (
        patch.object(rt, "aai_metrics", return_value=df_aai) as m1,
        patch.object(rt, "return_periods_metrics", return_value=df_rp) as m2,
        patch.object(rt, "aai_per_group_metrics") as m3,
    ):
        result = rt.per_date_risk_metrics(
            metrics=[AAI_METRIC_NAME, RETURN_PERIOD_METRIC_NAME]
        )
        m1.assert_called_once_with()
        m2.assert_called_once_with()
        m3.assert_not_called()
        expected = pd.concat([df_aai, df_rp])
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True), expected.reset_index(drop=True)
        )


# --- Risk Contributions Post Treatment ---


def test_risk_contributions_post_treatment(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
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
    result = rt._risk_contributions_post_treatment(pd.DataFrame(data))
    expected_risk = [100, 100, 100, 0, 50, 150, 0, 10, 30, 0, 5, 15, 0, 30, 90]
    assert result[RISK_COL_NAME].tolist() == expected_risk


# --- Reset Metrics ---


def test_reset_metrics(mock_snapshots):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    rt._eai_metrics = "dummy"  # type: ignore
    rt._aai_metrics = "dummy"  # type: ignore
    rt._reset_metrics()
    for metric in rt.POSSIBLE_METRICS:
        assert getattr(rt, f"_{metric}_metrics") is None


# --- Get Risk Periods ---


@pytest.fixture
def mock_risk_periods():
    periods = []
    dates = [(2020, 2021), (2021, 2022), (2022, 2023)]
    for start_year, end_year in dates:
        rp = Mock()
        rp.snapshot_start.date = datetime.date(start_year, 1, 1)
        rp.snapshot_end.date = datetime.date(end_year, 1, 1)
        periods.append(rp)
    return periods


@pytest.mark.parametrize(
    "start,end,strict,expected_indices",
    [
        (datetime.date(2020, 1, 1), datetime.date(2023, 1, 1), True, [0, 1, 2]),
        (datetime.date(2018, 1, 1), datetime.date(2024, 1, 1), True, [0, 1, 2]),
        (datetime.date(2021, 1, 1), datetime.date(2023, 1, 1), True, [1, 2]),
        (datetime.date(2021, 6, 1), datetime.date(2022, 6, 1), True, []),
        (datetime.date(2020, 1, 1), datetime.date(2023, 1, 1), False, [0, 1, 2]),
        (datetime.date(2018, 1, 1), datetime.date(2024, 1, 1), False, [0, 1, 2]),
        (datetime.date(2021, 1, 1), datetime.date(2023, 1, 1), False, [1, 2]),
        (datetime.date(2021, 6, 1), datetime.date(2022, 6, 1), False, [1, 2]),
        (datetime.date(2024, 6, 1), datetime.date(2026, 6, 1), False, []),
    ],
)
def test_get_risk_periods(mock_risk_periods, start, end, strict, expected_indices):
    result = InterpolatedRiskTrajectory._get_risk_periods(
        mock_risk_periods, start, end, strict=strict
    )
    assert result == [mock_risk_periods[i] for i in expected_indices]


def test_calc_waterfall_plot_data_default_dates(
    mock_snapshots, waterfall_contributions
):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(
        rt, "risk_contributions_metrics", return_value=waterfall_contributions
    ):
        result = rt._calc_waterfall_plot_data()

    assert result.index.name == DATE_COL_NAME
    assert result.columns.name == METRIC_COL_NAME
    assert set(result.columns) == set(waterfall_contributions[METRIC_COL_NAME].unique())
    # All three dates should be present
    assert len(result) == 3


def test_calc_waterfall_plot_data_custom_dates(mock_snapshots, waterfall_contributions):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(
        rt, "risk_contributions_metrics", return_value=waterfall_contributions
    ):
        result = rt._calc_waterfall_plot_data(
            start_date=datetime.date(2024, 1, 1),
            end_date=datetime.date(2025, 1, 1),
        )

    assert len(result) == 2
    assert pd.Timestamp("2023-01-01") not in result.index


def test_calc_waterfall_plot_data_returns_unstacked(
    mock_snapshots, waterfall_contributions
):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(
        rt, "risk_contributions_metrics", return_value=waterfall_contributions
    ):
        result = rt._calc_waterfall_plot_data()

    # Result should be a wide DataFrame (unstacked on METRIC_COL_NAME)
    assert isinstance(result, pd.DataFrame)
    assert result.index.name == DATE_COL_NAME
    assert METRIC_COL_NAME not in result.columns


def test_calc_waterfall_plot_data_calls_risk_contributions(
    mock_snapshots, waterfall_contributions
):
    rt = InterpolatedRiskTrajectory(mock_snapshots)
    with patch.object(
        rt, "risk_contributions_metrics", return_value=waterfall_contributions
    ) as mock_rc:
        rt._calc_waterfall_plot_data()
        mock_rc.assert_called_once_with()
