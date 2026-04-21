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

Test trajectories.

"""

import numpy as np
import pandas as pd
import pytest

from climada.engine.impact_calc import ImpactCalc
from climada.entity.disc_rates.base import DiscRates
from climada.test.conftest import CATEGORIES, EXPOSURE_REF_YEAR
from climada.trajectories import StaticRiskTrajectory
from climada.trajectories.constants import (
    DATE_COL_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RISK_COL_NAME,
    UNIT_COL_NAME,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import DEFAULT_RP

EXPOSURE_FUTURE_YEAR = 2040


@pytest.fixture
def snapshot_factory(
    exposures_factory,
    hazard_factory,
    impfset_factory,
):
    """
    Factory for Snapshot objects.

    Allows controlled construction of baseline / future / counterfactual
    scenarios by scaling exposure values, hazard intensity, and impact function.
    """

    def _make_snapshot(
        *,
        date=EXPOSURE_REF_YEAR,
        exposure_value_factor=1.0,
        hazard_intensity_factor=1.0,
        hazard_frequency_factor=1.0,
        paa_scale=1.0,
        group_id=None,
        negative_intensities=False,
    ):
        exposures = exposures_factory(
            value_factor=exposure_value_factor, ref_year=date, group_id=group_id
        )

        hazard = hazard_factory(
            intensity_scale=hazard_intensity_factor,
            frequency_scale=hazard_frequency_factor,
        )

        impfset = impfset_factory(
            paa_scale=paa_scale,
            negative_intensities=negative_intensities,
        )

        return Snapshot(
            exposure=exposures,
            hazard=hazard,
            impfset=impfset,
            date=str(date),
        )

    return _make_snapshot


@pytest.fixture
def snapshot_base(snapshot_factory):
    return snapshot_factory()


@pytest.fixture
def snapshot_future(snapshot_factory):
    return snapshot_factory(
        date=2040,
        exposure_value_factor=2.0,
        hazard_intensity_factor=2.0,
    )


def expected_static_metrics_from_snapshots(
    snapshots, return_periods=DEFAULT_RP, disc_rates=None
):
    rows = []
    if disc_rates is not None:
        discount_factor = pd.Series(index=disc_rates.years, data=1 + disc_rates.rates)
        discount_factor = 1 / ((discount_factor.shift(1, fill_value=1)).cumprod())
    else:
        discount_factor = None

    for snap in snapshots:
        imp = ImpactCalc(**snap.impact_calc_kwargs).impact()
        curve = imp.calc_freq_curve(return_periods)
        if discount_factor is not None:
            discount = discount_factor.loc[pd.Timestamp(str(snap.date)).year]
        else:
            discount = 1
        rows.append(
            [
                pd.Timestamp(str(snap.date)),
                "All",
                NO_MEASURE_VALUE,
                "aai",
                "USD",
                imp.aai_agg * discount,
            ]
        )

        rows.extend(
            [
                [
                    pd.Timestamp(str(snap.date)),
                    "All",
                    NO_MEASURE_VALUE,
                    f"rp_{rp}",
                    "USD",
                    val * discount,
                ]
                for rp, val in zip(curve.return_per, curve.impact)
            ]
        )
        if "group_id" in snap.exposure.gdf.columns:
            aai_per_group = [
                [
                    pd.Timestamp(str(snap.date)),
                    group,
                    NO_MEASURE_VALUE,
                    "aai",
                    "USD",
                    val * discount,
                ]
                for group, val in zip(snap.exposure.gdf["group_id"], imp.eai_exp)
            ]
            rows.extend(aai_per_group)

    res = pd.DataFrame(
        rows,
        columns=[
            DATE_COL_NAME,
            GROUP_COL_NAME,
            MEASURE_COL_NAME,
            METRIC_COL_NAME,
            UNIT_COL_NAME,
            RISK_COL_NAME,
        ],
    )

    res = res.groupby(
        [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            MEASURE_COL_NAME,
            METRIC_COL_NAME,
            UNIT_COL_NAME,
        ],
        as_index=False,
    ).sum()

    return res.set_index(
        [
            DATE_COL_NAME,
            GROUP_COL_NAME,
            MEASURE_COL_NAME,
            METRIC_COL_NAME,
            UNIT_COL_NAME,
        ]
    ).sort_index()


def test_static_trajectory(snapshot_factory):
    present_date = 2020
    future_date = 2040

    hazard_intensity_factor = 2.0
    exposure_value_factor = 10.0

    snapshot_base = snapshot_factory(
        date=present_date,
    )

    snapshot_fut = snapshot_factory(
        date=future_date,
        hazard_intensity_factor=hazard_intensity_factor,
        exposure_value_factor=exposure_value_factor,
    )

    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_fut]
    )
    static_traj = StaticRiskTrajectory([snapshot_base, snapshot_fut])
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )

    # --- Assertion ----------------------------------------------------------
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )


def test_static_trajectory_one_snap(snapshot_factory):
    present_date = 2020

    snapshot_base = snapshot_factory(
        date=present_date,
    )

    expected_static_metrics = expected_static_metrics_from_snapshots([snapshot_base])
    static_traj = StaticRiskTrajectory([snapshot_base])
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )

    # --- Assertion ----------------------------------------------------------
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )


def test_static_trajectory_with_group(snapshot_factory):
    present_date = 2020
    future_date = 2040

    hazard_intensity_factor = 2.0
    exposure_value_factor = 10.0

    snapshot_base = snapshot_factory(date=present_date, group_id=CATEGORIES)

    snapshot_fut = snapshot_factory(
        date=future_date,
        hazard_intensity_factor=hazard_intensity_factor,
        exposure_value_factor=exposure_value_factor,
        group_id=CATEGORIES,
    )

    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_fut]
    )
    static_traj = StaticRiskTrajectory([snapshot_base, snapshot_fut])
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )

    # --- Assertion ----------------------------------------------------------
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )


def test_static_trajectory_change_rp(snapshot_factory):
    present_date = 2020
    future_date = 2040

    hazard_intensity_factor = 2.0
    exposure_value_factor = 10.0

    snapshot_base = snapshot_factory(date=present_date, group_id=CATEGORIES)

    snapshot_fut = snapshot_factory(
        date=future_date,
        hazard_intensity_factor=hazard_intensity_factor,
        exposure_value_factor=exposure_value_factor,
        group_id=CATEGORIES,
    )

    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_fut], return_periods=[10, 60, 1000]
    )
    static_traj = StaticRiskTrajectory(
        [snapshot_base, snapshot_fut], return_periods=[10, 60, 1000]
    )
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )

    # --- Assertion ----------------------------------------------------------
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )

    # Also check change to other return period
    static_traj.return_periods = DEFAULT_RP
    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_fut], return_periods=DEFAULT_RP
    )
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )


def test_static_trajectory_risk_disc_rate(snapshot_base, snapshot_future):
    risk_disc_rate = DiscRates(
        years=np.array(range(EXPOSURE_REF_YEAR, EXPOSURE_FUTURE_YEAR + 1)),
        rates=np.ones(EXPOSURE_FUTURE_YEAR - EXPOSURE_REF_YEAR + 1) * 0.01,
    )
    static_traj = StaticRiskTrajectory(
        [snapshot_base, snapshot_future], risk_disc_rates=risk_disc_rate
    )
    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_future], disc_rates=risk_disc_rate
    )

    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )

    # Also check change to other disc_rate
    expected_static_metrics = expected_static_metrics_from_snapshots(
        [snapshot_base, snapshot_future]
    )

    static_traj.risk_disc_rates = None
    result = (
        static_traj.per_date_risk_metrics()
        .set_index(
            [
                DATE_COL_NAME,
                GROUP_COL_NAME,
                MEASURE_COL_NAME,
                METRIC_COL_NAME,
                UNIT_COL_NAME,
            ]
        )
        .sort_index()
    )
    pd.testing.assert_frame_equal(
        result,
        expected_static_metrics,
        check_index_type=False,
        check_categorical=False,
        check_like=False,
    )
