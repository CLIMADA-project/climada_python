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

import copy
from itertools import groupby
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from climada.engine.impact_calc import ImpactCalc
from climada.entity.disc_rates.base import DiscRates
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard
from climada.test.conftest import (
    CATEGORIES,
    DATES,
    EVENT_IDS,
    EVENT_NAMES,
    EXPOSURE_REF_YEAR,
    FREQUENCY,
    FREQUENCY_UNIT,
    HAZARD_MAX_INTENSITY,
    HAZARD_TYPE,
    HAZARD_UNIT,
    IMPF_ID,
    IMPF_NAME,
)
from climada.trajectories import InterpolatedRiskTrajectory, StaticRiskTrajectory
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    CONTRIBUTION_BASE_RISK_NAME,
    CONTRIBUTION_EXPOSURE_NAME,
    CONTRIBUTION_HAZARD_NAME,
    CONTRIBUTION_INTERACTION_TERM_NAME,
    CONTRIBUTION_VULNERABILITY_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    PERIOD_COL_NAME,
    RETURN_PERIOD_METRIC_NAME,
    RISK_COL_NAME,
    RP_VALUE_PREFIX,
    UNIT_COL_NAME,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import DEFAULT_RP

EXPOSURE_FUTURE_YEAR = 2040

from climada.trajectories.snapshot import Snapshot


@pytest.fixture(scope="session")
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
        )

        return Snapshot(
            exposure=exposures,
            hazard=hazard,
            impfset=impfset,
            date=str(date),
        )

    return _make_snapshot


@pytest.fixture(scope="session")
def snapshot_base(snapshot_factory):
    return snapshot_factory()


@pytest.fixture(scope="session")
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


# ----------- INTERPOLATED TRAJ ----------------


@pytest.fixture(scope="session")
def snapshot_future_interp(snapshot_factory):
    return snapshot_factory(
        date=2022,  # Closer date for less rows
        exposure_value_factor=6.0,
        hazard_intensity_factor=2.0,  # Different factor for contributors
    )


@pytest.fixture(scope="session")
def snapshot_future_interp_vulchange(snapshot_factory):
    return snapshot_factory(
        date=2022,  # Closer date for less rows
        exposure_value_factor=6.0,
        hazard_intensity_factor=2.0,  # Different factor for contributors
        paa_scale=0.5,
    )


@pytest.fixture(scope="session")
def expected_interp_metrics():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 18.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 94.5],
                  # Above should indeed not be 216+18 / 2 and slightly
                  # because as we interpolate each contributor separately,
                  # the interaction term grows slower.
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 216.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 500.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 2625.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 6000.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_250', 'USD', 3750.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_250', 'USD', 19687.5],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_250', 'USD', 45000.0]],
         'index_names': [None],
         'column_names': [None]
         },
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_interp_metrics_wgroup(expected_interp_metrics):
    return pd.concat(
        [
            expected_interp_metrics,
            # fmt: off
            pd.DataFrame.from_dict(
                {
                    "index": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                    "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                    "data": [
                        [pd.Period("2020"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 11.0,],
                        [pd.Period("2020"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 2.0,],
                        [pd.Period("2020"),  3, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 5.0,],
                        [pd.Period("2021"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 57.75,],
                        [pd.Period("2021"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 10.50,],
                        [pd.Period("2021"),  3, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 26.25,],
                        [pd.Period("2022"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 132.0,],
                        [pd.Period("2022"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 24.0,],
                        [pd.Period("2022"),  3, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 60.0,],
                    ],
                    "index_names": [None],
                    "column_names": [None],
                },
                orient="tight",
            ),
            # fmt: on
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="session")
def expected_period_metrics():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3],
         'columns': [PERIOD_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'aai', 'USD', 328.5/3],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_100', 'USD', 9125/3],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_250', 'USD', 68437.5/3],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  ],
         'index_names': [None],
         'column_names': [None]},
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_interp_period_wgroup(expected_period_metrics):
    return pd.concat(
        [
            # fmt: off
            pd.DataFrame.from_dict(
                {'index': [0, 1, 2],
                 'columns': [PERIOD_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
                 'data': [
                     [f"{EXPOSURE_REF_YEAR} to 2022", 1, NO_MEASURE_VALUE, 'aai', 'USD', 66.91666666666667],
                     [f"{EXPOSURE_REF_YEAR} to 2022", 2, NO_MEASURE_VALUE, 'aai', 'USD', 12.166666666666666],
                     [f"{EXPOSURE_REF_YEAR} to 2022", 3, NO_MEASURE_VALUE, 'aai', 'USD', 30.416666666666668],
                          ],
                 'index_names': [None],
                 'column_names': [None]},
                orient="tight"
            ),
            expected_period_metrics
            # fmt: on
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="session")
def expected_interp_metrics_rpchange():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 18.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 94.5],
                  # Above should indeed not be 216+18 / 2 and slightly
                  # because as we interpolate each contributor separately,
                  # the interaction term grows slower.
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 216.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_500', 'USD', 3750.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_500', 'USD', 19687.5],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_500', 'USD', 45000.0]],
         'index_names': [None],
         'column_names': [None]
         },
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_period_metrics_rpchange():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3],
         'columns': [PERIOD_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'aai', 'USD', 328.5/3],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_20', 'USD', 0.],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_500', 'USD', 22812.5],
                  ],
         'index_names': [None],
         'column_names': [None]},
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_interp_metrics_ratechange():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
         'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 18.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 90.0],
                  # Above should indeed not be 216+18 / 2 and slightly
                  # because as we interpolate each contributor separately,
                  # the interaction term grows slower.
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 195.9183673469],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 0.0],
                  [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 500.0],
                  [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 2500.0],
                  [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 5442.176870]],
         'index_names': [None],
         'column_names': [None]
         },
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_period_metrics_ratechange():
    # fmt: off
    return pd.DataFrame.from_dict(
        {'index': [0, 1, 2, 3],
         'columns': [PERIOD_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
         'data': [[f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'aai', 'USD', 101.3061224489],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_100', 'USD', 2814.0589],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                  [f"{EXPOSURE_REF_YEAR} to 2022", 'All', NO_MEASURE_VALUE, 'rp_50', 'USD', 0.],
                  ],
         'index_names': [None],
         'column_names': [None]},
        orient="tight"
    )
    # fmt: on


@pytest.fixture(scope="session")
def expected_interp_metrics_contributions():
    return pd.DataFrame.from_dict(
        # fmt: off
        {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
         'data': [
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 45.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 90.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 9.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 18.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 22.5],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 90.0]],
             'index_names': [None],
            'column_names': [None]},
        # fmt: on
        orient="tight",
    )


@pytest.fixture(scope="session")
def expected_interp_metrics_contributions_vulchange():
    return pd.DataFrame.from_dict(
        # fmt: off
        {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
         'data': [
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 18.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 45.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 90.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 9.0],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 18.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', -4.5],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', -9.0],
             [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 0.0],
             [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 3.375],
             [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', -9.0]],
             'index_names': [None],
            'column_names': [None]},
        # fmt: on
        orient="tight",
    )


def test_interpolated_trajectory(
    snapshot_base,
    snapshot_future_interp,
    expected_interp_metrics,
    expected_period_metrics,
):
    interp_traj = InterpolatedRiskTrajectory(
        [snapshot_base, snapshot_future_interp], return_periods=[50, 100, 250]
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_period_metrics,
        check_dtype=False,
        check_categorical=False,
    )


def test_interp_trajectory_with_group(
    snapshot_factory, expected_interp_metrics_wgroup, expected_interp_period_wgroup
):
    snapshot_base = snapshot_factory(
        group_id=CATEGORIES,
    )
    snapshot_future = snapshot_factory(
        date=2022,
        exposure_value_factor=6.0,
        hazard_intensity_factor=2.0,
        group_id=CATEGORIES,
    )
    interp_traj = InterpolatedRiskTrajectory(
        [snapshot_base, snapshot_future], return_periods=[50, 100, 250]
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics_wgroup,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_interp_period_wgroup,
        check_dtype=False,
        check_categorical=False,
    )


def test_interp_trajectory_change_rp(
    snapshot_base,
    snapshot_future_interp,
    expected_interp_metrics,
    expected_interp_metrics_rpchange,
    expected_period_metrics,
    expected_period_metrics_rpchange,
):
    interp_traj = InterpolatedRiskTrajectory(
        [snapshot_base, snapshot_future_interp], return_periods=[20, 50, 500]
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics_rpchange,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_period_metrics_rpchange,
        check_dtype=False,
        check_categorical=False,
    )

    # Also check change to other return period
    interp_traj.return_periods = [50, 100, 250]
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_period_metrics,
        check_dtype=False,
        check_categorical=False,
    )


def test_interp_trajectory_risk_disc_rate(
    snapshot_base,
    snapshot_future_interp,
    expected_interp_metrics,
    expected_interp_metrics_ratechange,
    expected_period_metrics,
    expected_period_metrics_ratechange,
):
    risk_disc_rate = DiscRates(
        years=np.array(range(2020, 2023)), rates=np.ones(3) * 0.05
    )
    interp_traj = InterpolatedRiskTrajectory(
        [snapshot_base, snapshot_future_interp], risk_disc_rates=risk_disc_rate
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics_ratechange,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_period_metrics_ratechange,
        check_dtype=False,
        check_categorical=False,
    )

    # Also check change to other return period
    interp_traj.return_periods = [50, 100, 250]
    interp_traj.risk_disc_rates = None
    pd.testing.assert_frame_equal(
        interp_traj.per_date_risk_metrics(),
        expected_interp_metrics,
        check_dtype=False,
        check_categorical=False,
    )
    pd.testing.assert_frame_equal(
        interp_traj.per_period_risk_metrics(),
        expected_period_metrics,
        check_dtype=False,
        check_categorical=False,
    )


def test_interp_trajectory_risk_contributions(
    snapshot_base, snapshot_future_interp, expected_interp_metrics_contributions
):
    interp_traj = InterpolatedRiskTrajectory([snapshot_base, snapshot_future_interp])
    pd.testing.assert_frame_equal(
        interp_traj.risk_contributions_metrics(),
        expected_interp_metrics_contributions,
        check_dtype=False,
        check_categorical=False,
    )


def test_interp_trajectory_risk_contributions_vulchange(
    snapshot_base,
    snapshot_future_interp_vulchange,
    expected_interp_metrics_contributions_vulchange,
):
    interp_traj = InterpolatedRiskTrajectory(
        [snapshot_base, snapshot_future_interp_vulchange]
    )
    pd.testing.assert_frame_equal(
        interp_traj.risk_contributions_metrics(),
        expected_interp_metrics_contributions_vulchange,
        check_dtype=False,
        check_categorical=False,
    )
