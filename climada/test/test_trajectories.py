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
            date=date,
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


def expected_static_metrics_from_snapshots(snapshots):
    rows = []
    group_p = False
    for snap in snapshots:
        imp = ImpactCalc(**snap.impact_calc_data).impact()
        curve = imp.calc_freq_curve(DEFAULT_RP)

        rows.append(
            [
                pd.Timestamp(str(snap.date)),
                "All",
                NO_MEASURE_VALUE,
                "aai",
                "USD",
                imp.aai_agg,
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
                    val,
                ]
                for rp, val in zip(curve.return_per, curve.impact)
            ]
        )
        if "group_id" in snap.exposure.gdf.columns:
            group_p = True
            aai_per_group = [
                [
                    pd.Timestamp(str(snap.date)),
                    group,
                    NO_MEASURE_VALUE,
                    "aai",
                    "USD",
                    val,
                ]
                for group, val in zip(snap.exposure.gdf["group_id"], imp.eai_exp)
            ]
            group_df = pd.DataFrame(
                aai_per_group,
                columns=[
                    DATE_COL_NAME,
                    GROUP_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
            )

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
    if group_p:
        res = pd.concat([res, group_df])

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


class TestStaticTrajectory:

    def test_static_trajectory_with_group(self):
        exp0 = reusable_minimal_exposures(group_id=CATEGORIES)
        exp1 = reusable_minimal_exposures(
            group_id=CATEGORIES, increase_value_factor=self.EXP_INCREASE_VALUE_FACTOR
        )
        snap0 = Snapshot(
            exposure=exp0,
            hazard=reusable_minimal_hazard(),
            impfset=reusable_minimal_impfset(),
            date=self.PRESENT_DATE,
        )
        snap1 = Snapshot(
            exposure=exp1,
            hazard=reusable_minimal_hazard(
                intensity_factor=self.HAZ_INCREASE_INTENSITY_FACTOR
            ),
            impfset=reusable_minimal_impfset(),
            date=self.FUTURE_DATE,
        )

        expected_static_metrics = pd.concat(
            [
                self.expected_static_metrics,
                pd.DataFrame.from_dict(
                    # fmt: off
                    {
                        "index": [8, 9, 10, 11],
                        "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                        "data": [
                            [pd.Timestamp(str(self.PRESENT_DATE)), 1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_base_imp.eai_exp[CATEGORIES == 1].sum(),],
                            [pd.Timestamp(str(self.PRESENT_DATE)), 2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_base_imp.eai_exp[CATEGORIES == 2].sum(),],
                            [pd.Timestamp(str(self.FUTURE_DATE)), 1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_future_imp.eai_exp[CATEGORIES == 1].sum(),],
                            [pd.Timestamp(str(self.FUTURE_DATE)), 2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_future_imp.eai_exp[CATEGORIES == 2].sum(),],
                        ],
                        "index_names": [None],
                        "column_names": [None],
                    },
                    # fmt: on
                    orient="tight",
                ),
            ]
        )

        static_traj = StaticRiskTrajectory([snap0, snap1])
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            expected_static_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_static_trajectory_change_rp(self):
        static_traj = StaticRiskTrajectory(
            [self.base_snapshot, self.future_snapshot], return_periods=[10, 60, 1000]
        )
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7],
                "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                "data": [
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_base_imp.aai_agg,],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_future_imp.aai_agg,],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, "rp_10", "USD", 0.0,],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, "rp_10", "USD", 0.0,],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, "rp_60", "USD", 700.0,],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, "rp_60", "USD", 14000.0,],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, "rp_1000", "USD", 1500.0,],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, "rp_1000", "USD", 30000.0,],
                ],
                "index_names": [None],
                "column_names": [None],
            },
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

        # Also check change to other return period
        static_traj.return_periods = DEFAULT_RP
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            self.expected_static_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_static_trajectory_risk_disc_rate(self):
        risk_disc_rate = DiscRates(
            years=np.array(range(self.PRESENT_DATE, 2041)), rates=np.ones(21) * 0.01
        )
        static_traj = StaticRiskTrajectory(
            [self.base_snapshot, self.future_snapshot], risk_disc_rates=risk_disc_rate
        )
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7],
                "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                "data": [
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_base_imp.aai_agg,],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", self.expected_future_imp.aai_agg * ((1 / (1 + 0.01)) ** 20),],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[0]}", "USD", self.expected_base_return_period_impacts[DEFAULT_RP[0]],],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[0]}", "USD", self.expected_future_return_period_impacts[DEFAULT_RP[0]] * ((1 / (1 + 0.01)) ** 20),],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[1]}", "USD", self.expected_base_return_period_impacts[DEFAULT_RP[1]],],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[1]}", "USD", self.expected_future_return_period_impacts[DEFAULT_RP[1]] * ((1 / (1 + 0.01)) ** 20),],
                    [pd.Timestamp(str(self.PRESENT_DATE)),"All",  NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[2]}", "USD", self.expected_base_return_period_impacts[DEFAULT_RP[2]],],
                    [pd.Timestamp(str(self.FUTURE_DATE)), "All", NO_MEASURE_VALUE, f"rp_{DEFAULT_RP[2]}", "USD", self.expected_future_return_period_impacts[DEFAULT_RP[2]] * ((1 / (1 + 0.01)) ** 20),],
                ],
                "index_names": [None],
                "column_names": [None],
            },
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

        # Also check change to other return period
        static_traj.risk_disc_rates = None
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            self.expected_static_metrics,
            check_dtype=False,
            check_categorical=False,
        )


class TestInterpolatedTrajectory(TestCase):
    PRESENT_DATE = 2020
    HAZ_INCREASE_INTENSITY_FACTOR = 2
    EXP_INCREASE_VALUE_FACTOR = 6
    FUTURE_DATE = 2022

    def setUp(self) -> None:
        self.base_snapshot = reusable_snapshot(date=self.PRESENT_DATE)
        self.future_snapshot = reusable_snapshot(
            hazard_intensity_increase_factor=self.HAZ_INCREASE_INTENSITY_FACTOR,
            exposure_value_increase_factor=self.EXP_INCREASE_VALUE_FACTOR,
            date=self.FUTURE_DATE,
        )

        self.expected_base_imp = ImpactCalc(
            **self.base_snapshot.impact_calc_data
        ).impact()
        self.expected_future_imp = ImpactCalc(
            **self.future_snapshot.impact_calc_data
        ).impact()
        # self.group_vector = self.base_snapshot.exposure.gdf[GROUP_ID_COL_NAME]
        self.expected_base_return_period_impacts = {
            rp: imp
            for rp, imp in zip(
                self.expected_base_imp.calc_freq_curve(DEFAULT_RP).return_per,
                self.expected_base_imp.calc_freq_curve(DEFAULT_RP).impact,
            )
        }
        self.expected_future_return_period_impacts = {
            rp: imp
            for rp, imp in zip(
                self.expected_future_imp.calc_freq_curve(DEFAULT_RP).return_per,
                self.expected_future_imp.calc_freq_curve(DEFAULT_RP).impact,
            )
        }

        # fmt: off
        self.expected_interp_metrics = pd.DataFrame.from_dict(
            {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
             'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
             'data': [[ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 20.0],
                      [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 105.0], # This should indeed not be 240+20 / 2 (because we interpolate each contributor separately)
                      [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'aai', 'USD', 240.0],
                      [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 500.0],
                      [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 2625.0],
                      [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_50', 'USD', 6000.0],
                      [ pd.Period(2020), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 1500.0],
                      [ pd.Period(2021), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 7875.0],
                      [ pd.Period(2022), 'All',NO_MEASURE_VALUE, 'rp_100', 'USD', 18000.0]],
             'index_names': [None],
             'column_names': [None]},
            orient="tight"
        )

        self.expected_period_metrics = pd.DataFrame.from_dict(
            {'index': [0, 1, 2, 3],
             'columns': [PERIOD_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME],
             'data': [[f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", 'All', NO_MEASURE_VALUE, 'aai', 'USD', 365.0/3],
                      [f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", 'All', NO_MEASURE_VALUE, 'rp_100', 'USD', 27375/3],
                      [f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", 'All', NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      [f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", 'All', NO_MEASURE_VALUE, 'rp_50', 'USD', 9125.0/3]],
             'index_names': [None],
             'column_names': [None]},
            orient="tight"
        )
        # fmt: on

    def test_interp_trajectory(self):
        interp_traj = InterpolatedRiskTrajectory(
            [self.base_snapshot, self.future_snapshot]
        )
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            self.expected_interp_metrics,
            check_dtype=False,
            check_categorical=False,
        )
        pd.testing.assert_frame_equal(
            interp_traj.per_period_risk_metrics(),
            self.expected_period_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_interp_trajectory_with_group(self):
        exp0 = reusable_minimal_exposures(group_id=CATEGORIES)
        exp1 = reusable_minimal_exposures(
            group_id=CATEGORIES, increase_value_factor=self.EXP_INCREASE_VALUE_FACTOR
        )
        snap0 = Snapshot(
            exposure=exp0,
            hazard=reusable_minimal_hazard(),
            impfset=reusable_minimal_impfset(),
            date=self.PRESENT_DATE,
        )
        snap1 = Snapshot(
            exposure=exp1,
            hazard=reusable_minimal_hazard(
                intensity_factor=self.HAZ_INCREASE_INTENSITY_FACTOR
            ),
            impfset=reusable_minimal_impfset(),
            date=self.FUTURE_DATE,
        )

        expected_interp_metrics = pd.concat(
            [
                self.expected_interp_metrics,
                # fmt: off
                pd.DataFrame.from_dict(
                    {
                        "index": [0, 1, 2, 3, 4, 5],
                        "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                        "data": [
                            [pd.Period("2020"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 15.0,],
                            [pd.Period("2020"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 5.0,],
                            [pd.Period("2021"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 78.75,],
                            [pd.Period("2021"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 26.25,],
                            [pd.Period("2022"),  1, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 180.0,],
                            [pd.Period("2022"),  2, NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 60.0,],
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

        interp_traj = InterpolatedRiskTrajectory([snap0, snap1])
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            expected_interp_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_interp_trajectory_change_rp(self):
        interp_traj = InterpolatedRiskTrajectory(
            [self.base_snapshot, self.future_snapshot], return_periods=[10, 60, 1000]
        )
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                "data": [
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 20.0,],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 105.0,],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 240.0,],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_60", "USD", 700.0],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_60", "USD", 3675.0],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_60", "USD", 8400.0],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_1000", "USD", 1500.0,],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_1000", "USD", 7875.0,],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_1000", "USD", 18000.0,],
                ],
                "index_names": [None],
                "column_names": [None],
            },
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

        # Also check change to other return period
        interp_traj.return_periods = DEFAULT_RP
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            self.expected_interp_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_interp_trajectory_risk_disc_rate(self):
        risk_disc_rate = DiscRates(
            years=np.array(range(2020, 2023)), rates=np.ones(3) * 0.05
        )  # Easy check for year 2021 -> 105.0 * 1/(1+0.05) == 100.
        interp_traj = InterpolatedRiskTrajectory(
            [self.base_snapshot, self.future_snapshot], risk_disc_rates=risk_disc_rate
        )
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "columns": [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
                "data": [
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 20.0,],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 100.0,],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, AAI_METRIC_NAME, "USD", 217.68707482993196,],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_50", "USD", 500.0],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_50", "USD", 2500.0],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_50", "USD", 5442.176870748299,],
                    [pd.Period(2020), "All", NO_MEASURE_VALUE, "rp_100", "USD", 1500.0],
                    [pd.Period(2021), "All", NO_MEASURE_VALUE, "rp_100", "USD", 7500.0],
                    [pd.Period(2022), "All", NO_MEASURE_VALUE, "rp_100", "USD", 16326.530612244896,],
                ],
                "index_names": [None],
                "column_names": [None],
            },
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

        # Also check change to other return period
        interp_traj.risk_disc_rates = None
        pd.testing.assert_frame_equal(
            interp_traj.per_date_risk_metrics(),
            self.expected_interp_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_interp_trajectory_risk_contributions(self):
        interp_traj = InterpolatedRiskTrajectory(
            [self.base_snapshot, self.future_snapshot]
        )
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
             'data': [
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 50.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 100.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 10.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 20.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 25.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 100.0]],
             'index_names': [None],
             'column_names': [None]},
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            interp_traj.risk_contributions_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

        # With changing vulnerability
        hazard = reusable_minimal_hazard()
        impfset1 = ImpactFuncSet(
            [
                ImpactFunc(
                    haz_type=hazard.haz_type,
                    intensity_unit=hazard.units,
                    name="linear",
                    intensity=np.array([0, 100 / 2, 100]),
                    mdd=np.array([0, 0.5, 1]),
                    paa=np.array([1, 1, 1]),
                    id=1,
                ),
            ]
        )
        impfset2 = ImpactFuncSet(
            [
                ImpactFunc(
                    haz_type=hazard.haz_type,
                    intensity_unit=hazard.units,
                    name="linear-half-paa",
                    intensity=np.array([0, 100 / 2, 100]),
                    mdd=np.array([0, 0.5, 1]),
                    paa=np.array([0.5, 0.5, 0.5]),
                    id=1,
                )
            ]
        )
        base_snapshot = Snapshot(
            exposure=reusable_minimal_exposures(),
            hazard=hazard,
            impfset=impfset1,
            date=2020,
        )
        future_snapshot = Snapshot(
            exposure=reusable_minimal_exposures(
                increase_value_factor=self.EXP_INCREASE_VALUE_FACTOR,
            ),
            hazard=reusable_minimal_hazard(
                intensity_factor=self.HAZ_INCREASE_INTENSITY_FACTOR
            ),
            impfset=impfset2,
            date=2022,
        )

        interp_traj = InterpolatedRiskTrajectory([base_snapshot, future_snapshot])
        expected = pd.DataFrame.from_dict(
            # fmt: off
            {'index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             'columns': [DATE_COL_NAME, GROUP_COL_NAME, MEASURE_COL_NAME, METRIC_COL_NAME, UNIT_COL_NAME, RISK_COL_NAME,],
             'data': [
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_BASE_RISK_NAME, 'USD', 20.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 50.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_EXPOSURE_NAME, 'USD', 100.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 10.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_HAZARD_NAME, 'USD', 20.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', -5.0],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_VULNERABILITY_NAME, 'USD', -10.0],
                 [pd.Period(str(2020)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 0.0],
                 [pd.Period(str(2021)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', 3.75],
                 [pd.Period(str(2022)), 'All', NO_MEASURE_VALUE, CONTRIBUTION_INTERACTION_TERM_NAME, 'USD', -10.0]],
             'index_names': [None],
             'column_names': [None]},
            # fmt: on
            orient="tight",
        )
        pd.testing.assert_frame_equal(
            interp_traj.risk_contributions_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )
