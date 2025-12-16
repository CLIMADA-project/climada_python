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

from climada.engine.impact_calc import ImpactCalc
from climada.entity.disc_rates.base import DiscRates
from climada.test.reusable import (
    CATEGORIES,
    reusable_minimal_exposures,
    reusable_minimal_hazard,
    reusable_minimal_impfset,
    reusable_snapshot,
)
from climada.trajectories import InterpolatedRiskTrajectory, StaticRiskTrajectory
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    AAI_PER_GROUP_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RETURN_PERIOD_METRIC_NAME,
    RISK_COL_NAME,
    RP_VALUE_PREFIX,
    UNIT_COL_NAME,
)
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import DEFAULT_RP


class TestStaticTrajectory(TestCase):
    PRESENT_DATE = 2020
    HAZ_INCREASE_INTENSITY_FACTOR = 2
    EXP_INCREASE_VALUE_FACTOR = 10
    FUTURE_DATE = 2040

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
        self.expected_static_metrics = pd.DataFrame.from_dict(
            {'index': [0, 1, 2, 3, 4, 5, 6, 7],
             'columns': ['group', 'date', 'measure', 'metric', 'unit', 'risk'],
             'data': [
                 ['All', pd.Timestamp(str(self.PRESENT_DATE)), NO_MEASURE_VALUE, 'aai', 'USD', self.expected_base_imp.aai_agg],
                 ['All', pd.Timestamp(str(self.FUTURE_DATE)),  NO_MEASURE_VALUE, 'aai', 'USD', self.expected_future_imp.aai_agg],
                 ['All', pd.Timestamp(str(self.PRESENT_DATE)), NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[0]}', 'USD', self.expected_base_return_period_impacts[DEFAULT_RP[0]]],
                 ['All', pd.Timestamp(str(self.FUTURE_DATE)),  NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[0]}', 'USD', self.expected_future_return_period_impacts[DEFAULT_RP[0]]],
                 ['All', pd.Timestamp(str(self.PRESENT_DATE)), NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[1]}', 'USD', self.expected_base_return_period_impacts[DEFAULT_RP[1]]],
                 ['All', pd.Timestamp(str(self.FUTURE_DATE)),  NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[1]}', 'USD', self.expected_future_return_period_impacts[DEFAULT_RP[1]]],
                 ['All', pd.Timestamp(str(self.PRESENT_DATE)), NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[2]}', 'USD', self.expected_base_return_period_impacts[DEFAULT_RP[2]]],
                 ['All', pd.Timestamp(str(self.FUTURE_DATE)),  NO_MEASURE_VALUE, f'rp_{DEFAULT_RP[2]}', 'USD', self.expected_future_return_period_impacts[DEFAULT_RP[2]]],
             ],
             'index_names': [None],
             'column_names': [None]},
            orient="tight"
        )
        # fmt: on

    def test_static_trajectory(self):
        static_traj = StaticRiskTrajectory([self.base_snapshot, self.future_snapshot])
        print(static_traj.per_date_risk_metrics())
        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            self.expected_static_metrics,
            check_dtype=False,
            check_categorical=False,
        )

    def test_static_trajectory_one_snap(self):
        static_traj = StaticRiskTrajectory([self.base_snapshot])
        expected = pd.DataFrame.from_dict(
            {
                "index": [0, 1, 2, 3],
                "columns": [
                    GROUP_COL_NAME,
                    DATE_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
                "data": [
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        self.expected_base_imp.aai_agg,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[0]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[0]],
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[1]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[1]],
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[2]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[2]],
                    ],
                ],
                "index_names": [None],
                "column_names": [None],
            },
            orient="tight",
        )

        pd.testing.assert_frame_equal(
            static_traj.per_date_risk_metrics(),
            expected,
            check_dtype=False,
            check_categorical=False,
        )

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
                    {
                        "index": [8, 9, 10, 11],
                        "columns": [
                            GROUP_COL_NAME,
                            DATE_COL_NAME,
                            MEASURE_COL_NAME,
                            METRIC_COL_NAME,
                            UNIT_COL_NAME,
                            RISK_COL_NAME,
                        ],
                        "data": [
                            [
                                1,
                                pd.Timestamp(str(self.PRESENT_DATE)),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                self.expected_base_imp.eai_exp[CATEGORIES == 1].sum(),
                            ],
                            [
                                2,
                                pd.Timestamp(str(self.PRESENT_DATE)),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                self.expected_base_imp.eai_exp[CATEGORIES == 2].sum(),
                            ],
                            [
                                1,
                                pd.Timestamp(str(self.FUTURE_DATE)),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                self.expected_future_imp.eai_exp[CATEGORIES == 1].sum(),
                            ],
                            [
                                2,
                                pd.Timestamp(str(self.FUTURE_DATE)),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                self.expected_future_imp.eai_exp[CATEGORIES == 2].sum(),
                            ],
                        ],
                        "index_names": [None],
                        "column_names": [None],
                    },
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
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7],
                "columns": [
                    GROUP_COL_NAME,
                    DATE_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
                "data": [
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        self.expected_base_imp.aai_agg,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        self.expected_future_imp.aai_agg,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_10",
                        "USD",
                        0.0,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_10",
                        "USD",
                        0.0,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_60",
                        "USD",
                        700.0,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_60",
                        "USD",
                        14000.0,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_1000",
                        "USD",
                        1500.0,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        "rp_1000",
                        "USD",
                        30000.0,
                    ],
                ],
                "index_names": [None],
                "column_names": [None],
            },
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
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7],
                "columns": [
                    GROUP_COL_NAME,
                    DATE_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
                "data": [
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        self.expected_base_imp.aai_agg,
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        self.expected_future_imp.aai_agg * ((1 / (1 + 0.01)) ** 20),
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[0]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[0]],
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[0]}",
                        "USD",
                        self.expected_future_return_period_impacts[DEFAULT_RP[0]]
                        * ((1 / (1 + 0.01)) ** 20),
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[1]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[1]],
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[1]}",
                        "USD",
                        self.expected_future_return_period_impacts[DEFAULT_RP[1]]
                        * ((1 / (1 + 0.01)) ** 20),
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.PRESENT_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[2]}",
                        "USD",
                        self.expected_base_return_period_impacts[DEFAULT_RP[2]],
                    ],
                    [
                        "All",
                        pd.Timestamp(str(self.FUTURE_DATE)),
                        NO_MEASURE_VALUE,
                        f"rp_{DEFAULT_RP[2]}",
                        "USD",
                        self.expected_future_return_period_impacts[DEFAULT_RP[2]]
                        * ((1 / (1 + 0.01)) ** 20),
                    ],
                ],
                "index_names": [None],
                "column_names": [None],
            },
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
             'columns': ['group', 'date', 'measure', 'metric', 'unit', 'risk'],
             'data': [['All', pd.Period(2020), NO_MEASURE_VALUE, 'aai', 'USD', 20.0],
                      ['All', pd.Period(2021), NO_MEASURE_VALUE, 'aai', 'USD', 105.0], # This should indeed not be 240+20 / 2 (because we interpolate each contributor separately)
                      ['All', pd.Period(2022), NO_MEASURE_VALUE, 'aai', 'USD', 240.0],
                      ['All', pd.Period(2020), NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      ['All', pd.Period(2021), NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      ['All', pd.Period(2022), NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      ['All', pd.Period(2020), NO_MEASURE_VALUE, 'rp_50', 'USD', 500.0],
                      ['All', pd.Period(2021), NO_MEASURE_VALUE, 'rp_50', 'USD', 2625.0],
                      ['All', pd.Period(2022), NO_MEASURE_VALUE, 'rp_50', 'USD', 6000.0],
                      ['All', pd.Period(2020), NO_MEASURE_VALUE, 'rp_100', 'USD', 1500.0],
                      ['All', pd.Period(2021), NO_MEASURE_VALUE, 'rp_100', 'USD', 7875.0],
                      ['All', pd.Period(2022), NO_MEASURE_VALUE, 'rp_100', 'USD', 18000.0]],
             'index_names': [None],
             'column_names': [None]},
            orient="tight"
        )

        self.expected_period_metrics = pd.DataFrame.from_dict(
            {'index': [0, 1, 2, 3],
             'columns': ['group', 'period', 'measure', 'metric', 'unit', 'risk'],
             'data': [['All', f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", NO_MEASURE_VALUE, 'aai', 'USD', 365.0/3],
                      ['All', f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", NO_MEASURE_VALUE, 'rp_100', 'USD', 9125/3],
                      ['All', f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", NO_MEASURE_VALUE, 'rp_20', 'USD', 0.0],
                      ['All', f"{self.PRESENT_DATE} to {self.FUTURE_DATE}", NO_MEASURE_VALUE, 'rp_50', 'USD', 27375.0]],
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
                pd.DataFrame.from_dict(
                    {
                        "index": [0, 1, 2, 3, 4, 5],
                        "columns": [
                            GROUP_COL_NAME,
                            DATE_COL_NAME,
                            MEASURE_COL_NAME,
                            METRIC_COL_NAME,
                            UNIT_COL_NAME,
                            RISK_COL_NAME,
                        ],
                        "data": [
                            [
                                1,
                                pd.Period("2020"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                15.0,
                            ],
                            [
                                2,
                                pd.Period("2020"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                5.0,
                            ],
                            [
                                1,
                                pd.Period("2021"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                78.75,
                            ],
                            [
                                2,
                                pd.Period("2021"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                26.25,
                            ],
                            [
                                1,
                                pd.Period("2022"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                180.0,
                            ],
                            [
                                2,
                                pd.Period("2022"),
                                NO_MEASURE_VALUE,
                                AAI_METRIC_NAME,
                                "USD",
                                60.0,
                            ],
                        ],
                        "index_names": [None],
                        "column_names": [None],
                    },
                    orient="tight",
                ),
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
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "columns": [
                    GROUP_COL_NAME,
                    DATE_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
                "data": [
                    [
                        "All",
                        pd.Period(2020),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        20.0,
                    ],
                    [
                        "All",
                        pd.Period(2021),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        105.0,
                    ],
                    [
                        "All",
                        pd.Period(2022),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        240.0,
                    ],
                    ["All", pd.Period(2020), NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    ["All", pd.Period(2021), NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    ["All", pd.Period(2022), NO_MEASURE_VALUE, "rp_10", "USD", 0.0],
                    ["All", pd.Period(2020), NO_MEASURE_VALUE, "rp_60", "USD", 700.0],
                    ["All", pd.Period(2021), NO_MEASURE_VALUE, "rp_60", "USD", 3675.0],
                    ["All", pd.Period(2022), NO_MEASURE_VALUE, "rp_60", "USD", 8400.0],
                    [
                        "All",
                        pd.Period(2020),
                        NO_MEASURE_VALUE,
                        "rp_1000",
                        "USD",
                        1500.0,
                    ],
                    [
                        "All",
                        pd.Period(2021),
                        NO_MEASURE_VALUE,
                        "rp_1000",
                        "USD",
                        7875.0,
                    ],
                    [
                        "All",
                        pd.Period(2022),
                        NO_MEASURE_VALUE,
                        "rp_1000",
                        "USD",
                        18000.0,
                    ],
                ],
                "index_names": [None],
                "column_names": [None],
            },
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
            {
                "index": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "columns": [
                    GROUP_COL_NAME,
                    DATE_COL_NAME,
                    MEASURE_COL_NAME,
                    METRIC_COL_NAME,
                    UNIT_COL_NAME,
                    RISK_COL_NAME,
                ],
                "data": [
                    [
                        "All",
                        pd.Period(2020),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        20.0,
                    ],
                    [
                        "All",
                        pd.Period(2021),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        100.0,
                    ],
                    [
                        "All",
                        pd.Period(2022),
                        NO_MEASURE_VALUE,
                        AAI_METRIC_NAME,
                        "USD",
                        217.68707482993196,
                    ],
                    ["All", pd.Period(2020), NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    ["All", pd.Period(2021), NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    ["All", pd.Period(2022), NO_MEASURE_VALUE, "rp_20", "USD", 0.0],
                    ["All", pd.Period(2020), NO_MEASURE_VALUE, "rp_50", "USD", 500.0],
                    ["All", pd.Period(2021), NO_MEASURE_VALUE, "rp_50", "USD", 2500.0],
                    [
                        "All",
                        pd.Period(2022),
                        NO_MEASURE_VALUE,
                        "rp_50",
                        "USD",
                        5442.176870748299,
                    ],
                    ["All", pd.Period(2020), NO_MEASURE_VALUE, "rp_100", "USD", 1500.0],
                    ["All", pd.Period(2021), NO_MEASURE_VALUE, "rp_100", "USD", 7500.0],
                    [
                        "All",
                        pd.Period(2022),
                        NO_MEASURE_VALUE,
                        "rp_100",
                        "USD",
                        16326.530612244896,
                    ],
                ],
                "index_names": [None],
                "column_names": [None],
            },
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
