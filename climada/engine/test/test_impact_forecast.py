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

Tests for Impact Forecast.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from climada.engine import Impact, ImpactForecast

from .test_impact import impact_kwargs as imp_kwargs


@pytest.fixture
def impact_kwargs():
    return imp_kwargs()


@pytest.fixture
def impact(impact_kwargs):
    return Impact(**impact_kwargs)


@pytest.fixture
def lead_time():
    return pd.timedelta_range(start="1 day", periods=6).to_numpy()


@pytest.fixture
def member():
    return np.arange(6)


@pytest.fixture
def impact_forecast(impact, lead_time, member):
    return ImpactForecast.from_impact(impact, lead_time=lead_time, member=member)


class TestImpactForecastInit:
    def assert_impact_kwargs(self, impact: Impact, **kwargs):
        for key, value in kwargs.items():
            attr = getattr(impact, key)
            if isinstance(value, (np.ndarray, list)):
                npt.assert_array_equal(attr, value)
            elif isinstance(value, csr_matrix):
                npt.assert_array_equal(attr.todense(), value.todense())
            else:
                assert attr == value

    def test_impact_forecast_init(self, impact_kwargs, lead_time, member):
        forecast1 = ImpactForecast(
            lead_time=lead_time,
            member=member,
            **impact_kwargs,
        )
        npt.assert_array_equal(forecast1.lead_time, lead_time)
        npt.assert_array_equal(forecast1.member, member)
        self.assert_impact_kwargs(forecast1, **impact_kwargs)

    def test_impact_forecast_from_impact(
        self, impact_forecast, impact_kwargs, lead_time, member
    ):
        npt.assert_array_equal(impact_forecast.lead_time, lead_time)
        npt.assert_array_equal(impact_forecast.member, member)
        self.assert_impact_kwargs(impact_forecast, **impact_kwargs)


def test_impact_forecast_select(impact_forecast, lead_time, member, impact_kwargs):
    """Check if Impact.select works on the derived class"""
    event_ids = impact_kwargs["event_id"][np.array([2, 0])]
    impact_fc = impact_forecast.select(event_ids=event_ids)
    # NOTE: Events keep their original order
    npt.assert_array_equal(
        impact_fc.event_id, impact_forecast.event_id[np.array([0, 2])]
    )
    npt.assert_array_equal(impact_fc.member, member[np.array([0, 2])])
    npt.assert_array_equal(impact_fc.lead_time, lead_time[np.array([0, 2])])


@pytest.mark.skip("Concat from base class does not work")
def test_impact_forecast_concat(impact_forecast, member):
    """Check if Impact.concat works on the derived class"""
    impact_fc = ImpactForecast.concat(
        [impact_forecast, impact_forecast], reset_event_ids=True
    )
    npt.assert_array_equal(impact_fc.member, np.concatenate([member, member]))


def test_impact_forecast_blocked_methods(impact_forecast):
    """Check if ImpactForecast.exceedance_freq_curve raises NotImplementedError"""
    with pytest.raises(NotImplementedError):
        impact_forecast.local_exceedance_impact(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.local_return_period(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.calc_freq_curve(np.array([10, 50, 100]))


def test_impact_forecast_mean_min_max(impact_forecast):
    """Check mean, min, and max methods for ImpactForecast"""
    imp_fcst_mean = impact_forecast.mean()
    imp_fcst_min = impact_forecast.min()
    imp_fcst_max = impact_forecast.max()
    # sparse.csr_matrix(
    #        np.array([[0, 0], [1, 1], [2, 2], [3, 3], [30, 30], [31, 31]])

    # assert imp_mat
    npt.assert_array_equal(
        imp_fcst_mean.imp_mat.todense(), impact_forecast.imp_mat.todense().mean(axis=0)
    )
    npt.assert_array_equal(imp_fcst_min.imp_mat.todense(), np.array([[0, 0]]))
    npt.assert_array_equal(imp_fcst_max.imp_mat.todense(), np.array([[31, 31]]))
    # assert at_event
    npt.assert_array_equal(
        imp_fcst_mean.at_event, impact_forecast.at_event.mean()
    )  # 134/6
    npt.assert_array_equal(imp_fcst_min.at_event, impact_forecast.at_event.min())
    npt.assert_array_equal(imp_fcst_max.at_event, impact_forecast.at_event.max())

    # check that attributes where reduced correctly
    assert imp_fcst_mean.event_name[0] == "mean"
    assert imp_fcst_min.event_name[0] == "min"
    assert imp_fcst_max.event_name[0] == "max"
    assert imp_fcst_mean.event_id[0] == 0
    assert imp_fcst_min.event_id[0] == 0
    assert imp_fcst_max.event_id[0] == 0
    assert imp_fcst_mean.frequency == 0
    assert imp_fcst_min.frequency == 0
    assert imp_fcst_max.frequency == 0
    assert imp_fcst_mean.date == impact_forecast.date.min()
    assert imp_fcst_min.date == impact_forecast.date.min()
    assert imp_fcst_max.date == impact_forecast.date.min()
