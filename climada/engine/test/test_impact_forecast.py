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


@pytest.mark.parametrize(
    "var, var_select",
    [("event_id", "event_ids"), ("event_name", "event_names"), ("date", "dates")],
)
def test_impact_forecast_select_events(
    impact_forecast, lead_time, member, impact_kwargs, var, var_select
):
    """Check if Impact.select works on the derived class"""
    select_mask = np.array([2, 1])
    ordered_select_mask = np.array([1, 2])
    if var == "date":
        # Date needs to be a valid delta
        select_mask = np.array([1, 2])
        ordered_select_mask = np.array([1, 2])

    var_value = np.array(impact_kwargs[var])[select_mask]
    # event_name is a list, convert to numpy array for indexing
    impact_fc = impact_forecast.select(**{var_select: var_value})
    # NOTE: Events keep their original order
    npt.assert_array_equal(
        impact_fc.event_id,
        impact_forecast.event_id[ordered_select_mask],
    )
    npt.assert_array_equal(
        impact_fc.event_name,
        np.array(impact_forecast.event_name)[ordered_select_mask],
    )
    npt.assert_array_equal(impact_fc.date, impact_forecast.date[ordered_select_mask])
    npt.assert_array_equal(
        impact_fc.frequency, impact_forecast.frequency[ordered_select_mask]
    )
    npt.assert_array_equal(impact_fc.member, member[ordered_select_mask])
    npt.assert_array_equal(impact_fc.lead_time, lead_time[ordered_select_mask])
    npt.assert_array_equal(
        impact_fc.imp_mat.todense(),
        impact_forecast.imp_mat.todense()[ordered_select_mask],
    )


def test_impact_forecast_select_exposure(
    impact_forecast, lead_time, member, impact_kwargs
):
    """Check if Impact.select works on the derived class"""
    exp_col = 0
    select_mask = np.array([exp_col])
    coord_exp = impact_kwargs["coord_exp"][select_mask]
    impact_fc = impact_forecast.select(coord_exp=coord_exp)
    npt.assert_array_equal(impact_fc.member, member)
    npt.assert_array_equal(impact_fc.lead_time, lead_time)
    npt.assert_array_equal(
        impact_fc.imp_mat.todense(), impact_forecast.imp_mat.todense()[:, exp_col]
    )


@pytest.mark.skip("Concat from base class does not work")
def test_impact_forecast_concat(impact_forecast, member):
    """Check if Impact.concat works on the derived class"""
    impact_fc = ImpactForecast.concat(
        [impact_forecast, impact_forecast], reset_event_ids=True
    )
    npt.assert_array_equal(impact_fc.member, np.concatenate([member, member]))
