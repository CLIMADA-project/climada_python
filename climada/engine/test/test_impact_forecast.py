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
    """Check if blocked methods raise NotImplementedError"""
    with pytest.raises(NotImplementedError):
        impact_forecast.local_exceedance_impact(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.local_return_period(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.calc_freq_curve(np.array([10, 50, 100]))
