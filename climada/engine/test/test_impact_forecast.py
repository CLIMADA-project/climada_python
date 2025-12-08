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


def assert_impact_kwargs(impact: Impact, **kwargs):
    for key, value in kwargs.items():
        attr = getattr(impact, key)
        if isinstance(value, (np.ndarray, list)):
            npt.assert_array_equal(attr, value)
        elif isinstance(value, csr_matrix):
            npt.assert_array_equal(attr.todense(), value.todense())
        else:
            assert attr == value


class TestImpactForecastInit:
    lead_time = pd.date_range("2000-01-01", "2000-01-02", periods=6).to_numpy()
    member = np.arange(6)

    def test_impact_forecast_init(self, impact_kwargs):
        forecast1 = ImpactForecast(
            lead_time=self.lead_time,
            member=self.member,
            **impact_kwargs,
        )
        npt.assert_array_equal(forecast1.lead_time, self.lead_time)
        npt.assert_array_equal(forecast1.member, self.member)
        assert_impact_kwargs(forecast1, **impact_kwargs)

    def test_impact_forecast_from_impact(self, impact, impact_kwargs):
        forecast = ImpactForecast.from_impact(
            impact, lead_time=self.lead_time, member=self.member
        )
        npt.assert_array_equal(forecast.lead_time, self.lead_time)
        npt.assert_array_equal(forecast.member, self.member)
        assert_impact_kwargs(forecast, **impact_kwargs)
