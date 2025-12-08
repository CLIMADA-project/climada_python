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

Tests for Hazard Forecast.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from climada.hazard.base import Hazard
from climada.hazard.forecast import HazardForecast
from climada.hazard.test.test_base import hazard_kwargs

# --- Examples for fixtures and test organization --- #


@pytest.fixture
def haz_kwargs():
    return hazard_kwargs()


@pytest.fixture
def hazard(haz_kwargs):
    return Hazard(**haz_kwargs)


@pytest.fixture
def lead_time():
    return pd.timedelta_range("1h", periods=6).to_numpy()


@pytest.fixture
def member():
    return np.arange(6)


@pytest.fixture
def haz_fc(lead_time, member, haz_kwargs):
    return HazardForecast(
        lead_time=lead_time,
        member=member,
        **haz_kwargs,
    )


def assert_hazard_kwargs(hazard: Hazard, **kwargs):
    for key, value in kwargs.items():
        attr = getattr(hazard, key)
        if isinstance(value, (np.ndarray, list)):
            npt.assert_array_equal(attr, value)
        elif isinstance(value, csr_matrix):
            npt.assert_array_equal(attr.todense(), value.todense())
        else:
            assert attr == value


def test_init_hazard_forecast(haz_fc, member, lead_time, haz_kwargs):
    assert isinstance(haz_fc, HazardForecast)
    npt.assert_array_equal(haz_fc.lead_time, lead_time)
    assert haz_fc.lead_time.dtype == lead_time.dtype
    npt.assert_array_equal(haz_fc.member, member)
    assert_hazard_kwargs(haz_fc, **haz_kwargs)


def test_from_hazard(lead_time, member, hazard, haz_kwargs):
    haz_fc_from_haz = HazardForecast.from_hazard(
        hazard, lead_time=lead_time, member=member
    )
    assert isinstance(haz_fc_from_haz, HazardForecast)
    npt.assert_array_equal(haz_fc_from_haz.lead_time, lead_time)
    npt.assert_array_equal(haz_fc_from_haz.member, member)
    assert_hazard_kwargs(haz_fc_from_haz, **haz_kwargs)


@pytest.mark.skip("Concat from base class does not work")
def test_hazard_forecast_concat(haz_fc, lead_time, member):
    haz_fc1 = haz_fc.select(event_id=[1, 2])
    haz_fc2 = haz_fc.select(event_id=[3, 4])
    haz_fc_concat = HazardForecast.concat([haz_fc1, haz_fc2])
    assert isinstance(haz_fc_concat, HazardForecast)
    npt.assert_array_equal(
        haz_fc_concat.lead_time, np.concatenate([lead_time, lead_time])
    )
    npt.assert_array_equal(haz_fc_concat.member, np.concatenate([member, member]))
