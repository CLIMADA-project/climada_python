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


@pytest.fixture
def haz_kwargs():
    return hazard_kwargs()


@pytest.fixture
def hazard(haz_kwargs):
    return Hazard(**haz_kwargs)


@pytest.fixture
def lead_time(haz_kwargs):
    return pd.timedelta_range("1h", periods=len(haz_kwargs["event_id"])).to_numpy()


@pytest.fixture
def member(haz_kwargs):
    return np.arange(len(haz_kwargs["event_id"]))


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


def test_init_hazard_forecast_error(hazard, member, lead_time, haz_kwargs):
    with pytest.raises(ValueError, match="Forecast.lead_time"):
        HazardForecast(lead_time=lead_time[:-2], member=member, **haz_kwargs)
    with pytest.raises(ValueError, match="Forecast.member"):
        HazardForecast.from_hazard(hazard, lead_time=lead_time, member=member[1:])


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


def test_hazard_forecast_select(haz_fc, lead_time, member):
    """Check if Hazard.select works on the derived class"""
    haz_fc_select = haz_fc.select(event_id=[4, 1])
    # NOTE: Events keep their original order
    npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[np.array([3, 0])])
    npt.assert_array_equal(haz_fc_select.member, member[np.array([3, 0])])
    npt.assert_array_equal(haz_fc_select.lead_time, lead_time[np.array([3, 0])])


def test_write_read_hazard_forecast(haz_fc, tmp_path):

    file_name = tmp_path / "test_hazard_forecast.h5"

    haz_fc.write_hdf5(file_name)
    haz_fc_read = HazardForecast.from_hdf5(file_name)

    assert haz_fc_read.lead_time.dtype.kind == np.dtype("timedelta64").kind

    for key in haz_fc.__dict__.keys():
        if key in ["intensity", "fraction"]:
            (haz_fc.__dict__[key] != haz_fc_read.__dict__[key]).nnz == 0
        else:
            # npt.assert_array_equal also works for comparing int, float or list
            npt.assert_array_equal(haz_fc.__dict__[key], haz_fc_read.__dict__[key])
