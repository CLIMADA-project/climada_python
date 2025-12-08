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
import pytest

from climada.hazard.base import Hazard
from climada.hazard.forecast import HazardForecast
from climada.hazard.test.test_base import hazard_kwargs

# --- Examples for fixtures and test organization --- #


@pytest.fixture
def haz_kwargs():
    return hazard_kwargs()


@pytest.fixture
def dummy_hazard(haz_kwargs):
    return Hazard(**haz_kwargs)


def test_init_hazard_forecast(haz_kwargs):
    haz_fc = HazardForecast(
        lead_time=np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"
        ),
        member=np.array([0, 1]),
        **haz_kwargs,
    )
    assert isinstance(haz_fc, HazardForecast)
    npt.assert_array_equal(
        haz_fc.lead_time,
        np.array(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"),
    )
    assert haz_fc.lead_time.dtype == "datetime64[s]"
    npt.assert_array_equal(haz_fc.member, np.array([0, 1]))
    assert haz_fc.haz_type == haz_kwargs["haz_type"]
    assert haz_fc.pool == haz_kwargs["pool"]
    assert haz_fc.units == haz_kwargs["units"]
    assert haz_fc.centroids == haz_kwargs["centroids"]
    npt.assert_array_equal(haz_fc.event_id, haz_kwargs["event_id"])
    npt.assert_array_equal(haz_fc.frequency, haz_kwargs["frequency"])
    assert haz_fc.frequency_unit == haz_kwargs["frequency_unit"]
    npt.assert_array_equal(haz_fc.event_name, haz_kwargs["event_name"])
    npt.assert_array_equal(haz_fc.date, haz_kwargs["date"])
    npt.assert_array_equal(haz_fc.orig, haz_kwargs["orig"])
    npt.assert_array_equal(
        haz_fc.intensity.todense(), haz_kwargs["intensity"].todense()
    )
    npt.assert_array_equal(haz_fc.fraction.todense(), haz_kwargs["fraction"].todense())


def test_from_hazard(dummy_hazard):
    lead_time = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"
    )
    member = np.array([0, 1])
    haz_fc_from_haz = HazardForecast.from_hazard(
        dummy_hazard, lead_time=lead_time, member=member
    )

    assert isinstance(haz_fc_from_haz, HazardForecast)
    npt.assert_array_equal(haz_fc_from_haz.lead_time, lead_time)
    npt.assert_array_equal(haz_fc_from_haz.member, member)
    assert haz_fc_from_haz.haz_type == dummy_hazard.haz_type
    assert haz_fc_from_haz.pool == dummy_hazard.pool
    assert haz_fc_from_haz.units == dummy_hazard.units
    assert haz_fc_from_haz.centroids == dummy_hazard.centroids
    npt.assert_array_equal(haz_fc_from_haz.event_id, dummy_hazard.event_id)
    npt.assert_array_equal(haz_fc_from_haz.frequency, dummy_hazard.frequency)
    assert haz_fc_from_haz.frequency_unit == dummy_hazard.frequency_unit
    npt.assert_array_equal(haz_fc_from_haz.event_name, dummy_hazard.event_name)
    npt.assert_array_equal(haz_fc_from_haz.date, dummy_hazard.date)
    npt.assert_array_equal(haz_fc_from_haz.orig, dummy_hazard.orig)
    npt.assert_array_equal(
        haz_fc_from_haz.intensity.todense(), dummy_hazard.intensity.todense()
    )
    npt.assert_array_equal(
        haz_fc_from_haz.fraction.todense(), dummy_hazard.fraction.todense()
    )


@pytest.fixture
def hazard():
    return Hazard()


def test_empty_hazard(hazard):
    assert hazard.size == 0
    assert hazard.haz_type == ""


class TestSomething:

    @pytest.fixture(autouse=True)
    def haz_type(self, hazard):
        hazard.haz_type = "foo"

    def test_haz_type(self, hazard):
        assert hazard.haz_type == "foo"
