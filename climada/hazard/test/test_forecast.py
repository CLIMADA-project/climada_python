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

from climada.hazard import Hazard, HazardForecast
from climada.hazard.test.test_base import hazard_kwargs

# --- Examples for fixtures and test organization --- #


@pytest.fixture
def haz_kwargs():
    return hazard_kwargs()


@pytest.fixture
def dummy_hazard(haz_kwargs):
    return Hazard(haz_kwargs())


def test_init_hazard_forecast():
    haz_fc = HazardForecast(
        lead_time=np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"
        ),
        member=np.array([0, 1]),
        **hazard_kwargs,
    )
    assert isinstance(haz_fc, HazardForecast)
    assert np.assert_array_equal(
        haz_fc.lead_time,
        np.array(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"),
    )
    assert haz_fc.lead_time.dtype == "datetime64[s]"
    assert haz_fc.member == np.array([0, 1])
    assert haz_fc.haz_type == haz_kwargs["haz_type"]
    assert haz_fc.pool == haz_kwargs["pool"]
    assert haz_fc.units == haz_kwargs["units"]
    assert haz_fc.centroids == haz_kwargs["centroids"]
    assert haz_fc.event_id == haz_kwargs["event_id"]
    assert haz_fc.frequency == haz_kwargs["frequency"]
    assert haz_fc.frequency_unit == haz_kwargs["frequency_unit"]
    assert haz_fc.event_name == haz_kwargs["event_name"]
    assert haz_fc.date == haz_kwargs["date"]
    assert haz_fc.orig == haz_kwargs["orig"]
    assert haz_fc.intensity == haz_kwargs["intensity"]
    assert haz_fc.fraction == haz_kwargs["fraction"]


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
