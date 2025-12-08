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
def hazard_kwargs_fixture():
    from climada.hazard.test.test_base import hazard_kwargs as get_hazard_kwargs

    return get_hazard_kwargs()


@pytest.fixture
def dummy_hazard(hazard_kwargs_fixture):
    return Hazard(**hazard_kwargs_fixture)


def test_init_hazard_forecast(hazard_kwargs_fixture):
    haz_fc = HazardForecast(
        lead_time=np.array(
            ["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"
        ),
        member=np.array([0, 1]),
        **hazard_kwargs_fixture,
    )
    assert isinstance(haz_fc, HazardForecast)
    npt.assert_array_equal(
        haz_fc.lead_time,
        np.array(["2024-01-01T00:00:00", "2024-01-01T00:01:00"], dtype="datetime64[s]"),
    )
    assert haz_fc.lead_time.dtype == "datetime64[s]"
    npt.assert_array_equal(haz_fc.member, np.array([0, 1]))
    assert haz_fc.haz_type == hazard_kwargs_fixture["haz_type"]
    assert haz_fc.pool == hazard_kwargs_fixture["pool"]
    assert haz_fc.units == hazard_kwargs_fixture["units"]
    assert haz_fc.centroids == hazard_kwargs_fixture["centroids"]
    npt.assert_array_equal(haz_fc.event_id, hazard_kwargs_fixture["event_id"])
    npt.assert_array_equal(haz_fc.frequency, hazard_kwargs_fixture["frequency"])
    assert haz_fc.frequency_unit == hazard_kwargs_fixture["frequency_unit"]
    npt.assert_array_equal(haz_fc.event_name, hazard_kwargs_fixture["event_name"])
    npt.assert_array_equal(haz_fc.date, hazard_kwargs_fixture["date"])
    npt.assert_array_equal(haz_fc.orig, hazard_kwargs_fixture["orig"])
    npt.assert_array_equal(
        haz_fc.intensity.todense(), hazard_kwargs_fixture["intensity"].todense()
    )
    npt.assert_array_equal(
        haz_fc.fraction.todense(), hazard_kwargs_fixture["fraction"].todense()
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
