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

import pytest

from climada.hazard import Hazard

# --- Examples for fixtures and test organization --- #


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
