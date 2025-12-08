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

Tests for Forecast base class.
"""

import numpy as np
import pytest

from climada.util.forecast import Forecast


class TestForecastInit:
    """Test Forecast initialization"""

    def test_init_with_none_values(self):
        """Test initialization with None values for lead_time and member"""
        forecast = Forecast(lead_time=None, member=None)

        # Check that lead_time is an empty numpy array
        assert isinstance(forecast.lead_time, np.ndarray)
        assert forecast.lead_time.size == 0

        # Check that member is an empty numpy array
        assert isinstance(forecast.member, np.ndarray)
        assert forecast.member.size == 0

    def test_init_with_empty_objects(self):
        """Test initialization with empty objects for lead_time and member"""
        forecast = Forecast(lead_time=[], member=[])

        # Check that lead_time is an empty numpy array
        assert isinstance(forecast.lead_time, np.ndarray)
        assert forecast.lead_time.size == 0

        # Check that member is a list (passed directly)
        assert isinstance(forecast.member, list)
        assert len(forecast.member) == 0

    def test_init_default(self):
        """Test initialization with no arguments (default behavior)"""
        forecast = Forecast()

        # Check that lead_time is an empty numpy array
        assert isinstance(forecast.lead_time, np.ndarray)
        assert forecast.lead_time.size == 0

        # Check that member is an empty numpy array
        assert isinstance(forecast.member, np.ndarray)
        assert forecast.member.size == 0
