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
import numpy.testing as npt

from climada.util.forecast import Forecast


def test_forecast_init():
    """Test initialization of Forecast class."""
    forecast = Forecast()
    npt.assert_array_equal(forecast.lead_time, np.array([]))
    npt.assert_array_equal(forecast.member, np.array([]))

    forecast = Forecast(member=np.array([1, 2]))
    npt.assert_array_equal(forecast.member, np.array([1, 2]), strict=True)

    forecast = Forecast(lead_time=np.array([1, 2]))
    npt.assert_array_equal(forecast.lead_time, np.array([1, 2]), strict=True)

    forecast = Forecast(lead_time=np.array([1, 2]), member=[3, 4])
    npt.assert_array_equal(forecast.lead_time, np.array([1, 2]), strict=True)
    npt.assert_array_equal(forecast.member, np.array([3, 4]), strict=True)
    assert isinstance(forecast.member, np.ndarray)

    # Test with datetime64 including seconds
    lead_times_seconds = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01"],
        dtype="datetime64[s]",
    )
    forecast = Forecast(lead_time=lead_times_seconds, member=[1, 2, 3])
    npt.assert_array_equal(forecast.lead_time, lead_times_seconds, strict=True)
    assert forecast.lead_time.dtype == np.dtype("datetime64[s]")
