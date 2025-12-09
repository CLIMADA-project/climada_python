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
import pandas as pd

from climada.util.forecast import Forecast


def test_forecast_init():
    """Test initialization of Forecast class."""
    forecast = Forecast()
    npt.assert_array_equal(forecast.lead_time, np.array([]))
    npt.assert_array_equal(forecast.member, np.array([]))

    forecast = Forecast(member=np.array([1, 2]))
    npt.assert_array_equal(forecast.member, np.array([1, 2]), strict=True)

    forecast = Forecast(lead_time=np.array([6, 12], dtype="timedelta64[h]"))
    npt.assert_array_equal(
        forecast.lead_time, np.array([6, 12], dtype="timedelta64[h]"), strict=True
    )

    forecast = Forecast(lead_time=np.array([1, 2]), member=[3, 4])
    npt.assert_array_equal(forecast.lead_time, np.array([1, 2]), strict=True)
    npt.assert_array_equal(forecast.member, np.array([3, 4]), strict=True)
    assert isinstance(forecast.member, np.ndarray)

    # Test with datetime64 including seconds
    lead_times_seconds = pd.timedelta_range(start="1 day", periods=4).to_numpy()
    forecast = Forecast(lead_time=lead_times_seconds, member=[1, 2, 3])
    npt.assert_array_equal(forecast.lead_time, lead_times_seconds, strict=True)
    assert forecast.lead_time.dtype == np.dtype("timedelta64[ns]")


def test_idx_member():
    """Test idx_member method of Forecast class."""
    forecast = Forecast(member=np.array([1, 2, 3, 4]))

    idx = forecast.idx_member(1)
    npt.assert_array_equal(idx, np.array([True, False, False, False]), strict=True)

    idx = forecast.idx_member(np.array([2, 4]))
    npt.assert_array_equal(idx, np.array([False, True, False, True]), strict=True)

    idx = forecast.idx_member([2, 4])
    npt.assert_array_equal(idx, np.array([False, True, False, True]), strict=True)

    idx = forecast.idx_member(None)
    npt.assert_array_equal(idx, np.array([False, False, False, False]), strict=True)

    # Try once with inconsitent types
    forecast = Forecast(member=np.array(["1", -2, np.nan]))
    npt.assert_array_equal(
        forecast.idx_member([np.nan, "1"]), np.array([True, False, True]), strict=True
    )


def test_idx_lead_time():
    """Test idx_lead_time method of Forecast class."""
    forecast = Forecast(
        lead_time=pd.timedelta_range(start="1 day", periods=4).to_numpy()
    )

    idx = forecast.idx_lead_time(
        pd.timedelta_range(start="1 day", periods=4).to_numpy()[::2]
    )
    npt.assert_array_equal(idx, np.array([True, False, True, False]), strict=True)

    idx = forecast.idx_lead_time(
        pd.timedelta_range(start="1 day", periods=4).to_numpy()[0]
    )
    npt.assert_array_equal(idx, np.array([True, False, False, False]), strict=True)

    idx = forecast.idx_lead_time(None)
    npt.assert_array_equal(idx, np.array([False, False, False, False]), strict=True)
