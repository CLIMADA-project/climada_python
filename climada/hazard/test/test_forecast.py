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

import datetime as dt
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
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

    # Check most hazard kwargs (excluding event_name and date which are auto-generated)
    check_kwargs = {
        k: v for k, v in haz_kwargs.items() if k not in ["event_name", "date"]
    }
    assert_hazard_kwargs(haz_fc_from_haz, **check_kwargs)

    # Check that event_name and date are auto-generated from lead_time and member
    assert len(haz_fc_from_haz.event_name) == len(lead_time)
    assert len(haz_fc_from_haz.date) == len(lead_time)
    # Date should be all zeros for forecast
    npt.assert_array_equal(haz_fc_from_haz.date, np.zeros(len(lead_time), dtype=int))
    # Event names should be formatted with lead_time and member
    assert haz_fc_from_haz.event_name[0] == f"lt_1h_m_{member[0]}"


def test_hazard_forecast_select(haz_fc, lead_time, member):
    """Check if Hazard.select works on the derived class"""
    haz_fc_select = haz_fc.select(event_id=[4, 1])
    # NOTE: Events keep their original order
    npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[np.array([3, 0])])
    npt.assert_array_equal(haz_fc_select.member, member[np.array([3, 0])])
    npt.assert_array_equal(haz_fc_select.lead_time, lead_time[np.array([3, 0])])


@pytest.fixture(scope="module")
def forecast_netcdf_file(tmp_path_factory):
    """Create a NetCDF file with forecast data structure"""
    tmpdir = tmp_path_factory.mktemp("forecast_data")
    netcdf_path = tmpdir / "forecast_data.nc"

    n_eps = 5
    n_lead_time = 4
    n_lat = 3
    n_lon = 4

    eps = np.array([3, 8, 13, 16, 20])
    ref_time = np.array([dt.datetime(2025, 12, 8, 6, 0, 0)], dtype="datetime64[ns]")
    lead_time_vals = pd.timedelta_range("3h", periods=n_lead_time, freq="2h").to_numpy()
    lon = np.array([10.0, 10.5, 11.0, 11.5])
    lat = np.array([45.0, 45.5, 46.0])

    valid_time = ref_time[0] + lead_time_vals

    np.random.seed(42)
    intensity = np.random.rand(n_eps, 1, n_lead_time, n_lat, n_lon) * 10

    # Create xarray Dataset
    dset = xr.Dataset(
        {
            "__xarray_dataarray_variable__": (
                ["eps", "ref_time", "lead_time", "lat", "lon"],
                intensity,
            ),
        },
        coords={
            "eps": eps,
            "ref_time": ref_time,
            "lead_time": lead_time_vals,
            "lon": lon,
            "lat": lat,
            "valid_time": (["lead_time"], valid_time),
        },
    )
    dset.to_netcdf(netcdf_path)

    return {
        "path": netcdf_path,
        "n_eps": n_eps,
        "n_lead_time": n_lead_time,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "eps": eps,
        "lead_time": lead_time_vals,
        "lon": lon,
        "lat": lat,
    }


def test_from_xarray_raster_basic(forecast_netcdf_file):
    """Test basic loading of forecast hazard from xarray"""
    haz_fc = HazardForecast.from_xarray_raster(
        forecast_netcdf_file["path"],
        hazard_type="PR",
        intensity_unit="mm/h",
        coordinate_vars={
            "longitude": "lon",
            "latitude": "lat",
            "lead_time": "lead_time",
            "member": "eps",
        },
    )

    # Check that it's a HazardForecast instance
    assert isinstance(haz_fc, HazardForecast)

    # Check dimensions - after stacking, we should have n_eps * n_lead_time events
    expected_n_events = (
        forecast_netcdf_file["n_eps"] * forecast_netcdf_file["n_lead_time"]
    )
    assert len(haz_fc.event_id) == expected_n_events
    assert len(haz_fc.lead_time) == expected_n_events
    assert len(haz_fc.member) == expected_n_events

    # Check that lead_time and member are correctly extracted
    npt.assert_array_equal(np.unique(haz_fc.member), forecast_netcdf_file["eps"])

    # Check intensity shape (events x centroids)
    expected_n_centroids = forecast_netcdf_file["n_lat"] * forecast_netcdf_file["n_lon"]
    assert haz_fc.intensity.shape == (expected_n_events, expected_n_centroids)

    # Check centroids
    assert len(haz_fc.centroids.lat) == expected_n_centroids
    assert len(haz_fc.centroids.lon) == expected_n_centroids


def test_from_xarray_raster_event_names(forecast_netcdf_file):
    """Test that event names are auto-generated from lead_time and member"""
    haz_fc = HazardForecast.from_xarray_raster(
        forecast_netcdf_file["path"],
        hazard_type="PR",
        intensity_unit="mm/h",
        coordinate_vars={
            "longitude": "lon",
            "latitude": "lat",
            "lead_time": "lead_time",
            "member": "eps",
        },
    )

    # Check that event names are generated with lead_time in hours
    expected_n_events = (
        forecast_netcdf_file["n_eps"] * forecast_netcdf_file["n_lead_time"]
    )
    assert len(haz_fc.event_name) == expected_n_events

    # First event should be for first lead_time and first member
    # Lead time should be in hours (e.g., "lt_3h_m_3")
    first_lead_hours = forecast_netcdf_file["lead_time"][0] / np.timedelta64(1, "h")
    expected_first_name = (
        f"lt_{first_lead_hours:.0f}h_m_{forecast_netcdf_file['eps'][0]}"
    )
    assert haz_fc.event_name[0] == expected_first_name


def test_from_xarray_raster_dates(forecast_netcdf_file):
    """Test that dates are set to 0 for forecast events"""
    haz_fc = HazardForecast.from_xarray_raster(
        forecast_netcdf_file["path"],
        hazard_type="PR",
        intensity_unit="mm/h",
        coordinate_vars={
            "longitude": "lon",
            "latitude": "lat",
            "lead_time": "lead_time",
            "member": "eps",
        },
    )

    # Check that all dates are 0 (undefined for forecast)
    expected_n_events = (
        forecast_netcdf_file["n_eps"] * forecast_netcdf_file["n_lead_time"]
    )
    npt.assert_array_equal(haz_fc.date, np.zeros(expected_n_events, dtype=int))
