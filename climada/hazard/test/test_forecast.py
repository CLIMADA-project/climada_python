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
from scipy import sparse
from scipy.sparse import csr_matrix

from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
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


class TestSelect:

    @pytest.mark.parametrize(
        "var, var_select",
        [("event_id", "event_id"), ("event_name", "event_names"), ("date", "date")],
    )
    def test_base_class_select(
        self, haz_fc, lead_time, member, haz_kwargs, var, var_select
    ):
        """Check if Hazard.select works on the derived class"""

        select_mask = np.array([3, 2])
        ordered_select_mask = np.array([3, 2])
        if var == "date":
            # Date needs to be a valid delta
            select_mask = np.array([2, 3])
            ordered_select_mask = np.array([2, 3])

        var_value = np.array(haz_kwargs[var])[select_mask]
        # event_name is a list, convert to numpy array for indexing
        haz_fc_sel = haz_fc.select(**{var_select: var_value})
        # Note: order is preserved
        npt.assert_array_equal(
            haz_fc_sel.event_id,
            haz_fc.event_id[ordered_select_mask],
        )
        npt.assert_array_equal(
            haz_fc_sel.event_name,
            np.array(haz_fc.event_name)[ordered_select_mask],
        )
        npt.assert_array_equal(haz_fc_sel.date, haz_fc.date[ordered_select_mask])
        npt.assert_array_equal(
            haz_fc_sel.frequency, haz_fc.frequency[ordered_select_mask]
        )
        npt.assert_array_equal(haz_fc_sel.member, member[ordered_select_mask])
        npt.assert_array_equal(haz_fc_sel.lead_time, lead_time[ordered_select_mask])
        npt.assert_array_equal(
            haz_fc_sel.intensity.todense(),
            haz_fc.intensity.todense()[ordered_select_mask],
        )
        npt.assert_array_equal(
            haz_fc_sel.fraction.todense(),
            haz_fc.fraction.todense()[ordered_select_mask],
        )

        assert haz_fc_sel.centroids == haz_fc.centroids

    def test_derived_select_single(self, haz_fc, lead_time, member):
        haz_fc_select = haz_fc.select(member=[3, 0])
        idx = np.array([0, 3])
        npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[idx])
        npt.assert_array_equal(haz_fc_select.member, member[idx])
        npt.assert_array_equal(haz_fc_select.lead_time, lead_time[idx])

        haz_fc_select = haz_fc.select(lead_time=lead_time[np.array([3, 0])])
        npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[idx])
        npt.assert_array_equal(haz_fc_select.member, member[idx])
        npt.assert_array_equal(haz_fc_select.lead_time, lead_time[idx])

    def test_derived_select_intersections(self, haz_fc, lead_time, member, haz_kwargs):
        haz_fc_select = haz_fc.select(event_id=[1, 4], member=[0, 1, 2])
        npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[np.array([0])])

        haz_fc_select = haz_fc.select(
            event_id=[1, 2, 4], member=[0, 1, 2], lead_time=lead_time[1:3]
        )
        npt.assert_array_equal(haz_fc_select.event_id, haz_fc.event_id[np.array([1])])

        # Test "outer"
        haz_fc2 = HazardForecast(
            lead_time=lead_time, member=np.zeros_like(member, dtype="int"), **haz_kwargs
        )
        haz_fc_select = haz_fc2.select(event_id=[1, 2, 4], member=[0])
        npt.assert_array_equal(haz_fc_select.event_id, [1, 2, 4])
        npt.assert_array_equal(haz_fc_select.member, [0, 0, 0])

    def test_derived_select_null(self, haz_fc, haz_kwargs):
        haz_fc_select = haz_fc.select()
        assert_hazard_kwargs(haz_fc_select, **haz_kwargs)

        with pytest.raises(IndexError):
            haz_fc.select(event_id=[-1])
        with pytest.raises(IndexError):
            haz_fc.select(member=[-1])
        with pytest.raises(IndexError):
            haz_fc.select(
                lead_time=[np.timedelta64("2", "Y").astype("timedelta64[ns]")]
            )


def test_check_sizes(haz_fc):
    """Test that _check_sizes validates matching lengths"""
    # Should pass with matching lengths
    haz_fc._check_sizes()

    # Test with mismatched member length - manipulate after creation
    haz_fc_bad = HazardForecast(
        lead_time=haz_fc.lead_time,
        member=haz_fc.member,
        event_id=haz_fc.event_id,
        event_name=haz_fc.event_name,
        date=haz_fc.date,
        haz_type=haz_fc.haz_type,
        units=haz_fc.units,
        centroids=haz_fc.centroids,
        intensity=haz_fc.intensity,
        fraction=haz_fc.fraction,
    )
    # Manipulate member array directly to bypass __init__ validation
    haz_fc_bad.member = haz_fc.member[:-1]
    with pytest.raises(ValueError, match="Forecast.member"):
        haz_fc_bad._check_sizes()

    # Test with mismatched lead_time length
    haz_fc_bad2 = HazardForecast(
        lead_time=haz_fc.lead_time,
        member=haz_fc.member,
        event_id=haz_fc.event_id,
        event_name=haz_fc.event_name,
        date=haz_fc.date,
        haz_type=haz_fc.haz_type,
        units=haz_fc.units,
        centroids=haz_fc.centroids,
        intensity=haz_fc.intensity,
        fraction=haz_fc.fraction,
    )
    # Manipulate lead_time array directly to bypass __init__ validation
    haz_fc_bad2.lead_time = haz_fc.lead_time[:-1]
    with pytest.raises(ValueError, match="Forecast.lead_time"):
        haz_fc_bad2._check_sizes()


def test_set_event_attrs_from_forecast_dims():
    """Test that _set_event_attrs_from_forecast_dims generates event attributes correctly"""
    lead_time = pd.timedelta_range("3h", periods=4, freq="2h").to_numpy()
    member = np.array([1, 2, 3, 4])

    # Create a HazardForecast without event_name and date (they will be auto-generated)
    haz_fc = HazardForecast(
        lead_time=lead_time,
        member=member,
        haz_type="TC",
        units="m/s",
        event_id=np.array([10, 20, 30, 40]),
        intensity=sparse.csr_matrix(np.random.rand(4, 3)),
        centroids=Centroids(lat=np.array([1, 2, 3]), lon=np.array([4, 5, 6])),
    )

    # Check that event_name was auto-generated
    assert len(haz_fc.event_name) == 4
    assert haz_fc.event_name[0] == "lt_3h_m_1"
    assert haz_fc.event_name[1] == "lt_5h_m_2"
    assert haz_fc.event_name[2] == "lt_7h_m_3"
    assert haz_fc.event_name[3] == "lt_9h_m_4"

    # Check that date was set to zeros
    npt.assert_array_equal(haz_fc.date, np.zeros(4, dtype=int))

    # Test that it raises error when lead_time and member have different lengths
    haz_fc_bad = HazardForecast(
        lead_time=lead_time,
        member=member,
        haz_type="TC",
        units="m/s",
        event_id=np.array([10, 20, 30, 40]),
        event_name=["a", "b", "c", "d"],  # Provide event_name to bypass auto-generation
        date=np.array([1, 2, 3, 4]),
        intensity=sparse.csr_matrix(np.random.rand(4, 3)),
        centroids=Centroids(lat=np.array([1, 2, 3]), lon=np.array([4, 5, 6])),
    )
    # Now manipulate arrays to create mismatch and call the method directly
    haz_fc_bad.member = member[:-1]
    with pytest.raises(ValueError, match="Length mismatch"):
        haz_fc_bad._set_event_attrs_from_forecast_dims()


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
