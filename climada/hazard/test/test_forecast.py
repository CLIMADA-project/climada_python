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

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

from climada.hazard.base import Hazard
from climada.hazard.forecast import HazardForecast, xarray_has_timedelta_bug
from climada.hazard.test.test_base import hazard_kwargs

# See https://docs.xarray.dev/en/stable/whats-new.html#id80
xarray_leadtime = pytest.mark.skipif(
    xarray_has_timedelta_bug(), reason="xarray timedelta bug"
)


@pytest.fixture(scope="module")
def haz_kwargs():
    return hazard_kwargs()


@pytest.fixture
def hazard(haz_kwargs):
    return Hazard(**haz_kwargs)


@pytest.fixture
def lead_time(haz_kwargs):
    return pd.timedelta_range("1h", periods=len(haz_kwargs["event_id"]))


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


class TestHazardForecastConcat:

    def test_concat(self, haz_fc, lead_time, member, haz_kwargs):
        haz_fc1 = haz_fc.select(event_id=[3])
        haz_fc2 = HazardForecast(
            haz_type=haz_kwargs["haz_type"], frequency_unit=haz_kwargs["frequency_unit"]
        )  # Empty hazard
        haz_fc3 = haz_fc.select(event_id=[1, 2])
        haz_fc_concat = HazardForecast.concat([haz_fc1, haz_fc2, haz_fc3])
        assert isinstance(haz_fc_concat, HazardForecast)
        npt.assert_array_equal(
            haz_fc_concat.lead_time, np.concatenate((lead_time[2:3], lead_time[0:2]))
        )
        npt.assert_array_equal(
            haz_fc_concat.member, np.concatenate((member[2:3], member[0:2]))
        )
        npt.assert_array_equal(haz_fc_concat.event_id, [3, 1, 2])

    def test_empty_list(self):
        haz_concat = HazardForecast.concat([])
        assert isinstance(haz_concat, HazardForecast)
        assert haz_concat.size == 0
        npt.assert_array_equal(haz_concat.lead_time, [])
        npt.assert_array_equal(haz_concat.event_id, [])

    def test_type_fail(self, haz_fc, hazard):
        with pytest.raises(TypeError, match="different classes"):
            HazardForecast.concat([haz_fc, hazard])
        with pytest.raises(TypeError, match="different classes"):
            Hazard.concat([haz_fc, hazard])


class TestXarrayReader:

    @pytest.fixture()
    def forecast_netcdf_file(self, tmp_path_factory):
        """Create a NetCDF file with forecast data structure"""
        tmpdir = tmp_path_factory.mktemp("forecast_data")
        netcdf_path = tmpdir / "forecast_data.nc"

        crs = "EPSG:4326"

        n_eps = 5
        n_lead_time = 4
        n_lat = 3
        n_lon = 4

        eps = np.array([3, 8, 13, 16, 20])
        ref_time = np.array([dt.datetime(2025, 12, 8, 6, 0, 0)], dtype="datetime64[ns]")
        lead_time_vals = pd.timedelta_range(
            "3h", periods=n_lead_time, freq="2h"
        ).to_numpy()
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
            "crs": crs,
        }

    @xarray_leadtime
    def test_from_xarray_raster_basic(self, forecast_netcdf_file):
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
        expected_n_centroids = (
            forecast_netcdf_file["n_lat"] * forecast_netcdf_file["n_lon"]
        )
        assert haz_fc.intensity.shape == (expected_n_events, expected_n_centroids)

        # Check centroids
        assert len(haz_fc.centroids.lat) == expected_n_centroids
        assert len(haz_fc.centroids.lon) == expected_n_centroids

    @xarray_leadtime
    def test_from_xarray_raster_event_names(self, forecast_netcdf_file):
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
            crs=forecast_netcdf_file["crs"],
        )

        # Check that event names are generated with lead_time in hours
        expected_n_events = (
            forecast_netcdf_file["n_eps"] * forecast_netcdf_file["n_lead_time"]
        )
        assert len(haz_fc.event_name) == expected_n_events

        event_names_expected = [
            f"lt_{lt / np.timedelta64(1, 'h'):.0f}h_m_{mm}"
            for lt, mm in zip(haz_fc.lead_time, haz_fc.member)
        ]
        npt.assert_array_equal(haz_fc.event_name, event_names_expected)

    @xarray_leadtime
    def test_from_xarray_raster_dates(self, forecast_netcdf_file):
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
            crs=forecast_netcdf_file["crs"],
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
            haz_fc.select(event_id=[])
        with pytest.raises(ValueError, match="Empty selection"):
            haz_fc.select(member=[-1])
        with pytest.raises(ValueError, match="Empty selection"):
            haz_fc.select(
                lead_time=[np.timedelta64("2", "Y").astype("timedelta64[ns]")]
            )


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


class TestReduce:
    @pytest.fixture
    def mat(self):
        return np.array([[0, -1, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1]], dtype="float")

    @pytest.fixture(autouse=True)
    def haz_fc_custom_intensity_fraction(self, mat, haz_fc):
        haz_fc.intensity = csr_matrix(mat)
        haz_fc.fraction = csr_matrix(mat)

    @pytest.fixture
    def haz_fc_dim_reduce(self, haz_fc):
        """Create hazard forecast where some members/leadtimes are duplicated"""
        haz_fc.member = np.array([1, 2, 1, 2])
        haz_fc.lead_time = np.array(
            [
                np.timedelta64(1, "h"),
                np.timedelta64(1, "h"),
                np.timedelta64(2, "h"),
                np.timedelta64(2, "h"),
            ]
        )
        return haz_fc

    @pytest.fixture
    def q(self):
        """Quantile to test"""
        return 0.25

    @pytest.fixture
    def reduction_results(self):
        return {
            "min": [0, -1, 0],
            "mean": [1.5, 0.5, 0.25],
            "max": [3, 2, 1],
            "median": [1.5, 0.5, 0.0],
            "quantile": [0.75, -0.25, 0.0],
        }

    @pytest.fixture
    def reduction_results_dim(self):
        return {
            "lead_time": {
                "min": [[0, -1, 0], [1, 0, 0]],
                "mean": [[1, 0, 0], [2, 1, 0.5]],
                "median": [[1, 0, 0], [2, 1, 0.5]],
                "max": [[2, 1, 0], [3, 2, 1]],
                "quantile": [[0.5, -0.5, 0], [1.5, 0.5, 0.25]],
            },
            "member": {
                "min": [[0, -1, 0], [2, 1, 0]],
                "mean": [[0.5, -0.5, 0], [2.5, 1.5, 0.5]],
                "median": [[0.5, -0.5, 0], [2.5, 1.5, 0.5]],
                "max": [[1, 0, 0], [3, 2, 1]],
                "quantile": [[0.25, -0.75, 0], [2.25, 1.25, 0.25]],
            },
        }

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "quantile", "median"])
    def test_reduce(self, haz_fc, q, reduction_results, attr):
        """Check reduction methods for HazardForecast"""
        kwargs = {"q": q} if attr == "quantile" else {}
        haz_fcst_reduced = getattr(haz_fc, attr)(**kwargs)

        # Test by checking results
        npt.assert_array_equal(
            haz_fcst_reduced.intensity.toarray().squeeze(), reduction_results[attr]
        )
        npt.assert_array_equal(
            haz_fcst_reduced.fraction.toarray().squeeze(), reduction_results[attr]
        )

        # Test by calling the same numpy function on the dense array
        npt.assert_array_equal(
            haz_fcst_reduced.intensity.toarray().squeeze(),
            getattr(np, attr)(haz_fc.intensity.toarray(), axis=0, **kwargs),
        )
        npt.assert_array_equal(
            haz_fcst_reduced.fraction.toarray().squeeze(),
            getattr(np, attr)(haz_fc.fraction.toarray(), axis=0, **kwargs),
        )

        # Check that attributes where reduced correctly
        attr_str = f"quantile_{q}" if attr == "quantile" else attr
        npt.assert_array_equal(haz_fcst_reduced.lead_time, [np.timedelta64("NaT")])
        npt.assert_array_equal(haz_fcst_reduced.member, [-1])
        npt.assert_array_equal(haz_fcst_reduced.event_name, [attr_str])
        npt.assert_array_equal(haz_fcst_reduced.event_id, [1])
        npt.assert_array_equal(haz_fcst_reduced.frequency, [1])
        npt.assert_array_equal(haz_fcst_reduced.date, [0])
        npt.assert_array_equal(haz_fcst_reduced.orig, [True])

    @pytest.mark.parametrize("quantile,reduce", [(0.0, "min"), (1.0, "max")])
    def test_quantile_min_max(self, haz_fc, quantile, reduce):
        """Compare min/max with quantiles 0/1"""
        haz_fcst_quantile = haz_fc.quantile(q=quantile)
        haz_fcst_reduce = getattr(haz_fc, reduce)()
        npt.assert_array_equal(
            haz_fcst_quantile.intensity.todense(), haz_fcst_reduce.intensity.todense()
        )

    def test_median_quantile(self, haz_fc):
        """Compare median with quantile 0.5"""
        haz_fcst_median = haz_fc.median()
        haz_fcst_quantile = haz_fc.quantile(q=0.5)
        npt.assert_array_equal(
            haz_fcst_median.intensity.todense(), haz_fcst_quantile.intensity.todense()
        )
        npt.assert_array_equal(
            haz_fcst_median.intensity.todense(),
            np.median(haz_fc.intensity.todense(), axis=0),
        )

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "median", "quantile"])
    @pytest.mark.parametrize("dim", ["lead_time", "member", "single"])
    def test_reduce_dim_unique_or_single(self, haz_fc, q, attr, dim):
        """Test that reduction over a dimension with all-unique values does nothing"""
        kwargs = {"q": q} if attr == "quantile" else {}
        if dim == "single":
            haz_fc = haz_fc.select(event_id=[haz_fc.event_id[0]])
            dim = None

        haz_fc_reduced = getattr(haz_fc, attr)(dim=dim, **kwargs)

        npt.assert_array_equal(haz_fc_reduced.member, haz_fc.member)
        npt.assert_array_equal(haz_fc_reduced.lead_time, haz_fc.lead_time)
        npt.assert_array_equal(
            haz_fc_reduced.intensity.todense(), haz_fc.intensity.todense()
        )
        npt.assert_array_equal(
            haz_fc_reduced.fraction.todense(), haz_fc.fraction.todense()
        )
        if dim == "single":
            npt.assert_array_equal(haz_fc_reduced.event_name, haz_fc.event_name)
            npt.assert_array_equal(haz_fc_reduced.event_id, haz_fc.event_id)
            npt.assert_array_equal(haz_fc_reduced.frequency, haz_fc.frequency)
            npt.assert_array_equal(haz_fc_reduced.date, haz_fc.date)
            npt.assert_array_equal(haz_fc_reduced.orig, haz_fc.orig)

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "quantile", "median"])
    def test_reduce_dim_error(self, haz_fc, q, attr):
        """Check reduction error message for invalid dimension name"""
        kwargs = {"q": q} if attr == "quantile" else {}
        with pytest.raises(ValueError, match=r"Cannot reduce over dim \'invalid_dim\'"):
            getattr(haz_fc, attr)(dim="invalid_dim", **kwargs)

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "median", "quantile"])
    @pytest.mark.parametrize("dim", ["lead_time", "member"])
    def test_reduce_dim(self, haz_fc_dim_reduce, q, reduction_results_dim, attr, dim):
        """Check reduction for HazardForecast with dim argument"""
        kwargs = {"q": q} if attr == "quantile" else {}
        haz_fc_reduced = getattr(haz_fc_dim_reduce, attr)(dim=dim, **kwargs)

        rdim = "member" if dim == "lead_time" else "lead_time"
        unique_rdim = np.unique(getattr(haz_fc_dim_reduce, rdim))
        npt.assert_array_equal(getattr(haz_fc_reduced, rdim), unique_rdim)

        if dim == "lead_time":
            unique_rdim = [1, 2]
            npt.assert_array_equal(
                haz_fc_reduced.lead_time, [np.timedelta64("NaT"), np.timedelta64("NaT")]
            )
            npt.assert_array_equal(haz_fc_reduced.member, unique_rdim)
        else:
            unique_rdim = [np.timedelta64(1, "h"), np.timedelta64(2, "h")]
            npt.assert_array_equal(
                haz_fc_reduced.lead_time,
                unique_rdim,
            )
            npt.assert_array_equal(haz_fc_reduced.member, [-1, -1])

        haz_fc_expected = haz_fc_dim_reduce.concat(
            [
                getattr(haz_fc_dim_reduce.select(**{rdim: val}), attr)(
                    dim=None, **kwargs
                )
                for val in unique_rdim
            ]
        )
        npt.assert_array_equal(
            haz_fc_reduced.intensity.toarray().squeeze(),
            reduction_results_dim[dim][attr],
        )
        npt.assert_array_equal(
            haz_fc_reduced.fraction.toarray().squeeze(),
            reduction_results_dim[dim][attr],
        )
        npt.assert_array_equal(
            haz_fc_reduced.intensity.todense(), haz_fc_expected.intensity.todense()
        )
        npt.assert_array_equal(
            haz_fc_reduced.fraction.todense(), haz_fc_expected.fraction.todense()
        )
