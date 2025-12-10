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

Tests for Impact Forecast.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

from climada.engine import Impact, ImpactForecast

from .test_impact import impact_kwargs as imp_kwargs


@pytest.fixture
def impact_kwargs():
    return imp_kwargs()


@pytest.fixture
def impact(impact_kwargs):
    return Impact(**impact_kwargs)


@pytest.fixture
def lead_time():
    return pd.timedelta_range(start="1 day", periods=6).to_numpy()


@pytest.fixture
def member():
    return np.arange(6)


@pytest.fixture
def impact_forecast(impact, lead_time, member):
    return ImpactForecast.from_impact(impact, lead_time=lead_time, member=member)


class TestImpactForecastInit:

    @pytest.fixture
    def lead_time(impact_kwargs):
        return pd.timedelta_range(
            start="1 day", periods=len(impact_kwargs["event_id"])
        ).to_numpy()

    @pytest.fixture
    def member(impact_kwargs):
        return np.arange(len(self, impact_kwargs["event_id"]))

    @pytest.fixture
    def impact_forecast(impact, lead_time, member):
        return ImpactForecast.from_impact(impact, lead_time=lead_time, member=member)


class TestImpactForecastInit:

    def assert_impact_kwargs(self, impact: Impact, **kwargs):
        for key, value in kwargs.items():
            attr = getattr(impact, key)
            if isinstance(value, (np.ndarray, list)):
                npt.assert_array_equal(attr, value)
            elif isinstance(value, csr_matrix):
                npt.assert_array_equal(attr.todense(), value.todense())
            else:
                assert attr == value

    def test_impact_forecast_init(self, impact_kwargs, lead_time, member):
        forecast1 = ImpactForecast(
            lead_time=lead_time,
            member=member,
            **impact_kwargs,
        )
        npt.assert_array_equal(forecast1.lead_time, lead_time)
        npt.assert_array_equal(forecast1.member, member)
        self.assert_impact_kwargs(forecast1, **impact_kwargs)

    def test_impact_forecast_init_error(self, impact, impact_kwargs, lead_time, member):
        with pytest.raises(ValueError, match="Forecast.lead_time"):
            ImpactForecast(lead_time=lead_time[:-2], member=member, **impact_kwargs)
        with pytest.raises(ValueError, match="Forecast.member"):
            ImpactForecast.from_impact(impact, lead_time=lead_time, member=member[1:])

    def test_impact_forecast_from_impact(
        self, impact_forecast, impact_kwargs, lead_time, member
    ):
        npt.assert_array_equal(impact_forecast.lead_time, lead_time)
        npt.assert_array_equal(impact_forecast.member, member)
        self.assert_impact_kwargs(impact_forecast, **impact_kwargs)


class TestSelect:

    @pytest.mark.parametrize(
        "var, var_select",
        [("event_id", "event_ids"), ("event_name", "event_names"), ("date", "dates")],
    )
    def test_base_class_select(
        self, impact_forecast, lead_time, member, impact_kwargs, var, var_select
    ):
        """Check if Impact.select works on the derived class"""
        select_mask = np.array([2, 1])
        ordered_select_mask = np.array([1, 2])
        if var == "date":
            # Date needs to be a valid delta
            select_mask = np.array([1, 2])
            ordered_select_mask = np.array([1, 2])

        var_value = np.array(impact_kwargs[var])[select_mask]
        # event_name is a list, convert to numpy array for indexing
        impact_fc = impact_forecast.select(**{var_select: var_value})
        # NOTE: Events keep their original order
        npt.assert_array_equal(
            impact_fc.event_id,
            impact_forecast.event_id[ordered_select_mask],
        )
        npt.assert_array_equal(
            impact_fc.event_name,
            np.array(impact_forecast.event_name)[ordered_select_mask],
        )
        npt.assert_array_equal(
            impact_fc.date, impact_forecast.date[ordered_select_mask]
        )
        npt.assert_array_equal(
            impact_fc.frequency, impact_forecast.frequency[ordered_select_mask]
        )
        npt.assert_array_equal(impact_fc.member, member[ordered_select_mask])
        npt.assert_array_equal(impact_fc.lead_time, lead_time[ordered_select_mask])
        npt.assert_array_equal(
            impact_fc.imp_mat.todense(),
            impact_forecast.imp_mat.todense()[ordered_select_mask],
        )

    def test_impact_forecast_select_exposure(
        self, impact_forecast, lead_time, member, impact_kwargs
    ):
        """Check if Impact.select works on the derived class"""
        exp_col = 0
        select_mask = np.array([exp_col])
        coord_exp = impact_kwargs["coord_exp"][select_mask]
        impact_fc = impact_forecast.select(coord_exp=coord_exp)
        npt.assert_array_equal(impact_fc.member, member)
        npt.assert_array_equal(impact_fc.lead_time, lead_time)
        npt.assert_array_equal(
            impact_fc.imp_mat.todense(), impact_forecast.imp_mat.todense()[:, exp_col]
        )

    def test_derived_select_single(self, impact_forecast, lead_time, member):
        imp_fc_select = impact_forecast.select(member=[2, 0])
        idx = np.array([0, 2])
        npt.assert_array_equal(imp_fc_select.event_id, impact_forecast.event_id[idx])
        npt.assert_array_equal(imp_fc_select.member, member[idx])
        npt.assert_array_equal(imp_fc_select.lead_time, lead_time[idx])

        imp_fc_select = impact_forecast.select(lead_time=lead_time[np.array([2, 0])])
        npt.assert_array_equal(imp_fc_select.event_id, impact_forecast.event_id[idx])
        npt.assert_array_equal(imp_fc_select.member, member[idx])
        npt.assert_array_equal(imp_fc_select.lead_time, lead_time[idx])

    def test_derived_select_intersections(
        self, impact_forecast, lead_time, member, impact_kwargs
    ):
        imp_fc_select = impact_forecast.select(event_ids=[10, 14], member=[0, 1, 2])
        npt.assert_array_equal(
            imp_fc_select.event_id, impact_forecast.event_id[np.array([0])]
        )

        imp_fc_select = impact_forecast.select(
            event_ids=[10, 11, 13], member=[0, 1, 2], lead_time=lead_time[1:3]
        )
        npt.assert_array_equal(
            imp_fc_select.event_id, impact_forecast.event_id[np.array([1])]
        )

        # Test "outer"
        impact_forecast2 = ImpactForecast(
            lead_time=lead_time,
            member=np.zeros_like(member, dtype="int"),
            **impact_kwargs,
        )
        imp_fc_select = impact_forecast2.select(event_ids=[10, 11, 13], member=[0])
        npt.assert_array_equal(imp_fc_select.event_id, [10, 11, 13])
        npt.assert_array_equal(imp_fc_select.member, [0, 0, 0])

    def test_no_select(self, impact_forecast, impact_kwargs):
        imp_fc_select = impact_forecast.select()
        npt.assert_array_equal(
            imp_fc_select.imp_mat.todense(), impact_forecast.imp_mat.todense()
        )

        num_centroids = len(impact_kwargs["coord_exp"])
        imp_fc_select = impact_forecast.select(event_names=["aaaaa", "foo"])
        assert imp_fc_select.imp_mat.shape == (0, num_centroids)
        imp_fc_select = impact_forecast.select(event_ids=[-1, 1002])
        assert imp_fc_select.imp_mat.shape == (0, num_centroids)
        imp_fc_select = impact_forecast.select(member=[-1])
        assert imp_fc_select.imp_mat.shape == (0, num_centroids)
        imp_fc_select = impact_forecast.select(np.timedelta64("3", "Y"))
        assert imp_fc_select.imp_mat.shape == (0, num_centroids)


def test_impact_forecast_concat(impact_forecast, member, lead_time):
    """Check if Impact.concat works on the derived class"""
    impact_fc = ImpactForecast.concat(
        [impact_forecast, impact_forecast], reset_event_ids=True
    )
    npt.assert_array_equal(impact_fc.member, np.concatenate([member, member]))
    npt.assert_array_equal(impact_fc.lead_time, np.concatenate([lead_time, lead_time]))
    npt.assert_array_equal(
        impact_fc.event_id, np.arange(impact_fc.imp_mat.shape[0]) + 1
    )
    npt.assert_array_equal(impact_fc.event_name, impact_forecast.event_name * 2)
    npt.assert_array_equal(
        impact_fc.imp_mat.toarray(),
        np.vstack(
            (impact_forecast.imp_mat.toarray(), impact_forecast.imp_mat.toarray())
        ),
    )


def test_impact_forecast_blocked_methods(impact_forecast):
    """Check if ImpactForecast.exceedance_freq_curve raises NotImplementedError"""
    with pytest.raises(NotImplementedError):
        impact_forecast.local_exceedance_impact(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.local_return_period(np.array([10, 50, 100]))

    with pytest.raises(NotImplementedError):
        impact_forecast.calc_freq_curve(np.array([10, 50, 100]))


@pytest.mark.parametrize("dense", [True, False])
def test_write_read_hdf5(impact_forecast, tmp_path, dense):

    file_name = tmp_path / "test_hazard_forecast.h5"
    # replace dummy_impact event_names with strings
    impact_forecast.event_name = [str(name) for name in impact_forecast.event_name]
    impact_forecast.write_hdf5(file_name, dense_imp_mat=dense)

    def compare_attr(obj, attr):
        actual = getattr(obj, attr)
        expected = getattr(impact_forecast, attr)
        if isinstance(actual, csr_matrix):
            npt.assert_array_equal(actual.todense(), expected.todense())
        else:
            npt.assert_array_equal(actual, expected)

    # Read ImpactForecast
    impact_forecast_read = ImpactForecast.from_hdf5(file_name)
    assert impact_forecast_read.lead_time.dtype.kind == np.dtype("timedelta64").kind
    for attr in impact_forecast.__dict__.keys():
        compare_attr(impact_forecast_read, attr)

    # Read Impact
    impact_read = Impact.from_hdf5(file_name)
    for attr in impact_read.__dict__.keys():
        compare_attr(impact_read, attr)
    assert "member" not in impact_read.__dict__
    assert "lead_time" not in impact_read.__dict__


@pytest.fixture
def impact_forecast_stats(impact_kwargs, lead_time, member):
    max_index = 4
    for key, val in impact_kwargs.items():
        if isinstance(val, (np.ndarray, list)):
            impact_kwargs[key] = val[:max_index]
        elif isinstance(val, csr_matrix):
            impact_kwargs[key] = val[:max_index, :]
    impact_kwargs["imp_mat"] = csr_matrix([[1, 0], [0, 1], [3, 2], [2, 3]])
    impact_kwargs["at_event"] = np.array([1, 1, 5, 5])
    return ImpactForecast(
        lead_time=lead_time[:max_index], member=member[:max_index], **impact_kwargs
    )


@pytest.mark.parametrize("attr", ["min", "mean", "max"])
def test_impact_forecast_min_mean_max(impact_forecast_stats, attr):
    """Check mean, min, and max methods for ImpactForecast"""
    imp_fc_reduced = getattr(impact_forecast_stats, attr)()

    # assert imp_mat
    npt.assert_array_equal(
        imp_fc_reduced.imp_mat.todense(),
        getattr(impact_forecast_stats.imp_mat.todense(), attr)(axis=0),
    )
    at_event_expected = {"min": [0], "mean": [3], "max": [6]}
    npt.assert_array_equal(imp_fc_reduced.at_event, at_event_expected[attr])

    # check that attributes where reduced correctly
    npt.assert_array_equal(np.isnat(imp_fc_reduced.lead_time), [True])
    npt.assert_array_equal(imp_fc_reduced.member, [-1])
    npt.assert_array_equal(imp_fc_reduced.event_name, [attr])
    npt.assert_array_equal(imp_fc_reduced.event_id, [0])
    npt.assert_array_equal(imp_fc_reduced.frequency, [1])
    npt.assert_array_equal(imp_fc_reduced.date, [0])


@pytest.mark.parametrize("quantile", [0.3, 0.6, 0.8])
def test_impact_forecast_quantile(impact_forecast, quantile):
    """Check quantile method for ImpactForecast"""
    imp_fcst_quantile = impact_forecast.quantile(q=quantile)

    # assert imp_mat
    npt.assert_array_equal(
        imp_fcst_quantile.imp_mat.toarray().squeeze(),
        np.quantile(impact_forecast.imp_mat.toarray(), quantile, axis=0),
    )
    # assert at_event
    npt.assert_array_equal(
        imp_fcst_quantile.at_event,
        np.quantile(impact_forecast.at_event, quantile, axis=0).sum(),
    )

    # check that attributes where reduced correctly
    npt.assert_array_equal(imp_fcst_quantile.member, np.array([-1]))
    npt.assert_array_equal(
        imp_fcst_quantile.lead_time, np.array([np.timedelta64("NaT")])
    )
    npt.assert_array_equal(imp_fcst_quantile.event_id, np.array([0]))
    npt.assert_array_equal(
        imp_fcst_quantile.event_name, np.array([f"quantile_{quantile}"])
    )
    npt.assert_array_equal(imp_fcst_quantile.frequency, np.array([1]))
    npt.assert_array_equal(imp_fcst_quantile.date, np.array([0]))


def test_median(impact_forecast):
    imp_fcst_median = impact_forecast.median()
    imp_fcst_quantile = impact_forecast.quantile(q=0.5)
    npt.assert_array_equal(
        imp_fcst_median.imp_mat.toarray(), imp_fcst_quantile.imp_mat.toarray()
    )
    npt.assert_array_equal(imp_fcst_median.imp_mat.toarray(), [[2.5, 2.5]])

    # check that attributes where reduced correctly
    npt.assert_array_equal(imp_fcst_median.member, np.array([-1]))
    npt.assert_array_equal(imp_fcst_median.lead_time, np.array([np.timedelta64("NaT")]))
    npt.assert_array_equal(imp_fcst_median.event_id, np.array([0]))
    npt.assert_array_equal(imp_fcst_median.event_name, np.array(["median"]))
    npt.assert_array_equal(imp_fcst_median.frequency, np.array([1]))
    npt.assert_array_equal(imp_fcst_median.date, np.array([0]))
