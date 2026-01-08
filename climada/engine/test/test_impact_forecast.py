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

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
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
def lead_time(impact_kwargs):
    return pd.timedelta_range(
        start="1 day", periods=len(impact_kwargs["event_id"])
    ).to_numpy()


@pytest.fixture
def member(impact_kwargs):
    return np.arange(len(impact_kwargs["event_id"]))


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


class TestReduce:

    @pytest.fixture
    def imp_fc_stats(self, impact_kwargs, lead_time, member):
        max_index = 4
        for key, val in impact_kwargs.items():
            if isinstance(val, (np.ndarray, list)):
                impact_kwargs[key] = val[:max_index]
            elif isinstance(val, csr_matrix):
                impact_kwargs[key] = val[:max_index, :]
        impact_kwargs["imp_mat"] = csr_matrix([[1, 0], [0, 1], [3, 2], [2, 5]])
        impact_kwargs["at_event"] = np.array([1, 1, 5, 7])
        return ImpactForecast(
            lead_time=lead_time[:max_index], member=member[:max_index], **impact_kwargs
        )

    @pytest.fixture
    def imp_fc_stats_dim_reduce(self, imp_fc_stats):
        """Create hazard forecast where some members/leadtimes are duplicated"""
        imp_fc_stats.member = np.array([1, 2, 1, 2])
        imp_fc_stats.lead_time = np.array(
            [
                np.timedelta64(1, "h"),
                np.timedelta64(1, "h"),
                np.timedelta64(2, "h"),
                np.timedelta64(2, "h"),
            ]
        )
        return imp_fc_stats

    @pytest.fixture
    def q(self):
        """Quantile to test"""
        return 0.25

    @pytest.fixture
    def reduction_results(self):
        return {
            "min": {"imp_mat": [0, 0], "at_event": [0]},
            "mean": {"imp_mat": [1.5, 2], "at_event": [3.5]},
            "max": {"imp_mat": [3, 5], "at_event": [8]},
            "median": {"imp_mat": [1.5, 1.5], "at_event": [3]},
            "quantile": {"imp_mat": [0.75, 0.75], "at_event": [1.5]},
        }

    @pytest.fixture
    def reduction_results_dim(self):
        return {
            "lead_time": {
                "min": {"imp_mat": [[1, 0], [0, 1]], "at_event": [1, 1]},
                "mean": {"imp_mat": [[2, 1], [1, 3]], "at_event": [3, 4]},
                "median": {"imp_mat": [[2, 1], [1, 3]], "at_event": [3, 4]},
                "max": {"imp_mat": [[3, 2], [2, 5]], "at_event": [5, 7]},
                "quantile": {"imp_mat": [[1.5, 0.5], [0.5, 2]], "at_event": [2, 2.5]},
            },
            "member": {
                "min": {"imp_mat": [[0, 0], [2, 2]], "at_event": [0, 4]},
                "mean": {"imp_mat": [[0.5, 0.5], [2.5, 3.5]], "at_event": [1, 6]},
                "median": {"imp_mat": [[0.5, 0.5], [2.5, 3.5]], "at_event": [1, 6]},
                "max": {"imp_mat": [[1, 1], [3, 5]], "at_event": [2, 8]},
                "quantile": {
                    "imp_mat": [[0.25, 0.25], [2.25, 2.75]],
                    "at_event": [0.5, 5],
                },
            },
        }

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "quantile", "median"])
    def test_reduce(self, imp_fc_stats, q, reduction_results, attr):
        """Check mean, min, and max methods for ImpactForecast"""
        kwargs = {"q": q} if attr == "quantile" else {}
        imp_fc_reduced = getattr(imp_fc_stats, attr)(**kwargs)

        # assert imp_mat
        npt.assert_array_equal(
            imp_fc_reduced.imp_mat.toarray().squeeze(),
            getattr(np, attr)(imp_fc_stats.imp_mat.toarray(), axis=0, **kwargs),
        )
        npt.assert_array_equal(
            imp_fc_reduced.imp_mat.toarray().squeeze(),
            reduction_results[attr]["imp_mat"],
        )
        npt.assert_array_equal(
            imp_fc_reduced.at_event, reduction_results[attr]["at_event"]
        )

        # check that attributes where reduced correctly
        attr_str = f"quantile_{q}" if attr == "quantile" else attr
        npt.assert_array_equal(imp_fc_reduced.lead_time, [np.timedelta64("NaT")])
        npt.assert_array_equal(imp_fc_reduced.member, [-1])
        npt.assert_array_equal(imp_fc_reduced.event_name, [attr_str])
        npt.assert_array_equal(imp_fc_reduced.event_id, [1])
        npt.assert_array_equal(imp_fc_reduced.frequency, [1])
        npt.assert_array_equal(imp_fc_reduced.date, [0])

    @pytest.mark.parametrize("quantile,reduce", [(0.0, "min"), (1.0, "max")])
    def test_quantile_min_max(self, imp_fc_stats, quantile, reduce):
        """Compare min/max with quantiles 0/1"""
        imp_fcst_quantile = imp_fc_stats.quantile(q=quantile)
        imp_fcst_reduce = getattr(imp_fc_stats, reduce)()
        npt.assert_array_equal(
            imp_fcst_quantile.imp_mat.toarray(), imp_fcst_reduce.imp_mat.toarray()
        )
        npt.assert_array_equal(imp_fcst_quantile.at_event, imp_fcst_reduce.at_event)

    def test_median_quantile(self, imp_fc_stats):
        """Compare median with quantile 0.5"""
        imp_fcst_median = imp_fc_stats.median()
        imp_fcst_quantile = imp_fc_stats.quantile(q=0.5)
        npt.assert_array_equal(
            imp_fcst_median.imp_mat.toarray(), imp_fcst_quantile.imp_mat.toarray()
        )
        npt.assert_array_equal(imp_fcst_median.at_event, imp_fcst_quantile.at_event)
        npt.assert_array_equal(
            imp_fcst_median.imp_mat.toarray().squeeze(),
            np.median(imp_fc_stats.imp_mat.toarray(), axis=0),
        )

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "median", "quantile"])
    @pytest.mark.parametrize("dim", ["lead_time", "member", "single"])
    def test_reduce_dim_unique_or_single(self, imp_fc_stats, q, attr, dim):
        """Test that reduction over a dimension with all-unique values does nothing"""
        kwargs = {"q": q} if attr == "quantile" else {}
        if dim == "single":
            imp_fc_stats = imp_fc_stats.select(event_ids=[imp_fc_stats.event_id[0]])
            dim = None

        imp_fc_stats_reduced = getattr(imp_fc_stats, attr)(dim=dim, **kwargs)

        npt.assert_array_equal(imp_fc_stats_reduced.member, imp_fc_stats.member)
        npt.assert_array_equal(imp_fc_stats_reduced.lead_time, imp_fc_stats.lead_time)
        npt.assert_array_equal(
            imp_fc_stats_reduced.imp_mat.toarray(),
            imp_fc_stats.imp_mat.toarray(),
        )
        npt.assert_array_equal(
            imp_fc_stats_reduced.at_event,
            imp_fc_stats.at_event,
        )
        if dim == "single":
            npt.assert_array_equal(
                imp_fc_stats_reduced.event_name,
                imp_fc_stats.event_name,
            )
            npt.assert_array_equal(imp_fc_stats_reduced.event_id, imp_fc_stats.event_id)
            npt.assert_array_equal(
                imp_fc_stats_reduced.frequency, imp_fc_stats.frequency
            )
            npt.assert_array_equal(imp_fc_stats_reduced.date, imp_fc_stats.date)

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "quantile", "median"])
    def test_reduce_dim_error(self, imp_fc_stats, q, attr):
        """Check reduction error message for invalid dimension name"""
        kwargs = {"q": q} if attr == "quantile" else {}
        with pytest.raises(ValueError, match=r"Cannot reduce over dim \'invalid_dim\'"):
            getattr(imp_fc_stats, attr)(dim="invalid_dim", **kwargs)

    @pytest.mark.parametrize("attr", ["min", "mean", "max", "median", "quantile"])
    @pytest.mark.parametrize("dim", ["lead_time", "member"])
    def test_reduce_dim(
        self, imp_fc_stats_dim_reduce, q, reduction_results_dim, attr, dim
    ):
        """Check reduction for HazardForecast with dim argument"""
        kwargs = {"q": q} if attr == "quantile" else {}
        imp_fc_reduced = getattr(imp_fc_stats_dim_reduce, attr)(dim=dim, **kwargs)

        rdim = "member" if dim == "lead_time" else "lead_time"
        unique_rdim = np.unique(getattr(imp_fc_stats_dim_reduce, rdim))
        npt.assert_array_equal(getattr(imp_fc_reduced, rdim), unique_rdim)

        if dim == "lead_time":
            unique_rdim = [1, 2]
            npt.assert_array_equal(
                imp_fc_reduced.lead_time, [np.timedelta64("NaT"), np.timedelta64("NaT")]
            )
            npt.assert_array_equal(imp_fc_reduced.member, unique_rdim)
        else:
            unique_rdim = [np.timedelta64(1, "h"), np.timedelta64(2, "h")]
            npt.assert_array_equal(
                imp_fc_reduced.lead_time,
                unique_rdim,
            )
            npt.assert_array_equal(imp_fc_reduced.member, [-1, -1])

        imp_fc_expected = imp_fc_stats_dim_reduce.concat(
            [
                getattr(imp_fc_stats_dim_reduce.select(**{rdim: val}), attr)(
                    dim=None, **kwargs
                )
                for val in unique_rdim
            ],
            reset_event_ids=True,
        )
        npt.assert_array_equal(
            imp_fc_reduced.imp_mat.toarray().squeeze(),
            reduction_results_dim[dim][attr]["imp_mat"],
        )
        npt.assert_array_equal(
            imp_fc_reduced.imp_mat.toarray(),
            imp_fc_expected.imp_mat.toarray(),
        )
        npt.assert_array_equal(
            imp_fc_reduced.at_event,
            reduction_results_dim[dim][attr]["at_event"],
        )
