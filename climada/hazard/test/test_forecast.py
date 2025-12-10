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
import pandas as pd
import pytest
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
        assert haz_fc_concat.size == 3
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


@pytest.mark.parametrize("attr", ["min", "mean", "max"])
def test_hazard_forecast_mean_min_max(haz_fc, attr):
    """Check mean, min, and max methods for ImpactForecast"""
    haz_fcst_reduced = getattr(haz_fc, attr)()

    # Assert sparse matrices
    npt.assert_array_equal(
        haz_fcst_reduced.intensity.todense(),
        getattr(haz_fc.intensity.todense(), attr)(axis=0),
    )
    npt.assert_array_equal(
        haz_fcst_reduced.fraction.todense(),
        getattr(haz_fc.fraction.todense(), attr)(axis=0),
    )

    # Check that attributes where reduced correctly
    npt.assert_array_equal(np.isnat(haz_fcst_reduced.lead_time), [True])
    npt.assert_array_equal(haz_fcst_reduced.member, [-1])
    npt.assert_array_equal(haz_fcst_reduced.event_name, [attr])
    npt.assert_array_equal(haz_fcst_reduced.event_id, [0])
    npt.assert_array_equal(haz_fcst_reduced.frequency, [1])
    npt.assert_array_equal(haz_fcst_reduced.date, [0])
    npt.assert_array_equal(haz_fcst_reduced.orig, [True])


def test_hazard_forecast_mean_min_max_dim(haz_fc):
    """Check mean, min, and max methods for ImpactForecast with dim argument"""
    for attr in ["min", "mean", "max"]:
        for dim, unique_vals in zip(
            ["member", "lead_time"],
            [np.unique(haz_fc.member), np.unique(haz_fc.lead_time)],
        ):
            haz_fcst_reduced = getattr(haz_fc, attr)(dim=dim)
            # Assert sparse matrices
            expected_intensity = []
            expected_fraction = []
            for val in unique_vals:
                mask = getattr(haz_fc, dim) == val
                expected_intensity.append(
                    getattr(haz_fc.intensity.todense()[mask], attr)(axis=0)
                )
                expected_fraction.append(
                    getattr(haz_fc.fraction.todense()[mask], attr)(axis=0)
                )
            npt.assert_array_equal(
                haz_fcst_reduced.intensity.todense(),
                np.vstack(expected_intensity),
            )
            npt.assert_array_equal(
                haz_fcst_reduced.fraction.todense(),
                np.vstack(expected_fraction),
            )

            # Check that attributes where reduced correctly
            if dim == "member":
                npt.assert_array_equal(haz_fcst_reduced.member, unique_vals)
                npt.assert_array_equal(
                    haz_fcst_reduced.lead_time,
                    np.array([np.timedelta64("NaT")] * len(unique_vals)),
                )
            else:  # dim == "lead_time"
                npt.assert_array_equal(haz_fcst_reduced.lead_time, unique_vals)
                npt.assert_array_equal(
                    haz_fcst_reduced.member,
                    np.array([-1] * len(unique_vals)),
                )
            npt.assert_array_equal(
                haz_fcst_reduced.event_name,
                np.array([attr] * len(unique_vals)),
            )
            npt.assert_array_equal(haz_fcst_reduced.event_id, [0] * len(unique_vals))
            npt.assert_array_equal(haz_fcst_reduced.frequency, [1] * len(unique_vals))
            npt.assert_array_equal(haz_fcst_reduced.date, [0] * len(unique_vals))
            npt.assert_array_equal(haz_fcst_reduced.orig, [True] * len(unique_vals))
    # TODO add test in case no reduction happens (e.g., all values along dim are unique)


def test_hazard_forecast_mean_max_min_dim_error(haz_fc):
    """Check mean, min, and max methods for ImpactForecast with dim argument"""
    for attr in ["min", "mean", "max"]:
        with pytest.raises(ValueError, match="not a valid dimension"):
            getattr(haz_fc, attr)(dim="invalid_dim")
