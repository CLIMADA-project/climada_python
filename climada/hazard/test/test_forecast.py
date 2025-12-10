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


@pytest.mark.parametrize(
    "var, var_select",
    [("event_id", "event_id"), ("event_name", "event_names"), ("date", "date")],
)
def test_hazard_forecast_select(haz_fc, lead_time, member, haz_kwargs, var, var_select):
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
    npt.assert_array_equal(haz_fc_sel.frequency, haz_fc.frequency[ordered_select_mask])
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


def test_hazard_forecast_quantile(haz_fc):
    """Check quantile method for HazardForecast"""
    for q in [0.0, 0.5, 0.8]:
        haz_fcst_quantile = haz_fc.quantile(q)

        # assert intensity
        npt.assert_array_equal(
            haz_fcst_quantile.intensity.toarray().squeeze(),
            np.quantile(haz_fc.intensity.toarray(), q, axis=0),
        )
        # assert fraction
        npt.assert_array_equal(
            haz_fcst_quantile.fraction.toarray().squeeze(),
            np.quantile(haz_fc.fraction.toarray(), q, axis=0),
        )

        # check that attributes where reduced correctly
        npt.assert_array_equal(
            haz_fcst_quantile.lead_time, np.array([np.timedelta64("NaT")])
        )
        npt.assert_array_equal(haz_fcst_quantile.member, np.array([-1]))
        npt.assert_array_equal(
            haz_fcst_quantile.event_name, np.array([f"quantile_{q}"])
        )
        npt.assert_array_equal(haz_fcst_quantile.event_id, np.array([0]))
        npt.assert_array_equal(haz_fcst_quantile.frequency, np.array([1]))
        npt.assert_array_equal(haz_fcst_quantile.date, np.array([0]))
        npt.assert_array_equal(haz_fcst_quantile.orig, np.array([True]))
