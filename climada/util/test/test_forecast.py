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

Tests for ForecastMixin class.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from climada.util.forecast import ForecastMixin, sparse_quantile_axis0


def test_forecast_init():
    """Test initialization of ForecastMixin class."""
    forecast = ForecastMixin()
    npt.assert_array_equal(forecast.lead_time, np.array([]))
    npt.assert_array_equal(forecast.member, np.array([]))

    forecast = ForecastMixin(member=np.array([1, 2]))
    npt.assert_array_equal(forecast.member, np.array([1, 2]), strict=True)

    forecast = ForecastMixin(lead_time=np.array([6, 12], dtype="timedelta64[h]"))
    npt.assert_array_equal(
        forecast.lead_time, np.array([6, 12], dtype="timedelta64[h]"), strict=True
    )

    forecast = ForecastMixin(lead_time=np.array([1, 2]), member=[3, 4])
    npt.assert_array_equal(forecast.lead_time, np.array([1, 2]), strict=True)
    npt.assert_array_equal(forecast.member, np.array([3, 4]), strict=True)
    assert isinstance(forecast.member, np.ndarray)

    # Test with datetime64 including seconds
    lead_times_seconds = pd.timedelta_range(start="1 day", periods=4).to_numpy()
    forecast = ForecastMixin(lead_time=lead_times_seconds, member=[1, 2, 3])
    npt.assert_array_equal(forecast.lead_time, lead_times_seconds, strict=True)
    assert forecast.lead_time.dtype == np.dtype("timedelta64[ns]")


def test_idx_member():
    """Test idx_member method of ForecastMixin class."""
    forecast = ForecastMixin(member=np.array([1, 2, 3, 4]))

    idx = forecast.idx_member(1)
    npt.assert_array_equal(idx, np.array([True, False, False, False]), strict=True)

    idx = forecast.idx_member(np.array([2, 4]))
    npt.assert_array_equal(idx, np.array([False, True, False, True]), strict=True)

    idx = forecast.idx_member([2, 4])
    npt.assert_array_equal(idx, np.array([False, True, False, True]), strict=True)

    idx = forecast.idx_member(None)
    npt.assert_array_equal(idx, np.array([False, False, False, False]), strict=True)

    # Try once with inconsitent types
    forecast = ForecastMixin(member=np.array(["1", -2, np.nan]))
    npt.assert_array_equal(
        forecast.idx_member([np.nan, "1"]), np.array([True, False, True]), strict=True
    )


def test_idx_lead_time():
    """Test idx_lead_time method of ForecastMixin class."""
    forecast = ForecastMixin(
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


class TestSparseQuantile:
    """Block-wise sparse quantile helper"""

    @pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize("block_size", [1, 3, 1000])
    def test_matches_dense_quantile(self, q, block_size):
        """Block-wise result must equal the dense reference exactly"""
        rng = np.random.default_rng(0)
        dense = rng.random((20, 37))
        dense[dense < 0.7] = 0.0  # sparse, with implicit zeros dominating
        mat = csr_matrix(dense)
        result = sparse_quantile_axis0(mat, q, block_size=block_size)
        npt.assert_allclose(result, np.quantile(dense, q, axis=0))

    def test_counts_implicit_zeros(self):
        """A column of [0,0,0,-1,9] has median 0, not 4 (the stored-only answer)"""
        mat = csr_matrix(np.array([[0.0], [0.0], [0.0], [-1.0], [9.0]]))
        npt.assert_allclose(sparse_quantile_axis0(mat, 0.5), [0.0])

    def test_negative_values(self):
        """Negative stored values must not be confused with implicit zeros"""
        dense = np.array([[-3.0, 0.0], [0.0, -1.0], [2.0, 0.0]])
        mat = csr_matrix(dense)
        npt.assert_allclose(
            sparse_quantile_axis0(mat, 0.5), np.quantile(dense, 0.5, axis=0)
        )

    def test_all_zero_column(self):
        """An entirely empty column yields zero, not NaN"""
        mat = csr_matrix((4, 3))
        npt.assert_allclose(sparse_quantile_axis0(mat, 0.5), np.zeros(3))

    @pytest.mark.parametrize("q", [-0.1, 1.5])
    def test_quantile_out_of_range(self, q):
        """An out-of-range quantile must raise, not return nonsense"""
        mat = csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        with pytest.raises(ValueError, match="Quantiles must be in the range"):
            sparse_quantile_axis0(mat, q)

    def test_single_row(self):
        """One event: the quantile is that event's values for any q"""
        dense = np.array([[5.0, 0.0, -2.0]])
        mat = csr_matrix(dense)
        npt.assert_allclose(sparse_quantile_axis0(mat, 0.3), [5.0, 0.0, -2.0])

    def test_does_not_densify_whole_matrix(self):
        """The full matrix must never be passed to toarray() (issue #1203)"""
        dense = np.zeros((6, 12))
        dense[1, 3] = 4.0
        dense[4, 9] = -2.0
        mat = csr_matrix(dense)
        densified = []
        original = csr_matrix.toarray

        def spy(self, *args, **kwargs):
            densified.append(self.shape)
            return original(self, *args, **kwargs)

        csr_matrix.toarray = spy
        try:
            sparse_quantile_axis0(mat, 0.5, block_size=4)
        finally:
            csr_matrix.toarray = original

        assert mat.shape not in densified, f"full matrix densified: {densified}"
