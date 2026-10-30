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


Unit tests for the helper functions.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from climada.entity.measures.helper import (
    change_impfset,
    composite_fun,
    helper_exposure,
    helper_hazard,
    helper_impfset,
    impact_intensity_rp_cutoff_helper,
    replace_hazard,
)
from climada.entity.measures.measure_config import (
    ExposuresModifierConfig,
    HazardModifierConfig,
    ImpfsetModifierConfig,
)
from climada.entity.measures.test.conftest import (
    HAZARD_MAX_INTENSITY,
    HAZARD_TYPE,
    IMPF_ID,
)


class TestCompositeFun:
    def test_single_function_applied(self):
        double = lambda x, **kw: x * 2
        composed = composite_fun(double)
        assert composed(3) == 6

    def test_two_functions_applied_right_to_left(self):
        add1 = lambda x, **kw: x + 1
        double = lambda x, **kw: x * 2
        # composite_fun(add1, double) => add1(double(x)) => 2*x + 1
        composed = composite_fun(add1, double)
        assert composed(3) == 7

    def test_three_functions_order(self):
        f = lambda x, **kw: x + 10
        g = lambda x, **kw: x * 2
        h = lambda x, **kw: x - 1
        # f(g(h(x))) = f(g(x-1)) = f(2*(x-1)) = 2*(x-1) + 10
        composed = composite_fun(f, g, h)
        assert composed(5) == 18

    def test_no_functions_returns_identity(self):
        composed = composite_fun()
        obj = object()
        assert composed(obj) is obj

    def test_kwargs_forwarded_to_all_functions(self):
        received_by_f, received_by_g = {}, {}

        def f(x, **kw):
            received_by_f.update(kw)
            return x

        def g(x, **kw):
            received_by_g.update(kw)
            return x

        composite_fun(f, g)(42, year=2030)
        assert received_by_f.get("year") == 2030
        assert received_by_g.get("year") == 2030


# ===========================================================================
# replace_hazard
# ===========================================================================


class TestReplaceHazard:
    def test_returns_new_hazard_ignoring_input(self):
        new_haz = MagicMock()
        original_haz = MagicMock()
        fn = replace_hazard(new_haz)
        result = fn(original_haz)
        assert result is new_haz

    def test_different_inputs_always_return_same_object(self):
        new_haz = MagicMock()
        fn = replace_hazard(new_haz)
        assert fn(MagicMock()) is new_haz
        assert fn(MagicMock()) is new_haz


# ===========================================================================
# helper_hazard  (no new hazard path, no rp cutoff)
# ===========================================================================


class TestHelperHazard:
    def _config(self, mult=1.0, add=0.0, path=None, rp=None):
        return HazardModifierConfig(
            haz_type="TEST",
            haz_int_mult=mult,
            haz_int_add=add,
            new_hazard_path=path,
            impact_rp_cutoff=rp,
        )

    def test_identity_transform_leaves_data_unchanged(self, hazard_factory):
        haz = hazard_factory()
        original_data = haz.intensity.data.copy()
        fn = helper_hazard(self._config(mult=1.0, add=0.0))
        result = fn(haz)
        np.testing.assert_allclose(result.intensity.data, original_data)

    def test_multiplicative_scaling(self, hazard_factory):
        haz = hazard_factory()
        original_data = haz.intensity.data.copy()
        fn = helper_hazard(self._config(mult=0.5))
        result = fn(haz)
        np.testing.assert_allclose(result.intensity.data, original_data * 0.5)

    def test_additive_shift_on_nonzero_entries(self, hazard_factory):
        haz = hazard_factory()
        original_data = haz.intensity.data.copy()
        fn = helper_hazard(self._config(add=10.0))
        result = fn(haz)
        np.testing.assert_allclose(result.intensity.data, original_data + 10.0)

    def test_negative_results_clipped_to_zero(self, hazard_factory):
        haz = hazard_factory()
        fn = helper_hazard(self._config(mult=-1.0))
        result = fn(haz)
        assert result.intensity.nnz == 0

    def test_negative_after_add_clipped_and_eliminated(self, hazard_factory):
        haz = hazard_factory()
        fn = helper_hazard(self._config(add=-HAZARD_MAX_INTENSITY - 1))
        result = fn(haz)
        assert (result.intensity.data >= 0).all()

    def test_combined_mult_and_add(self, hazard_factory):
        haz = hazard_factory()
        original_data = haz.intensity.data.copy()
        fn = helper_hazard(self._config(mult=2.0, add=5.0))
        result = fn(haz)
        np.testing.assert_allclose(result.intensity.data, original_data * 2.0 + 5.0)

    def test_returns_hazard_object(self, hazard_factory):
        from climada.hazard.base import Hazard

        haz = hazard_factory()
        fn = helper_hazard(self._config())
        assert isinstance(fn(haz), Hazard)


# ===========================================================================
# helper_hazard  (with rp_cutoff — exercises impact_intensity_rp_cutoff_helper)
# ===========================================================================


class TestHelperHazardRpCutoff:
    """
    Integration of helper_hazard + impact_intensity_rp_cutoff_helper.

    Using the conftest fixture setup:
      - Event 1 (freq 0.03): zero impact everywhere  → should be zeroed for any RP
      - Event 2 (freq 0.01): zero impact (hits centroid 0, value=0) → zeroed for any RP
      - Event 3 (freq 0.006): impact = 500           → RP ~167y
      - Event 4 (freq 0.004): impact = 3750          → RP = 250y
      - Event 5 (freq 0.0):   zero frequency         → never contributes
    """

    def _config_with_rp(self, rp):
        return HazardModifierConfig(
            haz_type=HAZARD_TYPE,
            impact_rp_cutoff=rp,
        )

    def test_very_large_rp_zeros_all_events(self, hazard_factory, exposures, impfset):
        """RP larger than any event → all intensities zeroed."""
        # Avoid the 0 freq of the default hazard_factory settings
        haz = hazard_factory(frequency_array=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        fn = helper_hazard(self._config_with_rp(10_000))
        result = fn(
            haz,
            base_exposures=exposures,
            base_impfset=impfset,
            base_hazard=haz,
        )
        assert result.intensity.nnz == 0

    def test_very_small_rp_keeps_all_events(self, hazard_factory, exposures, impfset):
        """RP smaller than all events → nothing is zeroed."""
        haz = hazard_factory()
        original_nnz = haz.intensity.nnz
        fn = helper_hazard(self._config_with_rp(1))
        result = fn(
            haz,
            base_exposures=exposures,
            base_impfset=impfset,
            base_hazard=haz,
        )
        assert result.intensity.nnz == original_nnz


# ===========================================================================
# impact_intensity_rp_cutoff_helper (directly)
# ===========================================================================


class TestImpactIntensityRpCutoffHelper:
    def test_returns_callable(self):
        fn = impact_intensity_rp_cutoff_helper(100)
        assert callable(fn)

    def test_region_id_filter_restricts_impact_computation(
        self, hazard_factory, exposures_factory, impfset
    ):
        """Passing exposures_region_id should not raise and should return a Hazard."""
        from climada.hazard.base import Hazard

        exp = exposures_factory()
        exp.gdf["region_id"] = [1, 1, 2, 2, 1, 1]
        haz = hazard_factory()

        fn = impact_intensity_rp_cutoff_helper(50)
        result = fn(
            haz,
            base_exposures=exp,
            base_impfset=impfset,
            base_hazard=haz,
            exposures_region_id=[1],
        )
        assert isinstance(result, Hazard)


# ===========================================================================
# change_impfset
# ===========================================================================


class TestChangeImpfset:
    def test_returns_new_impfset_ignoring_input(self, impfset_factory):
        new_ifs = impfset_factory()
        original_ifs = impfset_factory()
        fn = change_impfset(new_ifs)
        assert fn(original_ifs) is new_ifs

    def test_repeated_calls_return_same_object(self, impfset_factory):
        new_ifs = impfset_factory()
        fn = change_impfset(new_ifs)
        assert fn(impfset_factory()) is new_ifs
        assert fn(impfset_factory()) is new_ifs


# ===========================================================================
# helper_impfset
# ===========================================================================


class TestHelperImpfset:
    def _config(
        self,
        impf_ids=None,
        int_mult=1.0,
        int_add=0.0,
        mdd_mult=1.0,
        mdd_add=0.0,
        paa_mult=1.0,
        paa_add=0.0,
        path=None,
    ):
        return ImpfsetModifierConfig(
            haz_type=HAZARD_TYPE,
            impf_ids=impf_ids,
            impf_int_mult=int_mult,
            impf_int_add=int_add,
            impf_mdd_mult=mdd_mult,
            impf_mdd_add=mdd_add,
            impf_paa_mult=paa_mult,
            impf_paa_add=paa_add,
            new_impfset_path=path,
        )

    def test_identity_transform_leaves_functions_unchanged(self, impfset_factory):
        ifs = impfset_factory()
        original_mdd = ifs.get_func(haz_type=HAZARD_TYPE)[0].mdd.copy()
        fn = helper_impfset(self._config())
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].mdd, original_mdd
        )

    def test_mdd_multiplicative_scaling(self, impfset_factory):
        ifs = impfset_factory()
        original_mdd = ifs.get_func(haz_type=HAZARD_TYPE)[0].mdd.copy()
        fn = helper_impfset(self._config(mdd_mult=0.5))
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].mdd, original_mdd * 0.5
        )

    def test_paa_additive_shift(self, impfset_factory):
        ifs = impfset_factory()
        original_paa = ifs.get_func(haz_type=HAZARD_TYPE)[0].paa.copy()
        fn = helper_impfset(self._config(paa_add=0.1))
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].paa, original_paa + 0.1
        )

    def test_intensity_linear_transform(self, impfset_factory):
        ifs = impfset_factory()
        original_int = ifs.get_func(haz_type=HAZARD_TYPE)[0].intensity.copy()
        fn = helper_impfset(self._config(int_mult=2.0, int_add=5.0))
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].intensity,
            original_int * 2.0 + 5.0,
        )

    def test_specific_impf_id_targeted(self, impfset_factory):
        ifs = impfset_factory()
        original_mdd = ifs.get_func(haz_type=HAZARD_TYPE)[0].mdd.copy()
        fn = helper_impfset(self._config(impf_ids=IMPF_ID, mdd_mult=0.0))
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].mdd, original_mdd * 0.0
        )

    def test_non_matching_impf_id_not_modified(self, impfset_factory):
        ifs = impfset_factory()
        original_mdd = ifs.get_func(haz_type=HAZARD_TYPE)[0].mdd.copy()
        fn = helper_impfset(self._config(impf_ids=IMPF_ID + 99, mdd_mult=0.0))
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].mdd, original_mdd
        )

    def test_all_keyword_targets_every_function(self, impfset_factory):
        ifs = impfset_factory()
        fn = helper_impfset(self._config(impf_ids="all", paa_mult=0.0))
        result = fn(ifs)
        for impf in result.get_func(haz_type=HAZARD_TYPE):
            np.testing.assert_allclose(impf.paa, 0.0)

    def test_invalid_impf_ids_raises_value_error(self, impfset_factory):
        ifs = impfset_factory()
        fn = helper_impfset(self._config(impf_ids={"invalid": "dict"}))
        with pytest.raises(ValueError, match="invalid"):
            fn(ifs)

    def test_list_of_ids(self, impfset_factory):
        ifs = impfset_factory()
        fn = helper_impfset(self._config(impf_ids=[IMPF_ID], mdd_add=1.0))
        original_mdd = ifs.get_func(haz_type=HAZARD_TYPE)[0].mdd.copy()
        result = fn(ifs)
        np.testing.assert_allclose(
            result.get_func(haz_type=HAZARD_TYPE)[0].mdd, original_mdd + 1.0
        )


# ===========================================================================
# helper_exposure
# ===========================================================================


class TestHelperExposure:
    def _config(self, reassign=None, set_to_zero=None, path=None):
        return ExposuresModifierConfig(
            new_exposures_path=path,
            reassign_impf_id=reassign,
            set_to_zero=set_to_zero,
        )

    def test_identity_config_leaves_exposure_unchanged(self, exposures_factory):
        exp = exposures_factory()
        original_values = exp.gdf["value"].copy()
        fn = helper_exposure(self._config())
        result = fn(exp)
        np.testing.assert_array_equal(
            result.gdf["value"].values, original_values.values
        )

    def test_set_to_zero_by_boolean_mask(self, exposures_factory):
        exp = exposures_factory()
        mask = exp.gdf["value"] > 2000
        fn = helper_exposure(self._config(set_to_zero=mask))
        result = fn(exp)
        assert (result.gdf.loc[mask, "value"] == 0).all()

    def test_set_to_zero_does_not_affect_other_rows(self, exposures_factory):
        exp = exposures_factory()
        mask = exp.gdf["value"] > 4000
        original_below = exp.gdf.loc[~mask, "value"].copy()
        fn = helper_exposure(self._config(set_to_zero=mask))
        result = fn(exp)
        np.testing.assert_array_equal(
            result.gdf.loc[~mask, "value"].values, original_below.values
        )

    def test_reassign_impf_id_remaps_correctly(self, exposures_factory):
        exp = exposures_factory(impf_id=1)
        col = f"impf_{HAZARD_TYPE}"
        fn = helper_exposure(self._config(reassign={HAZARD_TYPE: {1: 2}}))
        result = fn(exp)
        assert (result.gdf[col] == 2).all()

    def test_reassign_impf_id_unknown_value_left_unchanged(self, exposures_factory):
        exp = exposures_factory(impf_id=1)
        col = f"impf_{HAZARD_TYPE}"
        fn = helper_exposure(self._config(reassign={HAZARD_TYPE: {99: 2}}))
        result = fn(exp)
        assert (result.gdf[col] == 1).all()

    def test_combined_set_to_zero_and_reassign(self, exposures_factory):
        exp = exposures_factory(impf_id=1)
        col = f"impf_{HAZARD_TYPE}"
        mask = exp.gdf["value"] > 3000
        fn = helper_exposure(
            self._config(
                set_to_zero=mask,
                reassign={HAZARD_TYPE: {1: 3}},
            )
        )
        result = fn(exp)
        assert (result.gdf.loc[mask, "value"] == 0).all()
        assert (result.gdf[col] == 3).all()
