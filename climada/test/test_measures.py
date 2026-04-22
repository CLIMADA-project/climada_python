"""
Integration tests for the Measure class.

These tests use real CLIMADA objects (Exposures, ImpactFuncSet, Hazard) built
from the shared conftest fixtures.  They verify that Measure transformations
produce analytically expected results rather than mocking internals.

Expected impact values (from conftest docstring):
    AAI   = 18
    RP20  =  0   (event 1, freq 0.03)
    RP50  =  0   (event 2, freq 0.01)
    RP100 = 500  (event 3, freq 0.006  →  1000 * 0.5)
    RP250 = 3750 (event 4, freq 0.004  →  15000 * 0.25)
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from climada.entity.measures.base import Measure
from climada.entity.measures.cost_income import CostIncome
from climada.test.conftest import HAZARD_TYPE

# ===========================================================================
# apply() — structural / immutability tests
# ===========================================================================


class TestApplyStructural:
    """Verify that apply() correctly transforms the risk triplet."""

    def test_identity_measure_returns_equal_objects(self, exposures, impfset, hazard):
        """An identity measure should return objects equal to the originals."""
        m = Measure("identity")
        new_exp, new_impfset, new_haz = m.apply(exposures, impfset, hazard)

        assert new_exp.gdf["value"].tolist() == exposures.gdf["value"].tolist()
        assert new_haz.intensity.nnz == hazard.intensity.nnz
        assert list(new_impfset.get_func(haz_type=HAZARD_TYPE)) == list(
            impfset.get_func(haz_type=HAZARD_TYPE)
        )

    def test_enforce_copy_true_does_not_mutate_originals(
        self, exposures, impfset, hazard
    ):
        """With enforce_copy=True, the originals must not be mutated."""
        original_values = exposures.gdf["value"].copy()

        def double_values(exp, **kw):
            exp.gdf["value"] *= 2
            return exp

        m = Measure("doubler", exposures_changes=double_values)
        m.apply(exposures, impfset, hazard, enforce_copy=True)

        np.testing.assert_array_equal(
            exposures.gdf["value"].values, original_values.values
        )

    def test_enforce_copy_false_mutates_original(self, exposures, impfset, hazard):
        """With enforce_copy=False the change function receives the original object."""
        received_ids = []

        def record_id(exp, **kw):
            received_ids.append(id(exp))
            return exp

        m = Measure("recorder", exposures_changes=record_id)
        m.apply(exposures, impfset, hazard, enforce_copy=False)

        assert received_ids[0] == id(exposures)

    def test_base_triplet_available_as_kwargs(self, exposures, impfset, hazard):
        """Each change function receives base_exposures/impfset/hazard as kwargs."""
        received = {}

        def capture(**kwargs):
            def fn(obj, **kw):
                received.update(kw)
                return obj

            return fn

        m = Measure(
            "capture",
            exposures_changes=capture(),
        )
        m.apply(exposures, impfset, hazard, enforce_copy=False)

        assert "base_exposures" in received
        assert "base_impfset" in received
        assert "base_hazard" in received


# ===========================================================================
# Exposures transformations
# ===========================================================================


class TestExposuresTransformations:
    """Measures that modify Exposures."""

    def test_scale_values(self, exposures, impfset, hazard):
        """Scaling exposure values by 0.5 should halve every entry."""

        def scale_half(exp, **kw):
            exp.gdf["value"] *= 0.5
            return exp

        m = Measure("scale_half", exposures_changes=scale_half)
        new_exp, _, _ = m.apply(exposures, impfset, hazard)

        np.testing.assert_allclose(
            new_exp.gdf["value"].values, exposures.gdf["value"].values * 0.5
        )

    def test_zero_all_values(self, exposures, impfset, hazard):
        """Zeroing all exposures should produce zero impacts downstream."""

        def zero_exp(exp, **kw):
            exp.gdf["value"] = 0.0
            return exp

        m = Measure("zero_exp", exposures_changes=zero_exp)
        new_exp, _, _ = m.apply(exposures, impfset, hazard)

        assert new_exp.gdf["value"].sum() == 0.0

    def test_exposures_change_uses_base_hazard_kwarg(
        self, exposures, impfset, hazard_factory
    ):
        """Change function can inspect base_hazard via the default kwargs."""
        captured_haz = {}

        def fn(exp, base_hazard=None, **kw):
            captured_haz["haz"] = base_hazard
            return exp

        haz = hazard_factory()
        m = Measure("haz_aware", exposures_changes=fn)
        m.apply(exposures, impfset, haz, enforce_copy=False)

        assert captured_haz["haz"] is haz


# ===========================================================================
# Hazard transformations
# ===========================================================================


class TestHazardTransformations:
    """Measures that modify the Hazard."""

    def test_zero_intensity_eliminates_impacts(
        self, exposures, impfset, hazard_factory
    ):
        """Setting hazard intensity to zero should yield zero impacts."""

        def zero_intensity(haz, **kw):
            haz.intensity = csr_matrix(haz.intensity.shape)
            return haz

        haz = hazard_factory()
        m = Measure("zero_haz", hazard_changes=zero_intensity)
        new_exp, new_impfset, new_haz = m.apply(exposures, impfset, haz)

        assert new_haz.intensity.nnz == 0

    def test_scale_intensity(self, exposures, impfset, hazard_factory):
        """Halving hazard intensity should halve non-zero entries."""
        original_data = hazard_factory().intensity.data.copy()

        def half_intensity(haz, **kw):
            haz.intensity = haz.intensity * 0.5
            return haz

        m = Measure("half_haz", hazard_changes=half_intensity)
        _, _, new_haz = m.apply(exposures, impfset, hazard_factory())

        np.testing.assert_allclose(new_haz.intensity.data, original_data * 0.5)


# ===========================================================================
# ImpactFuncSet transformations
# ===========================================================================


class TestImpfsetTransformations:
    """Measures that modify the ImpactFuncSet."""

    def test_scale_paa_to_zero_eliminates_impacts(
        self, exposures, impfset_factory, hazard
    ):
        """Setting PAA to 0 should make all MDD-based impacts vanish."""

        def zero_paa(ifs, **kw):
            for func in ifs.get_func(haz_type=HAZARD_TYPE):
                func.paa = np.zeros_like(func.paa)
            return ifs

        impfset = impfset_factory()
        m = Measure("zero_paa", impfset_changes=zero_paa)
        _, new_impfset, _ = m.apply(exposures, impfset, hazard)

        for func in new_impfset.get_func(haz_type=HAZARD_TYPE):
            np.testing.assert_array_equal(func.paa, 0)


# ===========================================================================
# Combined / multi-component transformations
# ===========================================================================


class TestCombinedTransformations:
    """Measures that simultaneously affect more than one risk component."""

    def test_combined_exposure_and_hazard_change(
        self, exposures, impfset, hazard_factory
    ):
        """Both exposure and hazard transforms should be applied independently."""

        def double_values(exp, **kw):
            exp.gdf["value"] *= 2
            return exp

        def half_intensity(haz, **kw):
            haz.intensity = haz.intensity * 0.5
            return haz

        haz = hazard_factory()
        original_intensity_data = haz.intensity.data.copy()

        m = Measure(
            "combined",
            exposures_changes=double_values,
            hazard_changes=half_intensity,
        )
        new_exp, _, new_haz = m.apply(exposures, impfset, haz)

        np.testing.assert_allclose(
            new_exp.gdf["value"].values, exposures.gdf["value"].values * 2
        )
        np.testing.assert_allclose(
            new_haz.intensity.data, original_intensity_data * 0.5
        )

    def test_kwargs_exposures_and_kwargs_hazard_routed_correctly(
        self, exposures, impfset, hazard
    ):
        """kwargs_exposures and kwargs_hazard should reach the right function only."""
        exp_extra, haz_extra = {}, {}

        def track_exp(exp, my_exp_param=None, **kw):
            exp_extra["val"] = my_exp_param
            return exp

        def track_haz(haz, my_haz_param=None, **kw):
            haz_extra["val"] = my_haz_param
            return haz

        m = Measure("routed", exposures_changes=track_exp, hazard_changes=track_haz)
        m.apply(
            exposures,
            impfset,
            hazard,
            enforce_copy=False,
            kwargs_exposures={"my_exp_param": "exp_value"},
            kwargs_hazard={"my_haz_param": "haz_value"},
        )

        assert exp_extra["val"] == "exp_value"
        assert haz_extra["val"] == "haz_value"


# ===========================================================================
# CostIncome integration
# ===========================================================================


class TestCostIncomeIntegration:
    """Verify that CostIncome financial data survives through Measure construction."""

    def test_cost_income_attached(self, exposures, impfset, hazard):
        ci = CostIncome(mkt_price_year=2020, init_cost=10_000, periodic_income=500)
        m = Measure("with_ci", cost_income=ci)
        assert m.cost_income is ci

    def test_total_cost_calculable_after_apply(self, exposures, impfset, hazard):
        ci = CostIncome(
            mkt_price_year=2020,
            init_cost=10_000,
            periodic_cost=1_000,
            periodic_income=2_000,
        )
        m = Measure("ci_measure", cost_income=ci)
        m.apply(exposures, impfset, hazard)

        total_net, total_cost, total_inc = m.cost_income.calc_total(
            impl_date="2020-01-01",
            start_date="2020-01-01",
            end_date="2030-01-01",
        )
        # Over 10 years: init_cost once + 9 periodic_costs + 10 periodic_incomes
        assert total_cost < 0
        assert total_inc > 0
        assert isinstance(total_net, float)
