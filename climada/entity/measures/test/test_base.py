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

Unit tests for the Measure class.
"""

import copy
from unittest.mock import MagicMock, patch

import pytest

from climada.entity.measures.base import Measure, allow_kwargs
from climada.entity.measures.cost_income import CostIncome


def _make_mock():
    """Return a simple MagicMock to stand in for Exposures / ImpactFuncSet / Hazard."""
    m = MagicMock()
    m.__deepcopy__ = lambda self: copy.copy(self)  # survive deepcopy
    return m


class TestAllowKwargs:
    def test_unknown_kwargs_are_filtered(self):
        @allow_kwargs
        def add(a, b):
            return a + b

        assert add(1, 2, extra=99) == 3

    def test_known_kwargs_are_passed(self):
        @allow_kwargs
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        assert greet("Alice", greeting="Hi", unused="x") == "Hi, Alice!"

    def test_func_with_var_keyword_receives_all(self):
        received = {}

        @allow_kwargs
        def sink(**kwargs):
            received.update(kwargs)

        sink(a=1, b=2)
        assert received == {"a": 1, "b": 2}

    def test_positional_args_still_work(self):
        @allow_kwargs
        def mul(x, y):
            return x * y

        assert mul(3, 4) == 12

    def test_wraps_preserves_name(self):
        def my_func(x):
            return x

        wrapped = allow_kwargs(my_func)
        assert wrapped.__name__ == "my_func"


class TestMeasureInit:
    def test_name_is_set(self):
        m = Measure("flood_barrier")
        assert m.name == "flood_barrier"

    def test_default_cost_income_created(self):
        m = Measure("test")
        assert isinstance(m.cost_income, CostIncome)

    def test_provided_cost_income_used(self):
        ci = CostIncome(init_cost=500)
        m = Measure("test", cost_income=ci)
        assert m.cost_income is ci

    def test_default_color_rgb(self):
        m = Measure("test")
        assert m.color_rgb == (0, 0, 0)

    def test_custom_color_rgb(self):
        m = Measure("test", color_rgb=(1.0, 0.5, 0.0))
        assert m.color_rgb == (1.0, 0.5, 0.0)

    def test_sub_measures_default_none(self):
        m = Measure("test")
        assert m.sub_measures is None

    def test_sub_measures_stored(self):
        m = Measure("combo", sub_measures=["a", "b"])
        assert m.sub_measures == ["a", "b"]

    def test_implementation_duration_stored(self):
        from pandas.tseries.offsets import DateOffset

        offset = DateOffset(years=2)
        m = Measure("test", implementation_duration=offset)
        assert m.implementation_duration == offset

    def test_change_functions_wrapped_with_allow_kwargs(self):
        def my_fn(obj):
            return obj

        m = Measure("test", exposures_changes=my_fn)
        # The wrapped function should not raise on extra kwargs
        stub = _make_mock()
        result = m.exposures_changes(stub, unexpected_kwarg=42)
        assert result is stub


class TestIsSerializable:
    def test_false_without_config(self):
        m = Measure("test")
        assert m.is_serializable is False

    def test_true_with_config(self):
        mock_config = MagicMock()
        m = Measure("test", _config=mock_config)
        assert m.is_serializable is True


class TestApplyExposuresChanges:
    def test_identity_does_not_deepcopy(self):
        m = Measure("test")  # defaults to identity_function
        exp = _make_mock()
        result = m.apply_exposures_changes(exp, enforce_copy=True)
        assert result is exp  # identity: no copy

    def test_custom_fn_deepcopies_when_enforce_copy(self):
        copied = _make_mock()
        original = _make_mock()

        def fn(obj, **kw):
            return obj

        with patch(
            "climada.entity.measures.base.copy.deepcopy", return_value=copied
        ) as mock_dc:
            m = Measure("test", exposures_changes=fn)
            result = m.apply_exposures_changes(original, enforce_copy=True)
            mock_dc.assert_called_once()
            assert result is copied

    def test_no_deepcopy_when_enforce_copy_false(self):
        original = _make_mock()

        def fn(obj, **kw):
            return obj

        with patch("climada.entity.measures.base.copy.deepcopy") as mock_dc:
            m = Measure("test", exposures_changes=fn)
            m.apply_exposures_changes(original, enforce_copy=False)
            mock_dc.assert_not_called()

    def test_kwargs_forwarded(self):
        received = {}

        def fn(obj, **kwargs):
            received.update(kwargs)
            return obj

        m = Measure("test", exposures_changes=fn)
        exp = _make_mock()
        m.apply_exposures_changes(exp, enforce_copy=False, year=2030)
        assert received.get("year") == 2030

    def test_missing_required_arg_raises_type_error_with_message(self):
        def fn(obj, required_arg):
            return obj

        m = Measure("test", exposures_changes=fn)
        with pytest.raises(TypeError, match="required positional argument"):
            m.apply_exposures_changes(_make_mock(), enforce_copy=False)


class TestApplyImpfsetChanges:
    def test_identity_returns_same_object(self):
        m = Measure("test")
        impfset = _make_mock()
        result = m.apply_impfset_changes(impfset, enforce_copy=True)
        assert result is impfset

    def test_custom_fn_deepcopies_when_enforce_copy(self):
        copied = _make_mock()

        def fn(obj, **kw):
            return obj

        with patch("climada.entity.measures.base.copy.deepcopy", return_value=copied):
            m = Measure("test", impfset_changes=fn)
            result = m.apply_impfset_changes(_make_mock(), enforce_copy=True)
            assert result is copied

    def test_kwargs_forwarded(self):
        received = {}

        def fn(obj, **kwargs):
            received.update(kwargs)
            return obj

        m = Measure("test", impfset_changes=fn)
        m.apply_impfset_changes(_make_mock(), enforce_copy=False, scenario="rcp85")
        assert received.get("scenario") == "rcp85"

    def test_missing_required_arg_raises_type_error_with_message(self):
        def fn(obj, required_arg):
            return obj

        m = Measure("test", impfset_changes=fn)
        with pytest.raises(TypeError, match="required positional argument"):
            m.apply_impfset_changes(_make_mock(), enforce_copy=False)


class TestApplyHazardChanges:
    def test_identity_returns_same_object(self):
        m = Measure("test")
        hazard = _make_mock()
        result = m.apply_hazard_changes(hazard, enforce_copy=True)
        assert result is hazard

    def test_custom_fn_deepcopies_when_enforce_copy(self):
        copied = _make_mock()

        def fn(obj, **kw):
            return obj

        with patch("climada.entity.measures.base.copy.deepcopy", return_value=copied):
            m = Measure("test", hazard_changes=fn)
            result = m.apply_hazard_changes(_make_mock(), enforce_copy=True)
            assert result is copied

    def test_no_deepcopy_when_enforce_copy_false(self):
        def fn(obj, **kw):
            return obj

        with patch("climada.entity.measures.base.copy.deepcopy") as mock_dc:
            m = Measure("test", hazard_changes=fn)
            m.apply_hazard_changes(_make_mock(), enforce_copy=False)
            mock_dc.assert_not_called()

    def test_kwargs_forwarded(self):
        received = {}

        def fn(obj, **kwargs):
            received.update(kwargs)
            return obj

        m = Measure("test", hazard_changes=fn)
        m.apply_hazard_changes(_make_mock(), enforce_copy=False, intensity_scale=0.8)
        assert received.get("intensity_scale") == 0.8

    def test_missing_required_arg_raises_type_error_with_message(self):
        def fn(obj, required_arg):
            return obj

        m = Measure("test", hazard_changes=fn)
        with pytest.raises(TypeError, match="required positional argument"):
            m.apply_hazard_changes(_make_mock(), enforce_copy=False)


class TestApply:
    def _make_measure_with_trackers(self):
        """Return a Measure whose change functions record received kwargs."""
        exp_kwargs, haz_kwargs, impfset_kwargs = {}, {}, {}

        def track_exp(obj, **kw):
            exp_kwargs.update(kw)
            return obj

        def track_haz(obj, **kw):
            haz_kwargs.update(kw)
            return obj

        def track_impfset(obj, **kw):
            impfset_kwargs.update(kw)
            return obj

        m = Measure(
            "tracker",
            exposures_changes=track_exp,
            hazard_changes=track_haz,
            impfset_changes=track_impfset,
        )
        return m, exp_kwargs, haz_kwargs, impfset_kwargs

    def test_returns_three_objects(self):
        m = Measure("test")
        exp, impfset, haz = _make_mock(), _make_mock(), _make_mock()
        result = m.apply(exp, impfset, haz)
        assert len(result) == 3

    def test_default_triplet_context_passed_as_kwargs(self):
        m, exp_kw, haz_kw, impfset_kw = self._make_measure_with_trackers()
        exp, impfset, haz = _make_mock(), _make_mock(), _make_mock()
        m.apply(exp, impfset, haz, enforce_copy=False)

        assert exp_kw["base_exposures"] is exp
        assert exp_kw["base_impfset"] is impfset
        assert exp_kw["base_hazard"] is haz

    def test_entity_specific_kwargs_override_defaults(self):
        m, exp_kw, _, _ = self._make_measure_with_trackers()
        exp, impfset, haz = _make_mock(), _make_mock(), _make_mock()
        custom = _make_mock()
        m.apply(
            exp,
            impfset,
            haz,
            enforce_copy=False,
            kwargs_exposures={"base_exposures": custom},
        )
        assert exp_kw["base_exposures"] is custom

    def test_entity_specific_kwargs_not_leaked_to_other_functions(self):
        m, exp_kw, haz_kw, _ = self._make_measure_with_trackers()
        exp, impfset, haz = _make_mock(), _make_mock(), _make_mock()
        m.apply(
            exp,
            impfset,
            haz,
            enforce_copy=False,
            kwargs_exposures={"exp_only_param": 42},
        )
        assert "exp_only_param" not in haz_kw
        assert "exp_only_param" in exp_kw

    def test_apply_with_all_identities_returns_originals(self):
        m = Measure("test")  # all identity functions
        exp, impfset, haz = _make_mock(), _make_mock(), _make_mock()
        new_exp, new_impfset, new_haz = m.apply(exp, impfset, haz)
        assert new_exp is exp
        assert new_impfset is impfset
        assert new_haz is haz

    def test_apply_order_is_exposures_hazard_impfset(self):
        """Verify the transformation order documented in the docstring."""
        call_order = []

        def track(name):
            def fn(obj, **kw):
                call_order.append(name)
                return obj

            return fn

        m = Measure(
            "order_test",
            exposures_changes=track("exposures"),
            hazard_changes=track("hazard"),
            impfset_changes=track("impfset"),
        )
        m.apply(_make_mock(), _make_mock(), _make_mock(), enforce_copy=False)
        assert call_order == ["exposures", "hazard", "impfset"]
