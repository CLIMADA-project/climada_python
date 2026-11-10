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

Unit tests for the MeasureSet class.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

from climada.entity.measures.cost_income import CostIncome
from climada.entity.measures.measure import Measure
from climada.entity.measures.measure_config import MeasureConfig
from climada.entity.measures.measure_set import MeasureSet

MODULE = "climada.entity.measures.measure_set"
NAMES = ["flood_barrier", "sea_wall", "early_warning"]


def _make_mock():
    """Return a MagicMock standing in for Exposures / ImpactFuncSet / Hazard."""
    return MagicMock()


def _make_measure(name, init_cost=0.0, **kwargs):
    """Return a lightweight Measure.

    Parameters
    ----------
    name : str
        Measure name.
    init_cost : float, optional
        Initial cost of the measure. Default is 0.
    **kwargs
        Forwarded to `Measure`.

    Returns
    -------
    Measure
    """
    return Measure(name, cost_income=CostIncome(init_cost=init_cost), **kwargs)


def _tag(label):
    """Return a change function appending `label` to a list-like object.

    Parameters
    ----------
    label : str
        Label appended when the function is called.

    Returns
    -------
    callable
        Function ``fn(obj, **kwargs)`` that appends `label` to `obj` and returns it.
    """

    def fn(obj, **kwargs):
        obj.append(label)
        return obj

    return fn


def _single_impf(impfset):
    """Return the only ImpactFunc of an ImpactFuncSet."""
    haz_dict = next(iter(impfset.get_func().values()))
    return next(iter(haz_dict.values()))


@pytest.fixture
def measure_set():
    return MeasureSet(
        [_make_measure(name, init_cost=100.0 * (i + 1)) for i, name in enumerate(NAMES)]
    )


class TestInit:
    def test_names_and_size(self, measure_set):
        assert measure_set.names == NAMES
        assert measure_set.size == 3

    def test_empty(self):
        ms = MeasureSet([])
        assert ms.names == []
        assert ms.size == 0

    def test_duplicate_names_keep_last(self):
        first, last = _make_measure("dup"), _make_measure("dup")
        ms = MeasureSet([first, last])
        assert ms.size == 1
        assert ms.select(["dup"])["dup"] is last

    def test_accepts_generator(self):
        ms = MeasureSet(_make_measure(n) for n in NAMES)
        assert ms.names == NAMES


class TestAppend:
    def test_adds_measure(self, measure_set):
        measure_set.append(_make_measure("new"))
        assert "new" in measure_set
        assert measure_set.size == 4

    def test_overwrites_same_name(self, measure_set):
        new = _make_measure(NAMES[0])
        measure_set.append(new)
        assert measure_set.size == 3
        assert measure_set.select([NAMES[0]])[NAMES[0]] is new

    def test_non_measure_raises_type_error(self, measure_set):
        with pytest.raises(TypeError, match="Expected Measure, got str"):
            measure_set.append("not a measure")


class TestSelect:
    def test_returns_requested_measures_in_order(self, measure_set):
        result = measure_set.select([NAMES[2], NAMES[0]])
        assert list(result) == [NAMES[2], NAMES[0]]
        assert all(isinstance(m, Measure) for m in result.values())

    def test_unknown_name_raises_value_error(self, measure_set):
        with pytest.raises(ValueError, match="No measure named"):
            measure_set.select(["unknown"])

    def test_empty_selection(self, measure_set):
        assert measure_set.select([]) == {}


class TestContains:
    def test_present(self, measure_set):
        assert NAMES[0] in measure_set

    def test_absent(self, measure_set):
        assert "unknown" not in measure_set


class TestCompose:
    def test_default_name_and_sub_measures(self, measure_set):
        result = measure_set.compose(NAMES[:2])
        assert result.name == f"{NAMES[0]}_{NAMES[1]}_composed"
        assert result.sub_measures == NAMES[:2]

    def test_custom_name(self, measure_set):
        assert measure_set.compose(NAMES[:2], combo_name="combo").name == "combo"

    def test_costs_are_summed(self, measure_set):
        result = measure_set.compose(NAMES[:2])
        assert result.cost_income.init_cost == -300.0

    def test_empty_names_raises_value_error(self, measure_set):
        with pytest.raises(ValueError, match="No measures found to compose"):
            measure_set.compose([])

    def test_unknown_name_raises_value_error(self, measure_set):
        with pytest.raises(ValueError, match="No measure named"):
            measure_set.compose(["unknown"])

    @pytest.mark.parametrize(
        "attr", ["exposures_changes", "impfset_changes", "hazard_changes"]
    )
    def test_execution_order_is_right_to_left(self, attr):
        ms = MeasureSet(
            [_make_measure(f"m{i}", **{attr: _tag(f"m{i}")}) for i in (1, 2, 3)]
        )
        composed = ms.compose(["m1", "m2", "m3"])
        assert getattr(composed, attr)([]) == ["m3", "m2", "m1"]


class TestCombineHazards:
    def test_takes_minimum_of_intensity_and_frequency(self, hazard_factory):
        ref = hazard_factory()
        low = hazard_factory(intensity_scale=0.5, frequency_scale=2.0)
        high = hazard_factory(intensity_scale=1.0, frequency_scale=0.5)

        result = MeasureSet._combine_hazards([low, high])

        np.testing.assert_allclose(
            result.intensity.toarray(), 0.5 * ref.intensity.toarray()
        )
        np.testing.assert_allclose(result.frequency, 0.5 * ref.frequency)
        assert result.fraction.shape == ref.fraction.shape

    def test_single_hazard_unchanged(self, hazard_factory):
        ref = hazard_factory()
        result = MeasureSet._combine_hazards([hazard_factory()])
        np.testing.assert_allclose(result.intensity.toarray(), ref.intensity.toarray())
        np.testing.assert_allclose(result.frequency, ref.frequency)


class TestCombineImpfsets:
    def test_safest_parameters_and_widest_intensity(self, impfset_factory):
        base = impfset_factory()
        mod_a = impfset_factory()
        mod_b = impfset_factory(max_intensity=200, paa_scale=0.5)
        _single_impf(mod_b).mdd = np.array([0, 0.25, 0.5])

        result = MeasureSet._combine_impfsets(base, [mod_a, mod_b])

        base_impf = _single_impf(base)
        impf = result.get_func(haz_type=base_impf.haz_type, fun_id=base_impf.id)
        np.testing.assert_allclose(impf.intensity, [0, 100, 200])
        np.testing.assert_allclose(impf.mdd, [0, 0.25, 0.5])
        np.testing.assert_allclose(impf.paa, [0.5, 0.5, 0.5])

    def test_metadata_taken_from_base(self, impfset_factory):
        base = impfset_factory()
        result = MeasureSet._combine_impfsets(base, [impfset_factory()])
        base_impf, impf = _single_impf(base), _single_impf(result)
        assert impf.intensity_unit == base_impf.intensity_unit
        assert impf.name == base_impf.name
        assert impf.id == base_impf.id


class TestCombineExposures:
    def test_merges_changes_on_distinct_columns(self, exposures_factory):
        categories = [1, 1, 2, 1, 1, 3]
        base = exposures_factory(categories=categories)
        mod_value = exposures_factory(categories=categories, value_factor=2.0)
        new_categories = list(categories)
        new_categories[2] = 9
        mod_category = exposures_factory(categories=new_categories)

        result = MeasureSet._combine_exposures(base, [mod_value, mod_category])

        np.testing.assert_allclose(result.gdf["value"], 2.0 * base.gdf["value"])
        np.testing.assert_allclose(result.gdf["category"], new_categories)
        assert result.ref_year == base.ref_year
        assert result.value_unit == base.value_unit
        assert result.description == base.description

    def test_no_change_returns_baseline_values(self, exposures_factory):
        base = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        same = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        result = MeasureSet._combine_exposures(base, [same])
        np.testing.assert_allclose(result.gdf["value"], base.gdf["value"])

    def test_conflicting_changes_raise_value_error(self, exposures_factory):
        base = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        mod_a = exposures_factory(categories=[1, 1, 2, 1, 1, 3], value_factor=2.0)
        mod_b = exposures_factory(categories=[1, 1, 2, 1, 1, 3], value_factor=3.0)
        with pytest.raises(ValueError, match="Conflict"):
            MeasureSet._combine_exposures(base, [mod_a, mod_b])

    def test_different_columns_raise_value_error(self, exposures_factory):
        base = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        mod = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        mod.gdf["extra"] = 1
        with pytest.raises(ValueError, match="identical column structure"):
            MeasureSet._combine_exposures(base, [mod])

    def test_different_index_raise_value_error(self, exposures_factory):
        base = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        mod = exposures_factory(categories=[1, 1, 2, 1, 1, 3])
        mod.gdf.index = range(10, 16)
        with pytest.raises(ValueError, match="same index"):
            MeasureSet._combine_exposures(base, [mod])


class TestCombine:
    def test_default_name_and_sub_measures(self, measure_set):
        result = measure_set.combine()
        assert result.name == "_".join(NAMES)
        assert result.sub_measures == NAMES

    def test_names_subset_and_custom_name(self, measure_set):
        result = measure_set.combine(NAMES[:2], combo_name="combo")
        assert result.name == "combo"
        assert result.sub_measures == NAMES[:2]

    def test_costs_are_summed(self, measure_set):
        assert measure_set.combine().cost_income.init_cost == -600.0

    def test_empty_names_raises_value_error(self, measure_set):
        with pytest.raises(ValueError, match="No measures found to combine"):
            measure_set.combine([])

    def test_unknown_name_raises_value_error(self, measure_set):
        with pytest.raises(ValueError, match="No measure named"):
            measure_set.combine(["unknown"])

    @pytest.mark.parametrize(
        "method, helper, has_base",
        [
            ("apply_exposures_changes", "_combine_exposures", True),
            ("apply_impfset_changes", "_combine_impfsets", True),
            ("apply_hazard_changes", "_combine_hazards", False),
        ],
    )
    def test_delegates_to_combination_helper(
        self, measure_set, method, helper, has_base
    ):
        obj = _make_mock()
        combo = measure_set.combine()

        with patch.object(MeasureSet, helper) as mock_helper:
            result = getattr(combo, method)(obj, enforce_copy=False)

        # identity sub-measures return the input, one modified object per measure
        modified = [obj] * len(NAMES)
        expected_args = (obj, modified) if has_base else (modified,)
        mock_helper.assert_called_once_with(*expected_args)
        assert result is mock_helper.return_value

    def test_kwargs_forwarded_to_sub_measures(self):
        received = {}

        def track(obj, **kwargs):
            received.update(kwargs)
            return obj

        ms = MeasureSet([_make_measure("a", hazard_changes=track)])
        with patch.object(MeasureSet, "_combine_hazards"):
            ms.combine().apply_hazard_changes(
                _make_mock(), enforce_copy=False, threshold=3.5
            )

        assert received.get("threshold") == 3.5


class TestFromExcel:
    @pytest.fixture
    def var_names(self):
        return {"sheet_name": "my_sheet", "col_name": {"cost": "init_cost"}}

    def test_columns_renamed_and_one_measure_per_row(self, var_names):
        df = pd.DataFrame({"name": ["a", "b"], "cost": [1.0, 2.0]})
        rows = []

        def fake_process(row):
            rows.append(row)
            return Measure(row["name"])

        with (
            patch(f"{MODULE}.pd.read_excel", return_value=df) as mock_read,
            patch.object(MeasureSet, "_process_excel_row", side_effect=fake_process),
        ):
            ms = MeasureSet.from_excel("file.xlsx", var_names=var_names)

        mock_read.assert_called_once()
        assert mock_read.call_args.args == ("file.xlsx",)
        assert mock_read.call_args.kwargs["sheet_name"] == "my_sheet"
        assert ms.names == ["a", "b"]
        assert [r["init_cost"] for r in rows] == [1.0, 2.0]

    def test_default_var_names_used(self):
        df = pd.DataFrame({"name": ["a"]})
        with (
            patch(f"{MODULE}.pd.read_excel", return_value=df) as mock_read,
            patch.object(
                MeasureSet, "_process_excel_row", side_effect=lambda r: Measure("a")
            ),
        ):
            MeasureSet.from_excel("file.xlsx")

        assert mock_read.call_args.kwargs["sheet_name"] == "measures"

    def test_process_excel_row_builds_measure_from_config(self):
        row = pd.Series({"name": "a"})
        with (
            patch(f"{MODULE}.MeasureConfig") as mock_config,
            patch(f"{MODULE}.Measure") as mock_measure,
        ):
            result = MeasureSet._process_excel_row(row)

        mock_config.from_row.assert_called_once_with(row)
        mock_measure.from_config.assert_called_once_with(
            mock_config.from_row.return_value
        )
        assert result is mock_measure.from_config.return_value


class TestToDict:
    def test_serializable_measures_are_dumped(self):
        ms = MeasureSet([_make_measure(n, _config=MagicMock()) for n in NAMES[:2]])
        with patch.object(
            Measure, "to_dict", autospec=True, side_effect=lambda m: {"name": m.name}
        ):
            result = ms.to_dict()

        assert result == {"measures": [{"name": n} for n in NAMES[:2]]}

    def test_non_serializable_measures_skipped_with_warning(self):
        ms = MeasureSet(
            [_make_measure("ok", _config=MagicMock()), _make_measure("function_only")]
        )
        with (
            patch.object(
                Measure,
                "to_dict",
                autospec=True,
                side_effect=lambda m: {"name": m.name},
            ),
            patch(f"{MODULE}.LOGGER") as mock_logger,
        ):
            result = ms.to_dict()

        assert result == {"measures": [{"name": "ok"}]}
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args.args[1] == ["function_only"]

    def test_no_warning_when_all_serializable(self):
        ms = MeasureSet([_make_measure("ok", _config=MagicMock())])
        with (
            patch.object(Measure, "to_dict", return_value={}),
            patch(f"{MODULE}.LOGGER") as mock_logger,
        ):
            ms.to_dict()

        mock_logger.warning.assert_not_called()


class TestFromDict:
    def test_builds_one_measure_per_entry(self):
        data = {"measures": [{"name": "a"}, {"name": "b"}]}
        with (
            patch.object(MeasureConfig, "from_dict", side_effect=lambda d: d),
            patch.object(
                Measure, "from_config", side_effect=lambda cfg: Measure(cfg["name"])
            ),
        ):
            ms = MeasureSet.from_dict(data)

        assert ms.names == ["a", "b"]

    def test_empty_measures(self):
        assert MeasureSet.from_dict({"measures": []}).size == 0


class TestFileIO:
    PAYLOAD = {"measures": [{"name": "a", "init_cost": 1.0}]}

    @pytest.mark.parametrize(
        "fmt, parse", [("yaml", yaml.safe_load), ("json", json.loads)]
    )
    def test_writes_to_dict_output(self, tmp_path, fmt, parse):
        path = tmp_path / f"measures.{fmt}"
        with patch.object(MeasureSet, "to_dict", return_value=self.PAYLOAD):
            getattr(MeasureSet([]), f"to_{fmt}")(path)

        assert parse(path.read_text(encoding="utf-8")) == self.PAYLOAD

    @pytest.mark.parametrize("fmt, dump", [("yaml", yaml.dump), ("json", json.dumps)])
    def test_reads_and_forwards_to_from_dict(self, tmp_path, fmt, dump):
        path = tmp_path / f"measures.{fmt}"
        path.write_text(dump(self.PAYLOAD), encoding="utf-8")

        with patch.object(MeasureSet, "from_dict") as mock_from_dict:
            result = getattr(MeasureSet, f"from_{fmt}")(path)

        mock_from_dict.assert_called_once_with(self.PAYLOAD)
        assert result is mock_from_dict.return_value
