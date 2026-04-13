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

Tests for MeasureConfig and related dataclasses.
"""

# tests/entity/measures/test_measure_config.py

import logging
import warnings
from datetime import datetime

import pandas as pd
import pytest

from climada.entity.measures.measure_config import (
    CostIncomeConfig,
    ExposuresModifierConfig,
    HazardModifierConfig,
    ImpfsetModifierConfig,
    MeasureConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_measure_dict():
    return {"name": "seawall", "haz_type": "TC"}


@pytest.fixture
def full_measure_dict():
    return {
        "name": "seawall",
        "haz_type": "TC",
        "haz_int_mult": 0.8,
        "haz_int_add": -0.1,
        "impf_mdd_mult": 0.9,
        "impf_paa_mult": 0.95,
        "impf_ids": [1, 2],
        "reassign_impf_id": {"TC": {1: 3}},
        "set_to_zero": [10, 20],
        "init_cost": 1000.0,
        "periodic_cost": 50.0,
        "color_rgb": [0.1, 0.5, 0.9],
        "implementation_duration": "2Y",
    }


@pytest.fixture
def basic_impfset_config():
    return ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.9)


@pytest.fixture
def basic_hazard_config():
    return HazardModifierConfig(haz_type="TC", haz_int_mult=0.8)


@pytest.fixture
def basic_exposures_config():
    return ExposuresModifierConfig(reassign_impf_id={"TC": {1: 2}})


@pytest.fixture
def basic_cost_income_config():
    return CostIncomeConfig(init_cost=1000.0, periodic_cost=50.0)


@pytest.fixture
def full_measure_config(full_measure_dict):
    return MeasureConfig.from_dict(full_measure_dict)


# ---------------------------------------------------------------------------
# _ModifierConfig (via concrete subclasses)
# ---------------------------------------------------------------------------


def test_modifier_config_to_dict_omits_defaults():
    config = ImpfsetModifierConfig(haz_type="TC")
    result = config.to_dict()
    assert result == {}


def test_modifier_config_to_dict_includes_non_defaults():
    config = ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.5, impf_paa_add=0.1)
    result = config.to_dict()
    assert result["impf_mdd_mult"] == 0.5
    assert result["impf_paa_add"] == 0.1


def test_modifier_config_from_dict_ignores_unknown_keys():
    d = {"haz_type": "TC", "unknown_field": 99, "another_unknown": "foo"}
    config = ImpfsetModifierConfig.from_dict(d)
    assert config.haz_type == "TC"
    assert not hasattr(config, "unknown_field")


def test_modifier_config_from_dict_roundtrip():
    config = ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.5, impf_paa_add=0.1)
    d = {**config.to_dict(), "haz_type": "TC"}
    recovered = ImpfsetModifierConfig.from_dict(d)
    assert recovered.impf_mdd_mult == config.impf_mdd_mult
    assert recovered.impf_paa_add == config.impf_paa_add


def test_modifier_config_filter_dict_to_fields_filters_extra_keys():
    d = {"haz_type": "TC", "impf_mdd_mult": 0.5, "not_a_field": 123}
    filtered = ImpfsetModifierConfig._filter_dict_to_fields(d)
    assert "not_a_field" not in filtered
    assert "haz_type" in filtered
    assert "impf_mdd_mult" in filtered
    assert filtered["haz_type"] == "TC"
    assert filtered["impf_mdd_mult"] == 0.5


def test_modifier_config_filter_out_default_fields_partitions_correctly():
    config = ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.5)
    non_defaults, defaults = config._filter_out_default_fields()
    assert "impf_mdd_mult" in non_defaults
    assert "impf_mdd_mult" not in defaults
    assert "impf_paa_mult" in defaults
    assert "impf_paa_mult" not in non_defaults
    from dataclasses import fields

    all_field_names = {f.name for f in fields(config) if f.name != "haz_type"}
    assert set(non_defaults) | set(defaults) == all_field_names


def test_modifier_config_repr_shows_non_defaults_prominently():
    config = ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.5)
    r = repr(config)
    assert "Non default fields" in r
    assert "impf_mdd_mult" in r


def test_modifier_config_repr_empty_when_all_defaults():
    config = ImpfsetModifierConfig(haz_type="TC")
    r = repr(config)
    assert "Non default fields" not in r


# ---------------------------------------------------------------------------
# ImpfsetModifierConfig
# ---------------------------------------------------------------------------


def test_impfset_modifier_config_defaults():
    config = ImpfsetModifierConfig(haz_type="TC")
    assert config.impf_ids is None
    assert config.impf_mdd_mult == 1.0
    assert config.impf_mdd_add == 0.0
    assert config.impf_paa_mult == 1.0
    assert config.impf_paa_add == 0.0
    assert config.impf_int_mult == 1.0
    assert config.impf_int_add == 0.0
    assert config.new_impfset_path is None


def test_impfset_modifier_config_from_dict_roundtrip():
    config = ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.8, impf_ids=[1, 2])
    d = {**config.to_dict(), "haz_type": "TC"}
    recovered = ImpfsetModifierConfig.from_dict(d)
    assert recovered.impf_mdd_mult == config.impf_mdd_mult
    assert recovered.impf_ids == config.impf_ids


def test_impfset_modifier_config_to_dict_roundtrip():
    d = {"haz_type": "TC", "impf_mdd_mult": 0.8, "impf_paa_add": 0.05}
    config = ImpfsetModifierConfig.from_dict(d)
    result = {**config.to_dict(), "haz_type": "TC"}
    assert result["impf_mdd_mult"] == d["impf_mdd_mult"]
    assert result["impf_paa_add"] == d["impf_paa_add"]


def test_impfset_modifier_config_warns_when_path_and_modifiers_combined():
    with pytest.warns(UserWarning):
        ImpfsetModifierConfig(
            haz_type="TC",
            new_impfset_path="path/to/file.xlsx",
            impf_mdd_mult=0.5,
        )


def test_impfset_modifier_config_no_warning_when_only_path():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ImpfsetModifierConfig(haz_type="TC", new_impfset_path="path/to/file.xlsx")


def test_impfset_modifier_config_no_warning_when_only_modifiers():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ImpfsetModifierConfig(haz_type="TC", impf_mdd_mult=0.5)


def test_impfset_modifier_config_impf_ids_accepts_int():
    config = ImpfsetModifierConfig(haz_type="TC", impf_ids=1)
    assert config.impf_ids == 1


def test_impfset_modifier_config_impf_ids_accepts_str():
    config = ImpfsetModifierConfig(haz_type="TC", impf_ids="1")
    assert config.impf_ids == "1"


def test_impfset_modifier_config_impf_ids_accepts_list():
    config = ImpfsetModifierConfig(haz_type="TC", impf_ids=[1, 2, "3"])
    assert config.impf_ids == [1, 2, "3"]


def test_impfset_modifier_config_impf_ids_accepts_none():
    config = ImpfsetModifierConfig(haz_type="TC", impf_ids=None)
    assert config.impf_ids is None


# ---------------------------------------------------------------------------
# HazardModifierConfig
# ---------------------------------------------------------------------------


def test_hazard_modifier_config_defaults():
    config = HazardModifierConfig(haz_type="TC")
    assert config.haz_int_mult == 1.0
    assert config.haz_int_add == 0.0
    assert config.new_hazard_path is None
    assert config.impact_rp_cutoff is None


def test_hazard_modifier_config_from_dict_roundtrip():
    config = HazardModifierConfig(haz_type="TC", haz_int_mult=0.8, haz_int_add=-0.1)
    d = {**config.to_dict(), "haz_type": "TC"}
    recovered = HazardModifierConfig.from_dict(d)
    assert recovered.haz_int_mult == config.haz_int_mult
    assert recovered.haz_int_add == config.haz_int_add


def test_hazard_modifier_config_to_dict_roundtrip():
    d = {"haz_type": "TC", "haz_int_mult": 0.7, "haz_int_add": -0.2}
    config = HazardModifierConfig.from_dict(d)
    result = {**config.to_dict(), "haz_type": "TC"}
    assert result["haz_int_mult"] == d["haz_int_mult"]
    assert result["haz_int_add"] == d["haz_int_add"]


def test_hazard_modifier_config_warns_when_path_and_modifiers_combined():
    with pytest.warns(UserWarning):
        HazardModifierConfig(
            haz_type="TC",
            new_hazard_path="path/to/hazard.h5",
            haz_int_mult=0.5,
        )


def test_hazard_modifier_config_warns_when_path_and_rp_cutoff_combined():
    with pytest.warns(UserWarning):
        HazardModifierConfig(
            haz_type="TC",
            new_hazard_path="path/to/hazard.h5",
            impact_rp_cutoff=100.0,
        )


def test_hazard_modifier_config_no_warning_when_only_path():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HazardModifierConfig(haz_type="TC", new_hazard_path="path/to/hazard.h5")


def test_hazard_modifier_config_no_warning_when_only_modifiers():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HazardModifierConfig(haz_type="TC", haz_int_mult=0.5)


# ---------------------------------------------------------------------------
# ExposuresModifierConfig
# ---------------------------------------------------------------------------


def test_exposures_modifier_config_defaults():
    config = ExposuresModifierConfig()
    assert config.reassign_impf_id is None
    assert config.set_to_zero is None
    assert config.new_exposures_path is None


def test_exposures_modifier_config_from_dict_roundtrip():
    config = ExposuresModifierConfig(
        reassign_impf_id={"TC": {1: 2}},
        set_to_zero=[10, 20],
    )
    d = config.to_dict()
    recovered = ExposuresModifierConfig.from_dict(d)
    assert recovered.reassign_impf_id == config.reassign_impf_id
    assert recovered.set_to_zero == config.set_to_zero


def test_exposures_modifier_config_to_dict_roundtrip():
    d = {"reassign_impf_id": {"TC": {1: 2}}, "set_to_zero": [5, 6]}
    config = ExposuresModifierConfig.from_dict(d)
    result = config.to_dict()
    assert result["reassign_impf_id"] == d["reassign_impf_id"]
    assert result["set_to_zero"] == d["set_to_zero"]


def test_exposures_modifier_config_warns_when_path_and_modifiers_combined():
    with pytest.warns(UserWarning):
        ExposuresModifierConfig(
            new_exposures_path="path/to/exp.h5",
            reassign_impf_id={"TC": {1: 2}},
        )


def test_exposures_modifier_config_no_warning_when_only_path():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExposuresModifierConfig(new_exposures_path="path/to/exp.h5")


def test_exposures_modifier_config_no_warning_when_only_modifiers():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExposuresModifierConfig(reassign_impf_id={"TC": {1: 2}})


def test_exposures_modifier_config_reassign_impf_id_accepts_int_keys():
    config = ExposuresModifierConfig(reassign_impf_id={"TC": {1: 2}})
    assert config.reassign_impf_id == {"TC": {1: 2}}


def test_exposures_modifier_config_reassign_impf_id_accepts_str_keys():
    config = ExposuresModifierConfig(reassign_impf_id={"TC": {"1": "2"}})
    assert config.reassign_impf_id == {"TC": {"1": "2"}}


def test_exposures_modifier_config_set_to_zero_accepts_none():
    config = ExposuresModifierConfig(set_to_zero=None)
    assert config.set_to_zero is None


def test_exposures_modifier_config_set_to_zero_accepts_list():
    config = ExposuresModifierConfig(set_to_zero=[1, 2, 3])
    assert config.set_to_zero == [1, 2, 3]


# ---------------------------------------------------------------------------
# CostIncomeConfig
# ---------------------------------------------------------------------------


def test_cost_income_config_defaults():
    config = CostIncomeConfig()
    assert config.init_cost == 0.0
    assert config.periodic_cost == 0.0
    assert config.periodic_income == 0.0
    assert config.cost_yearly_growth_rate == 0.0
    assert config.income_yearly_growth_rate == 0.0
    assert config.freq == "Y"
    assert config.custom_cash_flows is None


def test_cost_income_config_default_mkt_price_year_is_current_year():
    config = CostIncomeConfig()
    assert config.mkt_price_year == datetime.today().year


def test_cost_income_config_from_dict_roundtrip():
    config = CostIncomeConfig(init_cost=1000.0, periodic_cost=50.0, freq="M")
    d = config.to_dict()
    recovered = CostIncomeConfig.from_dict(d)
    assert recovered.init_cost == config.init_cost
    assert recovered.periodic_cost == config.periodic_cost
    assert recovered.freq == config.freq


def test_cost_income_config_to_dict_roundtrip():
    d = {"init_cost": 500.0, "periodic_income": 20.0, "freq": "M"}
    config = CostIncomeConfig.from_dict(d)
    result = config.to_dict()
    assert result["init_cost"] == d["init_cost"]
    assert result["periodic_income"] == d["periodic_income"]
    assert result["freq"] == d["freq"]


# ---------------------------------------------------------------------------
# MeasureConfig
# ---------------------------------------------------------------------------


def test_measure_config_from_dict_minimal(minimal_measure_dict):
    config = MeasureConfig.from_dict(minimal_measure_dict)
    assert config.name == "seawall"
    assert config.haz_type == "TC"
    assert config.impfset_modifier == ImpfsetModifierConfig(haz_type="TC")
    assert config.hazard_modifier == HazardModifierConfig(haz_type="TC")
    assert config.exposures_modifier == ExposuresModifierConfig()
    assert config.cost_income == CostIncomeConfig()


def test_measure_config_from_dict_full(full_measure_dict):
    config = MeasureConfig.from_dict(full_measure_dict)
    assert config.hazard_modifier.haz_int_mult == full_measure_dict["haz_int_mult"]
    assert config.impfset_modifier.impf_mdd_mult == full_measure_dict["impf_mdd_mult"]
    assert config.exposures_modifier.set_to_zero == full_measure_dict["set_to_zero"]
    assert config.cost_income.init_cost == full_measure_dict["init_cost"]
    assert config.color_rgb == tuple(full_measure_dict["color_rgb"])
    assert (
        config.implementation_duration == full_measure_dict["implementation_duration"]
    )


def test_measure_config_from_dict_ignores_unknown_keys(minimal_measure_dict):
    d = {**minimal_measure_dict, "completely_unknown": 42}
    config = MeasureConfig.from_dict(d)
    assert config.name == "seawall"
    assert not hasattr(config, "completely_unknown")


def test_measure_config_to_dict_roundtrip(full_measure_dict):
    config = MeasureConfig.from_dict(full_measure_dict)
    recovered = MeasureConfig.from_dict(config.to_dict())
    assert recovered.name == config.name
    assert recovered.haz_type == config.haz_type
    assert recovered.hazard_modifier == config.hazard_modifier
    assert recovered.impfset_modifier == config.impfset_modifier
    assert recovered.exposures_modifier == config.exposures_modifier
    assert recovered.color_rgb == config.color_rgb
    assert recovered.implementation_duration == config.implementation_duration


def test_measure_config_to_dict_color_rgb_none(minimal_measure_dict):
    config = MeasureConfig.from_dict(minimal_measure_dict)
    result = config.to_dict()
    assert result["color_rgb"] is None


def test_measure_config_to_dict_color_rgb_set(minimal_measure_dict):
    config = MeasureConfig.from_dict(
        {**minimal_measure_dict, "color_rgb": [0.1, 0.5, 0.9]}
    )
    result = config.to_dict()
    assert result["color_rgb"] == [0.1, 0.5, 0.9]


def test_measure_config_to_yaml_roundtrip(tmp_path, full_measure_dict):
    path = str(tmp_path / "measure.yaml")
    config = MeasureConfig.from_dict(full_measure_dict)
    config.to_yaml(path)
    recovered = MeasureConfig.from_yaml(path)
    assert recovered.name == config.name
    assert recovered.haz_type == config.haz_type
    assert recovered.hazard_modifier == config.hazard_modifier
    assert recovered.impfset_modifier == config.impfset_modifier
    assert recovered.color_rgb == config.color_rgb


def test_measure_config_from_yaml_reads_first_entry(tmp_path, full_measure_dict):
    import yaml

    second = {**full_measure_dict, "name": "second_measure"}
    path = str(tmp_path / "measures.yaml")
    with open(path, "w") as f:
        yaml.dump({"measures": [full_measure_dict, second]}, f)
    config = MeasureConfig.from_yaml(path)
    assert config.name == full_measure_dict["name"]


def test_measure_config_from_row_roundtrip(full_measure_dict):
    config = MeasureConfig.from_dict(full_measure_dict)
    row = pd.Series(config.to_dict())
    recovered = MeasureConfig.from_row(row)
    assert recovered.name == config.name
    assert recovered.hazard_modifier == config.hazard_modifier
    assert recovered.impfset_modifier == config.impfset_modifier


def test_measure_config_from_row_ignores_extra_columns(full_measure_dict):
    config = MeasureConfig.from_dict(full_measure_dict)
    d = {**config.to_dict(), "extra_column": "garbage"}
    row = pd.Series(d)
    recovered = MeasureConfig.from_row(row)
    assert recovered.name == config.name


def test_measure_config_sub_configs_correctly_dispatched(full_measure_dict):
    config = MeasureConfig.from_dict(full_measure_dict)
    assert config.hazard_modifier.haz_int_mult == full_measure_dict["haz_int_mult"]
    assert config.impfset_modifier.impf_mdd_mult == full_measure_dict["impf_mdd_mult"]
    assert (
        config.exposures_modifier.reassign_impf_id
        == full_measure_dict["reassign_impf_id"]
    )
    assert config.cost_income.init_cost == full_measure_dict["init_cost"]
    assert not hasattr(config.hazard_modifier, "impf_mdd_mult")
    assert not hasattr(config.impfset_modifier, "haz_int_mult")
