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

Define configuration dataclasses for Measure reading and writing.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import pandas as pd

from climada.util.string_parsers import parse_color, parse_mapping_string, parse_range

if TYPE_CHECKING:
    from climada.entity.measures.base import Measure
    from climada.entity.measures.cost_income import CostIncome

LOGGER = logging.getLogger(__name__)


@dataclass
class _ModifierConfig(ABC):
    def to_dict(self):
        # 1. Get the current values as a dict
        current_data = asdict(self)

        # 2. Identify fields where the current value differs from the default
        non_default_data = {}
        for f in fields(self):
            current_value = getattr(self, f.name)

            # Logic to get the default value (handling both default and default_factory)
            default_value = f.default
            if (
                f.default_factory is not field().default_factory
            ):  # Check if factory exists
                default_value = f.default_factory()

            if current_value != default_value:
                non_default_data[f.name] = current_data[f.name]

        non_default_data.pop("haz_type", None)
        return non_default_data

    @classmethod
    def from_dict(cls, d: dict):
        filtered = cls._filter_dict_to_fields(d)
        return cls(**filtered)

    @classmethod
    def _filter_dict_to_fields(cls, d: dict):
        """Filter out values that do not match the dataclass fields."""
        filtered = dict(
            filter(lambda k: k[0] in [f.name for f in fields(cls)], d.items())
        )
        return filtered

    def _filter_out_default_fields(self):
        non_defaults = {}
        defaults = {}
        for f in fields(self):
            val = getattr(self, f.name)
            default = f.default
            if f.default_factory is not field().default_factory:
                default = f.default_factory()

            if val != default:
                non_defaults[f.name] = val
            else:
                defaults[f.name] = val
        return non_defaults, defaults

    def __repr__(self) -> str:
        non_defaults, defaults = self._filter_out_default_fields()
        ndf_fields_str = (
            "\n\t\t\t".join(f"{k}={v!r}" for k, v in non_defaults.items())
            if non_defaults
            else None
        )
        fields_str = (
            "\n\t\t\t".join(f"{k}={v!r}" for k, v in defaults.items())
            if defaults
            else None
        )
        fields = (
            "(" "\n\t\tNon default fields:" f"\n\t\t\t{ndf_fields_str}"
            if ndf_fields_str
            else "()"
        )
        return f"{self.__class__.__name__}{fields}"


@dataclass(repr=False)
class ImpfsetModifierConfig(_ModifierConfig):
    """Configuration for impact function modifiers."""

    haz_type: str
    impf_ids: Optional[Union[int, str, list[Union[int, str]]]] = None
    impf_mdd_mult: float = 1.0
    impf_mdd_add: float = 0.0
    impf_paa_mult: float = 1.0
    impf_paa_add: float = 0.0
    impf_int_mult: float = 1.0
    impf_int_add: float = 0.0
    new_impfset_path: Optional[str] = None
    """Excel filepath for new impfset."""

    def __post_init__(self):
        if self.new_impfset_path is not None and any(
            [
                self.impf_mdd_add,
                self.impf_mdd_mult,
                self.impf_paa_add,
                self.impf_paa_mult,
                self.impf_int_add,
                self.impf_int_mult,
            ]
        ):
            LOGGER.warning(
                "Both new impfset object and impfset modifiers are provided, "
                "modifiers will be applied after changing the impfset."
            )


@dataclass(repr=False)
class HazardModifierConfig(_ModifierConfig):
    """Configuration for impact function modifiers."""

    haz_type: str
    haz_int_mult: Optional[float] = 1.0
    haz_int_add: Optional[float] = 0.0
    new_hazard_path: Optional[str] = None
    """HDF5 filepath for new hazard."""
    impact_rp_cutoff: Optional[float] = None

    def __post_init__(self):
        if self.new_hazard_path is not None and any(
            [self.haz_int_mult, self.haz_int_add, self.impact_rp_cutoff]
        ):
            LOGGER.warning(
                "Both new hazard object and hazard modifiers are provided, "
                "modifiers will be applied after changing the hazard."
            )


@dataclass(repr=False)
class ExposuresModifierConfig(_ModifierConfig):
    """Configuration for impact function modifiers."""

    reassign_impf_id: Optional[Dict[str, Dict[int | str, int | str]]] = None
    set_to_zero: Optional[list[int]] = None
    new_exposures_path: Optional[str] = None
    """HDF5 filepath for new exposure"""

    def __post_init__(self):
        if self.new_exposures_path is not None and any(
            [self.reassign_impf_id, self.set_to_zero]
        ):
            LOGGER.warning(
                "Both new exposures object and exposures modifiers are provided, "
                "modifiers will be applied after changing the exposures."
            )


@dataclass(repr=False)
class CostIncomeConfig(_ModifierConfig):
    """Serializable configuration for CostIncome."""

    mkt_price_year: Optional[int] = field(default_factory=lambda: datetime.today().year)
    init_cost: float = 0.0
    periodic_cost: float = 0.0
    periodic_income: float = 0.0
    cost_yearly_growth_rate: float = 0.0
    income_yearly_growth_rate: float = 0.0
    freq: str = "Y"
    custom_cash_flows: Optional[list[dict]] = None

    def to_cost_income(self) -> CostIncome:
        df = None
        if self.custom_cash_flows is not None:
            df = pd.DataFrame(self.custom_cash_flows)
            df["date"] = pd.to_datetime(df["date"])
        return CostIncome(
            mkt_price_year=self.mkt_price_year,
            init_cost=self.init_cost,
            periodic_cost=self.periodic_cost,
            periodic_income=self.periodic_income,
            cost_yearly_growth_rate=self.cost_yearly_growth_rate,
            income_yearly_growth_rate=self.income_yearly_growth_rate,
            custom_cash_flows=df,
            freq=self.freq,
        )

    @classmethod
    def from_cost_income(cls, ci: CostIncome) -> "CostIncomeConfig":
        """Round-trip from a live CostIncome object."""
        custom = None
        if ci.custom_cash_flows is not None:
            custom = (
                ci.custom_cash_flows.reset_index()
                .rename(columns={"index": "date"})
                .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        return cls(
            mkt_price_year=ci.mkt_price_year.year,  # datetime → int
            init_cost=abs(ci.init_cost),  # stored negative → positive
            periodic_cost=abs(ci.periodic_cost),
            periodic_income=ci.periodic_income,
            cost_yearly_growth_rate=ci.cost_growth_rate,
            income_yearly_growth_rate=ci.income_growth_rate,
            freq=ci.freq,
            custom_cash_flows=custom,
        )


@dataclass(repr=False)
class MeasureConfig(_ModifierConfig):
    name: str
    haz_type: str
    impfset_modifier: ImpfsetModifierConfig
    hazard_modifier: HazardModifierConfig
    exposures_modifier: ExposuresModifierConfig
    cost_income: CostIncomeConfig
    implementation_duration: Optional[str] = None
    color_rgb: Optional[Tuple[float, float, float]] = None

    def __repr__(self) -> str:
        fields_str = "\n\t".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}(\n\t{fields_str})"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "haz_type": self.haz_type,
            **self.impfset_modifier.to_dict(),
            **self.hazard_modifier.to_dict(),
            **self.exposures_modifier.to_dict(),
            **self.cost_income.to_dict(),
            "implementation_duration": self.implementation_duration,
            "color_rgb": list(self.color_rgb) if self.color_rgb is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MeasureConfig":
        color = d.get("color_rgb")
        return cls(
            name=d["name"],
            haz_type=d["haz_type"],
            impfset_modifier=ImpfsetModifierConfig.from_dict(d),
            hazard_modifier=HazardModifierConfig.from_dict(d),
            exposures_modifier=ExposuresModifierConfig.from_dict(d),
            cost_income=CostIncomeConfig.from_dict(d),
            implementation_duration=d.get("implementation_duration"),
            color_rgb=(
                tuple(color) if color is not None and not pd.isna(color) else None
            ),
        )

    def to_yaml(self, path: str) -> None:
        import yaml

        with open(path, "w") as f:
            yaml.dump(
                {"measures": [self.to_dict()]},
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "MeasureConfig":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f)["measures"][0])

    @classmethod
    def from_row(
        cls, row: pd.Series, haz_type: Optional[str] = None
    ) -> "MeasureConfig":
        """Build a MeasureConfig from a legacy Excel row."""
        row_dict = row.to_dict()
        return cls.from_dict(row_dict)


def _serialize_modifier_dict(d: dict) -> dict:
    """Stringify keys, convert tuples to lists for JSON."""
    return {str(k): list(v) for k, v in d.items()}


def _deserialize_modifier_dict(d: dict) -> dict:
    """Restore int keys where possible, values back to tuples."""
    return {
        (int(k) if isinstance(k, str) and k.isdigit() else k): tuple(v)
        for k, v in d.items()
    }
