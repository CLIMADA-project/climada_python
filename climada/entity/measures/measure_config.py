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
import warnings
from abc import ABC
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class _ModifierConfig(ABC):
    """
    Abstract base class for all modifier configuration dataclasses.

    Provides shared serialization, deserialization, and representation
    logic for all concrete modifier config subclasses. Not intended to
    be instantiated directly.
    """

    def _filter_out_default_fields(self):
        """
        Partition the instance's fields into non-default and default groups.

        The ``haz_type`` field is always excluded from the output, as it
        is managed at the ``MeasureConfig`` level.

        Returns
        -------
        non_defaults : dict
            Fields whose current value differs from the dataclass default.
        defaults : dict
            Fields whose current value equals the dataclass default.
        """

        non_defaults = {}
        defaults = {}
        for defined_field in fields(self):
            val = getattr(self, defined_field.name)
            default = defined_field.default
            if defined_field.default_factory is not field().default_factory:
                default = defined_field.default_factory()

            if val != default:
                non_defaults[defined_field.name] = val
            else:
                defaults[defined_field.name] = val

        if "haz_type" in non_defaults:
            non_defaults.pop("haz_type")
        return non_defaults, defaults

    def to_dict(self):
        """
        Serialize the config to a flat dictionary, omitting default values.

        The ``haz_type`` field is always excluded from the output, as it
        is managed at the ``MeasureConfig`` level.

        Returns
        -------
        dict
            Dictionary containing only fields whose values differ from
            their dataclass defaults.
        """
        non_default, _ = self._filter_out_default_fields()
        return non_default

    @classmethod
    def from_dict(cls, kwargs_dict: dict):
        """
        Instantiate a config from a dictionary, ignoring unknown keys.

        Parameters
        ----------
        kwargs_dict : dict
            Input dictionary. Keys not matching any dataclass field are
            silently discarded.

        Returns
        -------
        _ModifierConfig
            A new instance of the calling subclass.
        """

        filtered = cls._filter_dict_to_fields(kwargs_dict)
        return cls(**filtered)

    @classmethod
    def _filter_dict_to_fields(cls, to_filter: dict):
        """
        Filter a dictionary to only the keys matching the dataclass fields.

        Parameters
        ----------
        to_filter : dict
            Input dictionary, potentially containing extra keys.

        Returns
        -------
        dict
            A copy of ``to_filter`` restricted to keys that correspond to declared
            dataclass fields on this class.
        """

        filtered = dict(
            filter(lambda k: k[0] in [f.name for f in fields(cls)], to_filter.items())
        )
        return filtered

    def __repr__(self) -> str:
        """
        Return a human-readable representation highlighting non-default fields.

        Non-default fields are shown prominently; default fields are shown
        below them. This makes it easy to see at a glance what has been
        configured on an instance.

        Returns
        -------
        str
            A formatted string representation of the instance.
        """

        non_defaults, defaults = self._filter_out_default_fields()
        ndf_fields_str = (
            "\n\t\t\t".join(f"{k}={v!r}" for k, v in non_defaults.items())
            if non_defaults
            else None
        )
        _ = (
            "\n\t\t\t".join(f"{k}={v!r}" for k, v in defaults.items())
            if defaults
            else None
        )
        ndf_fields = (
            "(" "\n\t\tNon default fields:" f"\n\t\t\t{ndf_fields_str}" "\n)"
            if ndf_fields_str
            else "()"
        )
        return f"{self.__class__.__name__}{ndf_fields}"


@dataclass(repr=False)
class ImpfsetModifierConfig(_ModifierConfig):
    """
    Configuration for modifications to an impact function set.

    Supports scaling or shifting MDD, PAA, and intensity curves, as well
    as replacement of the impact function set, loaded from a file path. If
    both a new file path and modifier values are provided, modifiers are
    applied after the replacement (and a warning is issued).

    Parameters
    ----------
    haz_type : str
        Hazard type identifier (e.g. ``"TC"``) that this modifier targets.
    impf_ids : int or str or list of int or str, optional
        Impact function ID(s) to which modifications are applied.
        If ``None``, all impact functions are affected.
    impf_mdd_mult : float, optional
        Multiplicative factor applied to the mean damage degree (MDD) curve.
        Default is ``1.0`` (no change).
    impf_mdd_add : float, optional
        Additive offset applied to the MDD curve after multiplication.
        Default is ``0.0``.
    impf_paa_mult : float, optional
        Multiplicative factor applied to the percentage of affected assets
        (PAA) curve. Default is ``1.0``.
    impf_paa_add : float, optional
        Additive offset applied to the PAA curve after multiplication.
        Default is ``0.0``.
    impf_int_mult : float, optional
        Multiplicative factor applied to the intensity axis.
        Default is ``1.0``.
    impf_int_add : float, optional
        Additive offset applied to the intensity axis after multiplication.
        Default is ``0.0``.
    new_impfset_path : str, optional
        Path to an Excel file containing a replacement impact function set.
        If provided alongside modifier values, a warning is issued and
        modifiers are applied after loading the new set.

    Warns
    -----
    UserWarning
        If ``new_impfset_path`` is set alongside any non-default modifier
        values.
    """

    haz_type: str
    impf_ids: Optional[Union[int, str, list[Union[int, str]]]] = None
    impf_mdd_mult: float = 1.0
    impf_mdd_add: float = 0.0
    impf_paa_mult: float = 1.0
    impf_paa_add: float = 0.0
    impf_int_mult: float = 1.0
    impf_int_add: float = 0.0
    new_impfset_path: Optional[str] = None

    def __post_init__(self):
        config = self.to_dict()
        if "new_impfset_path" in config and any(
            key in config
            for key in [
                "impf_mdd_add",
                "impf_mdd_mult",
                "impf_paa_add",
                "impf_paa_mult",
                "impf_int_add",
                "impf_int_mult",
            ]
        ):
            warnings.warn(
                "Both new impfset object and impfset modifiers are provided, "
                "modifiers will be applied after changing the impfset."
            )


@dataclass(repr=False)
class HazardModifierConfig(_ModifierConfig):
    """
    Configuration for modifications to a hazard.

    Supports scaling or shifting hazard intensity, applying a return-period
    frequency cutoff, and replacement of the hazard, loaded from a file path.
    If both a new file path and modifier values are provided, modifiers are
    applied after the replacement.

    Parameters
    ----------
    haz_type : str
        Hazard type identifier (e.g. ``"TC"``) that this modifier targets.
    haz_int_mult : float, optional
        Multiplicative factor applied to hazard intensity.
        Default is ``1.0`` (no change).
    haz_int_add : float, optional
        Additive offset applied to hazard intensity after multiplication.
        Default is ``0.0``.
    new_hazard_path : str, optional
        Path to an HDF5 file containing a replacement hazard.
        If provided alongside modifier values, a warning is issued and
        modifiers are applied after loading the new hazard.
    impact_rp_cutoff : float, optional
        Return period (in years) below which hazard events are discarded.
        If ``None``, no cutoff is applied.

    Warns
    -----
    UserWarning
        If ``new_hazard_path`` is set alongside any non-default modifier
        values or a non-``None`` ``impact_rp_cutoff``.
    """

    haz_type: str
    haz_int_mult: Optional[float] = 1.0
    haz_int_add: Optional[float] = 0.0
    haz_freq_mult: Optional[float] = 1.0
    haz_freq_add: Optional[float] = 0.0
    new_hazard_path: Optional[str] = None
    impact_rp_cutoff: Optional[float] = None

    def __post_init__(self):
        config = self.to_dict()
        if "new_hazard_path" in config and any(
            key in config
            for key in [
                "haz_int_mult",
                "haz_int_add",
                "haz_freq_mult",
                "haz_freq_add",
                "impact_rp_cutoff",
            ]
        ):
            warnings.warn(
                "Both new hazard object and hazard modifiers are provided, "
                "modifiers will be applied after changing the hazard."
            )


@dataclass(repr=False)
class ExposuresModifierConfig(_ModifierConfig):
    """
    Configuration for modifications to an exposures object.

    Supports remapping impact function IDs, zeroing out selected regions,
    and replacement of the exposures from a new file. If both a new
    file path and modifier values are provided, modifiers are applied after
    the replacement.

    Parameters
    ----------
    reassign_impf_id : dict of {str: dict of {int or str: int or str}}, optional
        Nested mapping ``{haz_type: {old_id: new_id}}`` used to reassign
        impact function IDs in the exposures. If ``None``, no remapping
        is performed.
    set_to_zero : list of int, optional
        Region IDs for which exposure values are set to zero.
        If ``None``, no zeroing is applied.
    new_exposures_path : str, optional
        Path to an HDF5 file containing replacement exposures.
        If provided alongside modifier values, a warning is issued and
        modifiers are applied after loading the new exposures.

    Warns
    -----
    UserWarning
        If ``new_exposures_path`` is set alongside any non-``None``
        modifier values.
    """

    reassign_impf_id: Optional[Dict[str, Dict[int | str, int | str]]] = None
    set_to_zero: Optional[list[int]] = None
    new_exposures_path: Optional[str] = None

    def __post_init__(self):
        config = self.to_dict()
        if "new_exposures_path" in config and any(
            key in config for key in ["reassign_impf_id", "set_to_zero"]
        ):
            warnings.warn(
                "Both new exposures object and exposures modifiers are provided, "
                "modifiers will be applied after changing the exposures."
            )


@dataclass(repr=False)
class CostIncomeConfig(_ModifierConfig):
    """
    Serializable configuration for a ``CostIncome`` object.

    Encodes all parameters required to construct a ``CostIncome`` instance,
    including optional custom cash flow schedules.

    Parameters
    ----------
    mkt_price_year : int, optional
        Reference year for market prices. Defaults to the current year.
    init_cost : float, optional
        One-time initial investment cost (positive value). Default is ``0.0``.
    periodic_cost : float, optional
        Recurring cost per period (positive value). Default is ``0.0``.
    periodic_income : float, optional
        Recurring income per period. Default is ``0.0``.
    cost_yearly_growth_rate : float, optional
        Annual growth rate applied to periodic costs. Default is ``0.0``.
    income_yearly_growth_rate : float, optional
        Annual growth rate applied to periodic income. Default is ``0.0``.
    freq : str, optional
        Pandas offset alias defining the period length (e.g. ``"Y"`` for
        yearly, ``"M"`` for monthly). Default is ``"Y"``.
    custom_cash_flows : list of dict, optional
        Explicit cash flow schedule as a list of records with at minimum
        a ``"date"`` key (ISO 8601 string) and a value key. If provided,
        overrides the periodic cost/income logic.
    """

    mkt_price_year: Optional[int] = field(default_factory=lambda: datetime.today().year)
    init_cost: float = 0.0
    periodic_cost: float = 0.0
    periodic_income: float = 0.0
    cost_yearly_growth_rate: float = 0.0
    income_yearly_growth_rate: float = 0.0
    freq: str = "Y"
    custom_cash_flows: Optional[list[dict]] = None

    @classmethod
    def from_cost_income(cls, cost_income: "CostIncome") -> "CostIncomeConfig":
        """
        Construct a :class:`CostIncomeConfig` from a live
        :class:`CostIncome` object.

        Parameters
        ----------
        cost_income : CostIncome
            The live ``CostIncome`` instance to serialise.

        Returns
        -------
        CostIncomeConfig
            The config instance equivalent to the ``CostIncome``.
        """

        custom = None
        if cost_income.custom_cash_flows is not None:
            custom = (
                cost_income.custom_cash_flows.reset_index()
                .rename(columns={"index": "date"})
                .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
                .to_dict(orient="records")
            )
        return cls(
            mkt_price_year=cost_income.mkt_price_year.year,  # datetime → int
            init_cost=abs(cost_income.init_cost),  # stored negative → positive
            periodic_cost=abs(cost_income.periodic_cost),
            periodic_income=cost_income.periodic_income,
            cost_yearly_growth_rate=cost_income.cost_growth_rate,
            income_yearly_growth_rate=cost_income.income_growth_rate,
            freq=cost_income.freq,
            custom_cash_flows=custom,
        )


@dataclass(repr=False)
class MeasureConfig(_ModifierConfig):
    """
    Top-level serializable configuration for a single adaptation measure.

    Aggregates all modifier sub-configs (hazard, impact functions, exposures,
    cost/income) into a single object that can be round-tripped through dict,
    YAML, or a legacy Excel row.

    This class is the primary entry point for defining measures in a
    declarative, file-based workflow and serves as the serialization
    counterpart to :class:`~climada.entity.measures.base.Measure`.

    Parameters
    ----------
    name : str
        Unique name identifying this measure.
    haz_type : str
        Hazard type identifier (e.g. ``"TC"``) this measure is designed for.
    impfset_modifier : ImpfsetModifierConfig
        Configuration describing modifications to the impact function set.
    hazard_modifier : HazardModifierConfig
        Configuration describing modifications to the hazard.
    exposures_modifier : ExposuresModifierConfig
        Configuration describing modifications to the exposures.
    cost_income : CostIncomeConfig
        Financial parameters associated with implementing this measure.
    implementation_duration : str, optional
        Pandas offset alias (e.g. ``"2Y"``) representing the time before
        the measure is fully operational. If ``None``, the measure takes
        effect immediately.
    color_rgb : tuple of float, optional
        RGB colour triple in the range ``[0, 1]`` used for visualisation.
        If ``None``, defaults to black ``(0, 0, 0)``.
    """

    name: str
    haz_type: str
    impfset_modifier: ImpfsetModifierConfig
    hazard_modifier: HazardModifierConfig
    exposures_modifier: ExposuresModifierConfig
    cost_income: CostIncomeConfig
    implementation_duration: Optional[str] = None
    color_rgb: Optional[Tuple[float, float, float]] = None

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the measure configuration.

        All fields are shown, including sub-configs, with each on its own
        indented line.

        Returns
        -------
        str
            A formatted multi-line string representation.
        """

        fields_str = "\n\t".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}(\n\t{fields_str})"

    def to_dict(self) -> dict:
        """
        Serialize the measure configuration to a flat dictionary.

        Sub-config dictionaries are merged into the top-level dict (i.e.
        their keys are inlined, not nested). ``haz_type`` is always included
        at the top level. Fields with ``None`` values are preserved.

        Returns
        -------
        dict
            Flat dictionary representation suitable for YAML or Excel
            serialization.
        """

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
    def from_dict(cls, kwargs_dict: dict) -> "MeasureConfig":
        """
        Instantiate a :class:`MeasureConfig` from a flat dictionary.

        Delegates sub-config construction to the respective
        ``from_dict`` classmethods. Unknown keys are silently discarded
        by each sub-config parser.

        Parameters
        ----------
        kwargs_dict : dict
            Flat dictionary, as produced by :meth:`to_dict` or read from
            a legacy Excel row. Must contain at minimum ``"name"`` and
            ``"haz_type"``.

        Returns
        -------
        MeasureConfig
            A fully populated configuration instance.
        """

        return cls(
            name=kwargs_dict["name"],
            haz_type=kwargs_dict["haz_type"],
            impfset_modifier=ImpfsetModifierConfig.from_dict(kwargs_dict),
            hazard_modifier=HazardModifierConfig.from_dict(kwargs_dict),
            exposures_modifier=ExposuresModifierConfig.from_dict(kwargs_dict),
            cost_income=CostIncomeConfig.from_dict(kwargs_dict),
            implementation_duration=kwargs_dict.get("implementation_duration"),
            color_rgb=cls._normalize_color(kwargs_dict.get("color_rgb")),
        )

    @staticmethod
    def _normalize_color(color_rgb):
        # 1. Handle None and NaN (np.nan, pd.NA, float('nan'))
        if color_rgb is None or pd.isna(color_rgb) is True:
            return None

        # 2. Convert sequence types (list, np.array, tuple) to a standard tuple
        try:
            # Flatten in case it's a nested numpy array, then convert to tuple
            result = tuple(np.array(color_rgb).flatten().tolist())

            # 3. Enforce the length of three
            if len(result) != 3:
                raise ValueError(f"Expected 3 digits, got {len(result)}")

            return result

        except (TypeError, ValueError) as err:
            # Handle cases where input isn't iterable or wrong length
            raise ValueError(f"Invalid color format: {color_rgb}.") from err

    def to_yaml(self, path: str) -> None:
        """
        Write this configuration to a YAML file.

        The file is structured as ``{"measures": [<this config as dict>]}``,
        matching the expected format for :meth:`from_yaml`.

        Parameters
        ----------
        path : str
            Destination file path. Will be created or overwritten.
        """

        import yaml

        with open(path, "w") as opened_file:
            yaml.dump(
                {"measures": [self.to_dict()]},
                opened_file,
                default_flow_style=False,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "MeasureConfig":
        """
        Load a :class:`MeasureConfig` from a YAML file.

        Expects the file to contain a top-level ``"measures"`` list; reads
        only the first entry.

        Parameters
        ----------
        path : str
            Path to the YAML file to read.

        Returns
        -------
        MeasureConfig
            The configuration parsed from the first entry in
            ``measures``.
        """

        import yaml

        with open(path) as opened_file:
            return cls.from_dict(yaml.safe_load(opened_file)["measures"][0])

    @classmethod
    def from_row(cls, row: pd.Series) -> "MeasureConfig":
        """
        Construct a :class:`MeasureConfig` from a legacy Excel row.

        Converts the row to a dictionary and delegates to :meth:`from_dict`.
        This is the primary migration path for measures currently stored in
        the legacy Excel-based ``MeasureSet`` format.

        Parameters
        ----------
        row : pd.Series
            A single row from a legacy measures Excel sheet, with column
            names matching the flat dictionary keys expected by
            :meth:`from_dict`.

        Returns
        -------
        MeasureConfig
            A configuration instance populated from the row data.
        """

        row_dict = row.to_dict()
        return cls.from_dict(row_dict)
