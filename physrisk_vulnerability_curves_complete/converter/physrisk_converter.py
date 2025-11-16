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

Convert CLIMADA impact functions to physrisk vulnerability curves format.

This module provides functionality to export CLIMADA impact functions
in the format required by the OS-Climate physrisk library
(https://github.com/os-climate/physrisk).
"""

__all__ = ["ImpactFuncToPhysrisk"]

import logging
from typing import Dict, List, Optional, Union
import json

import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet

LOGGER = logging.getLogger(__name__)

# Hazard type mapping from CLIMADA to physrisk event types
# Source: physrisk documentation and CLIMADA hazard type conventions
HAZARD_TYPE_MAPPING = {
    "TC": "TropicalCyclone",
    "WS": "Windstorm",
    "FL": "RiverineInundation",
    "RF": "RiverineInundation",
    "CF": "CoastalInundation",
    "DR": "Drought",
    "EQ": "Earthquake",
    "WF": "Wildfire",
    "HS": "Hail",
    "LS": "Landslide",
}


class ImpactFuncToPhysrisk:
    """Converter for CLIMADA impact functions to physrisk vulnerability curves.

    This class converts CLIMADA ImpactFunc objects to the vulnerability curve
    format used by the OS-Climate physrisk library. The physrisk format uses
    Pydantic models with specific fields for asset type, location, event type,
    and impact distributions.

    physrisk VulnerabilityCurve schema (Pydantic BaseModel):
        - asset_type: str (e.g., "Buildings/Residential")
        - location: str (e.g., "North America", "Global")
        - event_type: str (e.g., "TropicalCyclone", "RiverineInundation")
        - impact_type: str ("Damage" or "Disruption")
        - intensity: List[float]
        - intensity_units: str
        - impact_mean: List[float] (mean damage/disruption, 0-1 scale)
        - impact_std: List[float] (standard deviation)

    CLIMADA ImpactFunc attributes:
        - haz_type: str (e.g., "TC", "WS")
        - id: int or str
        - name: str
        - intensity: np.array
        - mdd: np.array (Mean Damage Degree, 0-1)
        - paa: np.array (Percentage of Affected Assets, 0-1)
        - intensity_unit: str

    Key conversions:
        - impact_mean = mdd * paa (Mean Damage Ratio in CLIMADA)
        - impact_std = zeros (CLIMADA has no uncertainty quantification)
        - event_type = mapped from haz_type using HAZARD_TYPE_MAPPING
    """

    def __init__(
        self,
        default_location: str = "Global",
        default_asset_type: Optional[str] = None,
        default_impact_type: str = "Damage",
    ):
        """Initialize the converter.

        Parameters
        ----------
        default_location : str, optional
            Default location to use if not specified per impact function.
            Default: "Global"
        default_asset_type : str, optional
            Default asset type to use if not specified. If None, will use
            the ImpactFunc.name or ImpactFunc.id.
            Default: None
        default_impact_type : str, optional
            Default impact type: "Damage" or "Disruption".
            Default: "Damage"
        """
        self.default_location = default_location
        self.default_asset_type = default_asset_type
        self.default_impact_type = default_impact_type

    def convert_impact_func(
        self,
        impact_func: ImpactFunc,
        asset_type: Optional[str] = None,
        location: Optional[str] = None,
        impact_type: Optional[str] = None,
    ) -> Dict:
        """Convert a single CLIMADA ImpactFunc to physrisk VulnerabilityCurve format.

        Parameters
        ----------
        impact_func : ImpactFunc
            CLIMADA impact function to convert
        asset_type : str, optional
            Asset type for the vulnerability curve. If None, uses default or
            ImpactFunc.name/id
        location : str, optional
            Location for the vulnerability curve. If None, uses default_location
        impact_type : str, optional
            Impact type: "Damage" or "Disruption". If None, uses default_impact_type

        Returns
        -------
        dict
            Vulnerability curve in physrisk format (compatible with Pydantic model)

        Raises
        ------
        ValueError
            If impact function has empty intensity array or unmapped hazard type

        Examples
        --------
        >>> from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
        >>> impf = ImpfTropCyclone.from_emanuel_usa()
        >>> converter = ImpactFuncToPhysrisk()
        >>> vuln_curve = converter.convert_impact_func(
        ...     impf,
        ...     asset_type="Buildings/Residential",
        ...     location="North America"
        ... )
        >>> vuln_curve['event_type']
        'TropicalCyclone'
        """
        # Validate impact function
        if len(impact_func.intensity) == 0:
            raise ValueError(
                f"ImpactFunc {impact_func.id} has empty intensity array. "
                "Cannot convert to vulnerability curve."
            )

        # Map hazard type to event type
        event_type = HAZARD_TYPE_MAPPING.get(impact_func.haz_type)
        if event_type is None:
            if impact_func.haz_type:
                LOGGER.warning(
                    "Unmapped hazard type '%s' for ImpactFunc %s. "
                    "Using hazard type as-is for event_type.",
                    impact_func.haz_type,
                    impact_func.id,
                )
                event_type = impact_func.haz_type
            else:
                raise ValueError(
                    f"ImpactFunc {impact_func.id} has no haz_type. "
                    "Cannot determine event_type for physrisk."
                )

        # Determine asset type
        if asset_type is None:
            asset_type = self.default_asset_type
        if asset_type is None:
            # Use name or id as fallback
            asset_type = impact_func.name if impact_func.name else str(impact_func.id)

        # Determine location
        if location is None:
            location = self.default_location

        # Determine impact type
        if impact_type is None:
            impact_type = self.default_impact_type

        # Calculate impact_mean as MDR (Mean Damage Ratio = MDD * PAA)
        # This is the standard CLIMADA calculation
        impact_mean = (impact_func.mdd * impact_func.paa).tolist()

        # CLIMADA does not have uncertainty quantification,
        # so impact_std is set to zeros
        impact_std = np.zeros_like(impact_func.intensity).tolist()

        # Construct physrisk VulnerabilityCurve dict
        vulnerability_curve = {
            "asset_type": asset_type,
            "location": location,
            "event_type": event_type,
            "impact_type": impact_type,
            "intensity": impact_func.intensity.tolist(),
            "intensity_units": impact_func.intensity_unit,
            "impact_mean": impact_mean,
            "impact_std": impact_std,
        }

        return vulnerability_curve

    def convert_impact_func_set(
        self,
        impact_func_set: ImpactFuncSet,
        asset_type_mapping: Optional[Dict[Union[int, str], str]] = None,
        location_mapping: Optional[Dict[Union[int, str], str]] = None,
    ) -> List[Dict]:
        """Convert a CLIMADA ImpactFuncSet to list of physrisk VulnerabilityCurves.

        Parameters
        ----------
        impact_func_set : ImpactFuncSet
            CLIMADA impact function set to convert
        asset_type_mapping : dict, optional
            Mapping from ImpactFunc.id to asset_type string.
            Example: {1: "Buildings/Residential", 2: "Buildings/Commercial"}
        location_mapping : dict, optional
            Mapping from ImpactFunc.id to location string.
            Example: {1: "North America", 2: "Europe"}

        Returns
        -------
        list of dict
            List of vulnerability curves in physrisk format

        Examples
        --------
        >>> from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
        >>> impf_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()
        >>> converter = ImpactFuncToPhysrisk(default_asset_type="Buildings/Residential")
        >>> vuln_curves = converter.convert_impact_func_set(impf_set)
        >>> len(vuln_curves)  # Number of regional impact functions
        11
        """
        vulnerability_curves = []

        for haz_type, funcs in impact_func_set.get_func().items():
            for func_id, impact_func in funcs.items():
                # Get asset type from mapping or use default
                asset_type = None
                if asset_type_mapping:
                    asset_type = asset_type_mapping.get(func_id)

                # Get location from mapping or use default
                location = None
                if location_mapping:
                    location = location_mapping.get(func_id)

                try:
                    vuln_curve = self.convert_impact_func(
                        impact_func,
                        asset_type=asset_type,
                        location=location,
                    )
                    vulnerability_curves.append(vuln_curve)
                except ValueError as e:
                    LOGGER.warning(
                        "Skipping ImpactFunc %s (haz_type=%s): %s",
                        func_id,
                        haz_type,
                        str(e),
                    )

        return vulnerability_curves

    def to_json(
        self,
        impact_func_or_set: Union[ImpactFunc, ImpactFuncSet],
        file_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Convert CLIMADA impact function(s) to JSON format.

        Parameters
        ----------
        impact_func_or_set : ImpactFunc or ImpactFuncSet
            CLIMADA impact function(s) to convert
        file_path : str, optional
            If provided, write JSON to this file path
        **kwargs
            Additional keyword arguments passed to convert_impact_func or
            convert_impact_func_set

        Returns
        -------
        str
            JSON string representation

        Examples
        --------
        >>> from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
        >>> impf = ImpfTropCyclone.from_emanuel_usa()
        >>> converter = ImpactFuncToPhysrisk()
        >>> json_str = converter.to_json(
        ...     impf,
        ...     asset_type="Buildings/Residential",
        ...     location="North America"
        ... )
        """
        if isinstance(impact_func_or_set, ImpactFunc):
            data = self.convert_impact_func(impact_func_or_set, **kwargs)
        elif isinstance(impact_func_or_set, ImpactFuncSet):
            data = {
                "items": self.convert_impact_func_set(impact_func_or_set, **kwargs)
            }
        else:
            raise TypeError(
                f"Expected ImpactFunc or ImpactFuncSet, got {type(impact_func_or_set)}"
            )

        json_str = json.dumps(data, indent=2)

        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_str)
            LOGGER.info("Wrote vulnerability curve(s) to %s", file_path)

        return json_str
