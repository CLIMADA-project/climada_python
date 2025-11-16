# Exporting CLIMADA Impact Functions to physrisk Vulnerability Curves

## Overview

This guide explains how to export CLIMADA impact functions to the vulnerability curve format used by the [OS-Climate physrisk library](https://github.com/os-climate/physrisk).

The physrisk library is a climate physical risk calculation engine that assesses the impact of climate hazards on physical assets. It uses vulnerability curves to model the relationship between hazard intensity and asset damage/disruption.

## Background

### CLIMADA Impact Functions

CLIMADA represents impact functions with three key components:

- **Intensity**: Hazard intensity values (e.g., wind speed in m/s, flood depth in m)
- **MDD** (Mean Damage Degree): Average damage at each intensity level (0-1 scale)
- **PAA** (Percentage of Affected Assets): Fraction of assets exposed at each intensity level (0-1 scale)

The **Mean Damage Ratio (MDR)** is calculated as: `MDR = MDD × PAA`

### physrisk Vulnerability Curves

physrisk uses a Pydantic-based vulnerability curve model with these attributes:

- **intensity**: List of hazard intensity values
- **impact_mean**: List of mean impact values (damage or disruption, 0-1 scale)
- **impact_std**: List of standard deviations for uncertainty quantification
- **intensity_units**: Unit of intensity measurement
- **event_type**: Hazard event type (e.g., "TropicalCyclone", "RiverineInundation")
- **asset_type**: Type of asset (e.g., "Buildings/Residential")
- **location**: Geographic location (e.g., "North America", "Global")
- **impact_type**: Either "Damage" or "Disruption"

## Conversion Mapping

| CLIMADA | physrisk | Notes |
|---------|----------|-------|
| `intensity` | `intensity` | Direct copy, converted to list |
| `MDD × PAA` | `impact_mean` | CLIMADA's MDR calculation |
| N/A | `impact_std` | Set to zeros (CLIMADA has no uncertainty) |
| `intensity_unit` | `intensity_units` | Direct copy |
| `haz_type` | `event_type` | Mapped via `HAZARD_TYPE_MAPPING` |
| `id` / `name` | `asset_type` | User-provided or derived from name/id |
| N/A | `location` | User-provided |
| N/A | `impact_type` | Default: "Damage" |

### Hazard Type Mapping

| CLIMADA `haz_type` | physrisk `event_type` |
|-------------------|----------------------|
| TC | TropicalCyclone |
| WS | Windstorm |
| FL | RiverineInundation |
| RF | RiverineInundation |
| CF | CoastalInundation |
| DR | Drought |
| EQ | Earthquake |
| WF | Wildfire |
| HS | Hail |
| LS | Landslide |

## Usage

### Basic Example: Single Impact Function

```python
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk

# Create CLIMADA impact function
impf = ImpfTropCyclone.from_emanuel_usa()

# Create converter
converter = ImpactFuncToPhysrisk()

# Convert to physrisk format
vuln_curve = converter.convert_impact_func(
    impf,
    asset_type="Buildings/Residential",
    location="United States"
)

# Export to JSON
json_output = converter.to_json(
    impf,
    asset_type="Buildings/Residential",
    location="United States",
    file_path="vulnerability_curve.json"
)
```

### Converting Multiple Impact Functions

```python
from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk

# Load calibrated regional TC impact functions
impf_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()

# Define location mapping for regional functions
location_mapping = {
    1: "North America",
    2: "Central America & Caribbean",
    3: "South America",
    4: "Europe",
    5: "Africa",
    6: "Middle East",
    7: "South Asia",
    8: "East Asia",
    9: "Southeast Asia & Oceania",
    10: "Australia",
    11: "Rest of World",
}

# Convert all functions
converter = ImpactFuncToPhysrisk(
    default_asset_type="Buildings/Residential"
)

vuln_curves = converter.convert_impact_func_set(
    impf_set,
    location_mapping=location_mapping
)

# Export to JSON file
json_output = converter.to_json(
    impf_set,
    location_mapping=location_mapping,
    file_path="regional_tc_curves.json"
)
```

### Creating Custom Impact Functions

```python
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk

# Create custom sigmoid impact function for flood
impf = ImpactFunc.from_sigmoid_impf(
    intensity=(0, 10, 0.5),  # 0 to 10m, 0.5m steps
    L=1.0,                    # Max impact = 100%
    k=2.0,                    # Slope
    x0=3.0,                   # 50% impact at 3m depth
    haz_type="FL",
    impf_id=1,
    intensity_unit="m"
)

# Convert and export
converter = ImpactFuncToPhysrisk()
json_output = converter.to_json(
    impf,
    asset_type="Buildings/Residential",
    location="Global",
    file_path="flood_curve.json"
)
```

### Disruption Curves (Not Damage)

```python
import numpy as np
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk

# Create impact function for infrastructure disruption
impf = ImpactFunc(
    haz_type="FL",
    id=10,
    name="Road Disruption",
    intensity=np.array([0, 0.1, 0.3, 0.5, 1.0, 2.0]),
    mdd=np.array([0, 0.2, 0.5, 0.8, 1.0, 1.0]),
    paa=np.array([0, 0.5, 0.8, 0.9, 1.0, 1.0]),
    intensity_unit="m"
)

# Convert with impact_type="Disruption"
converter = ImpactFuncToPhysrisk()
vuln_curve = converter.convert_impact_func(
    impf,
    asset_type="Infrastructure/Roads",
    location="Global",
    impact_type="Disruption"  # Not "Damage"
)
```

## API Reference

### ImpactFuncToPhysrisk Class

```python
ImpactFuncToPhysrisk(
    default_location="Global",
    default_asset_type=None,
    default_impact_type="Damage"
)
```

**Parameters:**
- `default_location`: Default location if not specified per function
- `default_asset_type`: Default asset type if not specified (uses function name/id if None)
- `default_impact_type`: Default impact type ("Damage" or "Disruption")

#### Methods

##### convert_impact_func()

Convert a single CLIMADA ImpactFunc to physrisk format.

```python
converter.convert_impact_func(
    impact_func,
    asset_type=None,
    location=None,
    impact_type=None
) -> dict
```

**Returns:** Dictionary compatible with physrisk VulnerabilityCurve Pydantic model

##### convert_impact_func_set()

Convert a CLIMADA ImpactFuncSet to list of physrisk curves.

```python
converter.convert_impact_func_set(
    impact_func_set,
    asset_type_mapping=None,
    location_mapping=None
) -> list
```

**Parameters:**
- `asset_type_mapping`: Dict mapping function ID to asset type
- `location_mapping`: Dict mapping function ID to location

**Returns:** List of vulnerability curve dictionaries

##### to_json()

Export to JSON format.

```python
converter.to_json(
    impact_func_or_set,
    file_path=None,
    **kwargs
) -> str
```

**Parameters:**
- `impact_func_or_set`: ImpactFunc or ImpactFuncSet to convert
- `file_path`: Optional path to write JSON file
- `**kwargs`: Additional arguments for conversion methods

**Returns:** JSON string

## physrisk Compatibility Notes

### Uncertainty Quantification

CLIMADA impact functions do not include uncertainty quantification. The converter sets `impact_std` to zeros. If you need uncertainty in physrisk, you must add this separately or use physrisk's built-in uncertainty models.

### Event Type Naming

physrisk uses CamelCase event types (e.g., "TropicalCyclone", "RiverineInundation"). The converter automatically maps CLIMADA's abbreviated hazard types to physrisk's event types.

### Asset Type Conventions

physrisk commonly uses hierarchical asset types like:
- `Buildings/Residential`
- `Buildings/Commercial`
- `Infrastructure/Roads`
- `Infrastructure/PowerGeneration`

It's recommended to follow this convention for compatibility.

### JSON Schema

The exported JSON follows this schema for single curves:

```json
{
  "asset_type": "Buildings/Residential",
  "location": "North America",
  "event_type": "TropicalCyclone",
  "impact_type": "Damage",
  "intensity": [0, 25.7, 34.9, ...],
  "intensity_units": "m/s",
  "impact_mean": [0, 0.02, 0.08, ...],
  "impact_std": [0, 0, 0, ...]
}
```

For multiple curves (from ImpactFuncSet):

```json
{
  "items": [
    {
      "asset_type": "...",
      "location": "...",
      ...
    },
    ...
  ]
}
```

## Examples

See `script/applications/physrisk_export_example.py` for comprehensive examples including:

1. Simple step function conversion
2. Emanuel USA tropical cyclone function
3. Calibrated regional TC functions with location mapping
4. European windstorm functions
5. Custom sigmoid flood functions
6. File export
7. Disruption curves for infrastructure

## References

### CLIMADA Impact Functions

- Emanuel, K. (2011). Global Warming Effects on U.S. Hurricane Damage. *Weather, Climate, and Society*, 3(4), 261-268. https://doi.org/10.1175/WCAS-D-11-00007.1
- Eberenz, S., Stocker, D., Röösli, T., & Bresch, D. N. (2021). Asset exposure data for global physical risk assessment. *Earth System Science Data*, 13(2), 817-833. https://doi.org/10.5194/essd-13-817-2021
- Schwierz, C., et al. (2010). Modelling European winter wind storm losses in current and future climate. *Climatic Change*, 101, 485-514.

### physrisk

- OS-Climate physrisk: https://github.com/os-climate/physrisk
- physrisk documentation: https://physrisk.readthedocs.io/

## License

This conversion tool is part of CLIMADA and is released under the GNU General Public License v3.0.
