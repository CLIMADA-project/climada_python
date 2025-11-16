# Creating Custom Heat and Fire (FWI) Impact Functions for physrisk

## Overview

CLIMADA does **not** have pre-calibrated impact functions for Heat or Fire Weather Index (FWI) hazards. However, you **can create custom impact functions** using CLIMADA's generic builders and export them to physrisk format.

**This guide explains how to create empirically-grounded impact functions for these hazards.**

---

## Status of Heat and Fire in CLIMADA and physrisk

### Heat

| System | Status | Details |
|--------|--------|---------|
| **CLIMADA** | ❌ No pre-calibrated functions | No heat hazard module exists |
| **physrisk** | ✅ Has models | `ChronicHeatGZNModel` for labor productivity using degree days >32°C |

### Fire (FWI)

| System | Status | Details |
|--------|--------|---------|
| **CLIMADA** | ❌ No pre-calibrated functions | "WF" code mentioned but not implemented |
| **physrisk** | ❌ No models | No wildfire vulnerability models found |

**Conclusion**: You must create custom impact functions with your own calibration data.

---

## Heat Impact Functions

### Option 1: Chronic Heat (Degree Days)

**Intensity metric**: Cumulative annual degree days above threshold temperature (e.g., 32°C)

**Impact**: Fractional productivity loss or damage (0-1 scale)

**Example calibration**:
```python
from climada.entity.impact_funcs.base import ImpactFunc

impf_heat = ImpactFunc.from_sigmoid_impf(
    intensity=(0, 5000, 50),     # 0-5000 degree days, 50 DD steps
    L=0.5,                       # Max impact = 50% productivity loss
    k=0.002,                     # Steepness parameter
    x0=1500,                     # 25% impact at 1500 degree days
    haz_type="HT",               # Heat (custom code)
    intensity_unit="degree_days_above_32C"
)
```

**Calibration sources to consider**:
- **Labor productivity**:
  - Burke, M., Hsiang, S. M., & Miguel, E. (2015). Global non-linear effect of temperature on economic production. *Nature*, 527, 235-239. https://doi.org/10.1038/nature15725
  - Neidell, M., et al. (2021). Temperature and work performance in Indian manufacturing. *Journal of Development Economics*, 149, 102588.

- **Agriculture**:
  - Schlenker, W., & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. *PNAS*, 106(37), 15594-15598.

- **Mortality**:
  - Gasparrini, A., et al. (2015). Mortality risk attributable to high and low ambient temperature: a multicountry observational study. *The Lancet*, 386(9991), 369-375.

### Option 2: Acute Heatwave (Maximum Temperature)

**Intensity metric**: Maximum daily temperature (°C)

**Impact**: Infrastructure failure, heat-related mortality

**Example step function** (infrastructure damage threshold):
```python
impf_heatwave = ImpactFunc.from_step_impf(
    intensity=(30, 42, 50),      # Threshold at 42°C
    haz_type="HW",               # Heatwave (custom code)
    mdd=(0, 0.8),                # 80% damage severity above threshold
    paa=(0.3, 1.0),              # 30% → 100% assets affected
    intensity_unit="degC"
)
```

**Calibration thresholds**:
- **Rail buckling**: ~40-45°C (Australian standards)
- **Power transformer failure**: ~40°C ambient (load-dependent)
- **Road pavement damage**: ~50-60°C surface temperature
- **Human mortality**: Location-specific (e.g., 35°C in temperate climates)

---

## Fire Weather Index (FWI) Impact Functions

### Background: Fire Weather Index

The **Canadian Forest Fire Weather Index (FWI)** combines:
- Temperature
- Relative humidity
- Wind speed
- Precipitation

**FWI Scale**:
- 0-5: Low fire danger
- 5-10: Moderate
- 10-20: High
- 20-30: Very high
- 30+: Extreme

### Option 1: Sigmoid Function (Gradual Damage Increase)

**Suitable for**: Areas with firefighting capacity, building codes, defensible space

```python
impf_fire = ImpactFunc.from_sigmoid_impf(
    intensity=(0, 60, 1),        # FWI 0-60, 1-unit steps
    L=1.0,                       # Max impact = 100% loss
    k=0.15,                      # Steepness (calibrate!)
    x0=25,                       # 50% damage at FWI=25 (calibrate!)
    haz_type="WF",               # Wildfire
    intensity_unit="FWI"
)
```

### Option 2: Step Function (Threshold-Based)

**Suitable for**: High-risk areas, wooden structures, no firefighting

```python
impf_fire_step = ImpactFunc.from_step_impf(
    intensity=(0, 20, 60),       # FWI threshold at 20
    haz_type="WF",
    mdd=(0, 1.0),                # Total loss above threshold
    paa=(0.1, 1.0),              # 10% ember damage, 100% flame damage
    intensity_unit="FWI"
)
```

### Option 3: Custom Calibrated (Literature Data)

**Suitable for**: When you have empirical damage data

```python
import numpy as np

# Example: Replace with actual calibration data
fwi_values = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60])
mdd_values = np.array([0.0, 0.0, 0.05, 0.15, 0.35, 0.55, 0.75, 0.9, 0.95, 1.0])
paa_values = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.98, 0.99, 1.0])

impf_fire = ImpactFunc(
    haz_type="WF",
    id=1,
    intensity=fwi_values,
    mdd=mdd_values,
    paa=paa_values,
    intensity_unit="FWI"
)
```

### Calibration Sources for Fire

**Building damage**:
- Penman, T. D., et al. (2013). Modeling the determinants of destruction in the 2009 Australian bushfires. *International Journal of Wildland Fire*, 22(8), 1085-1097. https://doi.org/10.1071/WF12187
- Blanchi, R., et al. (2014). Environmental circumstances surrounding bushfire fatalities in Australia 1901-2011. *Environmental Science & Policy*, 37, 192-203.

**Forest/vegetation loss**:
- Parks, S. A., et al. (2018). Warmer and drier fire seasons contribute to increases in area burned at high severity. *Environmental Research Letters*, 13(6), 064006.

**Insurance data**:
- Industry loss curves (proprietary - check with local insurers)
- Government post-fire damage assessments

---

## Export to physrisk

After creating impact functions, export using the converter:

```python
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk

converter = ImpactFuncToPhysrisk()

# Export heat function
heat_curve = converter.convert_impact_func(
    impf_heat,
    asset_type="IndustrialActivity/Labor",
    location="Global",
    impact_type="Disruption"  # Productivity loss
)

# Save to JSON for physrisk
converter.to_json(
    impf_heat,
    asset_type="IndustrialActivity/Labor",
    location="Global",
    impact_type="Disruption",
    file_path="heat_vulnerability.json"
)

# Export fire function
fire_curve = converter.convert_impact_func(
    impf_fire,
    asset_type="Buildings/Residential",
    location="Australia",
    impact_type="Damage"
)

converter.to_json(
    impf_fire,
    asset_type="Buildings/Residential",
    location="Australia",
    impact_type="Damage",
    file_path="fire_fwi_vulnerability.json"
)
```

---

## physrisk Integration Notes

### Heat

**physrisk already has heat models** (`ChronicHeatGZNModel`) using:
- Intensity: `mean_degree_days/above/32c`
- Impact: Labor productivity loss (4.671 hours per degree day)
- Asset type: `IndustrialActivity`

**Options**:
1. Use physrisk's existing model (recommended if studying labor)
2. Create CLIMADA-based curves for other asset types (buildings, infrastructure, agriculture)

### Fire (FWI)

**physrisk has NO wildfire models**, so you MUST provide custom curves.

**Recommended asset types for physrisk**:
- `Buildings/Residential`
- `Buildings/Commercial`
- `Infrastructure/PowerGeneration`
- `Agriculture/Forestry`

**Location granularity**:
- Regional (e.g., "Australia", "California", "Mediterranean")
- Country-level
- Global (if calibration is universal)

---

## Critical Calibration Requirements

### ⚠️ WARNING

**The examples provided are PLACEHOLDERS with hypothetical parameters.**

**You MUST calibrate with real data from**:
1. **Peer-reviewed literature** (primary source)
2. **Insurance claims databases** (if accessible)
3. **Government damage assessments** (post-disaster reports)
4. **Field surveys** (empirical observations)

### Validation Requirements

After calibration, validate your impact functions by:

1. **Back-testing**: Compare modeled damage to historical events
2. **Expert review**: Have domain experts review parameter choices
3. **Sensitivity analysis**: Test parameter uncertainty
4. **Cross-validation**: Compare to independent datasets

### Documentation Requirements

For each impact function, document:
- **Calibration source** (paper, dataset, expert judgment)
- **Geographic applicability** (where is this valid?)
- **Asset type specificity** (building type, construction era)
- **Uncertainty bounds** (parameter confidence intervals)
- **Validation results** (back-testing metrics)

---

## Complete Example Script

See: `script/applications/create_heat_fire_impact_functions.py`

This script demonstrates:
1. Creating chronic heat impact function (degree days)
2. Creating acute heatwave impact function (temperature threshold)
3. Creating fire FWI sigmoid function
4. Creating fire FWI step function
5. Creating fire FWI custom calibrated function
6. Exporting all to physrisk JSON format

**Run with**:
```bash
python script/applications/create_heat_fire_impact_functions.py
```

---

## Summary: Is This Possible?

**YES**, you can create Heat and Fire FWI impact functions for physrisk using CLIMADA's methodology:

✅ **Heat (Chronic)**: Use sigmoid or custom functions with degree days
✅ **Heat (Acute)**: Use step functions for temperature thresholds
✅ **Fire (FWI)**: Use sigmoid, step, or custom functions with FWI values

**CRITICAL REQUIREMENT**: You must provide calibration parameters from empirical sources. CLIMADA provides the mathematical framework, not the calibration data for these hazards.

---

## References

### Heat
- Burke, M., Hsiang, S. M., & Miguel, E. (2015). *Nature*, 527, 235-239.
- Gasparrini, A., et al. (2015). *The Lancet*, 386(9991), 369-375.
- Schlenker, W., & Roberts, M. J. (2009). *PNAS*, 106(37), 15594-15598.

### Fire
- Penman, T. D., et al. (2013). *Int. J. Wildland Fire*, 22(8), 1085-1097.
- Blanchi, R., et al. (2014). *Environ. Sci. Policy*, 37, 192-203.
- Parks, S. A., et al. (2018). *Environ. Res. Lett.*, 13(6), 064006.

### physrisk
- OS-Climate physrisk: https://github.com/os-climate/physrisk
- Chronic heat model source: `src/physrisk/vulnerability_models/chronic_heat_models.py`
