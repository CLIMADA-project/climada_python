"""
Example: Creating custom Heat and Fire (FWI) impact functions for physrisk export.

This script demonstrates how to create impact functions for hazards that don't
have pre-calibrated functions in CLIMADA, then export them to physrisk format.

Author: CLIMADA Contributors
"""

import numpy as np
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk


def create_heat_impact_function_chronic():
    """
    Create impact function for chronic heat based on degree days above threshold.

    This example uses a sigmoid curve to model productivity/damage loss as a
    function of cumulative degree days above a threshold temperature.

    Calibration parameters shown are EXAMPLES and should be replaced with
    peer-reviewed values specific to your asset type and region.

    Returns
    -------
    ImpactFunc
        Heat impact function

    Notes
    -----
    Intensity metric: Cumulative degree days above 32°C (annual)
    Impact: Fractional productivity loss or damage (0-1)

    Example calibration sources to consider:
    - Labor productivity: Neidell et al. (2021), Burke et al. (2015)
    - Agriculture: Schlenker & Roberts (2009)
    - Infrastructure: Location-specific studies
    """

    # Example parameters - REPLACE WITH CALIBRATED VALUES
    # Based on physrisk ChronicHeatGZNModel concept:
    # ~4.67 hours lost per degree day above 32°C
    # Against ~107,460 annual hours = 0.0000435 fractional loss per DD

    impf_heat = ImpactFunc.from_sigmoid_impf(
        intensity=(0, 5000, 50),  # 0 to 5000 degree days, 50 DD steps
        L=0.5,                     # Max impact = 50% loss (calibrate this!)
        k=0.002,                   # Steepness (calibrate this!)
        x0=1500,                   # 25% impact at 1500 DD (calibrate this!)
        haz_type="HT",             # Heat (not a standard CLIMADA code)
        impf_id=1,
        name="Chronic Heat - Labor Productivity",
        intensity_unit="degree_days_above_32C"
    )

    return impf_heat


def create_heat_impact_function_heatwave():
    """
    Create impact function for acute heatwave events based on maximum temperature.

    This models short-term extreme heat impacts (e.g., infrastructure failure,
    health impacts, agricultural shock).

    Returns
    -------
    ImpactFunc
        Heatwave impact function

    Notes
    -----
    Intensity metric: Maximum daily temperature (°C)
    Impact: Fractional damage or mortality (0-1)

    Example calibration sources:
    - Human health: Gasparrini et al. (2015)
    - Infrastructure: Rail buckling thresholds, power generation limits
    - Agriculture: Crop heat stress thresholds
    """

    # Example step function for infrastructure damage
    # (e.g., rail buckling, power transformer failure)

    impf_heatwave = ImpactFunc.from_step_impf(
        intensity=(30, 42, 50),    # 30-50°C, threshold at 42°C
        haz_type="HW",             # Heatwave (not a standard CLIMADA code)
        mdd=(0, 0.8),              # 0% below, 80% above threshold
        paa=(0.3, 1.0),            # 30% affected below, 100% above
        impf_id=2,
        name="Heatwave - Infrastructure Failure",
        intensity_unit="degC"
    )

    return impf_heatwave


def create_fire_fwi_impact_function():
    """
    Create impact function for wildfire based on Fire Weather Index (FWI).

    FWI is a numeric rating of fire intensity combining temperature, humidity,
    wind speed, and precipitation (Canadian Forest Fire Weather Index System).

    FWI Scale:
    - 0-5: Low fire danger
    - 5-10: Moderate
    - 10-20: High
    - 20-30: Very high
    - 30+: Extreme

    Returns
    -------
    ImpactFunc
        Fire FWI impact function

    Notes
    -----
    Intensity metric: Fire Weather Index (dimensionless, 0-100+)
    Impact: Fractional building/forest damage (0-1)

    Calibration sources to consider:
    - Building damage: Penman et al. (2013), Blanchi et al. (2014)
    - Forest loss: Parks et al. (2018)
    - Insurance data: Industry-specific loss curves

    WARNING: This is an EXAMPLE. Real FWI-damage relationships depend heavily on:
    - Building construction (flame zone vs ember attack)
    - Vegetation type and fuel load
    - Firefighting capacity
    - Building codes and defensible space
    """

    # Example 1: Sigmoid function for building damage
    impf_fire_buildings = ImpactFunc.from_sigmoid_impf(
        intensity=(0, 60, 1),      # FWI from 0 to 60, 1-unit steps
        L=1.0,                     # Max impact = 100% loss
        k=0.15,                    # Steepness (calibrate!)
        x0=25,                     # 50% damage at FWI=25 (calibrate!)
        haz_type="WF",             # Wildfire
        impf_id=1,
        name="Wildfire FWI - Buildings",
        intensity_unit="FWI"
    )

    return impf_fire_buildings


def create_fire_fwi_impact_function_threshold():
    """
    Create step-function impact for fire (e.g., total loss above threshold).

    This models scenarios where assets have binary outcomes:
    - Below threshold: Asset survives
    - Above threshold: Asset destroyed

    Example: Wooden structures in flame zone

    Returns
    -------
    ImpactFunc
        Fire FWI step impact function
    """

    impf_fire_step = ImpactFunc.from_step_impf(
        intensity=(0, 20, 60),     # FWI threshold at 20
        haz_type="WF",             # Wildfire
        mdd=(0, 1.0),              # 0% below, 100% above
        paa=(0.1, 1.0),            # 10% affected below (ember attack), 100% above
        impf_id=2,
        name="Wildfire FWI - High-Risk Buildings (Step)",
        intensity_unit="FWI"
    )

    return impf_fire_step


def create_fire_fwi_impact_function_custom():
    """
    Create fully custom FWI impact function with manual calibration.

    This allows specifying exact intensity-damage pairs from calibration data
    or literature.

    Returns
    -------
    ImpactFunc
        Custom calibrated fire impact function
    """

    # Example: Hypothetical calibration data from literature or insurance claims
    # REPLACE WITH ACTUAL CALIBRATED VALUES

    fwi_values = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60])

    # Mean Damage Degree (MDD) - average damage conditional on being affected
    # Example values - MUST BE CALIBRATED
    mdd_values = np.array([0.0, 0.0, 0.05, 0.15, 0.35, 0.55, 0.75, 0.9, 0.95, 1.0])

    # Percentage of Affected Assets (PAA) - fraction of exposed assets damaged
    # Example values - MUST BE CALIBRATED
    paa_values = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 0.98, 0.99, 1.0])

    impf_fire_custom = ImpactFunc(
        haz_type="WF",
        id=3,
        name="Wildfire FWI - Custom Calibrated",
        intensity=fwi_values,
        mdd=mdd_values,
        paa=paa_values,
        intensity_unit="FWI"
    )

    return impf_fire_custom


def export_all_to_physrisk():
    """
    Create all impact functions and export to physrisk format.
    """
    print("=" * 80)
    print("Creating Heat and Fire Impact Functions for physrisk Export")
    print("=" * 80)

    # Create impact functions
    print("\n1. Creating chronic heat impact function...")
    impf_heat_chronic = create_heat_impact_function_chronic()

    print("2. Creating heatwave impact function...")
    impf_heatwave = create_heat_impact_function_heatwave()

    print("3. Creating fire FWI impact function (sigmoid)...")
    impf_fire_sigmoid = create_fire_fwi_impact_function()

    print("4. Creating fire FWI impact function (step)...")
    impf_fire_step = create_fire_fwi_impact_function_threshold()

    print("5. Creating fire FWI impact function (custom calibrated)...")
    impf_fire_custom = create_fire_fwi_impact_function_custom()

    # Initialize converter
    converter = ImpactFuncToPhysrisk()

    # Export chronic heat
    print("\n" + "=" * 80)
    print("CHRONIC HEAT (Degree Days)")
    print("=" * 80)

    heat_chronic_curve = converter.convert_impact_func(
        impf_heat_chronic,
        asset_type="IndustrialActivity/Labor",
        location="Global",
        impact_type="Disruption"  # Productivity loss
    )

    print(f"Asset Type: {heat_chronic_curve['asset_type']}")
    print(f"Intensity Range: {heat_chronic_curve['intensity'][0]} - "
          f"{heat_chronic_curve['intensity'][-1]} {heat_chronic_curve['intensity_units']}")
    print(f"Max Impact: {max(heat_chronic_curve['impact_mean']):.1%}")

    # Export to JSON
    json_output = converter.to_json(
        impf_heat_chronic,
        asset_type="IndustrialActivity/Labor",
        location="Global",
        impact_type="Disruption",
        file_path="/tmp/heat_chronic_vulnerability.json"
    )
    print(f"Exported to: /tmp/heat_chronic_vulnerability.json")

    # Export heatwave
    print("\n" + "=" * 80)
    print("HEATWAVE (Acute)")
    print("=" * 80)

    heatwave_curve = converter.convert_impact_func(
        impf_heatwave,
        asset_type="Infrastructure/Rail",
        location="Global",
        impact_type="Damage"
    )

    print(f"Asset Type: {heatwave_curve['asset_type']}")
    print(f"Intensity Range: {heatwave_curve['intensity'][0]} - "
          f"{heatwave_curve['intensity'][-1]} {heatwave_curve['intensity_units']}")
    print(f"Max Impact: {max(heatwave_curve['impact_mean']):.1%}")

    converter.to_json(
        impf_heatwave,
        asset_type="Infrastructure/Rail",
        location="Global",
        impact_type="Damage",
        file_path="/tmp/heatwave_vulnerability.json"
    )
    print(f"Exported to: /tmp/heatwave_vulnerability.json")

    # Export fire FWI (sigmoid)
    print("\n" + "=" * 80)
    print("FIRE FWI (Sigmoid)")
    print("=" * 80)

    fire_sigmoid_curve = converter.convert_impact_func(
        impf_fire_sigmoid,
        asset_type="Buildings/Residential",
        location="Australia",  # High wildfire risk region
        impact_type="Damage"
    )

    print(f"Asset Type: {fire_sigmoid_curve['asset_type']}")
    print(f"Intensity Range: {fire_sigmoid_curve['intensity'][0]} - "
          f"{fire_sigmoid_curve['intensity'][-1]} {fire_sigmoid_curve['intensity_units']}")
    print(f"Max Impact: {max(fire_sigmoid_curve['impact_mean']):.1%}")

    converter.to_json(
        impf_fire_sigmoid,
        asset_type="Buildings/Residential",
        location="Australia",
        impact_type="Damage",
        file_path="/tmp/fire_fwi_sigmoid_vulnerability.json"
    )
    print(f"Exported to: /tmp/fire_fwi_sigmoid_vulnerability.json")

    # Export fire FWI (step)
    print("\n" + "=" * 80)
    print("FIRE FWI (Step Function)")
    print("=" * 80)

    fire_step_curve = converter.convert_impact_func(
        impf_fire_step,
        asset_type="Buildings/Residential",
        location="California",
        impact_type="Damage"
    )

    print(f"Asset Type: {fire_step_curve['asset_type']}")
    print(f"Threshold: FWI = 20")
    print(f"Impact below threshold: {fire_step_curve['impact_mean'][0]:.1%}")
    print(f"Impact above threshold: {fire_step_curve['impact_mean'][-1]:.1%}")

    converter.to_json(
        impf_fire_step,
        asset_type="Buildings/Residential",
        location="California",
        impact_type="Damage",
        file_path="/tmp/fire_fwi_step_vulnerability.json"
    )
    print(f"Exported to: /tmp/fire_fwi_step_vulnerability.json")

    # Export fire FWI (custom)
    print("\n" + "=" * 80)
    print("FIRE FWI (Custom Calibrated)")
    print("=" * 80)

    fire_custom_curve = converter.convert_impact_func(
        impf_fire_custom,
        asset_type="Buildings/Residential",
        location="Mediterranean",
        impact_type="Damage"
    )

    print(f"Asset Type: {fire_custom_curve['asset_type']}")
    print(f"Calibration points: {len(fire_custom_curve['intensity'])}")
    print(f"Max Impact (MDR): {max(fire_custom_curve['impact_mean']):.1%}")

    converter.to_json(
        impf_fire_custom,
        asset_type="Buildings/Residential",
        location="Mediterranean",
        impact_type="Damage",
        file_path="/tmp/fire_fwi_custom_vulnerability.json"
    )
    print(f"Exported to: /tmp/fire_fwi_custom_vulnerability.json")

    print("\n" + "=" * 80)
    print("ALL VULNERABILITY CURVES CREATED AND EXPORTED")
    print("=" * 80)
    print("\nIMPORTANT: These are EXAMPLE calibrations!")
    print("You MUST replace parameters with calibrated values from:")
    print("  - Peer-reviewed literature")
    print("  - Insurance claims data")
    print("  - Government damage assessments")
    print("  - Field surveys")
    print("\nSuggested calibration sources:")
    print("  Heat: Burke et al. (2015), Neidell et al. (2021)")
    print("  Fire: Penman et al. (2013), Blanchi et al. (2014)")


if __name__ == "__main__":
    export_all_to_physrisk()
