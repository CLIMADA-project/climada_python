"""
Example script demonstrating export of CLIMADA impact functions to physrisk format.

This script shows how to convert CLIMADA impact functions to the vulnerability
curve format used by the OS-Climate physrisk library.

For more information about physrisk, see:
https://github.com/os-climate/physrisk
https://physrisk.readthedocs.io/

Author: CLIMADA Contributors
"""

import numpy as np
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone, ImpfSetTropCyclone
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
from climada.entity.impact_funcs.physrisk_converter import ImpactFuncToPhysrisk


def example_1_simple_conversion():
    """Example 1: Convert a single impact function to physrisk format."""
    print("=" * 80)
    print("Example 1: Simple Conversion of a Single Impact Function")
    print("=" * 80)

    # Create a simple step function for tropical cyclone
    impf = ImpactFunc.from_step_impf(
        intensity=(0, 50, 100),
        haz_type="TC",
        impf_id=1,
        name="Residential Buildings",
        intensity_unit="m/s",
    )

    print("\nCLIMADA Impact Function:")
    print(f"  Hazard Type: {impf.haz_type}")
    print(f"  ID: {impf.id}")
    print(f"  Name: {impf.name}")
    print(f"  Intensity: {impf.intensity}")
    print(f"  MDD: {impf.mdd}")
    print(f"  PAA: {impf.paa}")

    # Create converter
    converter = ImpactFuncToPhysrisk()

    # Convert to physrisk format
    vuln_curve = converter.convert_impact_func(
        impf,
        asset_type="Buildings/Residential",
        location="North America",
    )

    print("\nphysrisk Vulnerability Curve:")
    for key, value in vuln_curve.items():
        print(f"  {key}: {value}")

    # Export to JSON
    json_output = converter.to_json(
        impf,
        asset_type="Buildings/Residential",
        location="North America",
    )

    print("\nJSON Output (first 500 characters):")
    print(json_output[:500] + "...")


def example_2_emanuel_usa_tc():
    """Example 2: Convert Emanuel USA tropical cyclone impact function."""
    print("\n" + "=" * 80)
    print("Example 2: Emanuel USA Tropical Cyclone Impact Function")
    print("=" * 80)

    # Create Emanuel USA impact function
    impf = ImpfTropCyclone.from_emanuel_usa()

    print("\nEmanuel USA Impact Function:")
    print(f"  Hazard Type: {impf.haz_type}")
    print(f"  ID: {impf.id}")
    print(f"  Intensity Unit: {impf.intensity_unit}")
    print(f"  Number of intensity points: {len(impf.intensity)}")

    # Convert to physrisk format
    converter = ImpactFuncToPhysrisk()
    vuln_curve = converter.convert_impact_func(
        impf,
        asset_type="Buildings/Residential",
        location="United States",
    )

    print("\nphysrisk Vulnerability Curve:")
    print(f"  Event Type: {vuln_curve['event_type']}")
    print(f"  Asset Type: {vuln_curve['asset_type']}")
    print(f"  Location: {vuln_curve['location']}")
    print(f"  Intensity Points: {len(vuln_curve['intensity'])}")
    print(f"  Max Impact Mean: {max(vuln_curve['impact_mean']):.3f}")


def example_3_regional_tc_functions():
    """Example 3: Convert calibrated regional TC impact functions to physrisk."""
    print("\n" + "=" * 80)
    print("Example 3: Calibrated Regional Tropical Cyclone Impact Functions")
    print("=" * 80)

    # Load calibrated regional impact functions
    # Based on Eberenz et al. (2021)
    impf_set = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()

    print(f"\nLoaded {len(impf_set.get_func()['TC'])} regional impact functions")

    # Define location mapping for regional functions
    # IDs correspond to regions defined in trop_cyclone.py
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
        11: "Rest of World",  # ROW
    }

    # Convert all regional functions
    converter = ImpactFuncToPhysrisk(
        default_asset_type="Buildings/Residential",
        default_impact_type="Damage",
    )

    vuln_curves = converter.convert_impact_func_set(
        impf_set,
        location_mapping=location_mapping,
    )

    print(f"\nConverted {len(vuln_curves)} vulnerability curves")

    # Show sample of converted curves
    print("\nSample of converted curves:")
    for i, curve in enumerate(vuln_curves[:3]):
        print(f"\nCurve {i + 1}:")
        print(f"  Location: {curve['location']}")
        print(f"  Event Type: {curve['event_type']}")
        print(f"  Asset Type: {curve['asset_type']}")
        print(f"  Max Impact: {max(curve['impact_mean']):.3f}")

    # Export all curves to JSON
    json_output = converter.to_json(
        impf_set,
        location_mapping=location_mapping,
    )

    print(f"\nTotal JSON length: {len(json_output)} characters")


def example_4_storm_europe():
    """Example 4: Convert European windstorm impact function."""
    print("\n" + "=" * 80)
    print("Example 4: European Windstorm Impact Function")
    print("=" * 80)

    # Create Schwierz windstorm impact function
    impf = ImpfStormEurope.from_schwierz()

    print("\nSchwierz Windstorm Impact Function:")
    print(f"  Hazard Type: {impf.haz_type}")
    print(f"  ID: {impf.id}")
    print(f"  Intensity Unit: {impf.intensity_unit}")

    # Convert to physrisk format
    converter = ImpactFuncToPhysrisk()
    vuln_curve = converter.convert_impact_func(
        impf,
        asset_type="Buildings/Residential",
        location="Europe",
    )

    print("\nphysrisk Vulnerability Curve:")
    print(f"  Event Type: {vuln_curve['event_type']}")
    print(f"  Asset Type: {vuln_curve['asset_type']}")
    print(f"  Location: {vuln_curve['location']}")


def example_5_custom_impact_function():
    """Example 5: Create and convert a custom impact function."""
    print("\n" + "=" * 80)
    print("Example 5: Custom Sigmoid Impact Function")
    print("=" * 80)

    # Create a custom sigmoid impact function for flood
    impf = ImpactFunc.from_sigmoid_impf(
        intensity=(0, 10, 0.5),  # 0 to 10 meters, 0.5m steps
        L=1.0,  # Maximum impact = 100%
        k=2.0,  # Slope parameter
        x0=3.0,  # 50% impact at 3 meters depth
        haz_type="FL",
        impf_id=1,
        name="Flood Damage - Residential",
        intensity_unit="m",
    )

    print("\nCustom Flood Impact Function:")
    print(f"  Hazard Type: {impf.haz_type}")
    print(f"  Intensity Range: {impf.intensity[0]} - {impf.intensity[-1]} m")
    print(f"  50% damage at: ~3.0 m")

    # Convert to physrisk format
    converter = ImpactFuncToPhysrisk()
    vuln_curve = converter.convert_impact_func(
        impf,
        asset_type="Buildings/Residential",
        location="Global",
    )

    print("\nphysrisk Vulnerability Curve:")
    print(f"  Event Type: {vuln_curve['event_type']}")
    print(f"  Intensity Unit: {vuln_curve['intensity_units']}")

    # Show impact at key depths
    print("\nImpact at key flood depths:")
    for i in [0, 6, 12, 18]:  # Indices for ~0m, 3m, 6m, 9m
        if i < len(vuln_curve['intensity']):
            intensity = vuln_curve['intensity'][i]
            impact = vuln_curve['impact_mean'][i]
            print(f"  {intensity:.1f}m: {impact:.1%}")


def example_6_export_to_file():
    """Example 6: Export vulnerability curves to JSON file."""
    print("\n" + "=" * 80)
    print("Example 6: Export to JSON File")
    print("=" * 80)

    # Create a set of impact functions
    impf_set = ImpactFuncSet()

    # Add multiple hazard types
    impf_tc = ImpfTropCyclone.from_emanuel_usa()
    impf_ws = ImpfStormEurope.from_schwierz()

    impf_set.append(impf_tc)
    impf_set.append(impf_ws)

    print(f"\nCreated impact function set with {len(impf_set.get_func())} hazard types")

    # Define mappings
    asset_type_mapping = {
        1: "Buildings/Residential",  # TC function
        1: "Buildings/Residential",  # WS function
    }

    location_mapping = {
        1: "United States",  # TC function
        1: "Europe",  # WS function (note: same ID, different haz_type)
    }

    # Convert and export
    converter = ImpactFuncToPhysrisk()

    output_file = "/tmp/climada_physrisk_vulnerability_curves.json"

    json_output = converter.to_json(
        impf_set,
        asset_type_mapping=asset_type_mapping,
        location_mapping=location_mapping,
        file_path=output_file,
    )

    print(f"\nExported vulnerability curves to: {output_file}")
    print(f"File size: {len(json_output)} bytes")

    # Show file content structure
    import json

    data = json.loads(json_output)
    print(f"\nNumber of curves in file: {len(data['items'])}")


def example_7_disruption_curve():
    """Example 7: Create a disruption curve (not damage)."""
    print("\n" + "=" * 80)
    print("Example 7: Disruption Curve for Infrastructure")
    print("=" * 80)

    # Create an impact function representing service disruption
    # For example, road disruption due to flooding
    impf = ImpactFunc(
        haz_type="FL",
        id=10,
        name="Road Disruption",
        intensity=np.array([0, 0.1, 0.3, 0.5, 1.0, 2.0]),
        mdd=np.array([0, 0.2, 0.5, 0.8, 1.0, 1.0]),  # Disruption severity
        paa=np.array([0, 0.5, 0.8, 0.9, 1.0, 1.0]),  # Percentage affected
        intensity_unit="m",
    )

    print("\nRoad Disruption Impact Function:")
    print(f"  Hazard Type: {impf.haz_type}")
    print(f"  Intensity: {impf.intensity}")
    print(f"  MDD (disruption severity): {impf.mdd}")
    print(f"  PAA (% affected): {impf.paa}")

    # Convert to physrisk format as Disruption
    converter = ImpactFuncToPhysrisk()
    vuln_curve = converter.convert_impact_func(
        impf,
        asset_type="Infrastructure/Roads",
        location="Global",
        impact_type="Disruption",  # Not "Damage"
    )

    print("\nphysrisk Vulnerability Curve:")
    print(f"  Impact Type: {vuln_curve['impact_type']}")
    print(f"  Event Type: {vuln_curve['event_type']}")
    print(f"  Asset Type: {vuln_curve['asset_type']}")


if __name__ == "__main__":
    print("\n")
    print("*" * 80)
    print("CLIMADA to physrisk Vulnerability Curve Conversion Examples")
    print("*" * 80)

    # Run all examples
    example_1_simple_conversion()
    example_2_emanuel_usa_tc()
    example_3_regional_tc_functions()
    example_4_storm_europe()
    example_5_custom_impact_function()
    example_6_export_to_file()
    example_7_disruption_curve()

    print("\n" + "*" * 80)
    print("All examples completed!")
    print("*" * 80 + "\n")
