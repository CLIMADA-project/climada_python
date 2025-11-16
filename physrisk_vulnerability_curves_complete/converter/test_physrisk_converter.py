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

Unit tests for physrisk_converter module.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.impact_funcs.physrisk_converter import (
    ImpactFuncToPhysrisk,
    HAZARD_TYPE_MAPPING,
)


class TestImpactFuncToPhysrisk(unittest.TestCase):
    """Test ImpactFuncToPhysrisk converter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ImpactFuncToPhysrisk()

    def test_convert_simple_impact_func(self):
        """Test conversion of a simple impact function."""
        # Create a simple step impact function
        impf = ImpactFunc.from_step_impf(
            intensity=(0, 50, 100),
            haz_type="TC",
            impf_id=1,
            name="Test TC Impact Function",
            intensity_unit="m/s",
        )

        vuln_curve = self.converter.convert_impact_func(
            impf, asset_type="Buildings/Residential", location="North America"
        )

        # Verify required fields
        self.assertEqual(vuln_curve["asset_type"], "Buildings/Residential")
        self.assertEqual(vuln_curve["location"], "North America")
        self.assertEqual(vuln_curve["event_type"], "TropicalCyclone")
        self.assertEqual(vuln_curve["impact_type"], "Damage")
        self.assertEqual(vuln_curve["intensity_units"], "m/s")

        # Verify intensity array
        self.assertEqual(len(vuln_curve["intensity"]), 4)  # step function has 4 points
        self.assertIsInstance(vuln_curve["intensity"], list)

        # Verify impact arrays
        self.assertEqual(len(vuln_curve["impact_mean"]), 4)
        self.assertEqual(len(vuln_curve["impact_std"]), 4)

        # Verify impact_std is all zeros (CLIMADA has no uncertainty)
        self.assertTrue(all(x == 0 for x in vuln_curve["impact_std"]))

        # Verify impact_mean is MDR (mdd * paa)
        expected_mdr = (impf.mdd * impf.paa).tolist()
        np.testing.assert_array_almost_equal(
            vuln_curve["impact_mean"], expected_mdr
        )

    def test_convert_sigmoid_impact_func(self):
        """Test conversion of a sigmoid impact function."""
        impf = ImpactFunc.from_sigmoid_impf(
            intensity=(0, 100, 5),
            L=1.0,
            k=0.1,
            x0=50.0,
            haz_type="RF",
            impf_id=2,
            intensity_unit="m",
        )

        vuln_curve = self.converter.convert_impact_func(
            impf, asset_type="Infrastructure/Roads", location="Global"
        )

        self.assertEqual(vuln_curve["event_type"], "RiverineInundation")
        self.assertEqual(vuln_curve["intensity_units"], "m")
        self.assertEqual(len(vuln_curve["intensity"]), 20)  # (0, 100, 5) = 20 points

        # Impact mean should be sigmoid-shaped
        impact_mean = vuln_curve["impact_mean"]
        # Should be monotonically increasing
        self.assertTrue(all(impact_mean[i] <= impact_mean[i + 1]
                            for i in range(len(impact_mean) - 1)))

    def test_hazard_type_mapping(self):
        """Test that all hazard types are correctly mapped."""
        test_cases = [
            ("TC", "TropicalCyclone"),
            ("WS", "Windstorm"),
            ("FL", "RiverineInundation"),
            ("RF", "RiverineInundation"),
            ("CF", "CoastalInundation"),
            ("DR", "Drought"),
            ("EQ", "Earthquake"),
            ("WF", "Wildfire"),
        ]

        for haz_type, expected_event_type in test_cases:
            with self.subTest(haz_type=haz_type):
                impf = ImpactFunc(
                    haz_type=haz_type,
                    id=1,
                    intensity=np.array([0, 50, 100]),
                    mdd=np.array([0, 0.5, 1.0]),
                    paa=np.array([1, 1, 1]),
                    intensity_unit="unit",
                )
                vuln_curve = self.converter.convert_impact_func(
                    impf, asset_type="Test"
                )
                self.assertEqual(vuln_curve["event_type"], expected_event_type)

    def test_unmapped_hazard_type_warning(self):
        """Test that unmapped hazard types generate warning but still convert."""
        impf = ImpactFunc(
            haz_type="XX",  # Unmapped type
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
            intensity_unit="unit",
        )

        with self.assertLogs(level="WARNING") as log:
            vuln_curve = self.converter.convert_impact_func(impf, asset_type="Test")
            self.assertEqual(vuln_curve["event_type"], "XX")
            self.assertTrue(any("Unmapped hazard type" in msg for msg in log.output))

    def test_empty_intensity_raises_error(self):
        """Test that empty intensity array raises ValueError."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([]),
            mdd=np.array([]),
            paa=np.array([]),
        )

        with self.assertRaises(ValueError) as cm:
            self.converter.convert_impact_func(impf, asset_type="Test")

        self.assertIn("empty intensity", str(cm.exception))

    def test_no_hazard_type_raises_error(self):
        """Test that missing haz_type raises ValueError."""
        impf = ImpactFunc(
            haz_type="",
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        with self.assertRaises(ValueError) as cm:
            self.converter.convert_impact_func(impf, asset_type="Test")

        self.assertIn("no haz_type", str(cm.exception))

    def test_default_asset_type_from_name(self):
        """Test that asset_type defaults to ImpactFunc.name."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            name="Residential Buildings",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        vuln_curve = self.converter.convert_impact_func(impf)
        self.assertEqual(vuln_curve["asset_type"], "Residential Buildings")

    def test_default_asset_type_from_id(self):
        """Test that asset_type defaults to ImpactFunc.id when name is empty."""
        impf = ImpactFunc(
            haz_type="TC",
            id=42,
            name="",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        vuln_curve = self.converter.convert_impact_func(impf)
        self.assertEqual(vuln_curve["asset_type"], "42")

    def test_default_location(self):
        """Test that default location is used."""
        converter = ImpactFuncToPhysrisk(default_location="Asia")
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        vuln_curve = converter.convert_impact_func(impf, asset_type="Test")
        self.assertEqual(vuln_curve["location"], "Asia")

    def test_custom_impact_type(self):
        """Test custom impact type (Disruption instead of Damage)."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        vuln_curve = self.converter.convert_impact_func(
            impf, asset_type="Test", impact_type="Disruption"
        )
        self.assertEqual(vuln_curve["impact_type"], "Disruption")

    def test_mdr_calculation(self):
        """Test that impact_mean is correctly calculated as MDD * PAA."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([0, 25, 50, 75, 100]),
            mdd=np.array([0, 0.2, 0.5, 0.8, 1.0]),
            paa=np.array([0.5, 0.6, 0.8, 0.9, 1.0]),
        )

        vuln_curve = self.converter.convert_impact_func(impf, asset_type="Test")

        expected_mdr = [0, 0.12, 0.4, 0.72, 1.0]
        np.testing.assert_array_almost_equal(
            vuln_curve["impact_mean"], expected_mdr
        )


class TestConvertImpactFuncSet(unittest.TestCase):
    """Test conversion of ImpactFuncSet."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ImpactFuncToPhysrisk()

    def test_convert_impact_func_set(self):
        """Test conversion of an ImpactFuncSet."""
        # Create a simple impact function set
        impf_set = ImpactFuncSet()

        impf1 = ImpactFunc(
            haz_type="TC",
            id=1,
            name="Residential",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
            intensity_unit="m/s",
        )

        impf2 = ImpactFunc(
            haz_type="TC",
            id=2,
            name="Commercial",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.6, 1.0]),
            paa=np.array([1, 1, 1]),
            intensity_unit="m/s",
        )

        impf_set.append(impf1)
        impf_set.append(impf2)

        vuln_curves = self.converter.convert_impact_func_set(impf_set)

        self.assertEqual(len(vuln_curves), 2)
        self.assertEqual(vuln_curves[0]["asset_type"], "Residential")
        self.assertEqual(vuln_curves[1]["asset_type"], "Commercial")

    def test_convert_with_asset_type_mapping(self):
        """Test conversion with custom asset type mapping."""
        impf_set = ImpactFuncSet()

        impf1 = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf2 = ImpactFunc(
            haz_type="WS",
            id=2,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.6, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf_set.append(impf1)
        impf_set.append(impf2)

        asset_type_mapping = {1: "Buildings/Residential", 2: "Buildings/Commercial"}

        vuln_curves = self.converter.convert_impact_func_set(
            impf_set, asset_type_mapping=asset_type_mapping
        )

        self.assertEqual(vuln_curves[0]["asset_type"], "Buildings/Residential")
        self.assertEqual(vuln_curves[1]["asset_type"], "Buildings/Commercial")

    def test_convert_with_location_mapping(self):
        """Test conversion with custom location mapping."""
        impf_set = ImpactFuncSet()

        impf1 = ImpactFunc(
            haz_type="TC",
            id=1,
            name="NA",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf2 = ImpactFunc(
            haz_type="TC",
            id=2,
            name="EU",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.6, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf_set.append(impf1)
        impf_set.append(impf2)

        location_mapping = {1: "North America", 2: "Europe"}

        vuln_curves = self.converter.convert_impact_func_set(
            impf_set, location_mapping=location_mapping
        )

        self.assertEqual(vuln_curves[0]["location"], "North America")
        self.assertEqual(vuln_curves[1]["location"], "Europe")

    def test_convert_skips_invalid_functions(self):
        """Test that invalid impact functions are skipped with warning."""
        impf_set = ImpactFuncSet()

        # Valid function
        impf1 = ImpactFunc(
            haz_type="TC",
            id=1,
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        # Invalid function (empty intensity)
        impf2 = ImpactFunc(
            haz_type="TC",
            id=2,
            intensity=np.array([]),
            mdd=np.array([]),
            paa=np.array([]),
        )

        impf_set.append(impf1)
        impf_set.append(impf2)

        with self.assertLogs(level="WARNING") as log:
            vuln_curves = self.converter.convert_impact_func_set(impf_set)
            self.assertEqual(len(vuln_curves), 1)  # Only valid function converted
            self.assertTrue(any("Skipping" in msg for msg in log.output))


class TestJSONExport(unittest.TestCase):
    """Test JSON export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = ImpactFuncToPhysrisk()
        self.temp_dir = tempfile.mkdtemp()

    def test_to_json_single_function(self):
        """Test JSON export of a single impact function."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            name="Test",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
            intensity_unit="m/s",
        )

        json_str = self.converter.to_json(impf, asset_type="Buildings/Residential")

        # Parse JSON to verify structure
        data = json.loads(json_str)

        self.assertIn("asset_type", data)
        self.assertIn("intensity", data)
        self.assertIn("impact_mean", data)
        self.assertEqual(data["asset_type"], "Buildings/Residential")

    def test_to_json_impact_func_set(self):
        """Test JSON export of an ImpactFuncSet."""
        impf_set = ImpactFuncSet()

        impf1 = ImpactFunc(
            haz_type="TC",
            id=1,
            name="Test1",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf2 = ImpactFunc(
            haz_type="WS",
            id=2,
            name="Test2",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.6, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        impf_set.append(impf1)
        impf_set.append(impf2)

        json_str = self.converter.to_json(impf_set)

        # Parse JSON to verify structure
        data = json.loads(json_str)

        self.assertIn("items", data)
        self.assertIsInstance(data["items"], list)
        self.assertEqual(len(data["items"]), 2)
        self.assertIn("asset_type", data["items"][0])

    def test_to_json_file_write(self):
        """Test writing JSON to file."""
        impf = ImpactFunc(
            haz_type="TC",
            id=1,
            name="Test",
            intensity=np.array([0, 50, 100]),
            mdd=np.array([0, 0.5, 1.0]),
            paa=np.array([1, 1, 1]),
        )

        file_path = Path(self.temp_dir) / "test_vuln_curve.json"

        json_str = self.converter.to_json(
            impf, asset_type="Test", file_path=str(file_path)
        )

        # Verify file was created
        self.assertTrue(file_path.exists())

        # Verify file content matches returned string
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        self.assertEqual(file_content, json_str)

    def test_to_json_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with self.assertRaises(TypeError):
            self.converter.to_json("not an impact function")


# Run tests
if __name__ == "__main__":
    unittest.main()
