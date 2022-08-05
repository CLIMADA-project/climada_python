"""
This file is part of CLIMADA.

Copyright (C) 2022 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test xarray reading capabilities of Hazard base class.
"""

import os
import unittest
import datetime as dt
import numpy as np
from scipy.sparse import csr_matrix

import xarray as xr

from climada.hazard.base import Hazard
from pathlib import Path


class ReadDefaultNetCDF(unittest.TestCase):
    """Test reading a NetCDF file where the coordinates to read match the dimensions"""

    def setUp(self):
        """Write a simple NetCDF file to read"""
        self.netcdf_path = Path.cwd() / "default.nc"
        self.intensity = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
        self.time = np.array([dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)])
        self.latitude = np.array([0, 1])
        self.longitude = np.array([0, 1, 2])
        dset = xr.Dataset(
            {"intensity": (["time", "latitude", "longitude"], self.intensity),},
            dict(time=self.time, latitude=self.latitude, longitude=self.longitude),
        )
        dset.to_netcdf(self.netcdf_path)

    def tearDown(self):
        """Delete the NetCDF file"""
        self.netcdf_path.unlink()

    def _assert_default(self, hazard):
        """Assertions for the default hazard to be loaded"""
        self._assert_default_types(hazard)
        self._assert_default_values(hazard)

    def _assert_default_values(self, hazard):
        """Check the values of the default hazard to be loaded"""
        # Hazard data
        self.assertEqual(hazard.tag.haz_type, "")
        self.assertEqual(hazard.unit, "")
        np.testing.assert_array_equal(hazard.event_id, [1, 2])
        np.testing.assert_array_equal(
            hazard.event_name, [np.datetime64(val) for val in self.time]
        )
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in self.time]
        )
        np.testing.assert_array_equal(hazard.frequency, np.ones(hazard.event_id.size))

        # Centroids
        np.testing.assert_array_equal(hazard.centroids.lat, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(hazard.centroids.lon, [0, 1, 2, 0, 1, 2])

        # Intensity data
        np.testing.assert_array_equal(
            hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
        )

        # Fraction default
        np.testing.assert_array_equal(
            hazard.fraction.toarray(), [[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        )

    def _assert_default_types(self, hazard):
        """Check types of all hazard attributes"""
        self.assertIsInstance(hazard.unit, str)
        self.assertIsInstance(hazard.tag.haz_type, str)
        self.assertIsInstance(hazard.event_id, np.ndarray)
        self.assertIsInstance(hazard.event_name, list)
        self.assertIsInstance(hazard.frequency, np.ndarray)
        self.assertIsInstance(hazard.intensity, csr_matrix)
        self.assertIsInstance(hazard.fraction, csr_matrix)
        self.assertIsInstance(hazard.date, np.ndarray)

    def test_load_path(self):
        """Load the data with path as argument"""
        hazard = Hazard.from_raster_xarray(self.netcdf_path, "", "")
        self._assert_default(hazard)

        # Check wrong paths
        with self.assertRaises(FileNotFoundError) as cm:
            Hazard.from_raster_xarray("file-does-not-exist.nc", "", "")
        self.assertIn("file-does-not-exist.nc", str(cm.exception))
        with self.assertRaises(KeyError) as cm:
            Hazard.from_raster_xarray(
                self.netcdf_path, "", "", intensity="wrong-intensity-path"
            )
        self.assertIn("wrong-intensity-path", str(cm.exception))

    def test_load_dataset(self):
        """Load the data from an opened dataset as argument"""
        dataset = xr.open_dataset(self.netcdf_path)
        hazard = Hazard.from_raster_xarray(dataset, "", "")
        self._assert_default(hazard)

    def test_type_and_unit(self):
        """Test passing a custom type and unit"""
        hazard = Hazard.from_raster_xarray(
            self.netcdf_path, hazard_type="TC", intensity_unit="m/s"
        )
        self._assert_default_types(hazard)
        self.assertEqual(hazard.tag.haz_type, "TC")
        self.assertEqual(hazard.unit, "m/s")

    def test_data_vars(self):
        """Check handling of data variables"""
        dataset = xr.open_dataset(self.netcdf_path)
        size = dataset.sizes["time"]

        # Set optionals in the dataset
        frequency = np.ones(size) * 1.5
        event_id = np.array(range(size), dtype=np.int64) + 3
        event_name = ["bla"] * size
        date = np.array(range(size)) + 100
        dataset["event_id"] = event_id
        dataset["event_name"] = event_name
        dataset["date"] = date

        # Assign a proper coordinate for a change
        dataset = dataset.assign_coords(dict(frequency=("time", frequency)))

        # Assign fraction
        frac = xr.DataArray(
            np.array([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]]),
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=self.time, latitude=self.latitude, longitude=self.longitude
            ),
        )
        dataset["fraction"] = frac

        # Optionals should be read automatically
        hazard = Hazard.from_raster_xarray(dataset, "", "")
        self._assert_default_types(hazard)
        np.testing.assert_array_equal(hazard.frequency, frequency)
        np.testing.assert_array_equal(hazard.event_id, event_id)
        np.testing.assert_array_equal(hazard.event_name, event_name)
        np.testing.assert_array_equal(hazard.date, date)
        np.testing.assert_array_equal(
            hazard.fraction.toarray(), [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
        )

        # Ignore keys (should be default values)
        hazard = Hazard.from_raster_xarray(
            dataset,
            "",
            "",
            data_vars=dict(
                frequency="", event_id="", event_name="", date="", fraction=""
            ),
        )
        self._assert_default(hazard)

        # Wrong key
        with self.assertRaises(ValueError) as cm:
            Hazard.from_raster_xarray(
                dataset, "", "", data_vars=dict(wrong_key="stuff")
            )
        self.assertIn(
            "Unknown data variables passed: '['wrong_key']'.", str(cm.exception)
        )

        # Non-existent identifier
        with self.assertRaises(KeyError) as cm:
            Hazard.from_raster_xarray(
                dataset, "", "", data_vars=dict(frequency="freqqqqq")
            )
        self.assertIn("freqqqqq", str(cm.exception))

        # Wrong data length
        # NOTE: This also implicitly checks that 'frequency' is not read!
        dataset["freq"] = np.array(range(size + 1), dtype=np.float64)
        with self.assertRaises(RuntimeError) as cm:
            Hazard.from_raster_xarray(dataset, "", "", data_vars=dict(frequency="freq"))
        self.assertIn(
            f"'freq' must have shape ({size},), but shape is ({size + 1},)",
            str(cm.exception),
        )

        # Integer data assertions
        for key in ("event_id", "date"):
            dset = dataset.copy(deep=True)
            dset[key] = np.array(range(size), dtype=np.float64) + 3.5
            with self.assertRaises(TypeError) as cm:
                Hazard.from_raster_xarray(dset, "", "")
            self.assertIn(f"'{key}' data array must be integers", str(cm.exception))
            dset[key] = np.linspace(0, 10, size, dtype=np.int64)
            with self.assertRaises(ValueError) as cm:
                Hazard.from_raster_xarray(dset, "", "")
            self.assertIn(f"'{key}' data must be larger than zero", str(cm.exception))

    def test_nan(self):
        """Check handling of NaNs in original data"""
        dataset = xr.open_dataset(self.netcdf_path)
        intensity = xr.DataArray(
            np.array([[[0, np.nan, 2], [3, 4, 5]], [[6, np.nan, 8], [9, 10, 11]]]),
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=self.time, latitude=self.latitude, longitude=self.longitude
            ),
        )
        dataset["intensity"] = intensity
        fraction = xr.DataArray(
            np.array([[[0, 0, 0], [0, 0, 0]], [[1, np.nan, 1], [np.nan, 1, 1]]]),
            dims=["time", "latitude", "longitude"],
            coords=dict(
                time=self.time, latitude=self.latitude, longitude=self.longitude
            ),
        )
        dataset["fraction"] = fraction
        frequency = np.ones(dataset.sizes["time"])
        frequency[0] = np.nan
        dataset["frequency"] = frequency

        # Load hazard
        hazard = Hazard.from_raster_xarray(dataset, "", "")
        self._assert_default_types(hazard)

        # NaNs are set to zero in sparse data
        np.testing.assert_array_equal(
            hazard.intensity.toarray(), [[0, 0, 2, 3, 4, 5], [6, 0, 8, 9, 10, 11]],
        )
        np.testing.assert_array_equal(
            hazard.fraction.toarray(), [[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1]],
        )

        # NaNs are propagated in dense data
        np.testing.assert_array_equal(hazard.frequency, frequency)


class ReadDimsCoordsNetCDF(unittest.TestCase):
    """Checks for dimensions and coordinates with different names and shapes"""

    def setUp(self):
        """Write a NetCDF file with many coordinates"""
        self.netcdf_path = Path.cwd() / "coords.nc"
        self.intensity = np.array([[[0, 1, 2], [3, 4, 5]]])
        self.fraction = np.array([[[0, 0, 0], [1, 1, 1]]])
        self.time = np.array([dt.datetime(2000, 1, 1)])
        self.x = np.array([0, 1, 2])
        self.y = np.array([0, 1])
        self.lon = np.array([1, 2, 3])
        self.lat = np.array([1, 2])
        self.years = np.array([dt.datetime(1999, 2, 2)])
        self.longitude = np.array([[10, 11, 12], [10, 11, 12]])
        self.latitude = np.array([[100, 100, 100], [200, 200, 200]])

        dset = xr.Dataset(
            {
                "intensity": (["time", "y", "x"], self.intensity),
                "fraction": (["time", "y", "x"], self.fraction),
            },
            {
                "time": self.time,
                "x": self.x,
                "y": self.y,
                "lon": (["x"], self.lon),
                "lat": (["y"], self.lat),
                "years": (["time"], self.years),
                "latitude": (["y", "x"], self.latitude),
                "longitude": (["y", "x"], self.longitude),
            },
        )
        dset.to_netcdf(self.netcdf_path)

    def tearDown(self):
        """Delete the NetCDF file"""
        self.netcdf_path.unlink()

    def _assert_intensity_fraction(self, hazard):
        """Check if intensity and fraction data are read correctly"""
        np.testing.assert_array_equal(hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(hazard.fraction.toarray(), [[0, 0, 0, 1, 1, 1]])

    def test_dimension_naming(self):
        """Test if dimensions with different names can be read"""
        hazard = Hazard.from_raster_xarray(
            self.netcdf_path,
            "",
            "",
            coordinate_vars=dict(latitude="y", longitude="x"),  # 'time' stays default
        )
        np.testing.assert_array_equal(hazard.centroids.lat, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(hazard.centroids.lon, [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in self.time]
        )
        self._assert_intensity_fraction(hazard)

    def test_coordinate_naming(self):
        """Test if coordinates with different names than dimensions can be read"""
        hazard = Hazard.from_raster_xarray(
            self.netcdf_path,
            "",
            "",
            coordinate_vars=dict(latitude="lat", longitude="lon", event="years"),
        )
        np.testing.assert_array_equal(hazard.centroids.lat, [1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(hazard.centroids.lon, [1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in self.years]
        )
        self._assert_intensity_fraction(hazard)

    def test_2D_coordinates(self):
        """Test if read method correctly handles 2D coordinates"""
        hazard = Hazard.from_raster_xarray(
            self.netcdf_path,
            "",
            "",
            coordinate_vars=dict(latitude="latitude", longitude="longitude"),
        )
        np.testing.assert_array_equal(
            hazard.centroids.lat, [100, 100, 100, 200, 200, 200]
        )
        np.testing.assert_array_equal(hazard.centroids.lon, [10, 11, 12, 10, 11, 12])
        self._assert_intensity_fraction(hazard)

    def test_2D_time(self):
        """Test if stacking multiple time dimensions works out"""
        time = np.array(
            [
                [dt.datetime(1999, 1, 1), dt.datetime(1999, 2, 1)],
                [dt.datetime(2000, 1, 1), dt.datetime(2000, 2, 1)],
            ]
        )
        ds = xr.Dataset(
            {
                "intensity": (
                    ["year", "month", "latitude", "longitude"],
                    [[[[1]], [[2]]], [[[3]], [[4]]]],
                ),
                "fraction": (
                    ["year", "month", "latitude", "longitude"],
                    [[[[10]], [[20]]], [[[30]], [[40]]]],
                ),
            },
            {
                "latitude": [1],
                "longitude": [2],
                "year": [1999, 2000],
                "month": [1, 2],
                "time": (["year", "month"], time),
            },
        )
        hazard = Hazard.from_raster_xarray(ds, "", "")

        np.testing.assert_array_equal(hazard.intensity.toarray(), [[1], [2], [3], [4]])
        np.testing.assert_array_equal(
            hazard.fraction.toarray(), [[10], [20], [30], [40]]
        )
        np.testing.assert_array_equal(hazard.centroids.lat, [1])
        np.testing.assert_array_equal(hazard.centroids.lon, [2])
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in time.flat]
        )

    def test_errors(self):
        """Check if expected errors are thrown"""
        # Wrong coordinate key
        with self.assertRaises(ValueError) as cm:
            Hazard.from_raster_xarray(
                self.netcdf_path,
                "",
                "",
                coordinate_vars=dict(bar="latitude", longitude="baz"),
            )
        self.assertIn("Unknown coordinates passed: '['bar']'.", str(cm.exception))

        # Correctly specified, but the custom dimension does not exist
        with self.assertRaises(KeyError) as cm:
            Hazard.from_raster_xarray(
                self.netcdf_path, "", "", coordinate_vars=dict(latitude="lalalatitude"),
            )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(ReadDefaultNetCDF)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(ReadDimsCoordsNetCDF))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
