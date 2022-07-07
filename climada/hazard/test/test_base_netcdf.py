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

Test NetCDF reading capabilities of Hazard base class.
"""

import os
import unittest
import datetime as dt
import numpy as np
from scipy.sparse import csr_matrix

import xarray as xr

from climada.hazard.base import Hazard

THIS_DIR = os.path.dirname(__file__)


class ReadDimsCoordsNetCDF(unittest.TestCase):
    """Checks for dimensions and coordinates with different names and shapes"""

    def setUp(self):
        """Write a NetCDF file with many coordinates"""
        self.netcdf_path = os.path.join(THIS_DIR, "coords.nc")
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
        os.remove(self.netcdf_path)

    def _assert_intensity_fraction(self, hazard):
        """Check if intensity and fraction data are read correctly"""
        np.testing.assert_array_equal(hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(hazard.fraction.toarray(), [[0, 0, 0, 1, 1, 1]])

    def test_dimension_naming(self):
        """Test if dimensions with different names can be read"""
        hazard = Hazard.from_raster_netcdf(
            self.netcdf_path,
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
        hazard = Hazard.from_raster_netcdf(
            self.netcdf_path,
            coordinate_vars=dict(latitude="lat", longitude="lon", time="years"),
        )
        np.testing.assert_array_equal(hazard.centroids.lat, [1, 1, 1, 2, 2, 2])
        np.testing.assert_array_equal(hazard.centroids.lon, [1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in self.years]
        )
        self._assert_intensity_fraction(hazard)

    def test_2D_coordinates(self):
        """Test if read method correctly handles 2D coordinates"""
        hazard = Hazard.from_raster_netcdf(
            self.netcdf_path,
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
        hazard = Hazard.from_raster_netcdf(ds)

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
            Hazard.from_raster_netcdf(
                self.netcdf_path, coordinate_vars=dict(bar="latitude", longitude="baz"),
            )
        self.assertIn("Unknown coordinates passed: '['bar']'.", str(cm.exception))

        # Correctly specified, but the custom dimension does not exist
        with self.assertRaises(KeyError) as cm:
            Hazard.from_raster_netcdf(
                self.netcdf_path, coordinate_vars=dict(latitude="lalalatitude"),
            )


class ReadDefaultNetCDF(unittest.TestCase):
    def setUp(self):
        """Write a simple NetCDF file to read"""
        self.netcdf_path = os.path.join(THIS_DIR, "default.nc")
        self.intensity = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
        self.fraction = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]])
        self.time = np.array([dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)])
        self.latitude = np.array([0, 1])
        self.longitude = np.array([0, 1, 2])
        dset = xr.Dataset(
            {
                "intensity": (["time", "latitude", "longitude"], self.intensity),
                "fraction": (["time", "latitude", "longitude"], self.fraction),
            },
            dict(time=self.time, latitude=self.latitude, longitude=self.longitude),
        )
        dset.to_netcdf(self.netcdf_path)

    def tearDown(self):
        """Delete the NetCDF file"""
        os.remove(self.netcdf_path)

    def _assert_default(self, hazard):
        """Assertions for the default hazard to be loaded"""
        # Hazard data
        self.assertEqual(hazard.tag.haz_type, "")
        self.assertIsInstance(hazard.event_id, np.ndarray)
        np.testing.assert_array_equal(hazard.event_id, [1, 2])
        self.assertIsInstance(hazard.event_name, list)
        np.testing.assert_array_equal(
            hazard.event_name, [np.datetime64(val) for val in self.time]
        )
        self.assertIsInstance(hazard.date, np.ndarray)
        np.testing.assert_array_equal(
            hazard.date, [val.toordinal() for val in self.time]
        )

        # Centroids
        self.assertEqual(hazard.centroids.lat.size, 6)
        self.assertEqual(hazard.centroids.lon.size, 6)
        np.testing.assert_array_equal(hazard.centroids.lat, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(hazard.centroids.lon, [0, 1, 2, 0, 1, 2])

        # Intensity data
        self.assertIsInstance(hazard.intensity, csr_matrix)
        np.testing.assert_array_equal(hazard.intensity.shape, [2, 6])
        np.testing.assert_array_equal(hazard.intensity.toarray()[0], [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(
            hazard.intensity.toarray()[1], [6, 7, 8, 9, 10, 11]
        )

        # Fraction data
        self.assertIsInstance(hazard.fraction, csr_matrix)
        np.testing.assert_array_equal(hazard.fraction.shape, [2, 6])
        np.testing.assert_array_equal(hazard.fraction.toarray()[0], [0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(hazard.fraction.toarray()[1], [1, 1, 1, 1, 1, 1])

    def test_load_path(self):
        """Load the data with path as argument"""
        hazard = Hazard.from_raster_netcdf(self.netcdf_path)
        self._assert_default(hazard)

    def test_load_dataset(self):
        """Load the data from an opened dataset as argument"""
        dataset = xr.open_dataset(self.netcdf_path)
        hazard = Hazard.from_raster_netcdf(dataset)
        self._assert_default(hazard)

    def test_fraction_callable(self):
        """Test creating a fraction from a callable"""
        hazard = Hazard.from_raster_netcdf(
            self.netcdf_path, fraction=lambda x: np.where(x > 1, 1, 0)
        )
        self.assertIsInstance(hazard.fraction, csr_matrix)
        np.testing.assert_array_equal(hazard.fraction.shape, [2, 6])
        np.testing.assert_array_equal(hazard.fraction.toarray()[0], [0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(hazard.fraction.toarray()[1], [1, 1, 1, 1, 1, 1])

    def test_errors(self):
        """Check the errors thrown"""
        # TODO: Maybe move to 'test_load_path'
        # Wrong paths
        with self.assertRaises(FileNotFoundError):
            Hazard.from_raster_netcdf("file-does-not-exist.nc")
        with self.assertRaises(KeyError):
            Hazard.from_raster_netcdf(
                self.netcdf_path, intensity="wrong-intensity-path"
            )
        with self.assertRaises(KeyError):
            Hazard.from_raster_netcdf(self.netcdf_path, fraction="wrong-fraction-path")

        # TODO: Maybe move to 'test_fraction_callable'
        # Wrong type passed as fraction
        with self.assertRaises(TypeError) as cm:
            Hazard.from_raster_netcdf(self.netcdf_path, fraction=3)
        self.assertIn(
            "'fraction' parameter must be 'str' or Callable", str(cm.exception)
        )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(ReadDefaultNetCDF)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(ReadDimsCoordsNetCDF))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
