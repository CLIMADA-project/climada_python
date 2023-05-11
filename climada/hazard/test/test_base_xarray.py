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

import unittest
from unittest.mock import patch
import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.sparse import csr_matrix

import xarray as xr
from pyproj import CRS

from climada.hazard.base import Hazard
from climada.util.constants import DEF_CRS


class TestReadDefaultNetCDF(unittest.TestCase):
    """Test reading a NetCDF file where the coordinates to read match the dimensions"""

    @classmethod
    def setUpClass(cls):
        """Write a simple NetCDF file to read"""
        cls.tempdir = TemporaryDirectory()
        cls.netcdf_path = Path(cls.tempdir.name) / "default.nc"
        cls.intensity = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
        cls.time = np.array([dt.datetime(1999, 1, 1), dt.datetime(2000, 1, 1)])
        cls.latitude = np.array([0, 1])
        cls.longitude = np.array([0, 1, 2])
        dset = xr.Dataset(
            {
                "intensity": (["time", "latitude", "longitude"], cls.intensity),
            },
            dict(time=cls.time, latitude=cls.latitude, longitude=cls.longitude),
        )
        dset.to_netcdf(cls.netcdf_path)

    @classmethod
    def tearDownClass(cls):
        """Delete the NetCDF file"""
        cls.tempdir.cleanup()

    def _assert_default(self, hazard):
        """Assertions for the default hazard to be loaded"""
        self._assert_default_types(hazard)
        self._assert_default_values(hazard)

    def _assert_default_values(self, hazard):
        """Check the values of the default hazard to be loaded"""
        # Hazard data
        self.assertEqual(hazard.tag.haz_type, "")
        self.assertEqual(hazard.units, "")
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
        self.assertEqual(hazard.centroids.geometry.crs, DEF_CRS)

        # Intensity data
        np.testing.assert_array_equal(
            hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]
        )

        # Fraction default
        self.assertEqual(hazard.fraction.nnz, 0)
        np.testing.assert_array_equal(hazard.fraction.shape, hazard.intensity.shape)
        np.testing.assert_array_equal(
            hazard.fraction.toarray(), np.zeros_like(hazard.intensity.toarray())
        )

    def _assert_default_types(self, hazard):
        """Check types of all hazard attributes"""
        self.assertIsInstance(hazard.units, str)
        self.assertIsInstance(hazard.tag.haz_type, str)
        self.assertIsInstance(hazard.event_id, np.ndarray)
        self.assertIsInstance(hazard.event_name, list)
        self.assertIsInstance(hazard.frequency, np.ndarray)
        self.assertIsInstance(hazard.intensity, csr_matrix)
        self.assertIsInstance(hazard.fraction, csr_matrix)
        self.assertIsInstance(hazard.date, np.ndarray)

    def test_load_path(self):
        """Load the data with path as argument"""
        hazard = Hazard.from_xarray_raster_file(self.netcdf_path, "", "")
        self._assert_default(hazard)

        # Check wrong paths
        with self.assertRaises(FileNotFoundError) as cm:
            Hazard.from_xarray_raster_file("file-does-not-exist.nc", "", "")
        self.assertIn("file-does-not-exist.nc", str(cm.exception))
        with self.assertRaises(KeyError) as cm:
            Hazard.from_xarray_raster_file(
                self.netcdf_path, "", "", intensity="wrong-intensity-path"
            )
        self.assertIn("wrong-intensity-path", str(cm.exception))

    def test_load_dataset(self):
        """Load the data from an opened dataset as argument"""

        def _load_and_assert(chunks):
            with xr.open_dataset(self.netcdf_path, chunks=chunks) as dataset:
                hazard = Hazard.from_xarray_raster(dataset, "", "")
                self._assert_default(hazard)

        _load_and_assert(chunks=None)
        _load_and_assert(chunks=dict(latitude=1, longitude=1, time=1))

    def test_type_error(self):
        """Calling 'from_xarray_raster' with wrong data type should throw"""
        # Passing a path
        with self.assertRaises(TypeError) as cm:
            Hazard.from_xarray_raster(self.netcdf_path, "", "")
        self.assertIn(
            "Use Hazard.from_xarray_raster_file instead",
            str(cm.exception),
        )

        # Passing a DataArray
        with xr.open_dataset(self.netcdf_path) as dset, self.assertRaises(
            TypeError
        ) as cm:
            Hazard.from_xarray_raster(dset["intensity"], "", "")
        self.assertIn(
            "This method only supports xarray.Dataset as input data",
            str(cm.exception),
        )

    def test_type_and_unit(self):
        """Test passing a custom type and unit"""
        hazard = Hazard.from_xarray_raster_file(
            self.netcdf_path, hazard_type="TC", intensity_unit="m/s"
        )
        self._assert_default_types(hazard)
        self.assertEqual(hazard.tag.haz_type, "TC")
        self.assertEqual(hazard.units, "m/s")

    def test_data_vars(self):
        """Check handling of data variables"""
        with xr.open_dataset(self.netcdf_path) as dataset:
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
            hazard = Hazard.from_xarray_raster(dataset, "", "")
            self._assert_default_types(hazard)
            np.testing.assert_array_equal(hazard.frequency, frequency)
            np.testing.assert_array_equal(hazard.event_id, event_id)
            np.testing.assert_array_equal(hazard.event_name, event_name)
            np.testing.assert_array_equal(hazard.date, date)
            np.testing.assert_array_equal(
                hazard.fraction.toarray(), [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
            )

            # Ignore keys (should be default values)
            hazard = Hazard.from_xarray_raster(
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
                Hazard.from_xarray_raster(
                    dataset, "", "", data_vars=dict(wrong_key="stuff")
                )
            self.assertIn(
                "Unknown data variables passed: '['wrong_key']'.", str(cm.exception)
            )

            # Non-existent identifier
            with self.assertRaises(KeyError) as cm:
                Hazard.from_xarray_raster(
                    dataset, "", "", data_vars=dict(frequency="freqqqqq")
                )
            self.assertIn("freqqqqq", str(cm.exception))

            # Wrong data length
            # NOTE: This also implicitly checks that 'frequency' is not read!
            dataset["freq"] = np.array(range(size + 1), dtype=np.float64)
            with self.assertRaises(RuntimeError) as cm:
                Hazard.from_xarray_raster(
                    dataset, "", "", data_vars=dict(frequency="freq")
                )
            self.assertIn(
                f"'freq' must have shape ({size},), but shape is ({size + 1},)",
                str(cm.exception),
            )

            # Integer data assertions
            dset = dataset.copy(deep=True)
            dset["event_id"] = np.array(range(size), dtype=np.float64) + 3.5
            with self.assertRaises(TypeError) as cm:
                Hazard.from_xarray_raster(dset, "", "")
            self.assertIn("'event_id' data array must be integers", str(cm.exception))
            dset["event_id"] = np.linspace(0, 10, size, dtype=np.int64)
            with self.assertRaises(ValueError) as cm:
                Hazard.from_xarray_raster(dset, "", "")
            self.assertIn("'event_id' data must be larger than zero", str(cm.exception))

            # Date as datetime
            date_str = [f"2000-01-{i:02}" for i in range(1, size + 1)]
            dataset["date"] = date_str
            hazard = Hazard.from_xarray_raster(dataset, "", "")
            np.testing.assert_array_equal(
                hazard.date,
                [dt.datetime(2000, 1, i).toordinal() for i in range(1, size + 1)],
            )

    def test_data_vars_repeat(self):
        """Test if suitable data vars are repeated as expected"""
        with xr.open_dataset(self.netcdf_path) as dataset:
            size = dataset.sizes["time"]

            # Set optionals in the dataset
            frequency = [1.5]
            event_name = ["bla"]
            date = 1
            dataset["event_name"] = event_name
            dataset["date"] = date
            dataset["frequency"] = frequency

            # Check if single-valued arrays are repeated
            hazard = Hazard.from_xarray_raster(dataset, "", "")

        np.testing.assert_array_equal(hazard.date, [date] * size)
        np.testing.assert_array_equal(hazard.event_name, event_name * size)
        np.testing.assert_array_equal(hazard.frequency, frequency * size)

    def test_nan(self):
        """Check handling of NaNs in original data"""
        with xr.open_dataset(self.netcdf_path) as dataset:
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
            hazard = Hazard.from_xarray_raster(dataset, "", "")

        # Check types
        self._assert_default_types(hazard)

        # NaNs are set to zero in sparse data
        np.testing.assert_array_equal(
            hazard.intensity.toarray(),
            [[0, 0, 2, 3, 4, 5], [6, 0, 8, 9, 10, 11]],
        )
        np.testing.assert_array_equal(
            hazard.fraction.toarray(),
            [[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 1]],
        )

        # NaNs are propagated in dense data
        np.testing.assert_array_equal(hazard.frequency, frequency)

    def test_crs(self):
        """Check if different CRS inputs are handled correctly"""

        def test_crs_from_input(crs_input):
            crs = CRS.from_user_input(crs_input)
            hazard = Hazard.from_xarray_raster_file(
                self.netcdf_path, "", "", crs=crs_input
            )
            self.assertEqual(hazard.centroids.geometry.crs, crs)

        test_crs_from_input("EPSG:3857")
        test_crs_from_input(3857)
        test_crs_from_input("+proj=cea +lat_0=52.112866 +lon_0=5.150162 +units=m")

    def test_missing_dims(self):
        """Test if missing coordinates are expanded and correct errors are thrown"""
        # Drop time as dimension, but not as coordinate!
        with xr.open_dataset(self.netcdf_path) as ds:
            ds = ds.isel(time=0).squeeze()
            hazard = Hazard.from_xarray_raster(ds, "", "")
            self._assert_default_types(hazard)
            np.testing.assert_array_equal(
                hazard.event_name, [np.datetime64(self.time[0])]
            )
            np.testing.assert_array_equal(hazard.date, [self.time[0].toordinal()])
            np.testing.assert_array_equal(hazard.centroids.lat, [0, 0, 0, 1, 1, 1])
            np.testing.assert_array_equal(hazard.centroids.lon, [0, 1, 2, 0, 1, 2])
            self.assertEqual(hazard.centroids.geometry.crs, DEF_CRS)
            np.testing.assert_array_equal(
                hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5]]
            )
            self.assertEqual(hazard.fraction.nnz, 0)
            np.testing.assert_array_equal(
                hazard.fraction.toarray(), [[0, 0, 0, 0, 0, 0]]
            )

            # Now drop variable altogether, should raise an error
            ds = ds.drop_vars("time")
            with self.assertRaises(RuntimeError) as cm:
                Hazard.from_xarray_raster(ds, "", "")
            self.assertIn(
                "Dataset is missing dimension/coordinate: time", str(cm.exception)
            )

            # Expand time again
            ds = ds.expand_dims(time=[np.datetime64("2022-01-01")])
            hazard = Hazard.from_xarray_raster(ds, "", "")
            self._assert_default_types(hazard)
            np.testing.assert_array_equal(
                hazard.event_name, [np.datetime64("2022-01-01")]
            )
            np.testing.assert_array_equal(
                hazard.date, [dt.datetime(2022, 1, 1).toordinal()]
            )


class TestReadDimsCoordsNetCDF(unittest.TestCase):
    """Checks for dimensions and coordinates with different names and shapes"""

    @classmethod
    def setUpClass(cls):
        """Write a NetCDF file with many coordinates"""
        cls.tempdir = TemporaryDirectory()
        cls.netcdf_path = Path(cls.tempdir.name) / "coords.nc"
        cls.intensity = np.array([[[0, 1, 2], [3, 4, 5]]])
        cls.fraction = np.array([[[0, 0, 0], [1, 1, 1]]])
        cls.time = np.array([dt.datetime(2000, 1, 1)])
        cls.x = np.array([0, 1, 2])
        cls.y = np.array([0, 1])
        cls.lon = np.array([1, 2, 3])
        cls.lat = np.array([1, 2])
        cls.years = np.array([dt.datetime(1999, 2, 2)])
        cls.longitude = np.array([[10, 11, 12], [10, 11, 12]])
        cls.latitude = np.array([[100, 100, 100], [200, 200, 200]])

        dset = xr.Dataset(
            {
                "intensity": (["time", "y", "x"], cls.intensity),
                "fraction": (["time", "y", "x"], cls.fraction),
            },
            {
                "time": cls.time,
                "x": cls.x,
                "y": cls.y,
                "lon": (["x"], cls.lon),
                "lat": (["y"], cls.lat),
                "years": (["time"], cls.years),
                "latitude": (["y", "x"], cls.latitude),
                "longitude": (["y", "x"], cls.longitude),
            },
        )
        dset.to_netcdf(cls.netcdf_path)

    @classmethod
    def tearDownClass(cls):
        """Delete the NetCDF file"""
        cls.tempdir.cleanup()

    def _assert_intensity_fraction(self, hazard):
        """Check if intensity and fraction data are read correctly"""
        np.testing.assert_array_equal(hazard.intensity.toarray(), [[0, 1, 2, 3, 4, 5]])
        np.testing.assert_array_equal(hazard.fraction.toarray(), [[0, 0, 0, 1, 1, 1]])

    def test_dimension_naming(self):
        """Test if dimensions with different names can be read"""
        hazard = Hazard.from_xarray_raster_file(
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
        hazard = Hazard.from_xarray_raster_file(
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
        hazard = Hazard.from_xarray_raster_file(
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

    def test_load_dataset_rechunk(self):
        """Load the data from an opened dataset and force rechunking"""
        with xr.open_dataset(self.netcdf_path) as dataset:
            hazard = Hazard.from_xarray_raster(
                dataset,
                "",
                "",
                coordinate_vars=dict(latitude="latitude", longitude="longitude"),
                rechunk=True,
            )

        np.testing.assert_array_equal(
            hazard.centroids.lat, [100, 100, 100, 200, 200, 200]
        )
        np.testing.assert_array_equal(hazard.centroids.lon, [10, 11, 12, 10, 11, 12])
        self._assert_intensity_fraction(hazard)

        # Assert that .chunk is called the right way
        with patch("xarray.Dataset.chunk") as mock:
            with xr.open_dataset(self.netcdf_path) as dataset:
                mock.return_value = dataset
                Hazard.from_xarray_raster(
                    dataset,
                    "",
                    "",
                    coordinate_vars=dict(latitude="latitude", longitude="longitude"),
                    rechunk=True,
                )

            # First latitude dim, then longitude dim, then event dim
            mock.assert_called_once_with(chunks=dict(y=-1, x=-1, time="auto"))

            # Should not be called by default
            mock.reset_mock()
            with xr.open_dataset(self.netcdf_path) as dataset:
                mock.return_value = dataset
                Hazard.from_xarray_raster(
                    dataset,
                    "",
                    "",
                    coordinate_vars=dict(latitude="latitude", longitude="longitude"),
                )

            mock.assert_not_called()

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
        hazard = Hazard.from_xarray_raster(ds, "", "")

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
            Hazard.from_xarray_raster_file(
                self.netcdf_path,
                "",
                "",
                coordinate_vars=dict(bar="latitude", longitude="baz"),
            )
        self.assertIn("Unknown coordinates passed: '['bar']'.", str(cm.exception))

        # Correctly specified, but the custom dimension does not exist
        with self.assertRaises(RuntimeError) as cm:
            Hazard.from_xarray_raster_file(
                self.netcdf_path,
                "",
                "",
                coordinate_vars=dict(latitude="lalalatitude"),
            )
        self.assertIn(
            "Dataset is missing dimension/coordinate: lalalatitude.", str(cm.exception)
        )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReadDefaultNetCDF)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestReadDimsCoordsNetCDF)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
