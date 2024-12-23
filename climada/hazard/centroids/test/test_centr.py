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

Test CentroidsVector and CentroidsRaster classes.
"""

import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from cartopy.io import shapereader
from pyproj.crs.crs import CRS
from rasterio import Affine
from rasterio.windows import Window
from shapely.geometry.point import Point

import climada.util.coordinates as u_coord
from climada import CONFIG
from climada.entity import Exposures
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import DEF_CRS, HAZ_DEMO_FL

DATA_DIR = CONFIG.hazard.test_data.dir()

# Note: the coordinates are not directly on the cities, the region id and on land
# otherwise do not work correctly. It is only a close point.
LATLON = np.array(
    [
        [-21.1736, -175.1883],  # Tonga, Nuku'alofa, TON, 776
        [-18.133, 178.433],  # Fidji, Suva, FJI, 242  IN WATER IN NATURAL EARTH
        [-38.4689, 177.8642],  # New-Zealand, Te Karaka, NZL, 554
        [69.6833, 18.95],  # Norway, Tromso, NOR, 578 IN WATER IN NATURAL EARTH
        [78.84422, 20.82842],  # Norway, Svalbard, NOR, 578
        [1, 1],  # Ocean, 0  (0,0 is onland in Natural earth for testing reasons)
        [-77.85, 166.6778],  # Antarctica, McMurdo station, ATA, 010
        [-0.25, -78.5833],  # Ecuador, Quito, ECU, 218
    ]
)

VEC_LAT = LATLON[:, 0]
VEC_LON = LATLON[:, 1]

ON_LAND = np.array([True, False, True, False, True, False, True, True])
REGION_ID = np.array([776, 0, 554, 0, 578, 0, 10, 218])

TEST_CRS = "EPSG:4326"
ALT_CRS = "epsg:32632"  # UTM zone 32N (Central Europe, 6-12°E)


class TestCentroidsData(unittest.TestCase):
    """Test class for initialisation and value based creation of Centroids objects"""

    def setUp(self):
        self.lat = np.array([-10, 0, 10])
        self.lon = np.array([-170, -150, -130])
        self.region_id = np.array([1, 2, 3])
        self.on_land = np.array([True, False, False])
        self.crs = "epsg:32632"
        self.centr = Centroids(lat=VEC_LAT, lon=VEC_LON)

    def test_centroids_check_pass(self):
        """Test vector data in Centroids"""
        centr = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=ALT_CRS)

        self.assertTrue(u_coord.equal_crs(centr.crs, CRS.from_user_input(ALT_CRS)))
        self.assertEqual(
            list(centr.total_bounds),
            [VEC_LON.min(), VEC_LAT.min(), VEC_LON.max(), VEC_LAT.max()],
        )

        self.assertIsInstance(centr, Centroids)
        self.assertIsInstance(centr.lat, np.ndarray)
        self.assertIsInstance(centr.lon, np.ndarray)
        self.assertIsInstance(centr.coord, np.ndarray)
        self.assertTrue(np.array_equal(centr.lat, VEC_LAT))
        self.assertTrue(np.array_equal(centr.lon, VEC_LON))
        self.assertTrue(
            np.array_equal(centr.coord, np.array([VEC_LAT, VEC_LON]).transpose())
        )
        self.assertEqual(centr.size, VEC_LON.size)

    def test_init_pass(self):
        # Creating Centroids with latitude and longitude arrays
        # check instance - trivial...
        # Checking attributes
        np.testing.assert_array_equal(self.centr.lat, VEC_LAT)
        np.testing.assert_array_equal(self.centr.lon, VEC_LON)
        self.assertTrue(u_coord.equal_crs(self.centr.crs, DEF_CRS))

        # Creating Centroids with additional attributes
        centroids = Centroids(
            lat=VEC_LAT, lon=VEC_LON, region_id=REGION_ID, on_land=ON_LAND
        )

        # Checking additional attributes
        np.testing.assert_array_equal(centroids.region_id, REGION_ID)
        np.testing.assert_array_equal(centroids.on_land, ON_LAND)

    def test_init_defaults(self):
        """Checking default values for Centroids"""
        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON)
        # Checking defaults: nothing set for region_id, on_land
        self.assertFalse(centroids.region_id)
        self.assertFalse(centroids.on_land)
        # Guarantee a no-default TypeError for lon/lat
        with self.assertRaises(TypeError):
            Centroids()

    def test_init_properties(self):
        """Guarantee that Centroid objects have at least the properties:"""
        properties = [
            "gdf",
            "lon",
            "lat",
            "geometry",
            "on_land",
            "region_id",
            "crs",
            "shape",
            "size",
            "total_bounds",
            "coord",
        ]
        centroids = Centroids(lat=[], lon=[])
        [self.assertTrue(hasattr(centroids, prop)) for prop in properties]

    def test_init_kwargs(self):
        """Test default crs and kwargs forwarding"""
        centr = Centroids(
            lat=VEC_LAT,
            lon=VEC_LON,
            region_id=REGION_ID,
            on_land=ON_LAND,
        )
        self.assertTrue(u_coord.equal_crs(centr.crs, DEF_CRS))
        self.assertTrue(np.allclose(centr.region_id, REGION_ID))
        self.assertTrue(np.allclose(centr.on_land, ON_LAND))

        # make sure kwargs are properly forwarded to centroids.gdf
        np.random.seed(1000)
        randommask = np.random.choice([True, False], size=len(VEC_LON))
        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON, masked=randommask, ones=1)
        self.assertTrue(hasattr(centroids.gdf, "masked"))
        self.assertTrue(hasattr(centroids.gdf, "ones"))
        np.testing.assert_array_equal(randommask, centroids.gdf.masked)
        self.assertEqual(sum(centroids.gdf.ones), len(VEC_LON))

    def test_from_meta_pass(self):
        expected_lon = np.array([-30.0, -20.0, -10.0] * 3)
        expected_lat = np.repeat([30.0, 20.0, 10.0], 3)
        # Check metadata
        meta = dict(
            crs=DEF_CRS,
            height=3,
            width=3,
            transform=Affine(
                10,
                0,
                -35,
                0,
                -10,
                35,
            ),
        )
        centroids = Centroids.from_meta(meta)

        # check created object
        np.testing.assert_array_equal(centroids.lon, expected_lon)
        np.testing.assert_array_equal(centroids.lat, expected_lat)
        self.assertEqual(centroids.crs, DEF_CRS)
        # generally we assume that from_meta does not set region_ids and on_land flags
        self.assertFalse(centroids.region_id)
        self.assertFalse(centroids.on_land)

    def test_from_meta(self):
        """Test from_meta"""
        meta_ref = {
            "width": 10,
            "height": 8,
            "transform": rasterio.Affine(
                0.6,
                0,
                -0.1,
                0,
                -0.6,
                0.3,
            ),
            "crs": DEF_CRS,
        }

        lon_ref = np.array([0.2, 0.8, 1.4, 2.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6])
        lat_ref = np.array([0.0, -0.6, -1.2, -1.8, -2.4, -3.0, -3.6, -4.2])
        lon_ref, lat_ref = [ar.ravel() for ar in np.meshgrid(lon_ref, lat_ref)]

        centr = Centroids.from_meta(meta_ref)
        meta = centr.get_meta()
        self.assertTrue(u_coord.equal_crs(meta_ref["crs"], meta["crs"]))
        self.assertEqual(meta_ref["width"], meta["width"])
        self.assertEqual(meta_ref["height"], meta["height"])
        np.testing.assert_allclose(meta_ref["transform"], meta["transform"])

        centr = Centroids.from_meta(Centroids(lat=lat_ref, lon=lon_ref).get_meta())
        np.testing.assert_allclose(lat_ref, centr.lat)
        np.testing.assert_allclose(lon_ref, centr.lon)

        # `get_meta` enforces same resolution in x and y, and y-coordinates are decreasing.
        # For other cases, `from_meta` needs to be checked manually.
        meta_ref = {
            "width": 4,
            "height": 5,
            "transform": rasterio.Affine(
                0.5,
                0,
                0.2,
                0,
                0.6,
                -0.7,
            ),
            "crs": DEF_CRS,
        }
        lon_ref = np.array([0.45, 0.95, 1.45, 1.95])
        lat_ref = np.array([-0.4, 0.2, 0.8, 1.4, 2.0])
        lon_ref, lat_ref = [ar.ravel() for ar in np.meshgrid(lon_ref, lat_ref)]

        centr = Centroids.from_meta(meta_ref)
        np.testing.assert_allclose(lat_ref, centr.lat)
        np.testing.assert_allclose(lon_ref, centr.lon)

    def test_from_pnt_bounds(self):
        """Test from_pnt_bounds"""
        width, height = 26, 51
        left, bottom, right, top = 5, 0, 10, 10

        centr = Centroids.from_pnt_bounds((left, bottom, right, top), 0.2, crs=DEF_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, DEF_CRS))
        self.assertEqual(centr.size, width * height)
        np.testing.assert_allclose([5.0, 5.2, 5.0], centr.lon[[0, 1, width]], atol=0.1)
        np.testing.assert_allclose(
            [10.0, 10.0, 9.8], centr.lat[[0, 1, width]], atol=0.1
        )
        # generally we assume that from_meta does not set region_ids and on_land flags
        self.assertFalse(centr.region_id)
        self.assertFalse(centr.on_land)


class TestCentroidsTransformation(unittest.TestCase):
    """Test class for coordinate transformations of Centroid objects
    and modifications using set_ methods"""

    def setUp(self):
        self.lat = np.array([-10, 0, 10])
        self.lon = np.array([-170, -150, -130])
        self.region_id = np.array([1, 2, 3])
        self.on_land = np.array([True, False, False])
        self.crs = "epsg:32632"
        self.centr = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=TEST_CRS)

    def test_to_default_crs(self):
        # Creating Centroids with non-default CRS and
        # inplace transformation afterwards
        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=ALT_CRS)
        self.assertTrue(u_coord.equal_crs(centroids.crs, ALT_CRS))
        centroids.to_default_crs()
        # make sure CRS is DEF_CRS after transformation
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))
        # Checking that modification actually took place
        [self.assertNotEqual(x - y, 0) for x, y in zip(centroids.lon, VEC_LON)]
        [
            self.assertNotEqual(x - y, 0)
            for x, y in zip(centroids.lat, VEC_LAT)
            if not x == 0
        ]

    def test_to_default_crs_not_inplace(self):
        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=ALT_CRS)
        newcentr = centroids.to_default_crs(inplace=False)
        # make sure that new object has been created
        self.assertIsNot(centroids, newcentr)
        self.assertIsInstance(newcentr, Centroids)
        ## compare with inplace transformation
        centroids.to_default_crs()
        np.testing.assert_array_equal(centroids.lat, newcentr.lat)
        np.testing.assert_array_equal(centroids.lon, newcentr.lon)

    def test_to_crs(self):
        # Creating Centroids with default CRS
        centroids = Centroids(lat=self.lat, lon=self.lon, crs=DEF_CRS)

        # Transforming to another CRS
        new_crs = "epsg:3857"
        transformed_centroids = centroids.to_crs(new_crs)

        self.assertIsNot(centroids, transformed_centroids)
        self.assertFalse(centroids == transformed_centroids)

        # Checking CRS string after transformation
        self.assertTrue(u_coord.equal_crs(transformed_centroids.crs, new_crs))
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))

        # Checking correctness of transformation
        expected_lat = np.array([-1118889.974858, 0.0, 1118889.9748585])
        expected_lon = np.array([-18924313.434857, -16697923.618991, -14471533.803126])
        np.testing.assert_array_almost_equal(transformed_centroids.lat, expected_lat)
        np.testing.assert_array_almost_equal(transformed_centroids.lon, expected_lon)

    def test_to_crs_inplace(self):
        centroids = Centroids(lat=self.lat, lon=self.lon, crs=DEF_CRS)
        new_crs = "epsg:3857"
        transformed_centroids = centroids.to_crs(new_crs)

        # inplace transforming to another CRS
        centroids.to_crs(new_crs, inplace=True)

        self.assertTrue(centroids == transformed_centroids)

        expected_lat = np.array([-1118889.974858, 0.0, 1118889.9748585])
        expected_lon = np.array([-18924313.434857, -16697923.618991, -14471533.803126])
        np.testing.assert_array_almost_equal(centroids.lat, expected_lat)
        np.testing.assert_array_almost_equal(centroids.lon, expected_lon)

    def test_ne_crs_geom_pass(self):
        """Test _ne_crs_geom"""
        natural_earth_geom = self.centr._ne_crs_geom()
        self.assertEqual(natural_earth_geom.crs, u_coord.NE_CRS)

        centr = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=ALT_CRS)
        ne_geom = centr._ne_crs_geom()
        self.assertTrue(u_coord.equal_crs(ne_geom.crs, u_coord.NE_CRS))
        np.testing.assert_allclose(ne_geom.geometry[:].x.values, 4.5, atol=0.1)
        np.testing.assert_allclose(ne_geom.geometry[:].y.values, 0.0, atol=0.001)

    def test_set_on_land_pass(self):
        """Test set_on_land"""
        self.centr.set_on_land()
        np.testing.assert_array_equal(self.centr.on_land, ON_LAND)

        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON, on_land="natural_earth")
        np.testing.assert_array_equal(centroids.on_land, ON_LAND)

    def test_set_on_land_implementationerror(self):
        centroids = Centroids(lat=self.lat, lon=self.lon)

        with self.assertRaises(NotImplementedError):
            centroids.set_on_land(source="satellite", overwrite=True)

    def test_set_on_land_raster(self):
        """Test set_on_land"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_on_land()
        self.assertTrue(np.array_equal(centr_ras.on_land, np.ones(60 * 50, bool)))

    def test_set_region_id_pass(self):
        """Test set_region_id"""
        self.centr.set_region_id()
        np.testing.assert_array_equal(self.centr.region_id, REGION_ID)

        centroids = Centroids(lat=VEC_LAT, lon=VEC_LON, region_id="country")
        np.testing.assert_array_equal(centroids.region_id, REGION_ID)

    def test_set_region_id_raster(self):
        """Test set_region_id raster file"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_region_id()
        self.assertEqual(centr_ras.region_id.size, centr_ras.size)
        self.assertTrue(np.array_equal(np.unique(centr_ras.region_id), np.array([862])))

    def test_set_region_id_implementationerror(self):
        centroids = Centroids(lat=self.lat, lon=self.lon)

        with self.assertRaises(NotImplementedError):
            centroids.set_region_id(level="continent", overwrite=True)

    def test_set_geometry_points_pass(self):
        """Test geometry is set"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        x_flat = np.arange(-69.3326495969998, -68.88264959699978, 0.009000000000000341)
        y_flat = np.arange(10.423720966978939, 9.883720966978919, -0.009000000000000341)
        x_grid, y_grid = np.meshgrid(x_flat, y_flat)
        self.assertTrue(np.allclose(x_grid.flatten(), centr_ras.lon))
        self.assertTrue(np.allclose(y_grid.flatten(), centr_ras.lat))


class TestCentroidsReaderWriter(unittest.TestCase):
    """Test class for file based creation of Centroid objects and output"""

    def test_from_csv_def_crs(self):
        """Read a centroid csv file correctly and use default CRS."""
        # Create temporary csv file containing centroids data
        tmpfile = Path("test_write_csv.csv")
        lat = np.array([0, 90, -90, 0, 0])
        lon = np.array([0, 0, 0, 180, -180])
        df = pd.DataFrame({"lat": lat, "lon": lon})
        df.to_csv(tmpfile, index=False)

        # Read centroids using from_csv method
        centroids = Centroids.from_csv(tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, DEF_CRS)

        # delete file
        tmpfile.unlink()

    def test_from_csv(self):
        """Read a centroid csv file which contains CRS information."""
        tmpfile = Path("test_write_csv.csv")
        lat = np.array([0, 20048966.1, -20048966, 0, 0])
        lon = np.array([0, 0, 0, 20037508.34, -20037508.34])
        region_id = np.array([1, 2, 3, 4, 5])
        on_land = np.array([True, False, False, True, True])
        df = pd.DataFrame(
            {"lat": lat, "lon": lon, "region_id": region_id, "on_land": on_land}
        )
        df["crs"] = CRS.from_user_input(3857).to_wkt()
        df.to_csv(tmpfile, index=False)

        # Read centroids using from_csv method
        centroids = Centroids.from_csv(tmpfile)

        # Test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, "epsg:3857")
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)

        # Delete file
        tmpfile.unlink()

    def test_write_read_csv(self):
        """Write and read a Centroids CSV file correctly."""
        # Create Centroids with latitude and longitude arrays
        tmpfile = Path("test_write_csv.csv")
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        region_id = np.array([1, 2, 3])
        on_land = np.array([True, False, False])
        centroids_out = Centroids(
            lat=lat, lon=lon, region_id=region_id, on_land=on_land
        )

        # Write CSV file from Centroids using write_csv
        centroids_out.write_csv(tmpfile)

        # Read CSV file using read_csv
        centroids_in = Centroids.from_csv(tmpfile)

        # Test attributes
        np.testing.assert_array_equal(centroids_in.lat, centroids_out.lat)
        np.testing.assert_array_equal(centroids_in.lon, centroids_out.lon)
        self.assertTrue(u_coord.equal_crs(centroids_in.crs, centroids_out.crs))
        np.testing.assert_array_equal(centroids_in.region_id, centroids_out.region_id)
        np.testing.assert_array_equal(centroids_in.on_land, centroids_out.on_land)

        # delete file
        tmpfile.unlink()

    def test_from_excel_def_crs(self):
        """Read a centroid excel file correctly and use default CRS."""
        # Create temporary excel file containing centroids data
        tmpfile = Path("test_write_excel.xlsx")
        lat = np.array([0, 90, -90, 0, 0])
        lon = np.array([0, 0, 0, 180, -180])
        df = pd.DataFrame({"lat": lat, "lon": lon})
        df.to_excel(tmpfile, sheet_name="centroids", index=False)

        # Read centroids using from_excel method
        centroids = Centroids.from_excel(file_path=tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, DEF_CRS)

        # delete file
        tmpfile.unlink()

    def test_from_excel(self):
        """Read a centroid excel file correctly which contains CRS information."""
        # Create temporary excel file containing centroids data
        tmpfile = Path("test_write_excel.xlsx")
        lat = np.array([0, 20048966.1, -20048966, 0, 0])
        lon = np.array([0, 0, 0, 20037508.34, -20037508.34])
        region_id = np.array([1, 2, 3, 4, 5])
        on_land = np.array([True, False, False, True, True])
        df = pd.DataFrame(
            {"lat": lat, "lon": lon, "region_id": region_id, "on_land": on_land}
        )
        df["crs"] = CRS.from_user_input(3857).to_wkt()
        df.to_excel(tmpfile, sheet_name="centroids", index=False)

        # Read centroids using from_excel method
        centroids = Centroids.from_excel(file_path=tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, "epsg:3857")
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)

        # delete file
        tmpfile.unlink()

    def test_write_read_excel(self):
        """Write and read a Centroids Excel file correctly."""
        # Create Centroids with latitude and longitude arrays
        tmpfile = Path("test_write_excel.xlsx")
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        region_id = np.array([1, 2, 3])
        on_land = np.array([True, False, False])
        centroids_out = Centroids(
            lat=lat, lon=lon, region_id=region_id, on_land=on_land
        )

        # Write Excel file from Centroids using write_csv
        centroids_out.write_excel(tmpfile)

        # Read Excel file using read_csv
        centroids_in = Centroids.from_excel(tmpfile)

        # Test attributes
        np.testing.assert_array_equal(centroids_in.lat, centroids_out.lat)
        np.testing.assert_array_equal(centroids_in.lon, centroids_out.lon)
        self.assertTrue(u_coord.equal_crs(centroids_in.crs, centroids_out.crs))
        np.testing.assert_array_equal(centroids_in.region_id, centroids_out.region_id)
        np.testing.assert_array_equal(centroids_in.on_land, centroids_out.on_land)

        # delete file
        tmpfile.unlink()

    def test_from_raster_file(self):
        """Test from_raster_file"""
        width, height = 50, 60
        o_lat, o_lon = (10.42822096697894, -69.33714959699981)
        res_lat, res_lon = (-0.009000000000000341, 0.009000000000000341)

        centr_ras = Centroids.from_raster_file(
            HAZ_DEMO_FL, window=Window(0, 0, width, height)
        )
        self.assertTrue(u_coord.equal_crs(centr_ras.crs, DEF_CRS))
        self.assertEqual(centr_ras.size, width * height)
        np.testing.assert_allclose(
            [-69.333, -69.324, -69.333],
            centr_ras.lon[[0, 1, width]],
            atol=0.001,
        )
        np.testing.assert_allclose(
            [10.424, 10.424, 10.415],
            centr_ras.lat[[0, 1, width]],
            atol=0.001,
        )

    def test_from_vector_file(self):
        """Test from_vector_file and values_from_vector_files"""
        shp_file = shapereader.natural_earth(
            resolution="110m", category="cultural", name="populated_places_simple"
        )

        centr = Centroids.from_vector_file(shp_file, dst_crs=DEF_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, DEF_CRS))
        self.assertAlmostEqual(centr.lon[0], 12.453386544971766)
        self.assertAlmostEqual(centr.lon[-1], 114.18306345846304)
        self.assertAlmostEqual(centr.lat[0], 41.903282179960115)
        self.assertAlmostEqual(centr.lat[-1], 22.30692675357551)

        centr = Centroids.from_vector_file(shp_file, dst_crs=ALT_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, ALT_CRS))

    def test_from_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        crs = DEF_CRS
        lat = np.arange(170, 180)
        lon = np.arange(-50, -40)
        region_id = np.arange(1, 11)
        on_land = np.ones(10, dtype=bool)
        extra = np.full(10, "a")

        gdf = gpd.GeoDataFrame(
            {
                "geometry": gpd.points_from_xy(lon, lat),
                "region_id": region_id,
                "on_land": on_land,
                "extra": extra,
            },
            crs=crs,
        )

        centroids = Centroids.from_geodataframe(gdf)

        for name, array in zip(
            ["lat", "lon", "region_id", "on_land"],
            [lat, lon, region_id, on_land],
        ):
            np.testing.assert_array_equal(array, getattr(centroids, name))
        self.assertTrue("extra" in centroids.gdf.columns)
        self.assertTrue(u_coord.equal_crs(centroids.crs, crs))

    def test_from_geodataframe_invalid(self):

        # Creating an invalid GeoDataFrame with geometries that are not points
        invalid_geometry_gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    shapely.Point((2, 2)),
                    shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                    shapely.LineString([(0, 1), (1, 0)]),
                ],
            }
        )

        with self.assertRaises(ValueError):
            # Trying to create Centroids from invalid GeoDataFrame
            Centroids.from_geodataframe(invalid_geometry_gdf)

    def test_from_exposures_with_region_id(self):
        """
        Test that the `from_exposures` method correctly extracts
        centroids and region_id from an `Exposure` object with region_id.
        """
        # Create an Exposure object with region_id, on_land and custom crs
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        value = np.array([1, 1, 1])
        region_id = np.array([1, 2, 3])
        on_land = [False, True, True]
        crs = "epsg:32632"
        gdf = gpd.GeoDataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "value": value,
                "region_id": region_id,
                "on_land": on_land,
            }
        )
        exposures = Exposures(gdf, crs=crs)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)
        self.assertFalse(np.isin("value", centroids.gdf.columns))
        self.assertEqual(centroids.crs, crs)

    def test_from_exposures_without_region_id(self):
        """
        Test that the `from_exposures` method correctly extracts
        centroids from an `Exposure` object without region_id.
        """
        # Create an Exposure object without region_id and variables to ignore
        # and default crs
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        value = np.array([1, 1, 1])
        impf_TC = np.array([1, 2, 3])
        centr_TC = np.array([1, 2, 3])
        gdf = gpd.GeoDataFrame(
            {
                "latitude": lat,
                "longitude": lon,
                "value": value,
                "impf_tc": impf_TC,
                "centr_TC": centr_TC,
            }
        )
        exposures = Exposures(gdf)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        self.assertEqual(centroids.lat.tolist(), lat.tolist())
        self.assertEqual(centroids.lon.tolist(), lon.tolist())
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))
        self.assertEqual(centroids.region_id, None)
        self.assertEqual(centroids.on_land, None)
        np.testing.assert_equal(
            np.isin(["value", "impf_tc", "centr_tc"], centroids.gdf.columns),
            False,
        )

    def test_from_empty_exposures(self):
        gdf = gpd.GeoDataFrame({})
        exposures = Exposures(gdf)
        centroids = Centroids.from_exposures(exposures)
        self.assertEqual(
            centroids.gdf.shape, (0, 1)
        )  # there is an empty geometry column

    def test_read_write_hdf5(self):
        tmpfile = Path("test_write_hdf5.out.hdf5")
        crs = DEF_CRS
        centroids_w = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=crs)
        centroids_w.write_hdf5(tmpfile)
        centroids_r = Centroids.from_hdf5(tmpfile)
        self.assertTrue(centroids_w == centroids_r)
        tmpfile.unlink()

    def test_from_hdf5_nonexistent_file(self):
        """Test raising FileNotFoundError when creating Centroids object from a nonexistent HDF5 file"""
        file_name = "/path/to/nonexistentfile.h5"
        # prescribe that file does not exist
        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                Centroids.from_hdf5(file_name)


class TestCentroidsMethods(unittest.TestCase):
    """Test Centroids methods"""

    def setUp(self):
        self.centr = Centroids(lat=VEC_LAT, lon=VEC_LON, crs=TEST_CRS)

    def test_select_pass(self):
        """Test Centroids.select method"""
        region_id = np.zeros(VEC_LAT.size)
        region_id[[2, 4]] = 10
        centr = Centroids(lat=VEC_LAT, lon=VEC_LON, region_id=region_id)

        fil_centr = centr.select(reg_id=10)
        self.assertIsInstance(fil_centr, Centroids)
        self.assertEqual(fil_centr.size, 2)
        self.assertEqual(fil_centr.lat[0], VEC_LAT[2])
        self.assertEqual(fil_centr.lat[1], VEC_LAT[4])
        self.assertEqual(fil_centr.lon[0], VEC_LON[2])
        self.assertEqual(fil_centr.lon[1], VEC_LON[4])
        self.assertTrue(np.array_equal(fil_centr.region_id, np.ones(2) * 10))

    def test_select_extent_pass(self):
        """Test select extent"""
        centr = Centroids(
            lat=np.array([-5, -3, 0, 3, 5]),
            lon=np.array([-180, -175, -170, 170, 175]),
            region_id=np.zeros(5),
        )
        ext_centr = centr.select(extent=[-175, -170, -5, 5])
        self.assertIsInstance(ext_centr, Centroids)
        np.testing.assert_array_equal(ext_centr.lon, np.array([-175, -170]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-3, 0]))

        # Cross antimeridian, version 1
        ext_centr = centr.select(extent=[170, -175, -5, 5])
        np.testing.assert_array_equal(ext_centr.lon, np.array([-180, -175, 170, 175]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-5, -3, 3, 5]))

        # Cross antimeridian, version 2
        ext_centr = centr.select(extent=[170, 185, -5, 5])
        np.testing.assert_array_equal(ext_centr.lon, np.array([-180, -175, 170, 175]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-5, -3, 3, 5]))

    def test_append_pass(self):
        """Append points"""
        centr = self.centr
        centr_bis = Centroids(
            lat=np.array([1, 2, 3]), lon=np.array([4, 5, 6]), crs=DEF_CRS
        )
        with self.assertRaises(ValueError):
            # Different crs
            centr_bis.to_crs(ALT_CRS).append(centr)
        centr_bis.append(centr)
        self.assertAlmostEqual(centr_bis.lat[0], 1)
        self.assertAlmostEqual(centr_bis.lat[1], 2)
        self.assertAlmostEqual(centr_bis.lat[2], 3)
        self.assertAlmostEqual(centr_bis.lon[0], 4)
        self.assertAlmostEqual(centr_bis.lon[1], 5)
        self.assertAlmostEqual(centr_bis.lon[2], 6)
        self.assertTrue(np.array_equal(centr_bis.lat[3:], centr.lat))
        self.assertTrue(np.array_equal(centr_bis.lon[3:], centr.lon))

    def test_append(self):
        lat2, lon2 = np.array([6, 7, 8, 9, 10]), np.array([6, 7, 8, 9, 10])
        newcentr = Centroids(lat=lat2, lon=lon2)
        newcentr.append(self.centr)
        self.assertTrue(newcentr.size == len(self.centr.lon) + len(lon2))
        np.testing.assert_array_equal(
            newcentr.lon, np.concatenate([lon2, self.centr.lon])
        )
        np.testing.assert_array_equal(
            newcentr.lat, np.concatenate([lat2, self.centr.lat])
        )

    def test_append_dif_crs(self):
        lat2, lon2 = np.array([0, 0, 1, 2, 3, 4, 5]), np.array([0, 0, 1, 2, 3, 4, 5])
        centr2 = Centroids(lat=lat2, lon=lon2, crs="epsg:3857")

        # appending differing crs is not provided/possible
        with self.assertRaises(ValueError):
            self.centr.append(centr2)

    def test_append_multiple_arguments(self):
        """Test passing append() multiple arguments in the form of a list of Centroids."""
        # create a single centroid
        lat, lon = np.array([1, 2]), np.array([1, 2])
        centr = Centroids(lat=lat, lon=lon)
        # create a list of centroids
        coords = [(np.array([3, 4]), np.array([3, 4]))]
        centroids_list = [Centroids(lat=lat, lon=lon) for lat, lon in coords]

        centr.append(*centroids_list)

        np.testing.assert_array_equal(centr.lat, [1, 2, 3, 4])
        np.testing.assert_array_equal(centr.lon, [1, 2, 3, 4])

    def test_remove_duplicate_pass(self):
        """Test remove_duplicate_points"""
        centr = Centroids(
            lat=np.hstack([VEC_LAT, VEC_LAT]),
            lon=np.hstack([VEC_LON, VEC_LON]),
            crs=TEST_CRS,
        )
        self.assertTrue(centr.gdf.shape[0] == 2 * self.centr.gdf.shape[0])
        rem_centr = Centroids.remove_duplicate_points(centr)
        self.assertIsInstance(rem_centr, Centroids)
        self.assertTrue(self.centr == rem_centr)

    def test_remove_duplicates_dif_on_land(self):
        ### We currently expect that only the geometry of the gdf defines duplicates.
        ### If one geometry is duplicated with differences in other attributes e.g. on_land
        ### they get removed nevertheless. Only the first occurrence will be part of the new object
        ### this test is only here to guarantee this behaviour
        lat, lon = np.array([0, 0, 1, 2, 3, 4, 5]), np.array([0, 0, 1, 2, 3, 4, 5])
        centr = Centroids(lat=lat, lon=lon, on_land=[True] + [False] * 6)
        centr_subset = centr.remove_duplicate_points()
        # new object created
        self.assertFalse(centr == centr_subset)
        self.assertIsNot(centr, centr_subset)
        # duplicates removed
        self.assertTrue(centr_subset.size == len(lat) - 1)
        self.assertTrue(np.all(centr_subset.shape == (len(lat) - 1, len(lon) - 1)))
        np.testing.assert_array_equal(centr_subset.lon, np.unique(lon))
        np.testing.assert_array_equal(centr_subset.lat, np.unique(lat))
        # only first on_land (True) is selected
        self.assertTrue(centr_subset.on_land[0])

    def test_union(self):
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        lat2, lon2 = np.array([2, 3]), np.array([-2, 3])
        on_land2 = np.array([False, False])
        cent2 = Centroids(lat=lat2, lon=lon2, on_land=on_land2)

        lat3, lon3 = np.array([-1, -2]), np.array([1, 2])
        cent3 = Centroids(lat=lat3, lon=lon3)

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, np.concatenate([lat, lat2]))
        np.testing.assert_array_equal(cent.lon, np.concatenate([lon, lon2]))
        np.testing.assert_array_equal(cent.on_land, np.concatenate([on_land, on_land2]))

        cent = cent1.union(cent1, cent2)
        np.testing.assert_array_equal(cent.lat, np.concatenate([lat, lat2]))
        np.testing.assert_array_equal(cent.lon, np.concatenate([lon, lon2]))
        np.testing.assert_array_equal(cent.on_land, np.concatenate([on_land, on_land2]))

        cent = Centroids.union(cent1)
        np.testing.assert_array_equal(cent.lat, cent1.lat)
        np.testing.assert_array_equal(cent.lon, cent1.lon)
        np.testing.assert_array_equal(cent.on_land, cent1.on_land)

        cent = cent1.union(cent1)
        np.testing.assert_array_equal(cent.lat, cent1.lat)
        np.testing.assert_array_equal(cent.lon, cent1.lon)
        np.testing.assert_array_equal(cent.on_land, cent1.on_land)

        # if attributes are not part in one of the centroid objects it will be added as None in the union
        cent = Centroids.union(cent1, cent2, cent3)
        np.testing.assert_array_equal(cent.lat, np.concatenate([lat, lat2, lat3]))
        np.testing.assert_array_equal(cent.lon, np.concatenate([lon, lon2, lon3]))
        np.testing.assert_array_equal(
            cent.on_land, np.concatenate([on_land, on_land2, [None, None]])
        )

    def test_select_pass(self):
        """Test Centroids.select method"""
        region_id = np.zeros(VEC_LAT.size)
        region_id[[2, 4]] = 10
        centr = Centroids(lat=VEC_LAT, lon=VEC_LON, region_id=region_id)

        fil_centr = centr.select(reg_id=10)
        self.assertEqual(fil_centr.size, 2)
        self.assertEqual(fil_centr.lat[0], VEC_LAT[2])
        self.assertEqual(fil_centr.lat[1], VEC_LAT[4])
        self.assertEqual(fil_centr.lon[0], VEC_LON[2])
        self.assertEqual(fil_centr.lon[1], VEC_LON[4])
        self.assertTrue(np.array_equal(fil_centr.region_id, np.ones(2) * 10))

    def test_select_extent_pass(self):
        """Test select extent"""
        centr = Centroids(
            lat=np.array([-5, -3, 0, 3, 5]),
            lon=np.array([-180, -175, -170, 170, 175]),
            region_id=np.zeros(5),
        )
        ext_centr = centr.select(extent=[-175, -170, -5, 5])
        np.testing.assert_array_equal(ext_centr.lon, np.array([-175, -170]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-3, 0]))

        # Cross antimeridian, version 1
        ext_centr = centr.select(extent=[170, -175, -5, 5])
        np.testing.assert_array_equal(ext_centr.lon, np.array([-180, -175, 170, 175]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-5, -3, 3, 5]))

        # Cross antimeridian, version 2
        ext_centr = centr.select(extent=[170, 185, -5, 5])
        np.testing.assert_array_equal(ext_centr.lon, np.array([-180, -175, 170, 175]))
        np.testing.assert_array_equal(ext_centr.lat, np.array([-5, -3, 3, 5]))

    def test_get_meta(self):
        """
        Test that the `get_meta` method correctly generates metadata
        for a raster with a specified resolution.
        """
        # Create centroids with specified resolution
        lon = np.array([-10.0, -20.0, -30.0])
        lat = np.array([10.0, 20.0, 30.0])
        centroids = Centroids(lat=lat, lon=lon, crs=DEF_CRS)

        # Get metadata
        meta = centroids.get_meta()

        # Check metadata
        expected_meta = dict(
            crs=DEF_CRS,
            height=3,
            width=3,
            transform=Affine(
                10,
                0,
                -35,
                0,
                -10,
                35,
            ),
        )
        self.assertEqual(meta["height"], expected_meta["height"])
        self.assertEqual(meta["width"], expected_meta["width"])
        self.assertTrue(u_coord.equal_crs(meta["crs"], expected_meta["crs"]))
        self.assertTrue(meta["transform"].almost_equals(expected_meta["transform"]))

    def test_get_closest_point(self):
        """Test get_closest_point"""
        for n, (lat, lon) in enumerate(LATLON):
            x, y, idx = self.centr.get_closest_point(lon * 0.99, lat * 1.01)
            self.assertAlmostEqual(x, lon)
            self.assertAlmostEqual(y, lat)
            self.assertEqual(idx, n)
            self.assertEqual(self.centr.lon[n], x)
            self.assertEqual(self.centr.lat[n], y)

    def test_get_closest_point(self):
        """Test get_closest_point"""
        for y_sign in [1, -1]:
            meta = {
                "width": 10,
                "height": 20,
                "transform": rasterio.Affine(
                    0.5, 0, 0.1, 0, y_sign * 0.6, y_sign * (-0.3)
                ),
                "crs": DEF_CRS,
            }
            centr_ras = Centroids.from_meta(meta=meta)

            test_data = np.array(
                [
                    [0.4, 0.1, 0.35, 0.0, 0],
                    [-0.1, 0.2, 0.35, 0.0, 0],
                    [2.2, 0.1, 2.35, 0.0, 4],
                    [1.4, 2.5, 1.35, 2.4, 42],
                    [5.5, -0.1, 4.85, 0.0, 9],
                ]
            )
            test_data[:, [1, 3]] *= y_sign
            for x_in, y_in, x_out, y_out, idx_out in test_data:
                x, y, idx = centr_ras.get_closest_point(x_in, y_in)
                self.assertEqual(x, x_out)
                self.assertEqual(y, y_out)
                self.assertEqual(idx, idx_out)
                self.assertEqual(centr_ras.lon[idx], x)
                self.assertEqual(centr_ras.lat[idx], y)

        centr_ras = Centroids(
            lat=np.array([0, 0.2, 0.7]), lon=np.array([-0.4, 0.2, 1.1])
        )
        x, y, idx = centr_ras.get_closest_point(0.1, 0.0)
        self.assertEqual(x, 0.2)
        self.assertEqual(y, 0.2)
        self.assertEqual(idx, 1)

    def test_dist_coast_pass(self):
        """Test get_dist_coast"""
        dist_coast = self.centr.get_dist_coast()
        # Just checking that the output doesnt change over time.
        REF_VALUES = np.array(
            [
                860.0,
                200.0,
                25610.0,
                1000.0,
                4685.0,
                507500.0,
                500.0,
                150500.0,
            ]
        )
        self.assertIsInstance(dist_coast, np.ndarray)
        np.testing.assert_allclose(dist_coast, REF_VALUES, atol=1.0)

    def test_dist_coast_pass_raster(self):
        """Test get_dist_coast for centroids derived from a raster file"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        dist_coast = centr_ras.get_dist_coast()
        self.assertLess(abs(dist_coast[0] - 117000), 1000)
        self.assertLess(abs(dist_coast[-1] - 104000), 1000)

    def test_area_pass(self):
        """Test set_area"""
        ulx, xres, lrx = 60, 1, 90
        uly, yres, lry = 0, 1, 20
        xx, yy = np.meshgrid(
            np.arange(ulx + xres / 2, lrx, xres), np.arange(uly + yres / 2, lry, yres)
        )
        vec_data = gpd.GeoDataFrame(
            {
                "geometry": [
                    Point(xflat, yflat)
                    for xflat, yflat in zip(xx.flatten(), yy.flatten())
                ],
                "lon": xx.flatten(),
                "lat": yy.flatten(),
            },
            crs={"proj": "cea"},
        )
        centr = Centroids.from_geodataframe(vec_data)
        area_pixel = centr.get_area_pixel()
        self.assertTrue(np.allclose(area_pixel, np.ones(centr.size)))

    def test_area_pass_raster(self):
        """Test set_area"""
        window_size = (0, 0, 2, 3)
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(*window_size))
        area_pixel = centr_ras.get_area_pixel()

        # Result in the crs of the test file (ESPG:4326)
        # This is a wrong result as it should be projected to CEA (for correct area)
        res = 0.009000000000000341
        self.assertFalse(
            np.allclose(area_pixel, np.ones(window_size[2] * window_size[3]) * res**2)
        )

        # Correct result in CEA results in unequal pixel area
        test_area = np.array(
            [
                981010.32497514,
                981010.3249724,
                981037.92674855,
                981037.92674582,
                981065.50487659,
                981065.50487385,
            ]
        )
        np.testing.assert_allclose(area_pixel, test_area)

    def test_equal_pass(self):
        """Test equal"""
        centr_list = [
            Centroids(lat=VEC_LAT, lon=VEC_LON, crs=DEF_CRS),
            Centroids(lat=VEC_LAT, lon=VEC_LON, crs=ALT_CRS),
            Centroids(lat=VEC_LAT + 1, lon=VEC_LON + 1),
        ]
        for centr1, centr2 in itertools.combinations(centr_list, 2):
            self.assertFalse(centr2 == centr1)
            self.assertFalse(centr1 == centr2)
            self.assertTrue(centr1 == centr1)
            self.assertTrue(centr2 == centr2)

    def test_plot(self):
        "Test Centroids.plot()"
        centr = Centroids(
            lat=np.array([-5, -3, 0, 3, 5]),
            lon=np.array([-180, -175, -170, 170, 175]),
            region_id=np.zeros(5),
            crs=DEF_CRS,
        )
        centr.plot()

    def test_plot_non_def_crs(self):
        "Test Centroids.plot() with non-default CRS"
        centr = Centroids(
            lat=np.array([10.0, 20.0, 30.0]),
            lon=np.array([-10.0, -20.0, -30.0]),
            region_id=np.zeros(3),
            crs="epsg:32632",
        )
        centr.plot()


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsData)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReaderWriter)
    )
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsMethods))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
