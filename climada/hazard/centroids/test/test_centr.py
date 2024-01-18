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
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import csv
from pyproj.crs.crs import CRS

from climada import CONFIG
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coord
from climada.entity import Exposures
from rasterio import Affine


class TestCentroidsData(unittest.TestCase):

    def test_init(self):
        # Creating Centroids with latitude and longitude arrays
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        centroids = Centroids(lat=lat, lon=lon)

        # Checking attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))

        # Creating Centroids with additional properties
        region_id = np.array([1, 2, 3])
        on_land = np.array([True, False, False])
        centroids = Centroids(lat=lat, lon=lon, region_id=region_id, on_land=on_land)

        # Checking additional attributes
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)

    def test_to_default_crs(self):
        # Creating Centroids with non-default CRS
        crs = 'epsg:32632'
        lat = np.array([-10, 0, 10])
        lon = np.array([-170, -150, -130])
        centroids = Centroids(lat=lat, lon=lon, crs=crs)

        self.assertTrue(u_coord.equal_crs(centroids.crs, 'epsg:32632'))

        # Transforming to default CRS
        centroids.to_default_crs()

        # Checking CRS after transformation
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))

    def test_to_crs(self):
        # Creating Centroids with non-default CRS
        crs = 'epsg:4326'
        lat = np.array([-10, 0, 10])
        lon = np.array([-170, -150, -130])
        centroids = Centroids(lat=lat, lon=lon, crs=crs)

        # Transforming to another CRS
        new_crs = 'epsg:3857'
        transformed_centroids = centroids.to_crs(new_crs)

        # Checking CRS after transformation
        self.assertTrue(u_coord.equal_crs(transformed_centroids.crs, new_crs))
        self.assertTrue(u_coord.equal_crs(centroids.crs, crs))

        # Checking coordinates after transformation
        expected_lat = np.array([-1118889.974858, 0., 1118889.9748585])
        expected_lon = np.array([-18924313.434857, -16697923.618991, -14471533.803126])
        np.testing.assert_array_almost_equal(transformed_centroids.lat, expected_lat)
        np.testing.assert_array_almost_equal(transformed_centroids.lon, expected_lon)


class TestCentroidsReader(unittest.TestCase):
    """Test read functions Centroids"""

    def test_from_csv_def_crs(self):
        """Read a centroid csv file correctly and use default CRS."""
        # Create temporary csv file containing centroids data
        tmpfile = Path('test_write_csv.csv')
        lat = np.array([0, 90, -90, 0, 0])
        lon = np.array([0, 0, 0, 180, -180])
        df = pd.DataFrame({'lat':lat, 'lon':lon})
        df.to_csv(tmpfile, index=False)

        # Read centroids using from_csv method
        centroids = Centroids.from_csv(tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, DEF_CRS)

        #delete file
        Path(tmpfile).unlink()

    def test_from_csv(self):
        """Read a centroid csv file which contains CRS information."""
        tmpfile = Path('test_write_csv.csv')
        lat = np.array([0, 20048966.1, -20048966, 0, 0])
        lon = np.array([0, 0, 0, 20037508.34, -20037508.34])
        region_id = np.array([1, 2, 3, 4, 5])
        on_land = np.array([True, False, False, True, True])
        df = pd.DataFrame({'lat':lat, 'lon':lon, 'region_id':region_id, 'on_land':on_land})
        df['crs'] = CRS.from_user_input(3857).to_wkt()
        df.to_csv(tmpfile, index=False)

        # Read centroids using from_csv method
        centroids = Centroids.from_csv(tmpfile)

        # Test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, 'epsg:3857')
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)

        # Delete file
        Path(tmpfile).unlink()

    def test_write_read_csv(self):
        """Write and read a Centroids CSV file correctly."""
        # Create Centroids with latitude and longitude arrays
        tmpfile = Path('test_write_csv.csv')
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        region_id = np.array([1, 2, 3])
        on_land = np.array([True, False, False])
        centroids_out = Centroids(lat=lat, lon=lon, region_id=region_id, on_land=on_land)

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

        #delete file
        Path(tmpfile).unlink()

    def test_from_excel_def_crs(self):
        """Read a centroid excel file correctly and use default CRS."""
        # Create temporary excel file containing centroids data
        tmpfile = Path('test_write_excel.xlsx')
        lat = np.array([0, 90, -90, 0, 0])
        lon = np.array([0, 0, 0, 180, -180])
        df = pd.DataFrame({'lat':lat, 'lon':lon})
        df.to_excel(tmpfile, sheet_name = 'centroids', index=False)

        # Read centroids using from_excel method
        centroids = Centroids.from_excel(file_path=tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, DEF_CRS)

        #delete file
        Path(tmpfile).unlink()

    def test_from_excel(self):
        """Read a centroid excel file correctly which contains CRS information."""
        # Create temporary excel file containing centroids data
        tmpfile = Path('test_write_excel.xlsx')
        lat = np.array([0, 20048966.1, -20048966, 0, 0])
        lon = np.array([0, 0, 0, 20037508.34, -20037508.34])
        region_id = np.array([1, 2, 3, 4, 5])
        on_land = np.array([True, False, False, True, True])
        df = pd.DataFrame({'lat':lat, 'lon':lon, 'region_id':region_id, 'on_land':on_land})
        df['crs'] = CRS.from_user_input(3857).to_wkt()
        df.to_excel(tmpfile, sheet_name = 'centroids', index=False)

        # Read centroids using from_excel method
        centroids = Centroids.from_excel(file_path=tmpfile)

        # test attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        self.assertEqual(centroids.crs, 'epsg:3857')
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)

        #delete file
        Path(tmpfile).unlink()

    def test_write_read_excel(self):
        """Write and read a Centroids Excel file correctly."""
        # Create Centroids with latitude and longitude arrays
        tmpfile = Path('test_write_excel.xlsx')
        lat = np.array([10.0, 20.0, 30.0])
        lon = np.array([-10.0, -20.0, -30.0])
        region_id = np.array([1, 2, 3])
        on_land = np.array([True, False, False])
        centroids_out = Centroids(lat=lat, lon=lon, region_id=region_id, on_land=on_land)

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

        #delete file
        Path(tmpfile).unlink()

    def test_from_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        crs = DEF_CRS
        lat = np.arange(170, 180)
        lon = np.arange(-50, -40)
        region_id = np.arange(1, 11)
        on_land = np.ones(10).astype(bool)
        extra = np.repeat(str('a'), 10)

        gdf = gpd.GeoDataFrame({
            'geometry' : gpd.points_from_xy(lon, lat),
            'region_id' : region_id,
            'on_land': on_land,
            'extra' : extra
        }, crs=crs)

        centroids = Centroids.from_geodataframe(gdf)

        for name, array in zip(['lat', 'lon', 'region_id', 'on_land'],
                                [lat, lon, region_id, on_land]):
            np.testing.assert_array_equal(array, getattr(centroids, name))
        self.assertTrue('extra' in centroids.gdf.columns)
        self.assertTrue(u_coord.equal_crs(centroids.crs, crs))

    def test_from_geodataframe_invalid(self):

        # Creating an invalid GeoDataFrame with geometries that are not points
        invalid_geometry_gdf = gpd.GeoDataFrame({
            'geometry': [
                 shapely.Point((2,2)),
                 shapely.Polygon([(0, 0), (1, 1), (1, 0), (0, 0)]),
                 shapely.LineString([(0, 1), (1, 0)])
                 ]
            })

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
        crs = 'epsg:32632'
        gdf = gpd.GeoDataFrame({
            'latitude' : lat,
            'longitude': lon,
            'value': value,
            'region_id': region_id,
            'on_land': on_land
        })
        exposures = Exposures(gdf, crs=crs)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        np.testing.assert_array_equal(centroids.lat, lat)
        np.testing.assert_array_equal(centroids.lon, lon)
        np.testing.assert_array_equal(centroids.region_id, region_id)
        np.testing.assert_array_equal(centroids.on_land, on_land)
        self.assertFalse(np.isin('value', centroids.gdf.columns))
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
        gdf = gpd.GeoDataFrame({
            'latitude' : lat,
            'longitude': lon,
            'value': value,
            'impf_tc': impf_TC,
            'centr_TC': centr_TC
        })
        exposures = Exposures(gdf)

        # Extract centroids from exposures
        centroids = Centroids.from_exposures(exposures)

        # Check attributes
        self.assertEqual(centroids.lat.tolist(), lat.tolist())
        self.assertEqual(centroids.lon.tolist(), lon.tolist())
        self.assertTrue(u_coord.equal_crs(centroids.crs, DEF_CRS))
        self.assertEqual(centroids.region_id, None)
        self.assertEqual(centroids.on_land, None)
        self.assertFalse(
            np.all(np.isin(
                ['value', 'impf_tc', 'centr_tc'], centroids.gdf.columns
                 ))
        )




class TestCentroidsWriter(unittest.TestCase):

    def test_read_write_hdf5(self):
        tmpfile = Path('test_write_hdf5.out.hdf5')
        lat = np.arange(0,10)
        lon = np.arange(-10,0)
        crs = DEF_CRS
        centroids_w = Centroids(lat=lat, lon=lon, crs=crs)
        centroids_w.write_hdf5(tmpfile)
        centroids_r = Centroids.from_hdf5(tmpfile)
        self.assertTrue(centroids_w == centroids_r)
        Path(tmpfile).unlink()


class TestCentroidsMethods(unittest.TestCase):

    def test_union(self):
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        lat2, lon2 = np.array([2, 3]), np.array([-2, 3])
        on_land2 = np.array([False, False])
        cent2 = Centroids(lat=lat2, lon=lon2, on_land=on_land2)

        lat3, lon3 = np.array([-1, -2]), np.array([1, 2])
        cent3 = Centroids(lat=lat3,lon=lon3)

        cent = cent1.union(cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = cent1.union(cent1, cent2)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3])
        np.testing.assert_array_equal(cent.on_land, [True, True, False, False])

        cent = Centroids.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = cent1.union(cent1)
        np.testing.assert_array_equal(cent.lat, [0, 1])
        np.testing.assert_array_equal(cent.lon, [0, -1])
        np.testing.assert_array_equal(cent.on_land, [True, True])

        cent = Centroids.union(cent1, cent2, cent3)
        np.testing.assert_array_equal(cent.lat, [0, 1, 2, 3, -1, -2])
        np.testing.assert_array_equal(cent.lon, [0, -1, -2, 3, 1, 2])

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
            height= 3,
            width= 3,
            transform=Affine(10, 0, -35,
                              0, -10, 35)
        )
        self.assertEqual(meta['height'], expected_meta['height'])
        self.assertEqual(meta['width'], expected_meta['width'])
        self.assertTrue(u_coord.equal_crs(meta['crs'], expected_meta['crs']))
        self.assertTrue(meta['transform'].almost_equals(expected_meta['transform']))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroidsData)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsReader))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsMethods))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
