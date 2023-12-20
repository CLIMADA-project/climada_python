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
from climada.util.constants import HAZ_TEMPLATE_XLS
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

    def test_from_csv(self):
        """Read a centroid csv file correctly"""
        tmpfile = Path('test_write_csv.csv')
        test_crs = CRS.from_user_input(4326).to_wkt()
        sample_data = [
            ["lon", "lat", "crs"],
            [32.57, -25.95, test_crs],
            [32.46, -25.97],
            [34.84, -19.83],
            [39.24, -15.13],
            [33.47, -19.12],
            [40.68, -14.56],
            [36.89, -17.88],
            [33.58, -16.17],
            [33.64, -25.04],
            [36.98, -15.46],
            [35.34, -23.87],
            [35.24, -13.3],
            [40.48, -12.96],
            [34.73, -19.62],
            [39.91, -16.23]
            ]
        
        with open(tmpfile, 'w') as temp_csv:
            writer = csv.writer(temp_csv)
            writer.writerows(sample_data)

        centroids = Centroids.from_csv(tmpfile)

        n_centroids = 15
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -16.23)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 39.91)
        self.assertEqual(centroids.crs, 'epsg:4326')

        Path(tmpfile).unlink()
    
    def test_from_excel(self):
        """Read a centroid excel file correctly."""
        tmpfile = Path('test_write_excel.xlsx')
        test_crs = CRS.from_user_input(4326).to_wkt()
        sample_data = [
            [32.57, -25.95, test_crs],
            [32.46, -25.97],
            [34.84, -19.83],
            [39.24, -15.13],
            [33.47, -19.12],
            [40.68, -14.56],
            [36.89, -17.88],
            [33.58, -16.17],
            [33.64, -25.04],
            [36.98, -15.46],
            [35.34, -23.87],
            [35.24, -13.3],
            [40.48, -12.96],
            [34.73, -19.62],
            [39.91, -16.23]
            ]

        df = pd.DataFrame(sample_data, columns = ["lon", "lat", "crs"],)

        df.to_excel(tmpfile, sheet_name = 'centroids', index=False)
        
        centroids = Centroids.from_excel(file_path=tmpfile, sheet_name='centroids')

        n_centroids = 15
        self.assertEqual(centroids.coord.shape[0], n_centroids)
        self.assertEqual(centroids.coord.shape[1], 2)
        self.assertEqual(centroids.coord[0][0], -25.95)
        self.assertEqual(centroids.coord[0][1], 32.57)
        self.assertEqual(centroids.coord[n_centroids - 1][0], -16.23)
        self.assertEqual(centroids.coord[n_centroids - 1][1], 39.91)
        self.assertEqual(centroids.crs, 'epsg:4326')

        Path(tmpfile).unlink()

    def test_from_geodataframe(self):
        """Test that constructing a valid Centroids instance from gdf works."""
        crs = DEF_CRS
        lat = np.arange(170, 180)
        lon = np.arange(-50, -40)
        region_id = np.arange(1, 11)
        extra = np.repeat(str('a'), 10)

        gdf = gpd.GeoDataFrame({
            'geometry' : gpd.points_from_xy(lon, lat),
            'region_id' : region_id,
            'extra' : extra
        }, crs=crs)

        centroids = Centroids.from_geodataframe(gdf)

        for name, array in zip(['lat', 'lon', 'region_id'],
                                [lat, lon, region_id]):
            np.testing.assert_array_equal(array, getattr(centroids, name))
        np.testing.assert_array_equal(extra, centroids.gdf.extra.values)
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
