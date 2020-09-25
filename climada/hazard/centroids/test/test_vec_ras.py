"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test CentroidsVector and CentroidsRaster classes.
"""
import os
from cartopy.io import shapereader
from fiona.crs import from_epsg
import geopandas as gpd
import unittest
import numpy as np
from rasterio.windows import Window
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from climada.hazard.centroids.centr import Centroids
from climada.util.constants import HAZ_DEMO_FL, DEF_CRS
from climada.util.coordinates import NE_EPSG, equal_crs

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

VEC_LON = np.array([
    -59.6250000000000, -59.6250000000000, -59.6250000000000, -59.5416666666667, -59.5416666666667,
    -59.4583333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333,
    -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333,
    -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333,
    -60.2083333333333, -60.1250000000000, -60.1250000000000, -60.1250000000000, -60.1250000000000,
    -60.1250000000000, -60.1250000000000, -60.1250000000000, -60.1250000000000, -60.1250000000000,
    -60.1250000000000, -60.1250000000000, -60.1250000000000, -60.1250000000000, -60.1250000000000,
    -60.1250000000000, -60.1250000000000, -60.0416666666667, -60.0416666666667, -60.0416666666667,
    -60.0416666666667, -60.0416666666667, -60.0416666666667, -60.0416666666667, -60.0416666666667,
    -60.0416666666667, -60.0416666666667, -60.0416666666667, -60.0416666666667, -60.0416666666667,
    -60.0416666666667, -60.0416666666667, -60.0416666666667, -59.9583333333333, -59.9583333333333,
    -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.9583333333333,
    -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.9583333333333,
    -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.9583333333333, -59.8750000000000,
    -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000,
    -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000,
    -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000, -59.8750000000000,
    -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667,
    -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667,
    -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667,
    -59.7916666666667, -59.7083333333333, -59.7083333333333, -59.7083333333333, -59.7083333333333,
    -59.7083333333333, -59.7083333333333, -59.7083333333333, -59.7083333333333, -59.7083333333333,
    -59.7083333333333, -59.7083333333333, -59.7083333333333, -59.7083333333333, -59.7083333333333,
    -59.7083333333333, -59.7083333333333, -59.6250000000000, -59.6250000000000, -59.6250000000000,
    -59.6250000000000, -59.6250000000000, -59.6250000000000, -59.6250000000000, -59.6250000000000,
    -59.6250000000000, -59.6250000000000, -59.6250000000000, -59.6250000000000, -59.6250000000000,
    -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.5416666666667,
    -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.5416666666667,
    -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.5416666666667, -59.4583333333333,
    -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.4583333333333,
    -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.4583333333333,
    -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.4583333333333, -59.3750000000000,
    -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000,
    -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000,
    -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000, -59.3750000000000,
    -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667,
    -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667,
    -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667, -59.2916666666667,
    -59.2916666666667, -59.2083333333333, -59.2083333333333, -59.2083333333333, -59.2083333333333,
    -59.2083333333333, -59.2083333333333, -59.2083333333333, -59.2083333333333, -59.2083333333333,
    -59.2083333333333, -59.2083333333333, -59.2083333333333, -59.2083333333333, -59.2083333333333,
    -59.2083333333333, -59.2083333333333, -59.1250000000000, -59.1250000000000, -59.1250000000000,
    -59.1250000000000, -59.1250000000000, -59.1250000000000, -59.1250000000000, -59.1250000000000,
    -59.1250000000000, -59.1250000000000, -59.1250000000000, -59.1250000000000, -59.1250000000000,
    -59.1250000000000, -59.1250000000000, -59.1250000000000, -59.0416666666667, -59.0416666666667,
    -59.0416666666667, -59.0416666666667, -59.0416666666667, -59.0416666666667, -59.0416666666667,
    -59.0416666666667, -59.0416666666667, -59.0416666666667, -59.0416666666667, -59.0416666666667,
    -59.0416666666667, -59.0416666666667, -59.0416666666667, -58.9583333333333, -58.9583333333333,
    -58.9583333333333, -58.9583333333333, -58.9583333333333, -58.9583333333333, -58.9583333333333,
    -58.9583333333333, -58.9583333333333, -58.9583333333333, -58.9583333333333, -58.9583333333333,
    -58.9583333333333, -58.9583333333333, -61.0416666666667, -61.0416666666667, -61.0416666666667,
    -61.0416666666667, -61.0416666666667, -61.0416666666667, -61.0416666666667, -60.6250000000000,
    -60.6250000000000, -60.6250000000000, -60.6250000000000, -60.6250000000000, -60.6250000000000,
    -60.6250000000000, -60.2083333333333, -60.2083333333333, -60.2083333333333, -60.2083333333333,
    -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.7916666666667, -59.3750000000000,
    -59.3750000000000, -59.3750000000000, -59.3750000000000, -58.9583333333333, -58.9583333333333,
    -58.9583333333333, -58.9583333333333, -58.5416666666667, -58.5416666666667, -58.5416666666667,
    -58.5416666666667, -58.5416666666667, -58.5416666666667, -58.5416666666667, -58.1250000000000,
    -58.1250000000000, -58.1250000000000, -58.1250000000000, -58.1250000000000, -58.1250000000000,
    -58.1250000000000,
])

VEC_LAT = np.array([
    13.125, 13.20833333, 13.29166667, 13.125, 13.20833333, 13.125, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667, 13.375,
    13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333,
    12.79166667, 12.875, 12.95833333, 13.04166667, 13.375, 13.45833333, 13.54166667, 13.625,
    13.70833333, 13.79166667, 12.54166667, 12.625, 12.70833333, 12.79166667, 12.875, 12.95833333,
    13.04166667, 13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667,
    12.54166667, 12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667,
    12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667,
    12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667,
    12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667,
    12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 13.79166667, 12.54166667,
    12.625, 12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333,
    13.29166667, 13.375, 13.45833333, 13.54166667, 13.625, 13.70833333, 12.54166667, 12.625,
    12.70833333, 12.79166667, 12.875, 12.95833333, 13.04166667, 13.125, 13.20833333, 13.29166667,
    13.375, 13.45833333, 13.54166667, 13.625, 11.875, 12.29166667, 12.70833333, 13.125,
    13.54166667, 13.95833333, 14.375, 11.875, 12.29166667, 12.70833333, 13.125, 13.54166667,
    13.95833333, 14.375, 11.875, 12.29166667, 13.95833333, 14.375, 11.875, 12.29166667,
    13.95833333, 14.375, 11.875, 12.29166667, 13.95833333, 14.375, 11.875, 12.29166667,
    13.95833333, 14.375, 11.875, 12.29166667, 12.70833333, 13.125, 13.54166667, 13.95833333,
    14.375, 11.875, 12.29166667, 12.70833333, 13.125, 13.54166667, 13.95833333, 14.375
])

class TestVector(unittest.TestCase):
    """Test CentroidsVector class"""

    @staticmethod
    def data_vector():
        vec_data = gpd.GeoDataFrame(crs={'init': 'epsg:32632'})
        vec_data['geometry'] = list(zip(VEC_LON, VEC_LAT))
        vec_data['geometry'] = vec_data['geometry'].apply(Point)
        vec_data['lon'] = VEC_LON
        vec_data['lat'] = VEC_LAT
        vec_data['value'] = np.arange(VEC_LAT.size) + 100
        return vec_data.lat.values, vec_data.lon.values, vec_data.geometry

    def test_set_lat_lon_pass(self):
        """Test set_lat_lon"""
        centr = Centroids()
        centr.set_lat_lon(VEC_LAT, VEC_LON)
        self.assertTrue(np.allclose(centr.lat, VEC_LAT))
        self.assertTrue(np.allclose(centr.lon, VEC_LON))
        self.assertEqual(centr.crs, DEF_CRS)
        self.assertEqual(centr.geometry.crs, DEF_CRS)
        self.assertEqual(centr.geometry.size, 0)

        centr.set_area_pixel()
        self.assertEqual(centr.geometry.size, centr.lat.size)

    def test_ne_crs_geom_pass(self):
        """Test _ne_crs_geom"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        xy_vec = centr._ne_crs_geom()
        lon, lat = xy_vec.geometry[:].x.values, xy_vec.geometry[:].y.values
        self.assertAlmostEqual(4.51072194, lon[0])
        self.assertAlmostEqual(0.00011838, lat[0])
        self.assertAlmostEqual(4.5107354, lon[-1])
        self.assertAlmostEqual(0.0001297, lat[-1])

    def test_dist_coast_pass(self):
        """Test set_dist_coast"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        centr.set_dist_coast()
        self.assertAlmostEqual(2594.2070842031694, centr.dist_coast[1])
        self.assertAlmostEqual(166295.87602398323, centr.dist_coast[-2])

    def test_region_id_pass(self):
        """Test set_region_id"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        centr.set_region_id()
        self.assertEqual(np.count_nonzero(centr.region_id), 6)
        self.assertEqual(centr.region_id[0], 52)  # 052 for barbados

    def test_on_land(self):
        """Test set_on_land"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        centr.set_on_land()
        centr.set_region_id()
        centr.region_id[centr.region_id > 0] = 1
        self.assertTrue(np.array_equal(centr.on_land.astype(int),
                                       centr.region_id))

    def test_remove_duplicate_pass(self):
        """Test remove_duplicate_points"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        # create duplicates manually:
        centr.geometry.values[100] = centr.geometry.values[101]
        centr.geometry.values[120] = centr.geometry.values[101]
        centr.geometry.values[5] = Point([-59.7, 12.5])
        centr.geometry.values[133] = Point([-59.7, 12.5])
        centr.geometry.values[121] = Point([-59.7, 12.5])
        self.assertEqual(centr.size, 296)
        rem_centr = centr.remove_duplicate_points()
        self.assertEqual(centr.size, 296)
        self.assertEqual(rem_centr.size, 292)  # 5 centroids removed
        rem2_centr = rem_centr.remove_duplicate_points()
        self.assertEqual(rem_centr.size, 292)
        self.assertEqual(rem2_centr.size, 292)

    def test_area_pass(self):
        """Test set_area"""
        ulx, xres, lrx = 60, 1, 90
        uly, yres, lry = 0, 1, 20
        xx, yy = np.meshgrid(np.arange(ulx + xres / 2, lrx, xres),
                             np.arange(uly + yres / 2, lry, yres))
        vec_data = gpd.GeoDataFrame(crs={'proj': 'cea'})
        vec_data['geometry'] = list(zip(xx.flatten(), yy.flatten()))
        vec_data['geometry'] = vec_data['geometry'].apply(Point)
        vec_data['lon'] = xx.flatten()
        vec_data['lat'] = yy.flatten()

        centr = Centroids()
        centr.set_lat_lon(vec_data.lat.values, vec_data.lon.values)
        centr.geometry = vec_data.geometry
        centr.set_area_pixel()
        self.assertTrue(np.allclose(centr.area_pixel, np.ones(centr.size)))

    def test_size_pass(self):
        """Test size property"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        self.assertEqual(centr.size, 296)

    def test_get_closest_point(self):
        """Test get_closest_point"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        x, y, idx = centr.get_closest_point(-58.13, 14.38)
        self.assertAlmostEqual(x, -58.125)
        self.assertAlmostEqual(y, 14.375)
        self.assertEqual(idx, 295)
        self.assertEqual(centr.lon[idx], x)
        self.assertEqual(centr.lat[idx], y)

    def test_set_lat_lon_to_meta_pass(self):
        """Test set_lat_lon_to_meta"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}

        centr.set_lat_lon_to_meta()
        self.assertEqual(centr.meta['crs'], {'init': 'epsg:4326'})
        self.assertEqual(centr.meta['width'], 36)
        self.assertEqual(centr.meta['height'], 31)
        self.assertEqual(centr.meta['transform'][1], 0.0)
        self.assertEqual(centr.meta['transform'][3], 0.0)
        self.assertAlmostEqual(centr.meta['transform'][0], 0.08333333)
        self.assertAlmostEqual(centr.meta['transform'][2], -61.08333333)
        self.assertAlmostEqual(centr.meta['transform'][4], 0.08333333)
        self.assertAlmostEqual(centr.meta['transform'][5], 11.83333333)

    def test_get_pixel_polygons_pass(self):
        """Test calc_pixels_polygons"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        poly = centr.calc_pixels_polygons()
        self.assertIsInstance(poly[0], Polygon)
        self.assertTrue(np.allclose(poly.centroid[:].y.values, centr.lat))
        self.assertTrue(np.allclose(poly.centroid[:].x.values, centr.lon))

    def test_area_approx(self):
        """Test set_area_approx"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr.geometry.crs = {'init': 'epsg:4326'}
        with self.assertRaises(ValueError):
            centr.set_area_approx()

    def test_append_pass(self):
        """Append points"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr_bis = Centroids()
        centr_bis.set_lat_lon(np.array([1, 2, 3]), np.array([4, 5, 6]))
        with self.assertRaises(ValueError):
            centr_bis.append(centr)
        centr.geometry.crs = {'init': 'epsg:4326'}
        centr_bis.append(centr)
        self.assertAlmostEqual(centr_bis.lat[0], 1)
        self.assertAlmostEqual(centr_bis.lat[1], 2)
        self.assertAlmostEqual(centr_bis.lat[2], 3)
        self.assertAlmostEqual(centr_bis.lon[0], 4)
        self.assertAlmostEqual(centr_bis.lon[1], 5)
        self.assertAlmostEqual(centr_bis.lon[2], 6)
        self.assertTrue(np.array_equal(centr_bis.lat[3:], centr.lat))
        self.assertTrue(np.array_equal(centr_bis.lon[3:], centr.lon))

    def test_equal_pass(self):
        """Test equal"""
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = self.data_vector()
        centr_bis = Centroids()
        centr_bis.set_lat_lon(np.array([1, 2, 3]), np.array([4, 5, 6]))
        self.assertFalse(centr.equal(centr_bis))
        self.assertFalse(centr_bis.equal(centr))
        self.assertTrue(centr_bis.equal(centr_bis))
        self.assertTrue(centr.equal(centr))


class TestRaster(unittest.TestCase):
    """Test CentroidsRaster class"""

    def test_set_raster_pxl_pass(self):
        """Test set_raster_from_pxl_bounds from pixel borders"""
        centr = Centroids()
        xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon = 10, 5, -0.5, 0.2, 20, 25
        centr.set_raster_from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon)
        self.assertEqual(centr.meta['crs'], DEF_CRS)
        self.assertEqual(centr.meta['width'], n_lon)
        self.assertEqual(centr.meta['height'], n_lat)
        self.assertAlmostEqual(centr.meta['transform'][0], d_lon)
        self.assertAlmostEqual(centr.meta['transform'][1], 0.0)
        self.assertAlmostEqual(centr.meta['transform'][2], xo_lon)
        self.assertAlmostEqual(centr.meta['transform'][3], 0.0)
        self.assertAlmostEqual(centr.meta['transform'][4], d_lat)
        self.assertAlmostEqual(centr.meta['transform'][5], xf_lat)
        self.assertTrue('lat' in centr.__dict__.keys())
        self.assertTrue('lon' in centr.__dict__.keys())

    def test_set_raster_pnt_pass(self):
        """Test set_raster_from_pnt_bounds from point borders"""
        centr = Centroids()
        left, bottom, right, top = 5, 0, 10, 10
        centr.set_raster_from_pnt_bounds((left, bottom, right, top), 0.2)
        self.assertEqual(centr.meta['crs'], DEF_CRS)
        self.assertEqual(centr.meta['width'], 26)
        self.assertEqual(centr.meta['height'], 51)
        self.assertAlmostEqual(centr.meta['transform'][0], 0.2)
        self.assertAlmostEqual(centr.meta['transform'][1], 0.0)
        self.assertAlmostEqual(centr.meta['transform'][2], 5 - 0.2 / 2)
        self.assertAlmostEqual(centr.meta['transform'][3], 0.0)
        self.assertAlmostEqual(centr.meta['transform'][4], -0.2)
        self.assertAlmostEqual(centr.meta['transform'][5], 10 + 0.2 / 2)
        self.assertTrue('lat' in centr.__dict__.keys())
        self.assertTrue('lon' in centr.__dict__.keys())

    def test_read_all_pass(self):
        """Test centr_ras data"""
        centr_ras = Centroids()
        inten_ras = centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        self.assertAlmostEqual(centr_ras.meta['crs'], DEF_CRS)
        self.assertAlmostEqual(centr_ras.meta['transform'].c, -69.33714959699981)
        self.assertAlmostEqual(centr_ras.meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(centr_ras.meta['transform'].b, 0.0)
        self.assertAlmostEqual(centr_ras.meta['transform'].f, 10.42822096697894)
        self.assertAlmostEqual(centr_ras.meta['transform'].d, 0.0)
        self.assertAlmostEqual(centr_ras.meta['transform'].e, -0.009000000000000341)
        self.assertEqual(centr_ras.meta['height'], 60)
        self.assertEqual(centr_ras.meta['width'], 50)
        self.assertEqual(inten_ras.shape, (1, 60 * 50))

    def test_ne_crs_geom_pass(self):
        """Test _ne_crs_geom"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.meta['crs'] = {'init': 'epsg:32632'}

        xy_vec = centr_ras._ne_crs_geom()
        x_vec, y_vec = xy_vec.geometry[:].x.values, xy_vec.geometry[:].y.values
        self.assertAlmostEqual(4.51063496489, x_vec[0])
        self.assertAlmostEqual(9.40153761711e-05, y_vec[0])
        self.assertAlmostEqual(4.51063891581, x_vec[-1])
        self.assertAlmostEqual(8.92260922066e-05, y_vec[-1])

    def test_region_id_pass(self):
        """Test set_dist_coast"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_region_id()
        self.assertEqual(centr_ras.region_id.size, centr_ras.size)
        self.assertTrue(np.array_equal(np.unique(centr_ras.region_id), np.array([862])))

    def test_set_geometry_points_pass(self):
        """Test set_geometry_points"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_geometry_points()
        x_flat = np.arange(-69.3326495969998, -68.88264959699978, 0.009000000000000341)
        y_flat = np.arange(10.423720966978939, 9.883720966978919, -0.009000000000000341)
        x_grid, y_grid = np.meshgrid(x_flat, y_flat)
        self.assertTrue(np.allclose(x_grid.flatten(), centr_ras.lon))
        self.assertTrue(np.allclose(y_grid.flatten(), centr_ras.lat))

    def test_dist_coast_pass(self):
        """Test set_region_id"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_dist_coast()
        centr_ras.check()
        self.assertTrue(abs(centr_ras.dist_coast[0] - 117000) < 1000)
        self.assertTrue(abs(centr_ras.dist_coast[-1] - 104000) < 1000)

    def test_on_land(self):
        """Test set_on_land"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_on_land()
        centr_ras.check()
        self.assertTrue(np.array_equal(centr_ras.on_land, np.ones(60 * 50, bool)))

    def test_area_pass(self):
        """Test set_area"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.meta['crs'] = {'proj': 'cea'}
        centr_ras.set_area_pixel()
        centr_ras.check()
        self.assertTrue(
            np.allclose(centr_ras.area_pixel,
                        np.ones(60 * 50) * 0.009000000000000341 * 0.009000000000000341))

    def test_area_approx(self):
        """Test set_area_approx"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_area_approx()
        approx_dim = (centr_ras.meta['transform'][0] * 111 * 1000
                      * centr_ras.meta['transform'][0] * 111 * 1000)
        self.assertEqual(centr_ras.area_pixel.size, centr_ras.size)
        self.assertEqual(np.unique(centr_ras.area_pixel).size, 60)
        self.assertTrue(np.array_equal((approx_dim / np.unique(centr_ras.area_pixel)).astype(int),
                                       np.ones(60)))

    def test_size_pass(self):
        """Test size property"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        self.assertEqual(centr_ras.size, 50 * 60)

    def test_get_closest_point(self):
        """Test get_closest_point"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        x, y, idx = centr_ras.get_closest_point(-69.334, 10.42)
        self.assertAlmostEqual(x, -69.3326495969998)
        self.assertAlmostEqual(y, 10.423720966978939)
        self.assertEqual(idx, 0)
        centr_ras.set_meta_to_lat_lon()
        self.assertEqual(centr_ras.lon[idx], x)
        self.assertEqual(centr_ras.lat[idx], y)

    def test_set_meta_to_lat_lon_pass(self):
        """Test set_meta_to_lat_lon by using its inverse set_lat_lon_to_meta"""
        lat, lon, geometry = TestVector.data_vector()

        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = lat, lon, geometry

        centr.set_lat_lon_to_meta()
        meta = centr.meta
        centr.set_meta_to_lat_lon()
        self.assertEqual(centr.meta, meta)
        self.assertAlmostEqual(lat.max(), centr.lat.max(), 6)
        self.assertAlmostEqual(lat.min(), centr.lat.min(), 6)
        self.assertAlmostEqual(lon.max(), centr.lon.max(), 6)
        self.assertAlmostEqual(lon.min(), centr.lon.min(), 6)
        self.assertAlmostEqual(np.diff(centr.lon).max(), meta['transform'][0])
        self.assertAlmostEqual(np.diff(centr.lat).max(), meta['transform'][4])
        self.assertEqual(geometry.crs, centr.geometry.crs)

    def test_append_equal_pass(self):
        """Append raster"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_bis = Centroids()
        centr_bis.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_bis.append(centr_ras)
        self.assertAlmostEqual(centr_bis.meta['crs'], DEF_CRS)
        self.assertAlmostEqual(centr_bis.meta['transform'].c, -69.33714959699981)
        self.assertAlmostEqual(centr_bis.meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(centr_bis.meta['transform'].b, 0.0)
        self.assertAlmostEqual(centr_bis.meta['transform'].f, 10.42822096697894)
        self.assertAlmostEqual(centr_bis.meta['transform'].d, 0.0)
        self.assertAlmostEqual(centr_bis.meta['transform'].e, -0.009000000000000341)
        self.assertEqual(centr_bis.meta['height'], 60)
        self.assertEqual(centr_bis.meta['width'], 50)

    def test_append_diff_pass(self):
        """Append raster"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_bis = Centroids()
        centr_bis.set_raster_file(HAZ_DEMO_FL, window=Window(51, 61, 10, 10))
        centr_bis.append(centr_ras)
        self.assertAlmostEqual(centr_bis.meta['crs'], DEF_CRS)
        self.assertAlmostEqual(centr_bis.meta['transform'].c, -69.33714959699981)
        self.assertAlmostEqual(centr_bis.meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(centr_bis.meta['transform'].b, 0.0)
        self.assertAlmostEqual(centr_bis.meta['transform'].f, 10.42822096697894)
        self.assertAlmostEqual(centr_bis.meta['transform'].d, 0.0)
        self.assertAlmostEqual(centr_bis.meta['transform'].e, -0.009000000000000341)
        self.assertEqual(centr_bis.meta['height'], 71)
        self.assertEqual(centr_bis.meta['width'], 61)

    def test_equal_pass(self):
        """Test equal"""
        centr_ras = Centroids()
        centr_ras.set_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_bis = Centroids()
        centr_bis.set_raster_file(HAZ_DEMO_FL, window=Window(51, 61, 10, 10))
        self.assertFalse(centr_ras.equal(centr_bis))
        self.assertFalse(centr_bis.equal(centr_ras))
        self.assertTrue(centr_ras.equal(centr_ras))
        self.assertTrue(centr_bis.equal(centr_bis))


class TestCentroids(unittest.TestCase):
    """Test Centroids class"""

    def test_centroids_check_pass(self):
        """Test vector data in Centroids"""
        data_vec = TestVector.data_vector()
        centr = Centroids()
        centr.lat, centr.lon, centr.geometry = data_vec
        centr.check()

        self.assertEqual(centr.crs, data_vec[2].crs)
        self.assertIsInstance(centr.total_bounds, tuple)
        for i in range(4):
            self.assertEqual(centr.total_bounds[i], data_vec[2].total_bounds[i])

        self.assertIsInstance(centr.lat, np.ndarray)
        self.assertIsInstance(centr.lon, np.ndarray)
        self.assertIsInstance(centr.coord, np.ndarray)
        self.assertTrue(np.array_equal(centr.lat, VEC_LAT))
        self.assertTrue(np.array_equal(centr.lon, VEC_LON))
        self.assertTrue(np.array_equal(centr.coord, np.array([VEC_LAT, VEC_LON]).transpose()))
        self.assertEqual(centr.size, VEC_LON.size)

        centr.area_pixel = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.area_pixel = np.array([])
        centr.dist_coast = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.dist_coast = np.array([])
        centr.on_land = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.on_land = np.array([])
        centr.region_id = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.region_id = np.array([])
        centr.lat = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.lat = np.array([])
        centr.lon = np.array([1])
        with self.assertRaises(ValueError):
            centr.check()

        centr.lon = np.array([])
        centr.geometry = gpd.GeoSeries(np.ones(1))
        with self.assertRaises(ValueError):
            centr.check()

class TestReader(unittest.TestCase):
    """Test Centroids setter vector and raster methods"""
    def test_set_vector_file_wrong_fail(self):
        """Test set_vector_file with wrong centroids"""
        shp_file = shapereader.natural_earth(resolution='110m', category='cultural',
                                             name='populated_places_simple')
        centr = Centroids()
        inten = centr.set_vector_file(shp_file, ['pop_min', 'pop_max'])

        self.assertEqual(centr.geometry.crs, from_epsg(NE_EPSG))
        self.assertEqual(centr.geometry.size, centr.lat.size)
        self.assertEqual(centr.geometry.crs, from_epsg(NE_EPSG))
        self.assertAlmostEqual(centr.lon[0], 12.453386544971766)
        self.assertAlmostEqual(centr.lon[-1], 114.18306345846304)
        self.assertAlmostEqual(centr.lat[0], 41.903282179960115)
        self.assertAlmostEqual(centr.lat[-1], 22.30692675357551)

        self.assertEqual(inten.shape, (2, 243))
        # population min
        self.assertEqual(inten[0, 0], 832)
        self.assertEqual(inten[0, -1], 4551579)
        # population max
        self.assertEqual(inten[1, 0], 832)
        self.assertEqual(inten[1, -1], 7206000)

        shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                             name='populated_places_simple')
        with self.assertRaises(ValueError):
            centr.set_vector_file(shp_file, ['pop_min', 'pop_max'])

    def test_set_raster_file_wrong_fail(self):
        """Test set_raster_file with wrong centroids"""
        centr = Centroids()
        inten_ras = centr.set_raster_file(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        self.assertAlmostEqual(centr.meta['crs'], DEF_CRS)
        self.assertAlmostEqual(centr.meta['transform'].c, -69.2471495969998)
        self.assertAlmostEqual(centr.meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(centr.meta['transform'].b, 0.0)
        self.assertAlmostEqual(centr.meta['transform'].f, 10.248220966978932)
        self.assertAlmostEqual(centr.meta['transform'].d, 0.0)
        self.assertAlmostEqual(centr.meta['transform'].e, -0.009000000000000341)
        self.assertEqual(centr.meta['height'], 60)
        self.assertEqual(centr.meta['width'], 50)
        self.assertEqual(inten_ras.shape, (1, 60 * 50))
        self.assertAlmostEqual(inten_ras.reshape((60, 50)).tocsr()[25, 12], 0.056825936)

        with self.assertRaises(ValueError):
            centr.set_raster_file(HAZ_DEMO_FL, window=Window(10, 20, 52, 60))

    def test_write_read_raster_h5(self):
        """Write and read hdf5 format"""
        file_name = os.path.join(DATA_DIR, 'test_centr.h5')

        centr = Centroids()
        xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon = 10, 5, -0.5, 0.2, 20, 25
        centr.set_raster_from_pix_bounds(xf_lat, xo_lon, d_lat, d_lon, n_lat, n_lon)
        centr.write_hdf5(file_name)

        centr_read = Centroids()
        centr_read.read_hdf5(file_name)
        self.assertTrue(centr_read.meta)
        self.assertFalse(centr_read.lat.size)
        self.assertFalse(centr_read.lon.size)
        self.assertEqual(centr_read.meta['width'], centr.meta['width'])
        self.assertEqual(centr_read.meta['height'], centr.meta['height'])
        self.assertAlmostEqual(centr_read.meta['transform'].a, centr.meta['transform'].a)
        self.assertAlmostEqual(centr_read.meta['transform'].b, centr.meta['transform'].b)
        self.assertAlmostEqual(centr_read.meta['transform'].c, centr.meta['transform'].c)
        self.assertAlmostEqual(centr_read.meta['transform'].d, centr.meta['transform'].d)
        self.assertAlmostEqual(centr_read.meta['transform'].e, centr.meta['transform'].e)
        self.assertAlmostEqual(centr_read.meta['transform'].f, centr.meta['transform'].f)
        self.assertTrue(equal_crs(centr_read.meta['crs'], centr.meta['crs']))

    def test_write_read_points_h5(self):
        file_name = os.path.join(DATA_DIR, 'test_centr.h5')

        centr = Centroids()
        centr = Centroids()
        centr.set_lat_lon(VEC_LAT, VEC_LON)
        centr.write_hdf5(file_name)

        centr_read = Centroids()
        centr_read.read_hdf5(file_name)
        self.assertFalse(centr_read.meta)
        self.assertTrue(centr_read.lat.size)
        self.assertTrue(centr_read.lon.size)
        self.assertTrue(np.allclose(centr_read.lat, centr.lat))
        self.assertTrue(np.allclose(centr_read.lon, centr.lon))
        self.assertTrue(equal_crs(centr_read.crs, centr.crs))

class TestCentroidsFuncs(unittest.TestCase):
    """Test Centroids methods"""
    def test_select_pass(self):
        """Test set_vector"""
        centr = Centroids()
        centr.set_lat_lon(VEC_LAT, VEC_LON)

        centr.region_id = np.zeros(VEC_LAT.size)
        centr.region_id[[100, 200]] = 10

        fil_centr = centr.select(10)
        self.assertEqual(fil_centr.size, 2)
        self.assertEqual(fil_centr.lat[0], VEC_LAT[100])
        self.assertEqual(fil_centr.lat[1], VEC_LAT[200])
        self.assertEqual(fil_centr.lon[0], VEC_LON[100])
        self.assertEqual(fil_centr.lon[1], VEC_LON[200])
        self.assertTrue(np.array_equal(fil_centr.region_id, np.ones(2) * 10))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRaster)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVector))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroids))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
