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

Test coordinates module.
"""

from cartopy.io import shapereader
from fiona.crs import from_epsg
import geopandas as gpd
import unittest
import numpy as np
import shapely
import geopandas
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.warp import Resampling
from rasterio import Affine

from climada.util.constants import HAZ_DEMO_FL, DEF_CRS
from climada.util.coordinates import grid_is_regular, get_coastlines, \
get_land_geometry, nat_earth_resolution, coord_on_land, dist_to_coast, \
get_country_geometries, get_resolution, pts_to_raster_meta, read_vector, \
read_raster, NE_EPSG, equal_crs, set_df_geometry_points, points_to_raster, \
get_country_code, convert_wgs_to_utm, elevation_dem, DEM_NODATA

class TestFunc(unittest.TestCase):
    '''Test the auxiliary used with plot functions'''

    def test_is_regular_pass(self):
        """ Test is_regular function. """
        coord = np.array([[1, 2], [4.4, 5.4], [4, 5]])
        reg, hei, wid = grid_is_regular(coord)
        self.assertFalse(reg)
        self.assertEqual(hei, 1)
        self.assertEqual(wid, 1)

        coord = np.array([[1, 2], [4.4, 5], [4, 5]])
        reg, hei, wid = grid_is_regular(coord)
        self.assertFalse(reg)
        self.assertEqual(hei, 1)
        self.assertEqual(wid, 1)

        coord = np.array([[1, 2], [4, 5]])
        reg, hei, wid = grid_is_regular(coord)
        self.assertFalse(reg)
        self.assertEqual(hei, 1)
        self.assertEqual(wid, 1)

        coord = np.array([[1, 2], [4, 5], [1, 5], [4, 3]])
        reg, hei, wid = grid_is_regular(coord)
        self.assertFalse(reg)
        self.assertEqual(hei, 2)
        self.assertEqual(wid, 1)

        coord = np.array([[1, 2], [4, 5], [1, 5], [4, 2]])
        reg, hei, wid = grid_is_regular(coord)
        self.assertTrue(reg)
        self.assertEqual(hei, 2)
        self.assertEqual(wid, 2)

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 5),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = np.array([grid_x, grid_y]).transpose()
        reg, hei, wid = grid_is_regular(coord)
        self.assertTrue(reg)
        self.assertEqual(hei, 5)
        self.assertEqual(wid, 5)

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 4),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = np.array([grid_x, grid_y]).transpose()
        reg, hei, wid = grid_is_regular(coord)
        self.assertTrue(reg)
        self.assertEqual(hei, 5)
        self.assertEqual(wid, 4)

    def test_nat_earth_resolution_pass(self):
        """Correct resolution."""
        self.assertEqual(nat_earth_resolution(10), '10m')
        self.assertEqual(nat_earth_resolution(50), '50m')
        self.assertEqual(nat_earth_resolution(110), '110m')

    def test_nat_earth_resolution_fail(self):
        """Wrong resolution."""
        with self.assertRaises(ValueError):
            nat_earth_resolution(11)
        with self.assertRaises(ValueError):
            nat_earth_resolution(51)
        with self.assertRaises(ValueError):
            nat_earth_resolution(111)

    def test_get_coastlines_all_pass(self):
        '''Check get_coastlines function over whole earth'''
        coast = get_coastlines(resolution=110)
        tot_bounds = coast.total_bounds
        self.assertEqual((134, 1), coast.shape)
        self.assertAlmostEqual(tot_bounds[0], -180)
        self.assertAlmostEqual(tot_bounds[1], -85.60903777)
        self.assertAlmostEqual(tot_bounds[2], 180.00000044)
        self.assertAlmostEqual(tot_bounds[3], 83.64513)

    def test_get_coastlines_pass(self):
        '''Check get_coastlines function in defined extent'''
        bounds = (-100, -55, -20, 35)
        coast = get_coastlines(bounds, resolution=110)
        ex_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
        self.assertEqual(coast.shape[0], 14)
        self.assertTrue(coast.total_bounds[2] < 0)
        for row, line in coast.iterrows():
            if not ex_box.intersects(line.geometry):
                self.assertEqual(1, 0)

    def test_get_land_geometry_country_pass(self):
        """get_land_geometry with selected countries."""
        iso_countries = ['DEU', 'VNM']
        res = get_land_geometry(iso_countries, 110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        for res, ref in zip(res.bounds, (5.85248986800, 8.56557851800,
                                         109.47242272200, 55.065334377000)):
            self.assertAlmostEqual(res, ref)

        iso_countries = ['ESP']
        res = get_land_geometry(iso_countries, 110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        for res, ref in zip(res.bounds, (-18.16722571499986, 27.642238674000,
                                         4.337087436000, 43.793443101)):
            self.assertAlmostEqual(res, ref)

        iso_countries = ['FRA']
        res = get_land_geometry(iso_countries, 110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        for res, ref in zip(res.bounds, (-61.79784094999991, -21.37078215899993,
                                         55.854502800000034, 51.08754088371883)):
            self.assertAlmostEqual(res, ref)

    def test_get_land_geometry_extent_pass(self):
        """get_land_geometry with selected countries."""
        lat = np.array([28.203216, 28.555994, 28.860875])
        lon = np.array([-16.567489, -18.554130, -9.532476])
        res = get_land_geometry(extent=(np.min(lon), np.max(lon),
                                np.min(lat), np.max(lat)), resolution=10)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        self.assertAlmostEqual(res.bounds[0], -18.002186653)
        self.assertAlmostEqual(res.bounds[1], lat[0])
        self.assertAlmostEqual(res.bounds[2], np.max(lon))
        self.assertAlmostEqual(res.bounds[3], np.max(lat))

    def test_get_land_geometry_all_pass(self):
        """get_land_geometry with all earth."""
        res = get_land_geometry(resolution=110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        self.assertEqual(res.area, 21496.99098799273)

    def test_on_land_pass(self):
        """check point on land with 1:50.000.000 resolution."""
        lat = np.array([28.203216, 28.555994, 28.860875])
        lon = np.array([-16.567489, -18.554130, -9.532476])
        res = coord_on_land(lat, lon)
        self.assertEqual(res.size, 3)
        self.assertTrue(res[0])
        self.assertFalse(res[1])
        self.assertTrue(res[2])

    def test_dist_to_coast(self):
        """ Test point in coast and point not in coast """
        res = dist_to_coast(13.208333333333329, -59.625000000000014)
        self.assertAlmostEqual(2594.2071059573445, res[0])

        res = dist_to_coast(-12.497529, -58.849505)
        self.assertAlmostEqual(1382985.2459744606, res[0])

    def test_get_country_geometries_country_pass(self):
        """ get_country_geometries with selected countries. issues with the
        natural earth data should be caught by test_get_land_geometry_* since
        it's very similar """
        iso_countries = ['NLD', 'VNM']
        res = get_country_geometries(iso_countries, resolution=110)
        self.assertIsInstance(res,
                              geopandas.geodataframe.GeoDataFrame)

    def test_get_country_geometries_extent_pass(self):
        """get_country_geometries by selecting by extent"""
        lat = np.array([28.203216, 28.555994, 28.860875])
        lon = np.array([-16.567489, -18.554130, -9.532476])

        res = get_country_geometries(extent=(
            np.min(lon), np.max(lon),
            np.min(lat), np.max(lat)
        ))

        self.assertIsInstance(res, geopandas.geodataframe.GeoDataFrame)
        self.assertTrue(
            np.allclose(res.bounds.iloc[1, 1], lat[0])
        )
        self.assertTrue(
            np.allclose(res.bounds.iloc[0, 0], -11.800084333105298) or
            np.allclose(res.bounds.iloc[1, 0], -11.800084333105298)
        )
        self.assertTrue(
            np.allclose(res.bounds.iloc[0, 2], np.max(lon)) or
            np.allclose(res.bounds.iloc[1, 2], np.max(lon))
        )
        self.assertTrue(
            np.allclose(res.bounds.iloc[0, 3], np.max(lat)) or
            np.allclose(res.bounds.iloc[1, 3], np.max(lat))
        )

    def test_get_country_geometries_all_pass(self):
        """get_country_geometries with no countries or extent; i.e. the whole
        earth"""
        res = get_country_geometries(resolution=110)
        self.assertIsInstance(res, geopandas.geodataframe.GeoDataFrame)
        self.assertAlmostEqual(res.area[0], 1.639510995900778)

    def test_get_resolution_pass(self):
        """ Test _get_resolution method """
        lat = np.array([13.125, 13.20833333, 13.29166667, 13.125,
                        13.20833333, 13.125, 12.625, 12.70833333,
                        12.79166667, 12.875, 12.95833333, 13.04166667])
        lon = np.array([-59.6250000000000,-59.6250000000000,-59.6250000000000,-59.5416666666667,
                        -59.5416666666667,-59.4583333333333,-60.2083333333333,-60.2083333333333,
                        -60.2083333333333,-60.2083333333333,-60.2083333333333,-60.2083333333333])
        res_lat, res_lon = get_resolution(lat, lon)
        self.assertAlmostEqual(min(res_lat, res_lon), 0.0833333333333)

    def test_vector_to_raster_pass(self):
        """ Test vector_to_raster """
        xmin, ymin, xmax, ymax = -60, -5, -50, 10 # bounds of points == centers pixels
        points_bounds = (xmin, ymin, xmax, ymax)
        res = 0.5
        rows, cols, ras_trans = pts_to_raster_meta(points_bounds, res)
        self.assertEqual(xmin - res/2 + res * cols, xmax + res/2)
        self.assertEqual(ymax + res/2 - res * rows, ymin - res/2)
        self.assertEqual(ras_trans[0], res)
        self.assertEqual(ras_trans[4], -res)
        self.assertEqual(ras_trans[1], 0.0)
        self.assertEqual(ras_trans[3], 0.0)
        self.assertEqual(ras_trans[2], xmin - res/2)
        self.assertEqual(ras_trans[5], ymax + res/2)
        self.assertTrue(ymin >= ymax + res/2 - rows*res)
        self.assertTrue(xmax <= xmin - res/2 + cols*res)

    def test_pts_to_raster_irreg_pass(self):
        """ Test pts_to_raster_meta with irregular points """
        xmin, ymin, xmax, ymax = -124.19473, 32.81908, -114.4632, 42.020759999999996 # bounds of points == centers pixels
        points_bounds = (xmin, ymin, xmax, ymax)
        res = 0.013498920086393088
        rows, cols, ras_trans = pts_to_raster_meta(points_bounds, res)
        self.assertEqual(ras_trans[0], res)
        self.assertEqual(ras_trans[4], -res)
        self.assertEqual(ras_trans[1], 0.0)
        self.assertEqual(ras_trans[3], 0.0)
        self.assertEqual(ras_trans[2], xmin - res/2)
        self.assertEqual(ras_trans[5], ymax + res/2)
        self.assertTrue(ymin >= ymax + res/2 - rows*res)
        self.assertTrue(xmax <= xmin - res/2 + cols*res)

    def test_read_vector_pass(self):
        """ Test one columns data """
        shp_file = shapereader.natural_earth(resolution='110m', \
            category='cultural', name='populated_places_simple')
        lat, lon, geometry, intensity = read_vector(shp_file, ['pop_min', 'pop_max'])

        self.assertEqual(geometry.crs, from_epsg(NE_EPSG))
        self.assertEqual(geometry.size, lat.size)
        self.assertEqual(geometry.crs, from_epsg(NE_EPSG))
        self.assertAlmostEqual(lon[0], 12.453386544971766)
        self.assertAlmostEqual(lon[-1], 114.18306345846304)
        self.assertAlmostEqual(lat[0], 41.903282179960115)
        self.assertAlmostEqual(lat[-1], 22.30692675357551)

        self.assertEqual(intensity.shape, (2, 243))
        # population min
        self.assertEqual(intensity[0, 0], 832)
        self.assertEqual(intensity[0, -1], 4551579)
        # population max
        self.assertEqual(intensity[1, 0], 832)
        self.assertEqual(intensity[1, -1], 7206000)

    def test_window_raster_pass(self):
        """ Test window """
        meta, inten_ras = read_raster(HAZ_DEMO_FL, window=Window(10, 20, 50.1, 60))
        self.assertAlmostEqual(meta['crs'], DEF_CRS)
        self.assertAlmostEqual(meta['transform'].c, -69.2471495969998)
        self.assertAlmostEqual(meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(meta['transform'].b, 0.0)
        self.assertAlmostEqual(meta['transform'].f, 10.248220966978932)
        self.assertAlmostEqual(meta['transform'].d, 0.0)
        self.assertAlmostEqual(meta['transform'].e, -0.009000000000000341)
        self.assertEqual(meta['height'], 60)
        self.assertEqual(meta['width'], 50)
        self.assertEqual(inten_ras.shape, (1, 60*50))
        self.assertAlmostEqual(inten_ras.reshape((60, 50))[25, 12], 0.056825936)

    def test_poly_raster_pass(self):
        """ Test geometry """
        poly = box(-69.2471495969998, 9.708220966978912, -68.79714959699979, 10.248220966978932)
        meta, inten_ras = read_raster(HAZ_DEMO_FL, geometry=[poly])
        self.assertAlmostEqual(meta['crs'], DEF_CRS)
        self.assertAlmostEqual(meta['transform'].c, -69.2471495969998)
        self.assertAlmostEqual(meta['transform'].a, 0.009000000000000341)
        self.assertAlmostEqual(meta['transform'].b, 0.0)
        self.assertAlmostEqual(meta['transform'].f, 10.248220966978932)
        self.assertAlmostEqual(meta['transform'].d, 0.0)
        self.assertAlmostEqual(meta['transform'].e, -0.009000000000000341)
        self.assertEqual(meta['height'], 60)
        self.assertEqual(meta['width'], 50)
        self.assertEqual(inten_ras.shape, (1, 60*50))

    def test_crs_raster_pass(self):
        """ Test change projection """
        meta, inten_ras = read_raster(HAZ_DEMO_FL, dst_crs={'init':'epsg:2202'},
                                      resampling=Resampling.nearest)
        self.assertAlmostEqual(meta['crs'], {'init':'epsg:2202'})
        self.assertAlmostEqual(meta['transform'].c, 462486.8490210658)
        self.assertAlmostEqual(meta['transform'].a, 998.576177833903)
        self.assertAlmostEqual(meta['transform'].b, 0.0)
        self.assertAlmostEqual(meta['transform'].f, 1164831.4772731226)
        self.assertAlmostEqual(meta['transform'].d, 0.0)
        self.assertAlmostEqual(meta['transform'].e, -998.576177833903)
        self.assertEqual(meta['height'], 1081)
        self.assertEqual(meta['width'], 968)
        self.assertEqual(inten_ras.shape, (1, 1081*968))
        # TODO: NOT RESAMPLING WELL in this case!?
        self.assertAlmostEqual(inten_ras.reshape((1081, 968))[45, 22], 0)
        
    def test_crs_and_geometry_raster_pass(self):
        """ Test change projection and crop to geometry """
        ply = shapely.geometry.Polygon(((478080.8562247154, 1105419.13439131), (478087.5912452241, 1116475.583523723), (500000, 1116468.876713805), (500000, 1105412.49126517), (478080.8562247154, 1105419.13439131)))
        meta, inten_ras = read_raster(HAZ_DEMO_FL, dst_crs={'init':'epsg:2202'},
                                      geometry=[ply],resampling=Resampling.nearest)
        self.assertAlmostEqual(meta['crs'], {'init':'epsg:2202'})
        self.assertEqual(meta['height'], 12)
        self.assertEqual(meta['width'], 23)
        self.assertEqual(inten_ras.shape, (1, 12*23))
        # TODO: NOT RESAMPLING WELL in this case!?
        self.assertAlmostEqual(inten_ras.reshape((12, 23))[11, 12], 0.10063865780830383)

    def test_transform_raster_pass(self):
        meta, inten_ras = read_raster(HAZ_DEMO_FL,
            transform=Affine(0.009000000000000341, 0.0, -69.33714959699981,
            0.0, -0.009000000000000341, 10.42822096697894), height=500, width=501)

        left = meta['transform'].xoff
        top = meta['transform'].yoff
        bottom = top + meta['transform'][4]*meta['height']
        right = left + meta['transform'][0]*meta['width']

        self.assertAlmostEqual(left, -69.33714959699981)
        self.assertAlmostEqual(bottom, 5.928220966978939)
        self.assertAlmostEqual(right, -64.82814959699981)
        self.assertAlmostEqual(top, 10.42822096697894)
        self.assertEqual(meta['width'], 501)
        self.assertEqual(meta['height'], 500)
        self.assertEqual(meta['crs'].to_epsg(), 4326)
        self.assertEqual(inten_ras.shape, (1, 500*501))

        meta, inten_all = read_raster(HAZ_DEMO_FL, window=Window(0, 0, 501, 500))
        self.assertTrue(np.array_equal(inten_all, inten_ras))

    def test_compare_crs(self):
        """ Compare two crs """
        crs_one = {'init':'epsg:4326'}
        crs_two = {'init':'epsg:4326', 'no_defs': True}
        self.assertTrue(equal_crs(crs_one, crs_two))

    def test_set_df_geometry_points_pass(self):
        """ Test set_df_geometry_points """
        df_val = gpd.GeoDataFrame(crs={'init':'epsg:2202'})
        df_val['latitude'] = np.ones(10)*40.0
        df_val['longitude'] = np.ones(10)*0.50

        set_df_geometry_points(df_val)
        self.assertTrue(np.allclose(df_val.geometry[:].x.values, np.ones(10)*0.5))
        self.assertTrue(np.allclose(df_val.geometry[:].y.values, np.ones(10)*40.))

    def test_points_to_raster_pass(self):
        """ Test points_to_raster """
        df_val = gpd.GeoDataFrame(crs={'init':'epsg:2202'})
        x, y = np.meshgrid(np.linspace(0, 2, 5), np.linspace(40, 50, 10))
        df_val['latitude'] = y.flatten()
        df_val['longitude'] = x.flatten()
        df_val['value'] = np.ones(len(df_val))*10
        raster, meta = points_to_raster(df_val, val_names=['value'])
        self.assertTrue(equal_crs(meta['crs'], df_val.crs))
        self.assertAlmostEqual(meta['transform'][0], 0.5)
        self.assertAlmostEqual(meta['transform'][1], 0)
        self.assertAlmostEqual(meta['transform'][2], -0.25)
        self.assertAlmostEqual(meta['transform'][3], 0)
        self.assertAlmostEqual(meta['transform'][4], -0.5)
        self.assertAlmostEqual(meta['transform'][5], 50.25)
        self.assertEqual(meta['height'], 21)
        self.assertEqual(meta['width'], 5)

    def test_country_code_pass(self):
        """ Test set_region_id """

        lon = np.array([-59.6250000000000,-59.6250000000000,-59.6250000000000,-59.5416666666667,
                        -59.5416666666667,-59.4583333333333,-60.2083333333333,-60.2083333333333])
        lat = np.array([13.125,13.20833333,13.29166667,13.125,13.20833333,13.125,12.625,12.70833333])
        region_id = get_country_code(lat, lon)

        self.assertEqual(np.count_nonzero(region_id), 6)
        self.assertTrue(np.allclose(region_id[:6], np.ones(6)*52)) # 052 for barbados

    def test_convert_wgs_to_utm_pass(self):
        """ Test convert_wgs_to_utm """
        lat, lon = 17.346597, -62.768669
        epsg = convert_wgs_to_utm(lon, lat)
        self.assertEqual(epsg, 32620)

        lat, lon = 41.522410, 1.891026
        epsg = convert_wgs_to_utm(lon, lat)
        self.assertEqual(epsg, 32631)

    def test_elevation_pass(self):
        lon = np.array([-59.6250000000000,-59.6250000000000,-59.6250000000000,-59.5416666666667, \
            -59.5416666666667,-59.4583333333333,-60.2083333333333,-60.2083333333333])
        lat = np.array([13.125,13.20833333,13.29166667,13.125,13.20833333,13.125,12.625,12.70833333])
        elevation = elevation_dem(lon, lat, DEF_CRS, product='SRTM3', resampling=Resampling.nearest)
        self.assertEqual(elevation[0], 23)
        self.assertEqual(elevation[1], 92)
        self.assertEqual(elevation[2], 69)
        self.assertEqual(elevation[3], 77)
        self.assertEqual(elevation[4], 133)
        self.assertEqual(elevation[5], 41)
        self.assertEqual(elevation.min(), DEM_NODATA)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
