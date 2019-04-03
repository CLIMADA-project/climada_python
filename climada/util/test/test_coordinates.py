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

import unittest
import numpy as np
import shapely
from cartopy.io import shapereader
import geopandas

from climada.util.coordinates import grid_is_regular, get_coastlines, \
get_land_geometry, nat_earth_resolution, coord_on_land, shapely_to_pyshp,\
dist_to_coast, get_country_geometries

class TestFunc(unittest.TestCase):
    '''Test the auxiliary used with plot functions'''

    def test_is_regular_pass(self):
        """ Test is_regular function. """
        coord = np.array([[1, 2], [4.4, 5.4], [4, 5]])
        self.assertFalse(grid_is_regular(coord))

        coord = np.array([[1, 2], [4.4, 5], [4, 5]])
        self.assertFalse(grid_is_regular(coord))

        coord = np.array([[1, 2], [4, 5]])
        self.assertFalse(grid_is_regular(coord))

        coord = np.array([[1, 2], [4, 5], [1, 5], [4, 3]])
        self.assertFalse(grid_is_regular(coord))

        coord = np.array([[1, 2], [4, 5], [1, 5], [4, 2]])
        self.assertTrue(grid_is_regular(coord))

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 5),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = np.array([grid_x, grid_y]).transpose()
        self.assertTrue(grid_is_regular(coord))

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 4),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = np.array([grid_x, grid_y]).transpose()
        self.assertTrue(grid_is_regular(coord))

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
        self.assertEqual((5128, 2), coast.shape)

        self.assertEqual(-78.59566741324154, coast[0][0])
        self.assertEqual(73.60000000000001, coast[-1][0])

        self.assertEqual(-163.7128956777287, coast[0][1])
        self.assertEqual(-106.6, coast[-1][1])

    def test_get_coastlines_pass(self):
        '''Check get_coastlines function in defined extent'''
        extent = (-100, 95, -55, 35)
        coast = get_coastlines(extent, resolution=110)

        for lat_val, lon_val in coast:
            if lon_val < extent[0] or lon_val > extent[1]:
                self.assertTrue(False)
            if lat_val < extent[2] or lat_val > extent[3]:
                self.assertTrue(False)

        self.assertEqual((1234, 2), coast.shape)

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

    def test_shapely_to_pyshp_polygon_pass(self):
        """ Test shapely_to_pyshp with polygon."""
        shp_file = shapereader.natural_earth(resolution='110m',
                                             category='cultural',
                                             name='admin_0_countries')
        reader = shapereader.Reader(shp_file)
        countries = list(reader.records())
        cntry_geom = [country.geometry for country in countries]
        all_geom = shapely.ops.cascaded_union(cntry_geom)

        converted_shape = shapely_to_pyshp(all_geom)

        res =  shapereader._create_polygon(converted_shape)
        self.assertTrue(res.equals(all_geom))
        self.assertEqual(res.area, all_geom.area)

    def test_dist_to_coast(self):
        point = (13.208333333333329, -59.625000000000014)
        res = dist_to_coast(point)
        self.assertAlmostEqual(5.7988200982894105, res[0])

        point = (13.958333333333343, -58.125)
        res = dist_to_coast(point)
        self.assertAlmostEqual(166.36505441711506, res[0])

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


# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
unittest.TextTestRunner(verbosity=2).run(TESTS)
