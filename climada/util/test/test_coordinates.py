"""
Test GridPoints module.
"""

import os.path
import unittest
import numpy as np
import shapely
from cartopy.io import shapereader

from climada.util.constants import SYSTEM_DIR

from climada.util.coordinates import GridPoints, get_coastlines, \
get_land_geometry, nat_earth_resolution, coord_on_land, shapely_to_pyshp,\
GLOBE_LAND, dist_to_coast
    
class TestGridPoints(unittest.TestCase):
    ''' Test GridPoints class'''

    def test_shape_pass(self):
        """Check that shape returns expected value."""
        coord = GridPoints(np.array([[1, 2], [4.5, 5.5], [4, 5]]))
        self.assertEqual(coord.shape, (3,2))

        GridPoints(np.array([[1, 2], [4, 5], [4, 5]]))
        self.assertEqual(coord.shape, (3,2))

        coord = GridPoints()
        self.assertEqual(coord.shape, (0,2))

    def test_wrong_value_fail(self):
        """Check good values in constructor."""
        with self.assertRaises(ValueError):
            GridPoints(np.array([[1, 2], [4.3, 5], [4, 5]]).transpose())

    def test_resample_pass(self):
        """Check that resample works correctly."""
        coord_1 = GridPoints(np.array([[1, 2], [4.1, 5.1], [4, 5]]))
        coord_2 = coord_1
        result = coord_1.resample(coord_2)
        self.assertTrue(np.array_equal(result, np.array([ 0.,  1.,  2.])))

    def test_is_regular_pass(self):
        """ Test is_regular function. """
        coord = GridPoints(np.array([[1, 2], [4.4, 5.4], [4, 5]]))
        self.assertFalse(coord.is_regular())

        coord = GridPoints(np.array([[1, 2], [4.4, 5], [4, 5]]))
        self.assertFalse(coord.is_regular())

        coord = GridPoints(np.array([[1, 2], [4, 5]]))
        self.assertFalse(coord.is_regular())
        
        coord = GridPoints(np.array([[1, 2], [4, 5], [1, 5], [4, 3]]))
        self.assertFalse(coord.is_regular())
        
        coord = GridPoints(np.array([[1, 2], [4, 5], [1, 5], [4, 2]]))
        self.assertTrue(coord.is_regular())
        
        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 5),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = GridPoints(np.array([grid_x, grid_y]).transpose())
        self.assertTrue(coord.is_regular())

        grid_x, grid_y = np.mgrid[10 : 100 : complex(0, 4),
                                  0 : 10 : complex(0, 5)]
        grid_x = grid_x.reshape(-1,)
        grid_y = grid_y.reshape(-1,)
        coord = GridPoints(np.array([grid_x, grid_y]).transpose())
        self.assertTrue(coord.is_regular())

class TestFunc(unittest.TestCase):
    '''Test the auxiliary used with plot functions'''

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
        '''Check get_coastlines function in defined border'''
        border = (-100, 95, -55, 35)
        coast = get_coastlines(border, resolution=110)

        for lat_val, lon_val in coast:
            if lon_val < border[0] or lon_val > border[1]:
                self.assertTrue(False)
            if lat_val < border[2] or lat_val > border[3]:
                self.assertTrue(False)

        self.assertEqual((1234, 2), coast.shape)

    def test_get_land_geometry_country_pass(self):
        """get_land_geometry with selected countries."""
        iso_countries = ['DEU', 'VNM']
        res = get_land_geometry(iso_countries, 110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        self.assertEqual(res.bounds, (5.852489868000106, 8.565578518000152, 
                                      109.47242272200018, 55.06533437700001))

        iso_countries = ['ESP']
        res = get_land_geometry(iso_countries, 110)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        self.assertEqual(res.bounds, (-18.16722571499986, 27.64223867400007, 
                                      4.337087436000104, 43.793443100999994))

    def test_get_land_geometry_border_pass(self):
        """get_land_geometry with selected countries."""
        lat = np.array([28.203216, 28.555994, 28.860875])
        lon = np.array([-16.567489, -18.554130, -9.532476])
        res = get_land_geometry(border=(np.min(lon), np.max(lon),
                                np.min(lat), np.max(lat)), resolution=10)
        self.assertIsInstance(res, shapely.geometry.multipolygon.MultiPolygon)
        self.assertAlmostEqual(res.bounds[0], -18.002186653)
        self.assertAlmostEqual(res.bounds[1], lat[0])
        self.assertAlmostEqual(res.bounds[2], np.max(lon))
        self.assertAlmostEqual(res.bounds[3], np.max(lat))

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
        self.assertEqual(5.7988200982894105, res)
        
        point = (13.958333333333343, -58.125)
        res = dist_to_coast(point)
        self.assertEqual(166.36505441711506, res)        

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGridPoints)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunc))
unittest.TextTestRunner(verbosity=2).run(TESTS)
