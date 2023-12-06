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

from cartopy.io import shapereader
import geopandas as gpd
import numpy as np
from pyproj.crs import CRS
import rasterio
from rasterio.windows import Window
from shapely.geometry.point import Point

from climada import CONFIG
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import HAZ_DEMO_FL, DEF_CRS
import climada.util.coordinates as u_coord
import itertools

DATA_DIR = CONFIG.hazard.test_data.dir()


# Note: the coordinates are not directly on the cities, the region id and on land
# otherwise do not work correctly. It is only the closest point.
LATLON = np.array([
    [-21.1736, -175.1883], #Tonga, Nuku'alofa, TON, 776
    [-18.133, 178.433], #Fidji, Suva, FJI, 242  IN WATER IN NATURAL EARTH
    [-38.4689, 177.8642], #New-Zealand, Te Karaka, NZL, 554
    [69.6833, 18.95], #Norway, Tromso, NOR, 578 IN WATER IN NATURAL EARTH
    [78.84422,	20.82842],  #Norway, Svalbard, NOR, 578
    [1, 1], # Ocean, 0  (0,0 is onland in Natural earth for testing reasons)
    [-77.85, 166.6778], #Antarctica, McMurdo station, ATA, 010
    [-0.25, -78.5833] #Ecuador, Quito, ECU, 218
])

VEC_LAT = LATLON[:,0]
VEC_LON = LATLON[:,1]

ON_LAND = np.array([True, False, True, False, True, False, True, True])
REGION_ID = np.array([776, 0, 554, 0, 578, 0, 10, 218])

TEST_CRS = 'EPSG:4326'
ALT_CRS = 'epsg:32632' #Europe

class TestVector(unittest.TestCase):
    """Test CentroidsVector class"""

    def setUp(self):
        self.centr = Centroids(latitude=VEC_LAT,longitude=VEC_LON, crs=TEST_CRS)

    def test_init_pass(self):
        """Test from_lat_lon"""
        self.assertTrue(np.allclose(self.centr.lat, VEC_LAT))
        self.assertTrue(np.allclose(self.centr.lon, VEC_LON))
        self.assertTrue(u_coord.equal_crs(self.centr.crs, DEF_CRS))
        self.assertTrue(u_coord.equal_crs(self.centr.geometry.crs, DEF_CRS))

    def test_ne_crs_geom_pass(self):
        """Test _ne_crs_geom"""
        natural_earth_geom = self.centr._ne_crs_geom()
        self.assertEqual(natural_earth_geom.crs, u_coord.NE_CRS)

    def test_dist_coast_pass(self):
        """Test set_dist_coast"""
        dist_coast = self.centr.get_dist_coast()\
        # Just checking that the output doesnt change over time.
        REF_VALUES = np.array([
            5.55578093e+05, 1.64066475e+03, 2.00703835e+03,
            4.82264614e+02, 4.50037266e+03, 1.08610274e+05
            ])
        np.testing.assert_array_almost_equal(dist_coast, REF_VALUES, decimal=3)

    def test_region_id_pass(self):
        """Test set_region_id"""
        self.centr.set_region_id()
        np.testing.assert_array_equal(
            self.centr.region_id,
            REGION_ID
        )

    def test_on_land(self):
        """Test set_on_land"""
        self.centr.set_on_land()
        np.testing.assert_array_equal(
            self.centr.on_land,
            ON_LAND
        )

    def test_remove_duplicate_pass(self):
        """Test remove_duplicate_points"""
        centr = Centroids(latitude = np.hstack([VEC_LAT, VEC_LAT]),
                         longitude = np.hstack([VEC_LON , VEC_LON]),
                         crs=TEST_CRS)
        self.assertTrue(centr.gdf.shape[0] == 2*self.centr.gdf.shape[0])
        rem_centr = Centroids.remove_duplicate_points(centr)
        self.assertTrue(self.centr == rem_centr)

    def test_area_pass(self):
        """Test set_area"""
        ulx, xres, lrx = 60, 1, 90
        uly, yres, lry = 0, 1, 20
        xx, yy = np.meshgrid(np.arange(ulx + xres / 2, lrx, xres),
                             np.arange(uly + yres / 2, lry, yres))
        vec_data = gpd.GeoDataFrame({
            'geometry': [Point(xflat, yflat) for xflat, yflat in zip(xx.flatten(), yy.flatten())],
            'lon': xx.flatten(),
            'lat': yy.flatten(),
        }, crs={'proj': 'cea'})

        area_pixel = self.centr.get_area_pixel()
        self.assertTrue(np.allclose(area_pixel, np.ones(self.centr.size)))

    def test_size_pass(self):
        """Test size property"""
        centr = Centroids(latitude=VEC_LAT, longitude=VEC_LON)
        self.assertEqual(centr.size, len(VEC_LAT))

    def test_get_closest_point(self):
        """Test get_closest_point"""
        x, y, idx = self.centr.get_closest_point(-58.13, 14.38)
        self.assertAlmostEqual(x, -58.125)
        self.assertAlmostEqual(y, 14.375)
        self.assertEqual(idx, 295)
        self.assertEqual(self.centr.lon[idx], x)
        self.assertEqual(self.centr.lat[idx], y)

    def test_append_pass(self):
        """Append points"""
        centr = self.centr
        centr_bis = Centroids(latitude=np.array([1, 2, 3]), longitude=np.array([4, 5, 6]), crs=DEF_CRS)
        with self.assertRaises(ValueError):
            #Different crs
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

    def test_equal_pass(self):
        """Test equal"""
        centr_list = [
            Centroids(latitude=VEC_LAT, longitude=VEC_LON, crs=DEF_CRS),
            Centroids(latitude=VEC_LAT, longitude=VEC_LON, crs=ALT_CRS),
            Centroids(latitude=VEC_LAT+1, longitude=VEC_LON+1)
        ]
        for centr1, centr2 in itertools.combinations(centr_list, 2):
            self.assertFalse(centr2 == centr1)
            self.assertFalse(centr1 == centr2)
            self.assertTrue(centr1 == centr1)
            self.assertTrue(centr2 == centr2)


class TestRaster(unittest.TestCase):
    """Test CentroidsRaster class"""

    def test_from_pnt_bounds_pass(self):
        """Test from_pnt_bounds"""
        left, bottom, right, top = 5, 0, 10, 10
        centr = Centroids.from_pnt_bounds((left, bottom, right, top), 0.2, crs=DEF_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, DEF_CRS))

    def test_read_all_pass(self):
        """Test centr_ras data"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        self.assertTrue(u_coord.equal_crs(centr_ras.crs, DEF_CRS))

    def test_ne_crs_geom_pass(self):
        """Test _ne_crs_geom"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))

        xy_vec = centr_ras._ne_crs_geom()
        x_vec, y_vec = xy_vec.geometry[:].x.values, xy_vec.geometry[:].y.values
        self.assertAlmostEqual(4.51063496489, x_vec[0])
        self.assertAlmostEqual(9.40153761711e-05, y_vec[0])
        self.assertAlmostEqual(4.51063891581, x_vec[-1])
        self.assertAlmostEqual(8.92260922066e-05, y_vec[-1])

    def test_region_id_pass(self):
        """Test set_dist_coast"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_region_id()
        self.assertEqual(centr_ras.region_id.size, centr_ras.size)
        self.assertTrue(np.array_equal(np.unique(centr_ras.region_id), np.array([862])))

    def test_set_geometry_points_pass(self):
        """Test set_geometry_points"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        x_flat = np.arange(-69.3326495969998, -68.88264959699978, 0.009000000000000341)
        y_flat = np.arange(10.423720966978939, 9.883720966978919, -0.009000000000000341)
        x_grid, y_grid = np.meshgrid(x_flat, y_flat)
        self.assertTrue(np.allclose(x_grid.flatten(), centr_ras.lon))
        self.assertTrue(np.allclose(y_grid.flatten(), centr_ras.lat))

    def test_dist_coast_pass(self):
        """Test set_region_id"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        dist_coast = centr_ras.get_dist_coast()
        self.assertTrue(abs(dist_coast[0] - 117000) < 1000)
        self.assertTrue(abs(dist_coast[-1] - 104000) < 1000)

    def test_on_land(self):
        """Test set_on_land"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_ras.set_on_land()
        self.assertTrue(np.array_equal(centr_ras.on_land, np.ones(60 * 50, bool)))

    def test_area_pass(self):
        """Test set_area"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        #centr_ras.meta['crs'] = {'proj': 'cea'}
        area_pixel = centr_ras.get_area_pixel()
        self.assertTrue(
            np.allclose(area_pixel,
                        np.ones(60 * 50) * 0.009000000000000341 * 0.009000000000000341))

    def test_size_pass(self):
        """Test size property"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        self.assertEqual(centr_ras.size, 50 * 60)

    def test_get_closest_point(self):
        """Test get_closest_point"""
        for y_sign in [1, -1]:
            meta = {
                'width': 10,
                'height': 20,
                'transform': rasterio.Affine(0.5, 0, 0.1, 0, y_sign * 0.6, y_sign * (-0.3)),
                'crs': DEF_CRS,
            }
            centr_ras = Centroids.from_meta(meta=meta)

            test_data = np.array([
                [0.4, 0.1, 0.35, 0.0, 0],
                [-0.1, 0.2, 0.35, 0.0, 0],
                [2.2, 0.1, 2.35, 0.0, 4],
                [1.4, 2.5, 1.35, 2.4, 42],
                [5.5, -0.1, 4.85, 0.0, 9],
            ])
            test_data[:,[1,3]] *= y_sign
            for x_in, y_in, x_out, y_out, idx_out in test_data:
                x, y, idx = centr_ras.get_closest_point(x_in, y_in)
                self.assertEqual(x, x_out)
                self.assertEqual(y, y_out)
                self.assertEqual(idx, idx_out)
                self.assertEqual(centr_ras.lon[idx], x)
                self.assertEqual(centr_ras.lat[idx], y)

        centr_ras = Centroids(latitude=np.array([0, 0.2, 0.7]), longitude=np.array([-0.4, 0.2, 1.1]))
        x, y, idx = centr_ras.get_closest_point(0.1, 0.0)
        self.assertEqual(x, 0.2)
        self.assertEqual(y, 0.2)
        self.assertEqual(idx, 1)

    def test_set_meta_to_lat_lon_pass(self):
        """Test set_meta_to_lat_lon by using its inverse set_lat_lon_to_meta"""
        pass

    def test_equal_pass(self):
        """Test equal"""
        centr_ras = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(0, 0, 50, 60))
        centr_bis = Centroids.from_raster_file(HAZ_DEMO_FL, window=Window(51, 61, 10, 10))
        self.assertFalse(centr_ras == centr_bis)
        self.assertTrue(centr_ras == centr_ras)
        self.assertTrue(centr_bis == centr_bis)


class TestCentroids(unittest.TestCase):
    """Test Centroids class"""

    def test_centroids_check_pass(self):
        """Test vector data in Centroids"""
        centr = Centroids(latitude=VEC_LAT, longitude=VEC_LON, crs=ALT_CRS)

        self.assertTrue(u_coord.equal_crs(centr.crs, CRS.from_user_input(ALT_CRS)))
        self.assertEqual(list(centr.total_bounds),
                         [VEC_LON.min(), VEC_LAT.min(), VEC_LON.max(), VEC_LAT.max()])

        self.assertIsInstance(centr.lat, np.ndarray)
        self.assertIsInstance(centr.lon, np.ndarray)
        self.assertIsInstance(centr.coord, np.ndarray)
        self.assertTrue(np.array_equal(centr.lat, VEC_LAT))
        self.assertTrue(np.array_equal(centr.lon, VEC_LON))
        self.assertTrue(np.array_equal(centr.coord, np.array([VEC_LAT, VEC_LON]).transpose()))
        self.assertEqual(centr.size, VEC_LON.size)


class TestReader(unittest.TestCase):
    """Test Centroids setter vector and raster methods"""
    def setUp(self):
        self.centr = Centroids(latitude=VEC_LAT,longitude=VEC_LON, crs=TEST_CRS)

    def test_from_vector_file(self):
        """Test from_vector_file and values_from_vector_files"""
        shp_file = shapereader.natural_earth(resolution='110m', category='cultural',
                                             name='populated_places_simple')

        centr = Centroids.from_vector_file(shp_file, dst_crs=DEF_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, DEF_CRS))
        self.assertAlmostEqual(centr.lon[0], 12.453386544971766)
        self.assertAlmostEqual(centr.lon[-1], 114.18306345846304)
        self.assertAlmostEqual(centr.lat[0], 41.903282179960115)
        self.assertAlmostEqual(centr.lat[-1], 22.30692675357551)

        centr = Centroids.from_vector_file(shp_file, dst_crs=ALT_CRS)
        self.assertTrue(u_coord.equal_crs(centr.crs, ALT_CRS))


    def test_write_read_h5(self):
        """Write and read hdf5 format"""
        file_name = str(DATA_DIR.joinpath('test_centr.h5'))
        self.centr.write_hdf5(file_name)
        centr_read = Centroids.from_hdf5(file_name)
        self.assertEqual(self.centr, centr_read)


class TestCentroidsFuncs(unittest.TestCase):
    """Test Centroids methods"""
    def test_select_pass(self):
        """Test Centroids.select method"""
        region_id = np.zeros(VEC_LAT.size)
        region_id[[2, 4]] = 10
        centr = Centroids(latitude=VEC_LAT, longitude=VEC_LON, region_id=region_id)

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
            latitude=np.array([-5, -3, 0, 3, 5]),
            longitude=np.array([-180, -175, -170, 170, 175]),
            region_id=np.zeros(5))
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

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRaster)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVector))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroids))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroidsFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
