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

Test Exposure base class.
"""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import DistanceMetric
import rasterio
from rasterio.windows import Window

from climada import CONFIG
from climada.entity.exposures.base import Exposures, INDICATOR_IF, \
     INDICATOR_CENTR, add_sea, DEF_REF_YEAR, DEF_VALUE_UNIT
from climada.entity.tag import Tag
from climada.hazard.base import Hazard, Centroids
from climada.util.constants import ENT_TEMPLATE_XLS, ONE_LAT_KM, DEF_CRS, HAZ_DEMO_FL
from climada.util.coordinates import coord_on_land, equal_crs

DATA_DIR = CONFIG.exposures.test_data.dir()

def good_exposures():
    """Followng values are defined for each exposure"""
    data = {}
    data['latitude'] = np.array([1, 2, 3])
    data['longitude'] = np.array([2, 3, 4])
    data['value'] = np.array([1, 2, 3])
    data['deductible'] = np.array([1, 2, 3])
    data[INDICATOR_IF + 'NA'] = np.array([1, 2, 3])
    data['category_id'] = np.array([1, 2, 3])
    data['region_id'] = np.array([1, 2, 3])
    data[INDICATOR_CENTR + 'TC'] = np.array([1, 2, 3])

    expo = Exposures(gpd.GeoDataFrame(data=data))
    return expo

class TestFuncs(unittest.TestCase):
    """Check assign function"""

    def test_assign_pass(self):
        """Check that attribute `assigned` is correctly set."""
        np_rand = np.random.RandomState(123456789)

        haz = Hazard('FL')
        haz.set_raster([HAZ_DEMO_FL], window=Window(10, 20, 50, 60))
        haz.raster_to_vector()
        ncentroids = haz.centroids.size

        exp = Exposures()
        exp.gdf.crs = haz.centroids.crs

        # some are matching exactly, some are geographically close
        exp.gdf['longitude'] = np.concatenate([
            haz.centroids.lon, haz.centroids.lon + 0.001 * (-0.5 + np_rand.rand(ncentroids))])
        exp.gdf['latitude'] = np.concatenate([
            haz.centroids.lat, haz.centroids.lat + 0.001 * (-0.5 + np_rand.rand(ncentroids))])
        expected_result = np.concatenate([np.arange(ncentroids), np.arange(ncentroids)])

        exp.assign_centroids(haz)
        self.assertEqual(exp.gdf.shape[0], len(exp.gdf[INDICATOR_CENTR + 'FL']))
        np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values, expected_result)

    def test_read_raster_pass(self):
        """set_from_raster"""
        exp = Exposures()
        exp.set_from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        self.assertTrue(equal_crs(exp.crs, DEF_CRS))
        self.assertAlmostEqual(exp.gdf['latitude'].max(),
                               10.248220966978932 - 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp.gdf['latitude'].min(),
                               10.248220966978932 - 0.009000000000000341
                               / 2 - 59 * 0.009000000000000341)
        self.assertAlmostEqual(exp.gdf['longitude'].min(),
                               -69.2471495969998 + 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp.gdf['longitude'].max(),
                               -69.2471495969998 + 0.009000000000000341
                               / 2 + 49 * 0.009000000000000341)
        self.assertEqual(len(exp.gdf), 60 * 50)
        self.assertAlmostEqual(exp.gdf.value.values.reshape((60, 50))[25, 12], 0.056825936)

    def test_assign_raster_pass(self):
        """Test assign_centroids with raster hazard"""
        haz = Hazard('FL')

        # explicit, easy-to-understand raster centroids for hazard
        haz.centroids = Centroids()
        haz.centroids.meta = {
            'count': 1, 'crs': DEF_CRS,
            'width': 20, 'height': 10,
            'transform': rasterio.Affine(1.5, 0.0, -20, 0.0, -1.4, 8)
        }

        # explicit points with known results (see `expected_result` for details)
        exp = Exposures(crs=DEF_CRS)
        exp.gdf['longitude'] = np.array([
            -20.1, -20.0, -19.8, -19.0, -18.6, -18.4,
            -19.0, -19.0, -19.0, -19.0,
            -20.1, 0.0, 10.1, 10.1, 10.1, 0.0, -20.2, -20.3,
            -6.4, 9.8, 0.0,
        ])
        exp.gdf['latitude'] = np.array([
            7.3, 7.3, 7.3, 7.3, 7.3, 7.3,
            8.1, 7.9, 6.7, 6.5,
            8.1, 8.2, 8.3, 0.0, -6.1, -6.2, -6.3, 0.0,
            -1.9, -1.7, 0.0,
        ])
        exp.assign_centroids(haz)

        expected_result = [
            # constant y-value, varying x-value
            -1, 0, 0, 0, 0, 1,
            # constant x-value, varying y-value
            -1, 0, 0, 20,
            # out of bounds: topleft, top, topright, right, bottomright, bottom, bottomleft, left
            -1, -1, -1, -1, -1, -1, -1, -1,
            # some explicit points within the raster
            149, 139, 113,
        ]
        np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values, expected_result)


    def test_assign_raster_same_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures()
        exp.set_from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        haz = Hazard('FL')
        haz.set_raster([HAZ_DEMO_FL], window=Window(10, 20, 50, 60))
        exp.assign_centroids(haz)
        self.assertTrue(np.array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values,
                                       np.arange(haz.centroids.size, dtype=int)))

    def test_assign_large_hazard_subset_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures()
        exp.set_from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.gdf.latitude[[0, 1]] = exp.gdf.latitude[[1, 0]]
        exp.gdf.longitude[[0, 1]] = exp.gdf.longitude[[1, 0]]
        exp.check()
        haz = Hazard('FL')
        haz.set_raster([HAZ_DEMO_FL])
        haz.raster_to_vector()
        exp.assign_centroids(haz)
        assigned_centroids = haz.centroids.select(sel_cen=exp.gdf[INDICATOR_CENTR + 'FL'].values)
        np.testing.assert_array_equal(assigned_centroids.lat, exp.gdf.latitude)
        np.testing.assert_array_equal(assigned_centroids.lon, exp.gdf.longitude)


class TestChecker(unittest.TestCase):
    """Test logs of check function"""

    def test_info_logs_pass(self):
        """Correct exposures definition"""
        with self.assertLogs('climada.entity.exposures.base', level='INFO') as cm:
            expo = good_exposures()
            expo.check()
        self.assertIn('meta set to default value', cm.output[0])
        self.assertIn('tag set to default value', cm.output[1])
        self.assertIn('ref_year set to default value', cm.output[2])
        self.assertIn('value_unit set to default value', cm.output[3])
        self.assertIn('crs set to default value', cm.output[4])
        # the following lines are from an iteration over a set, so that the order may change:
        self.assertTrue(any(f'{var} not set' in o for o in cm.output[5:7])
                        for var in ['geometry', 'cover'])

    def test_error_logs_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.gdf.drop(['longitude'], inplace=True, axis=1)

        with self.assertLogs('climada.entity.exposures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('longitude missing', cm.output[0])

    def test_error_geometry_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.set_geometry_points()
        expo.gdf.latitude.values[0] = 5

        with self.assertRaises(ValueError):
            expo.check()

class TestIO(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def test_read_template_pass(self):
        """Wrong exposures definition"""
        df = pd.read_excel(ENT_TEMPLATE_XLS)
        exp_df = Exposures(df)
        # set metadata
        exp_df.ref_year = 2020
        exp_df.tag = Tag(ENT_TEMPLATE_XLS, 'ENT_TEMPLATE_XLS')
        exp_df.value_unit = 'XSD'
        exp_df.check()

    def test_io_hdf5_pass(self):
        """write and read hdf5"""
        exp_df = Exposures(pd.read_excel(ENT_TEMPLATE_XLS))
        exp_df.set_geometry_points()
        exp_df.check()
        # set metadata
        exp_df.ref_year = 2020
        exp_df.tag = Tag(ENT_TEMPLATE_XLS, 'ENT_TEMPLATE_XLS')
        exp_df.value_unit = 'XSD'

        file_name = DATA_DIR.joinpath('test_hdf5_exp.h5')
        exp_df.write_hdf5(file_name)

        exp_read = Exposures()
        exp_read.read_hdf5(file_name)

        self.assertEqual(exp_df.ref_year, exp_read.ref_year)
        self.assertEqual(exp_df.value_unit, exp_read.value_unit)
        self.assertEqual(exp_df.crs, exp_read.crs)
        self.assertEqual(exp_df.tag.file_name, exp_read.tag.file_name)
        self.assertEqual(exp_df.tag.description, exp_read.tag.description)
        self.assertTrue(np.array_equal(exp_df.gdf.latitude.values,    exp_read.gdf.latitude.values))
        self.assertTrue(np.array_equal(exp_df.gdf.longitude.values,   exp_read.gdf.longitude.values))
        self.assertTrue(np.array_equal(exp_df.gdf.value.values,       exp_read.gdf.value.values))
        self.assertTrue(np.array_equal(exp_df.gdf.deductible.values,  exp_read.gdf.deductible.values))
        self.assertTrue(np.array_equal(exp_df.gdf.cover.values,       exp_read.gdf.cover.values))
        self.assertTrue(np.array_equal(exp_df.gdf.region_id.values,   exp_read.gdf.region_id.values))
        self.assertTrue(np.array_equal(exp_df.gdf.category_id.values, exp_read.gdf.category_id.values))
        self.assertTrue(np.array_equal(exp_df.gdf.if_TC.values,       exp_read.gdf.if_TC.values))
        self.assertTrue(np.array_equal(exp_df.gdf.centr_TC.values,    exp_read.gdf.centr_TC.values))
        self.assertTrue(np.array_equal(exp_df.gdf.if_FL.values,       exp_read.gdf.if_FL.values))
        self.assertTrue(np.array_equal(exp_df.gdf.centr_FL.values,    exp_read.gdf.centr_FL.values))

        for point_df, point_read in zip(exp_df.gdf.geometry.values, exp_read.gdf.geometry.values):
            self.assertEqual(point_df.x, point_read.x)
            self.assertEqual(point_df.y, point_read.y)

class TestAddSea(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        exp = Exposures()
        exp.gdf['value'] = np.arange(0, 1.0e6, 1.0e5)
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp.gdf['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp.gdf['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp.gdf['region_id'] = np.ones(10)
        exp.gdf['if_TC'] = np.ones(10)
        exp.ref_year = 2015
        exp.value_unit = 'XSD'
        exp.check()

        sea_coast = 100
        sea_res_km = 50
        sea_res = (sea_coast, sea_res_km)
        exp_sea = add_sea(exp, sea_res)
        exp_sea.check()

        sea_coast /= ONE_LAT_KM
        sea_res_km /= ONE_LAT_KM

        min_lat = min_lat - sea_coast
        max_lat = max_lat + sea_coast
        min_lon = min_lon - sea_coast
        max_lon = max_lon + sea_coast
        self.assertEqual(np.min(exp_sea.gdf.latitude), min_lat)
        self.assertEqual(np.min(exp_sea.gdf.longitude), min_lon)
        self.assertTrue(np.array_equal(exp_sea.gdf.value.values[:10], np.arange(0, 1.0e6, 1.0e5)))
        self.assertEqual(exp_sea.ref_year, exp.ref_year)
        self.assertEqual(exp_sea.value_unit, exp.value_unit)

        on_sea_lat = exp_sea.gdf.latitude.values[11:]
        on_sea_lon = exp_sea.gdf.longitude.values[11:]
        res_on_sea = coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = ~res_on_sea
        self.assertTrue(np.all(res_on_sea))

        dist = DistanceMetric.get_metric('haversine')
        self.assertAlmostEqual(dist.pairwise([
            [exp_sea.gdf.longitude.values[-1], exp_sea.gdf.latitude.values[-1]],
            [exp_sea.gdf.longitude.values[-2], exp_sea.gdf.latitude.values[-2]],
        ])[0][1], sea_res_km)


class TestConcat(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def setUp(self):
        exp = Exposures(crs='epsg:3395')
        exp.gdf['value'] = np.arange(0, 1.0e6, 1.0e5)
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp.gdf['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp.gdf['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp.gdf['region_id'] = np.ones(10)
        exp.gdf['if_TC'] = np.ones(10)
        exp.ref_year = 2015
        exp.value_unit = 'XSD'
        self.dummy = exp

    def test_concat_pass(self):
        """Test condat function with fake data."""

        self.dummy.check()

        catexp = Exposures.concat([self.dummy, self.dummy.gdf, pd.DataFrame(self.dummy.gdf.values, columns=self.dummy.gdf.columns), self.dummy])
        self.assertEqual(self.dummy.gdf.shape, (10,5))
        self.assertEqual(catexp.gdf.shape, (40,5))
        self.assertEqual(catexp.gdf.crs, 'epsg:3395')

    def test_concat_fail(self):
        """Test failing concat function with fake data."""

        with self.assertRaises(TypeError):
            Exposures.concat([self.dummy, self.dummy.gdf, self.dummy.gdf.values, self.dummy])


class TestGeoDFFuncs(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def test_copy_pass(self):
        """Test copy function."""
        exp = good_exposures()
        exp.check()
        exp_copy = exp.copy()
        self.assertIsInstance(exp_copy, Exposures)
        self.assertEqual(exp_copy.crs, exp.crs)
        self.assertEqual(exp_copy.ref_year, exp.ref_year)
        self.assertEqual(exp_copy.value_unit, exp.value_unit)
        self.assertEqual(exp_copy.tag.description, exp.tag.description)
        self.assertEqual(exp_copy.tag.file_name, exp.tag.file_name)
        self.assertTrue(np.array_equal(exp_copy.gdf.latitude.values, exp.gdf.latitude.values))
        self.assertTrue(np.array_equal(exp_copy.gdf.longitude.values, exp.gdf.longitude.values))

    def test_to_crs_inplace_pass(self):
        """Test to_crs function inplace."""
        exp = good_exposures()
        exp.set_geometry_points()
        exp.check()
        exp.to_crs({'init': 'epsg:3395'}, inplace=True)
        self.assertIsInstance(exp, Exposures)
        self.assertEqual(exp.crs, {'init': 'epsg:3395'})
        self.assertEqual(exp.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp.tag.description, '')
        self.assertEqual(exp.tag.file_name, '')

    def test_to_crs_pass(self):
        """Test to_crs function copy."""
        exp = good_exposures()
        exp.set_geometry_points()
        exp.check()
        exp_tr = exp.to_crs({'init': 'epsg:3395'})
        self.assertIsInstance(exp, Exposures)
        self.assertEqual(exp.crs, DEF_CRS)
        self.assertEqual(exp_tr.crs, {'init': 'epsg:3395'})
        self.assertEqual(exp_tr.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp_tr.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp_tr.tag.description, '')
        self.assertEqual(exp_tr.tag.file_name, '')

    def test_constructor_pass(self):
        """Test initialization with input GeiDataFrame"""
        in_gpd = gpd.GeoDataFrame()
        in_gpd['value'] = np.zeros(10)
        in_gpd.ref_year = 2015
        in_exp = Exposures(in_gpd, ref_year=2015)
        self.assertEqual(in_exp.ref_year, 2015)
        self.assertTrue(np.array_equal(in_exp.gdf.value, np.zeros(10)))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecker)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuncs))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAddSea))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGeoDFFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
