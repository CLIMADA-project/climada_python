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
import os
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import DistanceMetric
from climada.util.coordinates import coord_on_land
from rasterio.windows import Window

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IF, \
INDICATOR_CENTR, add_sea, DEF_REF_YEAR, DEF_VALUE_UNIT
from climada.hazard.base import Hazard
from climada.util.constants import ENT_TEMPLATE_XLS, ONE_LAT_KM, DEF_CRS, HAZ_DEMO_FL
from climada.util.coordinates import equal_crs

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

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
        """Check that assigned attribute is correctly set."""
        # Fill with dummy values
        expo = good_exposures()
        expo.check()
        # Fill with dummy values the centroids
        haz = Hazard('TC')
        haz.centroids.set_lat_lon(np.ones(expo.shape[0] + 6), np.ones(expo.shape[0] + 6))
        # assign
        expo.assign_centroids(haz)

        # check assigned variable has been set with correct length
        self.assertEqual(expo.shape[0], len(expo[INDICATOR_CENTR + 'TC']))

    def test_read_raster_pass(self):
        """set_from_raster"""
        exp = Exposures()
        exp.set_from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        self.assertTrue(equal_crs(exp.crs, DEF_CRS))
        self.assertAlmostEqual(exp['latitude'].max(),
                               10.248220966978932 - 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp['latitude'].min(),
                               10.248220966978932 - 0.009000000000000341
                               / 2 - 59 * 0.009000000000000341)
        self.assertAlmostEqual(exp['longitude'].min(),
                               -69.2471495969998 + 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp['longitude'].max(),
                               -69.2471495969998 + 0.009000000000000341
                               / 2 + 49 * 0.009000000000000341)
        self.assertEqual(len(exp), 60 * 50)
        self.assertAlmostEqual(exp.value.values.reshape((60, 50))[25, 12], 0.056825936)

    def test_assign_raster_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures()
        exp['longitude'] = np.array([-69.235, -69.2427, -72, -68.8016496, 30])
        exp['latitude'] = np.array([10.235, 10.226, 2, 9.71272097, 50])
        exp.crs = DEF_CRS
        haz = Hazard('FL')
        haz.set_raster([HAZ_DEMO_FL], window=Window(10, 20, 50, 60))
        exp.assign_centroids(haz)
        self.assertEqual(exp[INDICATOR_CENTR + 'FL'][0], 51)
        self.assertEqual(exp[INDICATOR_CENTR + 'FL'][1], 100)
        self.assertEqual(exp[INDICATOR_CENTR + 'FL'][2], -1)
        self.assertEqual(exp[INDICATOR_CENTR + 'FL'][3], 3000 - 1)
        self.assertEqual(exp[INDICATOR_CENTR + 'FL'][4], -1)


    def test_assign_raster_same_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures()
        exp.set_from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        haz = Hazard('FL')
        haz.set_raster([HAZ_DEMO_FL], window=Window(10, 20, 50, 60))
        exp.assign_centroids(haz)
        self.assertTrue(np.array_equal(exp[INDICATOR_CENTR + 'FL'].values,
                                       np.arange(haz.centroids.size, dtype=int)))

class TestChecker(unittest.TestCase):
    """Test logs of check function"""

    def test_info_logs_pass(self):
        """Wrong exposures definition"""
        expo = good_exposures()

        with self.assertLogs('climada.entity.exposures.base', level='INFO') as cm:
            expo.check()
        self.assertIn('crs set to default value', cm.output[0])
        self.assertIn('tag metadata set to default value', cm.output[1])
        self.assertIn('ref_year metadata set to default value', cm.output[2])
        self.assertIn('value_unit metadata set to default value', cm.output[3])
        self.assertIn('meta metadata set to default value', cm.output[4])
        self.assertIn('geometry not set', cm.output[6])
        self.assertIn('cover not set', cm.output[5])

    def test_error_logs_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo = expo.drop(['longitude'], axis=1)

        with self.assertLogs('climada.entity.exposures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('longitude missing', cm.output[0])

    def test_error_geometry_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.set_geometry_points()
        expo.latitude.values[0] = 5

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

        file_name = os.path.join(DATA_DIR, 'test_hdf5_exp.h5')
        exp_df.write_hdf5(file_name)

        exp_read = Exposures()
        exp_read.read_hdf5(file_name)

        self.assertEqual(exp_df.ref_year, exp_read.ref_year)
        self.assertEqual(exp_df.value_unit, exp_read.value_unit)
        self.assertEqual(exp_df.crs, exp_read.crs)
        self.assertEqual(exp_df.tag.file_name, exp_read.tag.file_name)
        self.assertEqual(exp_df.tag.description, exp_read.tag.description)
        self.assertTrue(np.array_equal(exp_df.latitude.values, exp_read.latitude.values))
        self.assertTrue(np.array_equal(exp_df.longitude.values, exp_read.longitude.values))
        self.assertTrue(np.array_equal(exp_df.value.values, exp_read.value.values))
        self.assertTrue(np.array_equal(exp_df.deductible.values, exp_read.deductible.values))
        self.assertTrue(np.array_equal(exp_df.cover.values, exp_read.cover.values))
        self.assertTrue(np.array_equal(exp_df.region_id.values, exp_read.region_id.values))
        self.assertTrue(np.array_equal(exp_df.category_id.values, exp_read.category_id.values))
        self.assertTrue(np.array_equal(exp_df.if_TC.values, exp_read.if_TC.values))
        self.assertTrue(np.array_equal(exp_df.centr_TC.values, exp_read.centr_TC.values))
        self.assertTrue(np.array_equal(exp_df.if_FL.values, exp_read.if_FL.values))
        self.assertTrue(np.array_equal(exp_df.centr_FL.values, exp_read.centr_FL.values))

        for point_df, point_read in zip(exp_df.geometry.values, exp_read.geometry.values):
            self.assertEqual(point_df.x, point_read.x)
            self.assertEqual(point_df.y, point_read.y)

class TestAddSea(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        exp = Exposures()
        exp['value'] = np.arange(0, 1.0e6, 1.0e5)
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp['region_id'] = np.ones(10)
        exp['if_TC'] = np.ones(10)
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
        self.assertEqual(np.min(exp_sea.latitude), min_lat)
        self.assertEqual(np.min(exp_sea.longitude), min_lon)
        self.assertTrue(np.array_equal(exp_sea.value.values[:10], np.arange(0, 1.0e6, 1.0e5)))
        self.assertEqual(exp_sea.ref_year, exp.ref_year)
        self.assertEqual(exp_sea.value_unit, exp.value_unit)

        on_sea_lat = exp_sea.latitude.values[11:]
        on_sea_lon = exp_sea.longitude.values[11:]
        res_on_sea = coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = ~res_on_sea
        self.assertTrue(np.all(res_on_sea))

        dist = DistanceMetric.get_metric('haversine')
        self.assertAlmostEqual(dist.pairwise([
            [exp_sea.longitude.values[-1], exp_sea.latitude.values[-1]],
            [exp_sea.longitude.values[-2], exp_sea.latitude.values[-2]],
        ])[0][1], sea_res_km)


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
        self.assertTrue(np.array_equal(exp_copy.latitude.values, exp.latitude.values))
        self.assertTrue(np.array_equal(exp_copy.longitude.values, exp.longitude.values))

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

    def test_constructoer_pass(self):
        """Test initialization with input GeiDataFrame"""
        in_gpd = gpd.GeoDataFrame()
        in_gpd['value'] = np.zeros(10)
        in_gpd.ref_year = 2015
        in_exp = Exposures(in_gpd)
        self.assertEqual(in_exp.ref_year, 2015)
        self.assertTrue(np.array_equal(in_exp.value, np.zeros(10)))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecker)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuncs))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAddSea))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGeoDFFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
