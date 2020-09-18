"""
This file is part of CLIMADA.

Copyright (C) 2019 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

Integration Test for open_street_map.py and 3 time consuming unit tests
"""

import math
import unittest
from climada.entity import Exposures
from climada.entity.exposures import open_street_map as OSM
import os
import random
import geopandas
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class TestOpenStreetMapModule(unittest.TestCase):
    def test_get_features_osm(self):
        """test get_features_osm"""
        Low_Value_gdf_47_8 = OSM.get_features_OSM([47.2, 8.0, 47.3, 8.07],
                                                  {'waterway', 'landuse=forest'},
                                                  DATA_DIR, check_plot=0)
        self.assertIsInstance(Low_Value_gdf_47_8, geopandas.GeoDataFrame)
        self.assertNotIn('LineString', Low_Value_gdf_47_8.geometry.type)
        self.assertTrue('waterway' in Low_Value_gdf_47_8.Item.unique())
        self.assertTrue('landuse=forest' in Low_Value_gdf_47_8.Item.unique())
        self.assertEqual(len(Low_Value_gdf_47_8.columns), 6)

    def test_get_highValueArea(self):
        """test get_highValueArea"""
        Low_Value_gdf_47_8 = OSM.get_features_OSM([47.2, 8.0, 47.3, 8.07],
                                                  {'waterway', 'landuse=forest'},
                                                  DATA_DIR, check_plot=0)
        High_Value_gdf_47_8 = OSM.get_highValueArea([47.2, 8.0, 47.3, 8.07], DATA_DIR,
                                                    DATA_DIR + '/OSM_features_47_8.shp',
                                                    check_plot=0)
        self.assertTrue(math.isclose(47.2, High_Value_gdf_47_8.bounds.miny, rel_tol=0.05))
        self.assertTrue(math.isclose(8.07, High_Value_gdf_47_8.bounds.maxx, rel_tol=0.05))
        self.assertIsInstance(High_Value_gdf_47_8, geopandas.GeoDataFrame)

    def test_get_osmstencil_litpop(self):
        """test for get_osmstencil_litpop"""
        for mode in ['proportional', 'nearest', 'even']:
            exposure_high_47_8 = OSM.get_osmstencil_litpop(
                [47.2, 8.0, 47.3, 8.07], 'CHE', mode,
                os.path.join(DATA_DIR, 'High_Value_Area_47_8.shp'),
                DATA_DIR, check_plot=0)
            self.assertIsInstance(exposure_high_47_8, Exposures)
            self.assertEqual(len(exposure_high_47_8.columns), 8)
            self.assertGreater(
                exposure_high_47_8.iloc[random.randint(0, len(exposure_high_47_8))].value,
                0)

    def test_make_osmexposure(self):
        """test for make_osmexposure"""
        # With default 5400 Chf / m2 values
        buildings_47_8_default = OSM.make_osmexposure(
            os.path.join(DATA_DIR, 'buildings_47_8.shp'),
            mode='default', save_path=DATA_DIR, check_plot=0)
        self.assertIsInstance(buildings_47_8_default, Exposures)
        self.assertEqual(len(buildings_47_8_default.columns), 12)
        self.assertEqual(
            buildings_47_8_default.loc[
                random.randint(0, len(buildings_47_8_default))].geometry.type,
            "Point")
        self.assertGreater(
            buildings_47_8_default.loc[random.randint(0, len(buildings_47_8_default))].value,
            0)
        self.assertGreater(
            buildings_47_8_default.loc[
                random.randint(0, len(buildings_47_8_default))].projected_area,
            0)

        # With LitPop values
        buildings_47_8_LitPop = OSM.make_osmexposure(os.path.join(DATA_DIR, 'buildings_47_8.shp'),
                                                     country='CHE', mode="LitPop",
                                                     save_path=DATA_DIR, check_plot=0)
        self.assertIsInstance(buildings_47_8_LitPop, Exposures)
        self.assertEqual(len(buildings_47_8_LitPop.columns), 12)
        self.assertEqual(
            buildings_47_8_LitPop.loc[random.randint(0, len(buildings_47_8_LitPop))].geometry.type,
            "Point")
        self.assertGreater(
            buildings_47_8_LitPop.loc[random.randint(0, len(buildings_47_8_LitPop))].value,
            0)
        self.assertGreater(
            buildings_47_8_LitPop.loc[
                random.randint(0, len(buildings_47_8_LitPop))].projected_area,
            0)

        for mode in ['default', 'LitPop']:
            os.remove(DATA_DIR + '/exposure_buildings_' + mode + '_47_7.h5')


class TestOSMlongUnitTests(unittest.TestCase):

    def test_get_litpop_bbox(self):
        """test _get_litpop_bbox within get_osmstencil_litpop function"""
        # Define and load parameters
        country = 'CHE'
        highValueArea = geopandas.read_file(os.path.join(DATA_DIR, 'High_Value_Area_47_8.shp'))
        # Execute function
        exp_sub = OSM._get_litpop_bbox(country, highValueArea)
        self.assertTrue(
            math.isclose(min(exp_sub.latitude), highValueArea.bounds.miny, rel_tol=1e-2))
        self.assertTrue(
            math.isclose(max(exp_sub.longitude), highValueArea.bounds.maxx, rel_tol=1e-2))

    def test_split_exposure_highlow(self):
        """test _split_exposure_highlow within get_osmstencil_litpop function"""
        # Define and load parameters:
        country = 'CHE'  # this takes too long for unit test probably
        highValueArea = geopandas.read_file(os.path.join(DATA_DIR, 'High_Value_Area_47_8.shp'))
        exp_sub = OSM._get_litpop_bbox(country, highValueArea)
        High_Value_Area_gdf = geopandas.read_file(os.path.join(DATA_DIR,
                                                               'High_Value_Area_47_8.shp'))
        # execute function
        for mode in {'proportional', 'even'}:
            print('testing mode %s' % mode)
            exp_sub_high = OSM._split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf)
            self.assertTrue(math.isclose(sum(exp_sub_high.value),
                                         sum(exp_sub.value),
                                         rel_tol=0.01))
        print('testing mode nearest neighbour')
        exp_sub_high = OSM._split_exposure_highlow(exp_sub, "nearest", High_Value_Area_gdf)
        self.assertTrue(math.isclose(sum(exp_sub_high.value), sum(exp_sub.value), rel_tol=0.1))

    def test_assign_values_exposure(self):
        """test _assign_values_exposure within make_osmexposure function"""
        # Define and load parameters:
        # function tested previously
        building_gdf = OSM._get_midpoints(os.path.join(DATA_DIR, 'buildings_47_8.shp'))
        mode = 'LitPop'     # mode LitPop takes too long for unit test, moved to integration test
        country = 'CHE'
        # Execute function
        High_Value_Area_gdf = OSM._assign_values_exposure(building_gdf, mode, country)
        self.assertGreater(
            High_Value_Area_gdf.loc[random.randint(0, len(High_Value_Area_gdf))].value,
            0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOpenStreetMapModule)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOSMlongUnitTests))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
