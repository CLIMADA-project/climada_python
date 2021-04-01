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

Unit Tests on Open Street Map exposures.
"""

import unittest
import random

import pandas as pd
import geopandas

from climada import CONFIG
from climada.entity.exposures import open_street_map as OSM

DATA_DIR = CONFIG.test_data.dir()

class TestOSMFunctions(unittest.TestCase):
    """Test OSM Class methods"""

    def test_osm_api_query(self):
        """test _osm_api_query within get_features_OSM function"""
        bbox = [47.2, 8.03, 47.3, 8.07]
        item = 'landuse=forest'
        result_NodesFromWays, result_NodesWaysFromRels = OSM._osm_api_query(item, bbox)
        self.assertGreater(len(result_NodesFromWays.nodes), 0)
        self.assertGreater(len(result_NodesFromWays.ways), 0)
        self.assertGreater(len(result_NodesWaysFromRels.relations), 0)

    def test_format_shape_osm(self):
        """test _format_shape_osm function within get_features_OSM function:"""
        # define input parameters
        bbox = [47.2, 8.03, 47.3, 8.07]
        item = 'landuse=forest'
        result_NodesFromWays, result_NodesWaysFromRels = OSM._osm_api_query(item, bbox)

        # Execute function to be tested
        globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))] = \
        OSM._format_shape_osm(bbox, result_NodesFromWays, result_NodesWaysFromRels, item, DATA_DIR)
        # check that shapes were found both from relations and from way/nodes query
        self.assertGreater(len(
            globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))]),
            len(result_NodesWaysFromRels.relations))
        # check that geometry row exists and contains polygons / multipolygons
        self.assertEqual(
            globals()[
                str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))
            ].iloc[0].geometry.type, 'Polygon')
        self.assertEqual(
            globals()[
                str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))
            ].iloc[-1].geometry.type, 'MultiPolygon')

    def test_combine_dfs_osm(self):
        """test _combine_dfs_osm function within get_features_OSM function:"""
        # define input parameters
        types = {'landuse=forest', 'waterway'}
        bbox = [47.2, 8.03, 47.3, 8.07]
        for item in types:
            print(item)
            # strictly, this doesnt belong here, but needs to be invoked (tested before)
            result_NodesFromWays, result_NodesWaysFromRels = OSM._osm_api_query(item, bbox)
            globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))] = \
            OSM._format_shape_osm(bbox, result_NodesFromWays, result_NodesWaysFromRels,
                                  item, DATA_DIR)
        # Execute function
        OSM_features_gdf_combined = \
            geopandas.GeoDataFrame(
                pd.DataFrame(columns=['Item', 'Name', 'Type', 'Natural_Type', 'geometry']),
                crs='epsg:4326', geometry='geometry')
        for item in types:
            print('adding results from %s ...' % item)
            OSM_features_gdf_combined = \
            OSM_features_gdf_combined.append(
                globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))],
                ignore_index=True)
        i = 0
        for geom in OSM_features_gdf_combined.geometry:
            if geom.type == 'LineString':
                OSM_features_gdf_combined.geometry[i] = geom.buffer(0.000045)
            i += 1
        # waterway_gdf_all_47_8 is in the global scope as defined in line 76 :|
        self.assertGreater(len(OSM_features_gdf_combined), len(waterway_gdf_all_47_8))
        self.assertNotIn('LineString', OSM_features_gdf_combined.geometry.type)

    def test_makeUnion(self):
        """test makeUnion function within get_highValueArea function"""
        gdf_all = geopandas.read_file(DATA_DIR.joinpath('OSM_features_47_8.shp'))

        # Execute function
        Low_Value_Union = OSM._makeUnion(gdf_all)
        self.assertEqual(Low_Value_Union.type, 'MultiPolygon')
        self.assertTrue(Low_Value_Union.is_valid)

# this takes too long for unit test (loads CHE LitPop exposure!) Moved to integration test
#    def test_get_litpop_bbox(self):
#        """test _get_litpop_bbox within get_osmstencil_litpop function"""
#        # Define and load parameters
#        country = 'CHE'
#        highValueArea = geopandas.read_file(DATA_DIR.joinpath('High_Value_Area_47_8.shp'))
#
#        # Execute function
#        exp_sub = OSM._get_litpop_bbox(country, highValueArea)
#        self.assertTrue(math.isclose(min(exp_sub.latitude), highValueArea.bounds.miny, rel_tol=1e-2))
#        self.assertTrue(math.isclose(max(exp_sub.longitude), highValueArea.bounds.maxx, rel_tol=1e-2))

# this takes too long, since loadas CHE LitPop exposure. Moved to integration test
#    def test_split_exposure_highlow(self):
#        """test _split_exposure_highlow within get_osmstencil_litpop function"""
#        # Define and load parameters:
#        country = 'CHE' #this takes too long for unit test probably
#        highValueArea = geopandas.read_file(DATA_DIR.joinpath('High_Value_Area_47_8.shp'))
#        exp_sub = OSM._get_litpop_bbox(country, highValueArea)
#        High_Value_Area_gdf = geopandas.read_file(DATA_DIR.joinpath('High_Value_Area_47_8.shp'))
#
#        # execute function
#        for mode in {'proportional','even'}:
#            print('testing mode %s' %mode)
#            exp_sub_high = OSM._split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf)
#            self.assertAlmostEqual(sum(exp_sub_high.value), sum(exp_sub.value))
#        print('testing mode nearest neighbour')
#        exp_sub_high = OSM._split_exposure_highlow(exp_sub, "nearest", High_Value_Area_gdf)
#        self.assertTrue(math.isclose(sum(exp_sub_high.value), sum(exp_sub.value), rel_tol=0.1))

    def test_get_midpoints(self):
        """test _get_midpoints within make_osmexposure function"""
        # Define and load parameters:
        building_path = DATA_DIR.joinpath('buildings_47_8.shp')

        building_gdf = OSM._get_midpoints(building_path)
        self.assertEqual(building_gdf.loc[random.randint(0, len(building_gdf))].geometry.type,
                         'Point')
        self.assertGreater(building_gdf.loc[random.randint(0, len(building_gdf))].projected_area,
                           0)

    def test_assign_values_exposure(self):
        """test _assign_values_exposure within make_osmexposure function"""
        # Define and load parameters:
        # function tested previously
        building_gdf = OSM._get_midpoints(DATA_DIR.joinpath('buildings_47_8.shp'))
        # mode LitPop takes too long for unit test, since loads entire CH-litpop exposure!
        # moved to integration test
        mode = 'default'
        country = 'CHE'
        # Execute Function
        High_Value_Area_gdf = OSM._assign_values_exposure(building_gdf, mode, country)
        self.assertGreater(
            High_Value_Area_gdf.loc[random.randint(0, len(High_Value_Area_gdf))].value,
            0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOSMFunctions)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLitPopClass))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
