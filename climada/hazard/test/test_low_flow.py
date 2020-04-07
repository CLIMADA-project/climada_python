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

Test low flow module.
"""
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd
from climada.hazard.low_flow import LowFlow, unique_clusters
from climada.util.constants import DATA_DIR

INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FN_STR_DEMO = 'co2_dis_global_daily_DEMO_FR'



def init_test_data_unqiue_clusters():
    """creates sandbox test data for 2D cluster IDs for test of identification of
    unique 3D clusters"""

    df = pd.DataFrame(columns=['target_cluster', 'cluster_id', 'c_lat_lon', \
                               'c_lat_dt_month', 'c_lon_dt_month'])
    
    df.c_lon_dt_month = np.array([1, 1, 1, 1, 2, 2, 3, 4, 5, 4, 4, 5, 6, -1, -1])
    df.c_lat_dt_month = np.array([1, -1, 2, 2, 2, 3, 5, 3, 4, 6, 6, 5, 7, -1, 1])
    df.c_lat_lon = np.array([1, 3, 1, 3, 3, 3, 5, 3, 5, 3, 4, 5, 2, -1, -1])
    df.target_cluster = [ 1, -1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 3, -1, -1]
    df.cluster_id = np.zeros(len(df.target_cluster), dtype=int)-1
    return df

def init_test_data_clustering():
    """creates sandbox test data for monthly days below threshold data
    for testing clustering"""
    
    df = pd.DataFrame(columns=['lat', 'lon', 'dis', \
                              'dt_month', 'target_cluster'])

    df.lat = np.array([-0, -0, -.5, -.5, -1, -.5, -1, -0, -.5, -1, -1, -1.5, -2.5])
    df.lon = np.array([ 0,  1, 0,   1.5,  2, 0,    0,  1, 1.5,  0,  2,  0,    2.5])
    df.dt_month = np.array([1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3])
    # df.target_cluster = [1, 2, 1, 2, 2, 1, 1, 3, 3, 1, 3, 1, 4]
    df.dis = [5, 11, 5, 11, 11, 10, 10, 22, 22, 20, 22, 20, 1]
    # df.ass_c = np.zeros(len(df.tar_c))-1

    df['iter_ev'] = np.ones(len(df), bool)
    df['cons_id'] = np.zeros(len(df), int)-1
    return df


class TestLowFlow(unittest.TestCase):
    """Test for defining low flow event from discharge data file"""
    def test_load_FR_all(self):
        """Test defining low flow hazard from complete demo file (France)
        and keep monthly data"""
        haz = LowFlow()
        haz.set_from_nc(input_dir=INPUT_DIR, percentile=2.5,
                    yearrange=(2001, 2005), yearrange_ref=(2001, 2005),
                    gh_model='h08', cl_model='gfdl-esm2m',
                    scenario='historical', scenario_ref='historical', soc='histsoc', \
                    soc_ref='histsoc', fn_str_var=FN_STR_DEMO, keep_dis_data=True)
        self.assertEqual(haz.data.shape[0], 1653)
        self.assertEqual(haz.data.shape[1], 13)
        self.assertEqual(haz.data.dis.max(), 31.0)
        self.assertAlmostEqual(haz.data.dis.mean(), 10.588021778584393)
        self.assertEqual(haz.centroids.lon.min(), -4.75)
        self.assertEqual(haz.centroids.lon.max(), 8.25)
        self.assertEqual(haz.centroids.lat.min(), 42.25)
        self.assertEqual(haz.centroids.lat.max(), 51.25)
        self.assertEqual(haz.intensity.shape, (66, 513))
        self.assertEqual(haz.event_id.size, 66)
        self.assertEqual(haz.intensity.max(), 46.0)
        self.assertEqual(haz.data.cluster_id.unique().size, haz.event_id.size)
     
    def test_unique_clusters(self):
        """Test unique 3D cluster identification from 2D cluster data"""
        data = init_test_data_unqiue_clusters()
        data = unique_clusters(data)
        self.assertEqual(data.size, 75)
        self.assertListEqual(list(data.cluster_id), list(data.target_cluster))
 
    def test_identify_clusters_default(self):   
        """test clustering event from monthly days below threshold data"""
        haz = LowFlow()
        # 1) direct neighbors only (allowing over cross in space):
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=1.5, clus_thresh_t=1, min_samples=1)
        target_cluster = [1, 2, 1, 2, 2, 1, 1, 3, 3, 1, 3, 1, 4]
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

        # as (1), but allowing 1 month break in between:
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=1.5, clus_thresh_t=2, min_samples=1)
        target_cluster = [1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 3]        
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

        # as (1), but allowing 1 gridcell break in between:
        haz.data = init_test_data_clustering()
        haz.identify_clusters(clus_thres_xy=2., clus_thresh_t=1, min_samples=1)
        target_cluster = [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 3]        
        self.assertListEqual(list(haz.data.cluster_id), target_cluster)

    # def test_exact_area_selection(self):
    #     testCentroids = RiverFlood.select_exact_area(['LIE'])

    #     self.assertEqual(testCentroids.lon.shape[0], 13)
    #     self.assertAlmostEqual(testCentroids.lon[0], 9.5206968)
    #     self.assertAlmostEqual(testCentroids.lon[1], 9.5623634)
    #     self.assertAlmostEqual(testCentroids.lon[2], 9.60403)
    #     self.assertAlmostEqual(testCentroids.lon[3], 9.5206968)
    #     self.assertAlmostEqual(testCentroids.lon[4], 9.5623634)
    #     self.assertAlmostEqual(testCentroids.lon[5], 9.60403)
    #     self.assertAlmostEqual(testCentroids.lon[6], 9.5206968)
    #     self.assertAlmostEqual(testCentroids.lon[7], 9.5623634)
    #     self.assertAlmostEqual(testCentroids.lon[8], 9.60403)
    #     self.assertAlmostEqual(testCentroids.lon[9], 9.5206968)
    #     self.assertAlmostEqual(testCentroids.lon[10], 9.5623634)
    #     self.assertAlmostEqual(testCentroids.lon[11], 9.5206968)
    #     self.assertAlmostEqual(testCentroids.lon[12], 9.5623634)

    #     self.assertAlmostEqual(testCentroids.lat[0], 47.0622474)
    #     self.assertAlmostEqual(testCentroids.lat[1], 47.0622474)
    #     self.assertAlmostEqual(testCentroids.lat[2], 47.0622474)
    #     self.assertAlmostEqual(testCentroids.lat[3], 47.103914)
    #     self.assertAlmostEqual(testCentroids.lat[4], 47.103914)
    #     self.assertAlmostEqual(testCentroids.lat[5], 47.103914)
    #     self.assertAlmostEqual(testCentroids.lat[6], 47.1455806)
    #     self.assertAlmostEqual(testCentroids.lat[7], 47.1455806)
    #     self.assertAlmostEqual(testCentroids.lat[8], 47.1455806)
    #     self.assertAlmostEqual(testCentroids.lat[9], 47.1872472)
    #     self.assertAlmostEqual(testCentroids.lat[10], 47.1872472)
    #     self.assertAlmostEqual(testCentroids.lat[11], 47.2289138)
    #     self.assertAlmostEqual(testCentroids.lat[12], 47.2289138)

    #     self.assertEqual(testCentroids.id[0], 0)
    #     self.assertEqual(testCentroids.id[5], 5)
    #     self.assertEqual(testCentroids.id[12], 12)

    # def test_flooded_area(self):
    #     dph_path = HAZ_DEMO_FLDDPH
    #     frc_path = HAZ_DEMO_FLDFRC

    #     testRFset = RiverFlood()
    #     testRFset.set_from_nc(countries=['AFG'], dph_path=dph_path,
    #                           frc_path=frc_path)
    #     years = [2000, 2001, 2002]
    #     manipulated_dates = [730303, 730669, 731034]
    #     for i in range(len(years)):
    #         testRFaddset = RiverFlood()
    #         testRFaddset.set_from_nc(countries=['AFG'])
    #         testRFaddset.date = [manipulated_dates[i]]
    #         if i == 0:
    #             testRFaddset.event_name = ['2000_2']
    #         else:
    #             testRFaddset.event_name = [str(years[i])]
    #         testRFset.append(testRFaddset)

    #     testRFset.set_flooded_area()
    #     self.assertEqual(testRFset.units, 'm')

    #     self.assertEqual(testRFset.fla_event.shape[0], 4)
    #     self.assertEqual(testRFset.fla_annual.shape[0], 3)
    #     self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[0]),
    #                            17200498.22927546)
    #     self.assertEqual(np.argmax(testRFset.fla_ev_centr[0]),
    #                      32610)
    #     self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[2]),
    #                            17200498.22927546)
    #     self.assertEqual(np.argmax(testRFset.fla_ev_centr[2]),
    #                      32610)

    #     self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[0]),
    #                            34400996.45855092)
    #     self.assertEqual(np.argmax(testRFset.fla_ann_centr[0]),
    #                      32610)
    #     self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[2]),
    #                            17200498.22927546)
    #     self.assertEqual(np.argmax(testRFset.fla_ann_centr[2]),
    #                      32610)

    #     self.assertAlmostEqual(testRFset.fla_event[0],
    #                            6244242013.5826435, 4)
    #     self.assertAlmostEqual(testRFset.fla_annual[0],
    #                            12488484027.165287, 3)
    #     self.assertAlmostEqual(testRFset.fla_ann_av,
    #                            8325656018.110191, 4)
    #     self.assertAlmostEqual(testRFset.fla_ev_av,
    #                            6244242013.5826435, 4)

    # def test_select_model_run(self):
    #     testRFModel = RiverFlood()
    #     flood_dir = '/home/test/flood/'
    #     rf_model = 'LPJmL'
    #     cl_model = 'wfdei'
    #     prot_std = 'flopros'
    #     scenario = 'historical'

    #     self.assertEqual(testRFModel._select_model_run(flood_dir, rf_model,
    #                                                    cl_model,
    #                                                    scenario, prot_std)[0],
    #                      '/home/test/flood/flddph_LPJmL_wfdei_' +
    #                      'flopros_gev_0.1.nc')
    #     self.assertEqual(testRFModel._select_model_run(flood_dir, rf_model,
    #                                                    cl_model, scenario,
    #                                                    prot_std, proj=True)[0],
    #                      '/home/test/flood/flddph_LPJmL_wfdei_' +
    #                      'historical_flopros_gev_picontrol_2000_0.1.nc')

    # def test_set_centroids_from_file(self):
    #     testRFCentr = RiverFlood()
    #     lon = [1, 2, 3]
    #     lat = [1, 2, 3]
    #     testRFCentr._set_centroids_from_file(lon, lat)
    #     test_centroids_lon = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    #     test_centroids_lat = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    #     self.assertTrue(np.array_equal(testRFCentr.centroids.lon,
    #                                    test_centroids_lon))
    #     self.assertTrue(np.array_equal(testRFCentr.centroids.lat,
    #                                    test_centroids_lat))

    # def test_select_events(self):
    #     testRFTime = RiverFlood()
    #     tt1 = dt.datetime.strptime('1988-07-02', '%Y-%m-%d')
    #     tt2 = dt.datetime.strptime('2010-04-01', '%Y-%m-%d')
    #     tt3 = dt.datetime.strptime('1997-07-02', '%Y-%m-%d')
    #     tt4 = dt.datetime.strptime('1990-07-02', '%Y-%m-%d')
    #     years = [2010, 1997]
    #     test_time = np.array([tt1, tt2, tt3, tt4])
    #     self.assertTrue(np.array_equal(
    #                     testRFTime._select_event(test_time, years), [1, 2]))
    #     years = [1988, 1990]
    #     self.assertTrue(np.array_equal(
    #                     testRFTime._select_event(test_time, years), [0, 3]))

    # def test_cut_window(self):

    #     testRFCut = RiverFlood()
    #     centr = RiverFlood.select_window_area(['AUT'])
    #     testRFCut.centroids.lon = centr.lon
    #     testRFCut.centroids.lat = centr.lat
    #     lon = np.arange(7, 20, 0.2)
    #     lat = np.arange(40, 50, 0.2)
    #     test_window = [[4, 24], [55, 45]]
    #     self.assertTrue(np.array_equal(testRFCut._cut_window(lon, lat),
    #                                    test_window))


#
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLowFlow)
unittest.TextTestRunner(verbosity=2).run(TESTS)
