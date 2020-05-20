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
    df.dis = [5, 11, 5, 11, 11, 10, 10, 22, 22, 20, 22, 20, 1]

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
        self.assertEqual(haz.data.shape[1], 14)
        self.assertEqual(haz.data.ndays.max(), 31.0)
        self.assertAlmostEqual(haz.data.ndays.mean(), 10.588021778584393)
        self.assertAlmostEqual(haz.data.relative_dis.max(), 0.41278067)
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

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLowFlow)
unittest.TextTestRunner(verbosity=2).run(TESTS)
