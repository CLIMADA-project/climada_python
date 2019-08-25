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

Test Bush fire class
"""

import os
import unittest
import numpy as np
from scipy import sparse
from rasterio import Affine

from climada.hazard.bush_fire import BushFire
from climada.hazard.centroids.centr import Centroids
from climada.util import get_resolution
from climada.util.constants import ONE_LAT_KM

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_Soberanes_2016_viirs.csv")

description = ''

def def_ori_centroids(firms, centr_res_factor):
    # res_data in km
    if firms['instrument'].any() == 'MODIS':
        res_data = 1.0
    else:
        res_data = 0.375 # For VIIRS data

    centroids = Centroids()
    dlat_km = abs(firms['latitude'].min() - firms['latitude'].max()) * ONE_LAT_KM
    dlon_km = abs(firms['longitude'].min() - firms['longitude'].max()) * ONE_LAT_KM* \
        np.cos(np.radians((abs(firms['latitude'].min() - firms['latitude'].max()))/2))
    nb_centr_lat = int(dlat_km/res_data * centr_res_factor)
    nb_centr_lon = int(dlon_km/res_data * centr_res_factor)
    coord = (np.mgrid[firms['latitude'].min() : firms['latitude'].max() : complex(0, nb_centr_lat), \
        firms['longitude'].min() : firms['longitude'].max() : complex(0, nb_centr_lon)]). \
        reshape(2, nb_centr_lat*nb_centr_lon).transpose()
    centroids.set_lat_lon(coord[:, 0], coord[:, 1])

    centroids.set_area_approx()
    centroids.set_on_land()
    centroids.empty_geometry_points()
    return centroids, res_data

DEF_CENTROIDS = def_ori_centroids(BushFire._clean_firms_csv(TEST_FIRMS), 1/2)

class TestMethodsFirms(unittest.TestCase):
    """Test loading functions from the BushFire class"""

    def test_clean_firms_pass(self):
        """ Test _clean_firms_csv """
        bf = BushFire()
        firms = bf._clean_firms_csv(TEST_FIRMS)

        self.assertEqual(firms['latitude'][0], 36.46245)
        self.assertEqual(firms['longitude'][0], -121.8989)
        self.assertEqual(firms['latitude'].iloc[-1], 36.17266)
        self.assertEqual(firms['longitude'].iloc[-1], -121.61211000000002)
        self.assertEqual(firms['datenum'].iloc[-1], 736245)
        self.assertTrue(np.array_equal(firms['iter_ev'].values, np.ones(len(firms), bool)))
        self.assertTrue(np.array_equal(firms['cons_id'].values, np.zeros(len(firms), int)-1))
        self.assertTrue(np.array_equal(firms['clus_id'].values, np.zeros(len(firms), int)-1))

    def test_firms_cons_days_1_pass(self):
        """ Test _firms_cons_days """
        bf = BushFire()
        firms_ori = bf._clean_firms_csv(TEST_FIRMS)
        firms = bf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.zeros(9325, int)
        cons_id[8176:-4] = 1
        cons_id[-4:] = 2
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_cons_days_2_pass(self):
        """ Test _firms_cons_days """
        bf = BushFire()
        firms_ori = bf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms = bf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.ones(9325, int)
        cons_id[100] = 0
        cons_id[8176:-4] = 2
        cons_id[-4:] = 3
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_cons_days_3_pass(self):
        """ Test _firms_cons_days """
        ori_thres = BushFire.days_thres
        BushFire.days_thres = 3
        bf = BushFire()
        firms_ori = bf._clean_firms_csv(TEST_FIRMS)
        firms = bf._firms_cons_days(firms_ori.copy())
        self.assertEqual(len(firms), 9325)
        cons_id = np.zeros(9325, int)
        cons_id[-4:] = 1
        self.assertTrue(np.allclose(firms.cons_id.values, cons_id))
        for col_name in firms.columns:
            if col_name not in ('cons_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'cons_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)
        BushFire.days_thres = ori_thres

    def test_firms_cluster_pass(self):
        """ Test _firms_clustering """
        bf = BushFire()
        firms_ori = bf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms_ori = bf._firms_cons_days(firms_ori)
        firms = bf._firms_clustering(firms_ori.copy(), 0.375/2/15, 15)
        self.assertEqual(len(firms), 9325)
        self.assertTrue(np.allclose(firms.clus_id.values[-4:], np.zeros(4, int)))
        self.assertEqual(firms.clus_id.values[100], 0)
        self.assertEqual(firms.clus_id.values[2879], 3)
        self.assertEqual(firms.clus_id.values[0], 2)
        self.assertEqual(firms.clus_id.values[2], 2)
        self.assertEqual(firms.clus_id.values[9320], 0)
        self.assertEqual(firms.clus_id.values[4254], 2)
        self.assertEqual(firms.clus_id.values[4255], 0)
        self.assertEqual(firms.clus_id.values[8105], 1)
        self.assertTrue(np.allclose(np.unique(firms.clus_id.values), np.arange(4)))
        for col_name in firms.columns:
            if col_name not in ('clus_id', 'acq_date', 'confidence', 'instrument', 'satellite'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name != 'clus_id':
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_firms_event_pass(self):
        """ Test _firms_event """
        bf = BushFire()
        firms_ori = bf._clean_firms_csv(TEST_FIRMS)
        firms_ori['datenum'].values[100] = 7000
        firms_ori = bf._firms_cons_days(firms_ori)
        firms_ori = bf._firms_clustering(firms_ori, 0.375/2/15, 15)
        firms = firms_ori.copy()
        bf._firms_event(2, firms.cons_id.values, firms.clus_id.values,
            firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
        self.assertEqual(len(firms), 9325)
        self.assertTrue(np.allclose(np.unique(firms.event_id), np.arange(7)))
        self.assertTrue(np.allclose(firms.event_id[-4:], np.ones(4)*6))
        self.assertEqual(firms.event_id.values[100], 0)
        self.assertEqual(firms.event_id.values[4255], 1)
        self.assertEqual(firms.event_id.values[8105], 2)
        self.assertEqual(firms.event_id.values[0], 3)
        self.assertEqual(firms.event_id.values[2], 3)
        self.assertEqual(firms.event_id.values[4254], 3)
        self.assertEqual(firms.event_id.values[2879], 4)
        self.assertEqual(firms.event_id.values[9320], 5)

        self.assertFalse(firms.iter_ev.any())

        for ev_id in np.unique(firms.event_id):
            if np.unique(firms[firms.event_id == ev_id].cons_id).size != 1:
                self.assertEqual(1, 0)
            if np.unique(firms[firms.event_id == ev_id].clus_id).size != 1:
                self.assertEqual(1, 0)

        for col_name in firms.columns:
            if col_name not in ('event_id', 'acq_date', 'confidence', 'instrument', 'satellite', 'iter_ev'):
                self.assertTrue(np.allclose(firms[col_name].values, firms_ori[col_name].values))
            elif col_name not in ('event_id', 'iter_ev'):
                for elem_l, elem_r in zip(firms[col_name].values, firms_ori[col_name].values):
                    self.assertEqual(elem_l, elem_r)

    def test_iter_events_pass(self):
        """ Test identification of events """
        ori_thres = BushFire.days_thres
        bf = BushFire()
        firms = bf._clean_firms_csv(TEST_FIRMS)
        firms['datenum'].values[100] = 7000
        i_iter = 0
        BushFire.days_thres = 3
        while firms.iter_ev.any():
            # Compute cons_id: consecutive events in current iteration
            bf._firms_cons_days(firms)
            # Compute clus_id: cluster identifier inside cons_id
            bf._firms_clustering(firms, 0.375, 15)
            # compute event_id
            BushFire.days_thres = 2
            bf._firms_event(BushFire.days_thres, firms.cons_id.values, firms.clus_id.values, 
                firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
            i_iter += 1
        self.assertEqual(i_iter, 2)
        BushFire.days_thres = ori_thres

    def test_centroids_resolution_pass(self):
        """ Test _centroids_resolution """
        bf = BushFire()
        res_centr = bf._centroids_resolution(DEF_CENTROIDS[0])
        res = get_resolution(DEF_CENTROIDS[0].lat, DEF_CENTROIDS[0].lon)
        self.assertAlmostEqual((res[0]+res[1])/2, res_centr)

        centroids = Centroids()
        centroids.meta = {'transform': Affine(0.5, 0, -180, 0, -0.5, 90)}
        res_centr = bf._centroids_resolution(centroids)
        self.assertAlmostEqual(0.5, res_centr)

    def test_calc_bright_pass(self):
        """ Test _calc_brightness """
#        from pathos.pools import ProcessPool as Pool
#        pool = Pool()
        bf = BushFire()
        firms = bf._clean_firms_csv(TEST_FIRMS)
        firms['datenum'].values[100] = 7000
        firms = bf._firms_cons_days(firms)
        firms = bf._firms_clustering(firms, DEF_CENTROIDS[1]/2/15, 15)
        bf._firms_event(2, firms.cons_id.values, firms.clus_id.values,
            firms.event_id.values, firms.iter_ev.values, firms.datenum.values)
        firms.latitude[8169] = firms.loc[16]['latitude']
        firms.longitude[8169] = firms.loc[16]['longitude']
        bf._calc_brightness(firms, DEF_CENTROIDS[0], DEF_CENTROIDS[1])
        bf.check()

        self.assertEqual(bf.tag.haz_type, 'BF')
        self.assertEqual(bf.tag.description, '')
        self.assertEqual(bf.units, 'K')
        self.assertEqual(bf.centroids.size, 19454)
        self.assertTrue(np.allclose(bf.event_id, np.arange(1, 8)))
        self.assertEqual(bf.event_name, ['1', '2', '3', '4', '5', '6', '7'])
        self.assertTrue(bf.frequency.size, 0)
        self.assertTrue(isinstance(bf.intensity, sparse.csr_matrix))
        self.assertTrue(isinstance(bf.fraction, sparse.csr_matrix))
        self.assertEqual(bf.intensity.shape, (7, 19454))
        self.assertEqual(bf.fraction.shape, (7, 19454))
        self.assertEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.intensity[0, 16618], firms.loc[100].brightness)
        self.assertAlmostEqual(bf.intensity[1, 123], max(firms.loc[6721].brightness, firms.loc[6722].brightness))
        self.assertAlmostEqual(bf.intensity[2, :].max(), firms.loc[8105].brightness)
        self.assertAlmostEqual(bf.intensity[4, 19367], firms.loc[2879].brightness)
        self.assertAlmostEqual(bf.intensity[5, 10132], firms.loc[8176].brightness)
        self.assertAlmostEqual(bf.intensity[6, 10696], max(firms.loc[9322].brightness, firms.loc[9324].brightness))
        self.assertAlmostEqual(bf.intensity[6, 10697], max(firms.loc[9321].brightness, firms.loc[9323].brightness))
        self.assertEqual((bf.intensity[0, :]>0).sum(), 1)
        self.assertEqual((bf.intensity[1, :]>0).sum(), 424)
        self.assertEqual((bf.intensity[2, :]>0).sum(), 1)
        self.assertEqual((bf.intensity[3, :]>0).sum(), 1112)
        self.assertEqual((bf.intensity[4, :]>0).sum(), 1)
        self.assertEqual((bf.intensity[5, :]>0).sum(), 264)
        self.assertEqual((bf.intensity[6, :]>0).sum(), 2)

    def test_random_one_pass(self):
        """ Test _random_bushfire_one_event """
        np.random.seed(8)
        bf = BushFire()
        bf.centroids = DEF_CENTROIDS[0]
        bf.date = np.ones(1)
        rnd_num = np.random.randint(2, size=1000)
        bf.intensity = sparse.lil_matrix(np.zeros((1, bf.centroids.size)))
        bf.intensity[0, :1000] = rnd_num
        bf.intensity[0, np.logical_not(bf.centroids.on_land)] = 0
        bf.intensity = bf.intensity.tocsr()
        np.random.seed(8)
        syn_haz = bf._random_bushfire_one_event(0, 5)
        self.assertTrue(syn_haz.intensity.data.size >= bf.intensity.data.size)
        self.assertTrue(syn_haz.centroids.area_pixel.sum() >= bf.centroids.area_pixel.sum())
        self.assertAlmostEqual(syn_haz.intensity.max(), bf.intensity.max())
        self.assertAlmostEqual(syn_haz.intensity.min(), bf.intensity.min())

    def test_set_frequency_pass(self):
        """ Test _set_frequency """
        bf = BushFire()
        bf.date = np.array([736167, 736234, 736235])
        bf.orig = np.ones(3, bool)
        bf.event_id = np.arange(3)
        bf._set_frequency()
        self.assertTrue(np.allclose(bf.frequency, np.ones(3, float)))

        bf = BushFire()
        bf.date = np.array([736167, 736167, 736167, 736234, 736234, 736234, 736235, 736235, 736235])
        bf.orig = np.zeros(9, bool)
        bf.orig[[0, 3, 6]] = True
        bf.event_id = np.arange(9)
        bf._set_frequency()
        self.assertTrue(np.allclose(bf.frequency, np.ones(9, float)/3))

    def test_centroids_pass(self):
        """ Test _centroids_creation """
        bf = BushFire()
        firms = bf._clean_firms_csv(TEST_FIRMS)
        centroids = bf._centroids_creation(firms, 0.375/ONE_LAT_KM, 1/2)
        self.assertEqual(centroids.meta['width'], 144)
        self.assertEqual(centroids.meta['height'], 138)
        self.assertEqual(centroids.meta['crs']['init'], 'epsg:4326')
        self.assertAlmostEqual(centroids.meta['transform'][0], 0.006749460043196544)
        self.assertAlmostEqual(centroids.meta['transform'][1], 0.0)
        self.assertTrue(centroids.meta['transform'][2]<= firms.longitude.min())
        self.assertAlmostEqual(centroids.meta['transform'][3], 0.0)
        self.assertAlmostEqual(centroids.meta['transform'][4], -centroids.meta['transform'][0])
        self.assertTrue(centroids.meta['transform'][5] >= firms.latitude.max())
        self.assertTrue(firms.latitude.max() <= centroids.total_bounds[3])
        self.assertTrue(firms.latitude.min() >= centroids.total_bounds[1])
        self.assertTrue(firms.longitude.max() <= centroids.total_bounds[2])
        self.assertTrue(firms.longitude.min() >= centroids.total_bounds[0])
        self.assertTrue(centroids.lat.size)
        self.assertTrue(centroids.area_pixel.size)
        self.assertTrue(centroids.on_land.size)

    def test_firms_resolution_pass(self):
        """ Test _firms_resolution """
        bf = BushFire()
        firms = bf._clean_firms_csv(TEST_FIRMS)
        self.assertAlmostEqual(bf._firms_resolution(firms), 0.375/ONE_LAT_KM)
        firms['instrument'][0] = 'MODIS'
        self.assertAlmostEqual(bf._firms_resolution(firms), 1.0/ONE_LAT_KM)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestMethodsFirms)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
