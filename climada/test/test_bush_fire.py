"""
Test Bush fire class
"""

import os
import unittest
import numpy as np

from climada.hazard.bush_fire import BushFire
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import ONE_LAT_KM

DATA_DIR = os.path.join(os.path.dirname(__file__), '../hazard/test/data')
TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_Soberanes_2016_viirs.csv")

description = ''

def def_ori_centroids(min_lon, min_lat, max_lon, max_lat, centr_res_factor):

    res_data = 0.375 # For VIIRS data

    centroids = Centroids()
    dlat_km = abs(min_lat - max_lat) * ONE_LAT_KM
    dlon_km = abs(min_lon - max_lon) * ONE_LAT_KM*np.cos(np.radians((abs(min_lat - max_lat))/2))
    nb_centr_lat = int(dlat_km/res_data * centr_res_factor)
    nb_centr_lon = int(dlon_km/res_data * centr_res_factor)
    coord = (np.mgrid[min_lat : max_lat : complex(0, nb_centr_lat), \
             min_lon : max_lon : complex(0, nb_centr_lon)]). \
             reshape(2, nb_centr_lat*nb_centr_lon).transpose()
    centroids.set_lat_lon(coord[:, 0], coord[:, 1])

    # Calculate the area attributed to each centroid
    centroids.set_area_approx()
    # Calculate if the centroids is on land or not
    centroids.set_on_land()
    # Create on land grid
    centroids.empty_geometry_points()
    return centroids

class TestCaliforniaFirms(unittest.TestCase):
    """Test loading functions from the BushFire class"""

    def test_centr_hist_pass(self):
        """ Test set_hist_events """
        centroids = def_ori_centroids(-121.92908999999999, 35.66364, -120.96468, 36.59146, 1/2)
        bf = BushFire()
        bf.set_hist_events(TEST_FIRMS, centroids=centroids)
        bf.check()

        self.assertEqual(bf.tag.haz_type, 'BF')
        self.assertEqual(bf.units, 'K')
        self.assertTrue(np.allclose(bf.event_id, np.arange(1, 10)))
        self.assertTrue(np.allclose(bf.date,
            np.array([736190, 736218, 736167, 736180, 736221, 736245, 736221, 736225, 736228])))
        self.assertTrue(np.allclose(bf.orig, np.ones(9, bool)))
        self.assertEqual(bf.event_name, ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertTrue(np.allclose(bf.frequency, np.ones(9)))
        self.assertEqual(bf.intensity.shape, (9, 19454))
        self.assertEqual(bf.fraction.shape, (9, 19454))
        self.assertEqual(bf.intensity[1, :].nonzero()[1][0], 6764)
        self.assertEqual(bf.intensity[3, :].nonzero()[1][0], 19367)
        self.assertEqual(bf.intensity[5, :].nonzero()[1][0], 10696)
        self.assertEqual(bf.intensity[5, :].nonzero()[1][1], 10697)
        self.assertEqual(bf.intensity[6, :].nonzero()[1][0], 10132)
        self.assertEqual(bf.intensity[7, :].nonzero()[1][0], 10834)
        self.assertAlmostEqual(bf.intensity[0, 3257], 299.1)
        self.assertAlmostEqual(bf.intensity[2, 13650], 367.0)
        self.assertAlmostEqual(bf.intensity[2, 17328], 357.9)
        self.assertAlmostEqual(bf.intensity[7, 10834], 336.7)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)

    def test_hist_pass(self):
        """ Test set_hist_events """
        bf = BushFire()
        bf.set_hist_events(TEST_FIRMS, centr_res_factor=1/2)
        bf.check()

        self.assertEqual(bf.tag.haz_type, 'BF')
        self.assertEqual(bf.units, 'K')
        self.assertTrue(np.allclose(bf.event_id, np.arange(1, 10)))
        self.assertTrue(np.allclose(bf.date,
            np.array([736190, 736218, 736167, 736180, 736221, 736245, 736221, 736225, 736228])))
        self.assertTrue(np.allclose(bf.orig, np.ones(9, bool)))
        self.assertEqual(bf.event_name, ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertTrue(np.allclose(bf.frequency, np.ones(9)))
        self.assertEqual(bf.intensity.shape, (9, 19872))
        self.assertEqual(bf.fraction.shape, (9, 19872))
        self.assertEqual(bf.intensity[1, :].nonzero()[1][0], 13052)
        self.assertEqual(bf.intensity[3, :].nonzero()[1][0], 56)
        self.assertEqual(bf.intensity[5, :].nonzero()[1][0], 8975)
        self.assertEqual(bf.intensity[5, :].nonzero()[1][1], 8976)
        self.assertEqual(bf.intensity[5, :].nonzero()[1].size, 2)
        self.assertEqual(bf.intensity[6, :].nonzero()[1][0], 9555)
        self.assertEqual(bf.intensity[6, :].nonzero()[1].size, 1)
        self.assertEqual(bf.intensity[7, :].nonzero()[1][0], 8682)
        self.assertEqual(bf.intensity[7, :].nonzero()[1].size, 3)
        self.assertAlmostEqual(bf.intensity[0, 19003], 317.6)
        self.assertAlmostEqual(bf.intensity[2, 2008], 0.0)
        self.assertAlmostEqual(bf.intensity[2, 9409], 350.9)
        self.assertAlmostEqual(bf.intensity[7, 8683], 307.9)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)

    def test_centr_synth_pass(self):
        """ Test probabilistic set_proba_events """
        centroids = def_ori_centroids(-121.92908999999999, 35.66364, -120.96468, 36.59146, 1/2)
        bf = BushFire()
        bf.set_hist_events(TEST_FIRMS, centroids=centroids)
        bf.set_proba_events(ens_size=2, seed=8)
        bf.check()

        self.assertEqual(bf.size, 9*3)
        orig = np.zeros(27, bool)
        orig[:9] = True
        self.assertTrue(np.allclose(bf.orig, orig))
        self.assertTrue(np.allclose(bf.date[:9], np.array([736190, 736218, 736167, 736180,
                         736221, 736245, 736221, 736225, 736228])))
        self.assertTrue(np.allclose(bf.date[9:11], np.array([736190, 736190])))
        self.assertTrue(np.allclose(bf.date[11:13], np.array([736218, 736218])))
        self.assertTrue(np.allclose(bf.date[-2:], np.array([736228, 736228])))
        self.assertEqual(bf.event_name[:9], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertEqual(bf.event_name[9:11], ['1_gen1', '1_gen2'])
        self.assertEqual(bf.event_name[-2:], ['9_gen1', '9_gen2'])
        self.assertEqual(bf.intensity.shape, (27, 19454))
        self.assertEqual(bf.fraction.shape, (27, 19454))
        self.assertAlmostEqual(bf.intensity[26, 10128], 301.5)
        self.assertAlmostEqual(bf.intensity[26, 10269], 301.5)
        self.assertAlmostEqual(bf.intensity[26, 10271], 295.4)
        self.assertEqual(bf.intensity[26, :].nonzero()[0].size, 3)
        self.assertEqual(bf.intensity.nonzero()[0].size, 5445)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)

    def test_synth_pass(self):
        """ Test probabilistic set_proba_events """
        bf = BushFire()
        bf.set_hist_events(TEST_FIRMS, 1/2)
        bf.set_proba_events(ens_size=2, seed=9)
        bf.check()

        self.assertEqual(bf.size, 9*3)
        orig = np.zeros(27, bool)
        orig[:9] = True
        self.assertTrue(np.allclose(bf.orig, orig))
        self.assertTrue(np.allclose(bf.date[:9], np.array([736190, 736218, 736167, 736180,
                         736221, 736245, 736221, 736225, 736228])))
        self.assertTrue(np.allclose(bf.date[9:11], np.array([736190, 736190])))
        self.assertTrue(np.allclose(bf.date[11:13], np.array([736218, 736218])))
        self.assertTrue(np.allclose(bf.date[-2:], np.array([736228, 736228])))
        self.assertEqual(bf.event_name[:9], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertEqual(bf.event_name[9:11], ['1_gen1', '1_gen2'])
        self.assertEqual(bf.event_name[-2:], ['9_gen1', '9_gen2'])
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)
        self.assertAlmostEqual(bf.intensity[26, 9994], 301.5)
        self.assertAlmostEqual(bf.intensity[26, 9995], 326.7)
        self.assertAlmostEqual(bf.intensity[26, 10138], 326.7)
        self.assertAlmostEqual(bf.intensity[26, 10281], 326.7)
        self.assertEqual(bf.intensity[26, :].nonzero()[0].size, 4)
        self.assertEqual(bf.intensity.nonzero()[0].size, 5517)
        self.assertEqual(bf.intensity.shape, (27, 19872))
        self.assertEqual(bf.fraction.shape, (27, 19872))

    def test_pool_pass(self):
        """ Test reproducibility with pool """
        from pathos.pools import ProcessPool as Pool
        pool = Pool()
        bf = BushFire(pool)
        bf.set_hist_events(TEST_FIRMS, 1/2)
        bf.set_proba_events(ens_size=2, seed=9)
        bf.check()

        self.assertEqual(bf.size, 9*3)
        orig = np.zeros(27, bool)
        orig[:9] = True
        self.assertTrue(np.allclose(bf.orig, orig))
        self.assertTrue(np.allclose(bf.date[:9], np.array([736190, 736218, 736167, 736180,
                         736221, 736245, 736221, 736225, 736228])))
        self.assertTrue(np.allclose(bf.date[9:11], np.array([736190, 736190])))
        self.assertTrue(np.allclose(bf.date[11:13], np.array([736218, 736218])))
        self.assertTrue(np.allclose(bf.date[-2:], np.array([736228, 736228])))
        self.assertEqual(bf.event_name[:9], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        self.assertEqual(bf.event_name[9:11], ['1_gen1', '1_gen2'])
        self.assertEqual(bf.event_name[-2:], ['9_gen1', '9_gen2'])
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)
        self.assertAlmostEqual(bf.intensity[26, 9994], 301.5)
        self.assertAlmostEqual(bf.intensity[26, 9995], 326.7)
        self.assertAlmostEqual(bf.intensity[26, 10138], 326.7)
        self.assertAlmostEqual(bf.intensity[26, 10281], 326.7)
        self.assertEqual(bf.intensity[26, :].nonzero()[0].size, 4)
        self.assertEqual(bf.intensity.nonzero()[0].size, 5517)
        self.assertEqual(bf.intensity.shape, (27, 19872))
        self.assertEqual(bf.fraction.shape, (27, 19872))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCaliforniaFirms)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
