"""
Test Bush fire class
"""

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse



from climada.hazard.bush_fire import BushFire

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
#TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_firms.csv")
#TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_2016_viirs.csv")
TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_Soberanes_2016_viirs.csv")
#TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_King_2014_viirs.csv")

description = ''

class TestReaderFirms(unittest.TestCase):
    """Test loading functions from the BushFire class"""
        
#    def test_random_brightness_one_pass(self):
#        bf = BushFire()
#        firms, description = bf._read_firms_csv(TEST_FIRMS)
#        
#        firms = bf._clean_firms_csv(firms)
#        
#        self.assertEqual(firms['latitude'][0], 36.46245)
#        self.assertEqual(firms['longitude'][0], -121.8989)
#        self.assertEqual(firms['latitude'].iloc[-1], 36.17266)
#        self.assertEqual(firms['longitude'].iloc[-1], -121.61211000000002)
#         
#        centroids, res_data = bf._centroids_creation(firms, 1)
#        firms = bf._firms_cons_days(firms)
#        
#        self.assertEqual(firms['cons_id'][9320], 0)
#        self.assertEqual(firms['cons_id'][9321], 1)
#        
#        firms = bf._firms_clustering(firms, res_data)
#        
#        self.assertEqual(firms['clus_id'][1621], 0)
#        self.assertEqual(firms['clus_id'][1622], 1)
#        self.assertEqual(firms['clus_id'][1623], 2)
#        
#        firms = bf._firms_event(firms)
#        
#        self.assertEqual(firms['event_id'][1621], 1)
#        self.assertEqual(firms['event_id'][1622], 2)
#        self.assertEqual(firms['event_id'][1623], 3)
#        self.assertEqual(firms['event_id'][9319], 3)
#        self.assertEqual(firms['event_id'][9320], 4)
#        
#        bf._calc_brightness(firms, centroids)
#        bf.check()
#        
#        self.assertEqual(bf.tag.haz_type, 'BF')
#        self.assertEqual(bf.tag.description, '')
#        self.assertEqual(bf.units, 'K')
#        self.assertEqual(bf.centroids.id.size, 78090)
#        self.assertEqual(bf.event_id.size, 5)
#        self.assertEqual(bf.event_id[0], 1)
#        self.assertEqual(bf.event_name, ['1.0', '2.0', '3.0', '4.0', '5.0'])
#        self.assertTrue(np.array_equal(bf.frequency, np.array([1, 1, 1, 1, 1])))
#        self.assertTrue(isinstance(bf.intensity, sparse.csr_matrix))
#        self.assertTrue(isinstance(bf.fraction, sparse.csr_matrix))
#        self.assertEqual(bf.intensity.shape, (5, 78090))
#        self.assertEqual(bf.fraction.shape, (5, 78090))
#        
#        area_2016 = bf._area_one_year(2016)
#        
#        ev_id = 3
#        area_hull_one_event = bf._hull_burned_area(ev_id)
#        
#        self.assertAlmostEqual(area_hull_one_event, 60421, delta = 500)
#        
#        prob_event = bf._random_bushfire_one_event(ev_id, 1)
#        prob_event.check()
#        
#        self.assertEqual(prob_event.tag.haz_type, 'BF')
#        self.assertEqual(prob_event.tag.description, '')
#        self.assertEqual(prob_event.units, 'K')
#        self.assertEqual(prob_event.centroids.id.size, 78090)
#        self.assertEqual(prob_event.event_id.size, 1)
#        self.assertEqual(prob_event.event_id[0], 1)
#        self.assertEqual(prob_event.event_name, ['3_gen1'])
#        self.assertTrue(np.array_equal(prob_event.frequency, np.array([1])))
#        self.assertTrue(isinstance(prob_event.intensity, sparse.csr_matrix))
#        self.assertTrue(isinstance(prob_event.fraction, sparse.csr_matrix))
#        self.assertEqual(prob_event.intensity.shape, (1, 78090))
#        self.assertEqual(prob_event.fraction.shape, (1, 78090))
#
#        self.assertAlmostEqual(prob_event.intensity.nonzero()[0].size, 3928, delta = 10)
#        
#        prob_event.plot_intensity(event = '3_gen0')
#        
#        self.assertAlmostEqual(bf.intensity[ev_id - 1].data.mean(),prob_event.intensity[0].data.mean(), delta = 2)
#        self.assertAlmostEqual(bf.intensity[ev_id - 1].data.std(), prob_event.intensity[0].data.std(), delta = 2)
        
         # fill in Hazard file
#    def test_bush_fire_one_pass(self):
#         bf = BushFire()
#         centr_res_factor = 1
#         
#         bf.set_bush_fire (TEST_FIRMS, centr_res_factor, seed = 8)
#         
#         bf.check()
#        
#         self.assertEqual(bf.tag.haz_type, 'BF')
#         self.assertEqual(bf.tag.description, '')
#         self.assertEqual(bf.units, 'K')
#         self.assertEqual(bf.centroids.id.size, 78090)
#         self.assertEqual(bf.event_id.size, 5)
#         self.assertEqual(bf.event_id[0], 1)
#         self.assertEqual(bf.event_name, ['1.0', '2.0', '3.0', '4.0', '5.0'])
#         self.assertTrue(np.array_equal(bf.frequency, np.array([1, 1, 1, 1, 1])))
#         self.assertTrue(isinstance(bf.intensity, sparse.csr_matrix))
#         self.assertTrue(isinstance(bf.fraction, sparse.csr_matrix))
#         self.assertEqual(bf.intensity.shape, (5, 78090))
#         self.assertEqual(bf.fraction.shape, (5, 78090))
#         
#         bf.plot_intensity(event=0)
#         
#    def test_bush_fire_random_one_pass(self):
#         bf = BushFire()
#         centr_res_factor = 1
#         
#         bf.set_bush_fire (TEST_FIRMS, centr_res_factor, seed = 8)
#         bf_haz = bf.set_proba_one_event(ev_id = 3, ens_size = 3)
#         
#         prob_haz = BushFire()
#         prob_haz._append_all(bf_haz)
#         bf.append(prob_haz)
#         
#         bf.check()
#         
#         self.assertEqual(bf.tag.haz_type, 'BF')
#         self.assertEqual(bf.tag.description, '')
#         self.assertEqual(bf.units, 'K')
#         self.assertEqual(bf.centroids.id.size, 78090)
#         self.assertEqual(bf.event_id.size, 8)
#         self.assertEqual(bf.event_id[0], 1)
#         self.assertEqual(bf.event_name, ['1.0', '2.0', '3.0', '4.0', '5.0', '3_gen0', '3_gen1', '3_gen2'])
#         self.assertTrue(np.array_equal(bf.frequency, np.array([1, 1, 1, 1, 1, 1, 1, 1])))
#         self.assertTrue(isinstance(bf.intensity, sparse.csr_matrix))
#         self.assertTrue(isinstance(bf.fraction, sparse.csr_matrix))
#         self.assertEqual(bf.intensity.shape, (8, 78090))
#         self.assertEqual(bf.fraction.shape, (8, 78090))
#         
#         bf.plot_intensity('3_gen0')
#         
#         
    def test_bush_fire_one_pass(self):
         bf = BushFire()
         centr_res_factor = 1
         
         bf.set_bush_fire (TEST_FIRMS, centr_res_factor, seed = 8)
         bf.set_proba_all_event(ens_size = 3)
         bf.check()
         
         self.assertEqual(bf.tag.haz_type, 'BF')
         self.assertEqual(bf.tag.description, '')
         self.assertEqual(bf.units, 'K')
         self.assertEqual(bf.centroids.id.size, 78090)
         self.assertEqual(bf.event_id.size, 20)
         self.assertEqual(bf.event_id[0], 1)
         self.assertEqual(bf.event_name, ['1.0', '2.0', '3.0', '4.0', '5.0', 
                                          '1.0_gen0', '1.0_gen1', '1.0_gen2',
                                          '2.0_gen0', '2.0_gen1', '2.0_gen2',
                                          '3.0_gen0', '3.0_gen1', '3.0_gen2',
                                          '4.0_gen0', '4.0_gen1', '4.0_gen2',
                                          '5.0_gen0', '5.0_gen1', '5.0_gen2'])
         self.assertTrue(np.array_equal(bf.frequency, np.array([0.25, 0.25, 0.25, 0.25, 0.25, 
                                                                0.25, 0.25, 0.25,
                                                                0.25, 0.25, 0.25,
                                                                0.25, 0.25, 0.25,
                                                                0.25, 0.25, 0.25,
                                                                0.25, 0.25, 0.25])))
         self.assertTrue(isinstance(bf.intensity, sparse.csr_matrix))
         self.assertTrue(isinstance(bf.fraction, sparse.csr_matrix))
         self.assertEqual(bf.intensity.shape, (20, 78090))
         self.assertEqual(bf.fraction.shape, (20, 78090)) 
         bf.plot_intensity(event='3.0')
         bf.plot_intensity(event='3.0_gen0')
         bf.plot_intensity(event='3.0_gen1')
         bf.plot_intensity(event='3.0_gen2')
         


# Execute Tests
#TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderFirms)
#unittest.TextTestRunner(verbosity=2).run(TESTS)
