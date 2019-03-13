#"""
#Test Bush fire class
#"""
#
#import os
#import unittest
#import numpy as np
#import pandas as pd
#from datetime import datetime
#
#
#from climada.hazard.bush_fire import BushFire
#
#DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
#TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_firms.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_2016.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "California_firms_Soberanes_2016_viirs.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_2002-2003.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_janv-feb_2013.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "Spain_2013-2018_firms.csv")
#description = ''
#
#class TestReaderFirms(unittest.TestCase):
#    """Test loading functions from the BushFire class"""
#
##    def test_read_one_pass(self):
##         bf = BushFire()
##         firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
##
##         self.assertEqual(firms['latitude'][0], -38.104)
##         self.assertEqual(firms['longitude'][0], 146.388)
##         self.assertEqual(firms['latitude'].iloc[-1], -36.016)
##         self.assertEqual(firms['longitude'].iloc[-1], 146.97)
##         self.assertFalse(firms['datenum'][0] == firms['datenum'][29])
##
##    def test_cons_days(self):
##        bf = BushFire()
##        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
##        firms = bf._firms_cons_days(firms)
##
##        self.assertEqual(firms['cons_id'][27], 0)
##        self.assertEqual(firms['cons_id'][28], 1)
##        self.assertEqual(firms['cons_id'][194], 1)
##        self.assertEqual(firms['cons_id'][195], 2)
##        self.assertEqual(firms['cons_id'][197], 2)
##        self.assertEqual(firms['cons_id'][198], 3)
##        self.assertEqual(firms['cons_id'][2136], 3)
##        self.assertEqual(firms['cons_id'][2137], 4)
##
##    def test_clustering(self):
##         bf = BushFire()
##         firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
##         #add cons_id
##         firms = bf._firms_cons_days(firms)
##         #add clus_id
##         firms = bf._firms_clustering(firms)
##
##         self.assertEqual(max((firms['clus_id'][:28]).values), 2)
##         self.assertEqual(max((firms['clus_id'][195:198]).values), 1)
##
##    def test_event_one_pass(self):
##        bf = BushFire()
##        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
##        firms = bf._firms_cons_days(firms)
##        firms = bf._firms_clustering(firms)
##        #add event_id
##        firms = bf._firms_event(firms)
##        self.assertEqual(max((firms['event_id'][:28]).values), 3)
##        self.assertEqual(max((firms['event_id'][28:195]).values), 10)
#
##    def test_brightness_one_pass(self):
##        bf = BushFire()
##        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
##        firms = bf._firms_cons_days(firms)
##        firms = bf._firms_clustering(firms)
##        firms = bf._firms_event(firms)
##        centroids = bf._centroids_creation(firms)
##
##        brightness, num_centr, latlon = bf._calc_brightness(firms, centroids)
#        
##        bf.centroids = centroids
##        bf.event_id = np.array(np.unique(firms['event_id'].values))
##        bf.event_name = np.array(np.unique(firms['event_id']))
##        bf.intensity = brightness
##        bf.plot_intensity(event=0)
##        bf.plot_intensity(event=16)
##        bf.plot_intensity(event=37)
##        bf.plot_intensity(event=47)
#
#        
#
#         #fill in Hazard file
#    def test_bush_fire_one_pass(self):
#         bf = BushFire()
#
#         bf.set_bush_fire (TEST_FIRMS, description)
#         bf.check()
#         
#         bf.plot_intensity(event=0)
#         
#
#
#
## Execute Tests
#TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderFirms)
#unittest.TextTestRunner(verbosity=2).run(TESTS)
