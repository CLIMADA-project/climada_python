#"""
#Test Bush fire class
#"""
#
#import os
#import unittest
#import pandas as pd
#from datetime import datetime
#import numpy as np
#
#
#from climada.hazard.bush_fire import BushFire
#
#DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
#TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_firms.csv")
##TEST_FIRMS = os.path.join(DATA_DIR, "Spain_2013-2018_firms.csv")
#description = ''
#
#class TestReaderFirms(unittest.TestCase):
#    """Test loading functions from the BushFire class"""
#
#    def test_read_one_pass(self):
#         bf = BushFire()
#         firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
#
#         self.assertEqual(firms['latitude'][0], -38.104)
#         self.assertEqual(firms['longitude'][0], 146.388)
#         self.assertEqual(firms['latitude'].iloc[-1], -36.016)
#         self.assertEqual(firms['longitude'].iloc[-1], 146.97)
#         self.assertFalse(firms['datenum'][0] == firms['datenum'][29])
#
#    def test_cons_days(self):
#        bf = BushFire()
#        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
#        firms = bf._firms_cons_days(firms)
#
#        self.assertEqual(firms['cons_id'][27], 0)
#        self.assertEqual(firms['cons_id'][28], 1)
#        self.assertEqual(firms['cons_id'][194], 1)
#        self.assertEqual(firms['cons_id'][195], 2)
#        self.assertEqual(firms['cons_id'][197], 2)
#        self.assertEqual(firms['cons_id'][198], 3)
#        self.assertEqual(firms['cons_id'][2136], 3)
#        self.assertEqual(firms['cons_id'][2137], 4)
#
#    def test_clustering(self):
#         bf = BushFire()
#         firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
#         #add cons_id
#         firms = bf._firms_cons_days(firms)
#         #add clus_id
#         firms = bf._firms_clustering(firms)
#
#         self.assertEqual(max((firms['clus_id'][:28]).values), 2)
#         self.assertEqual(max((firms['clus_id'][195:198]).values), 1)
#
#    def test_event_one_pass(self):
#        bf = BushFire()
#        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
#        firms = bf._firms_cons_days(firms)
#        firms = bf._firms_clustering(firms)
#        #add event_id
#        firms = bf._firms_event(firms)
#        self.assertEqual(max((firms['event_id'][:28]).values), 3)
#        self.assertEqual(max((firms['event_id'][28:195]).values), 10)
#
#    def test_brightness_one_pass(self):
#        bf = BushFire()
#        firms, csv_firms, description = bf._read_firms_csv(TEST_FIRMS)
#        firms = bf._firms_cons_days(firms)
#        firms = bf._firms_clustering(firms)
#        firms = bf._firms_event(firms)
#        centroids = bf._centroids_creation(firms)
#
#
#        brightness = bf._calc_brightness(firms, centroids)
#
## Execute Tests
#TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderFirms)
#unittest.TextTestRunner(verbosity=2).run(TESTS)
#
#class TestReaderDF(unittest.TestCase):
#    """Test loading functions from the BushFire class for synthetic data"""
#
#    def _read_firms_synth(firms, description=''):
#        """Read synthetic files from bushfire data.
#
#        Parameters:
#            firms: dataframe with latitude, longitude, acquisition date (acq_date)
#            description (str, optional): description of the events
#
#        Returns:
#            firms, description
#        """
#        firms = pd.DataFrame.from_dict(firms)
#        for index, acq_date in enumerate(firms['acq_date'].values):
#            datenum = datetime.strptime(acq_date, '%Y-%M-%d').toordinal()
#            firms.at[index, 'datenum'] = datenum
#        return firms, description
#
#    def test_read_one_pass(self):
#        bf = BushFire()
#        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.093, -38.095,
#             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
#             -38.095, -37.45],
#                 'longitude' : [146.388, 146.388, 146.388, 146.397, 146.386, 142.43,
#                       142.442, 142.428, 145.361, 146.388, 146.388, 146.388,
#                       146.397, 145.361],
#        'brightness' : [316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
#             359.6, 312.9, 100, 400, 500, 300, 250],
#        'acq_date' : ['2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
#             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
#             '2006-01-28', '2006-01-28', '2006-01-30']}
#
#        firms, description = TestReaderDF._read_firms_synth(firms)
#
#        self.assertEqual(firms['latitude'][0], -38.104)
#        self.assertEqual(firms['longitude'][0], 146.388)
#        self.assertEqual(firms['latitude'].iloc[-1], -37.45)
#        self.assertEqual(firms['longitude'].iloc[-1], 145.361)
#        self.assertFalse(firms['datenum'][0] == firms['datenum'][9])
#
#    def test_cons_days(self):
#        bf = BushFire()
#        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.093, -38.095,
#             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
#             -38.095, -37.45],
#        'longitude' : [146.388, 146.388, 146.388, 146.397, 146.386, 142.43,
#             142.442, 142.428, 145.361, 146.388, 146.388, 146.388,
#             146.397, 145.361],
#        'brightness' : [316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
#             359.6, 312.9, 100, 400, 500, 300, 250],
#        'acq_date' : ['2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
#             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
#             '2006-01-28', '2006-01-28', '2006-01-30']}
#        firms, description = TestReaderDF._read_firms_synth(firms)
#        # add cons_id
#        firms = bf._firms_cons_days(firms)
#
#        self.assertEqual(firms['cons_id'][0], 0)
#        self.assertEqual(firms['cons_id'][7], 0)
#        self.assertEqual(firms['cons_id'][8], 0)
#        self.assertEqual(firms['cons_id'][9], 0)
#        self.assertEqual(firms['cons_id'][10], 0)
#        self.assertEqual(firms['cons_id'][11], 1)
#        self.assertEqual(firms['cons_id'][12], 1)
#        self.assertEqual(firms['cons_id'][13], 2)
#
#    def clustering(self):
#        bf = BushFire()
#        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.093, -38.095,
#             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
#             -38.095, -37.45],
#        'longitude' : [146.388, 146.388, 146.388, 146.397, 146.386, 142.43,
#             142.442, 142.428, 145.361, 146.388, 146.388, 146.388,
#             146.397, 145.361],
#        'brightness' : [316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
#             359.6, 312.9, 100, 400, 500, 300, 250],
#        'acq_date' : ['2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
#             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
#             '2006-01-28', '2006-01-28', '2006-01-30']}
#        firms, description = TestReaderDF._read_firms_synth(firms)
#        firms = bf._firms_cons_days(firms)
#         #add clus_id
#        firms = bf._firms_clustering(firms)
#
#        self.assertEqual(max((firms['clus_id'][:10]).values), 2)
#        self.assertEqual(max((firms['clus_id'][11:13]).values), 0)
#
#    def test_event_one_pass(self):
#        bf = BushFire()
#        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.093, -38.095,
#             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
#             -38.095, -37.45],
#        'longitude' : [146.388, 146.388, 146.388, 146.397, 146.386, 142.43,
#             142.442, 142.428, 145.361, 146.388, 146.388, 146.388,
#             146.397, 145.361],
#        'brightness' : [316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
#             359.6, 312.9, 100, 400, 500, 300, 250],
#        'acq_date' : ['2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
#             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
#             '2006-01-28', '2006-01-28', '2006-01-30']}
#        firms, description = TestReaderDF._read_firms_synth(firms)
#        firms = bf._firms_cons_days(firms)
#        firms = bf._firms_clustering(firms)
#        #add event_id
#        firms = bf._firms_event(firms)
#        centroids = bf._centroids_creation(firms)
#
#        self.assertEqual(max((firms['event_id'][:10]).values), 3)
#        self.assertEqual(firms['event_id'][11], 4)
#        self.assertEqual(firms['event_id'][12], 4)
#        self.assertEqual(firms['event_id'][13], 5)
#
#    def test_brightness_one_pass(self):
#        bf = BushFire()
#        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.093, -38.095,
#             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
#             -38.095, -37.45],
#        'longitude' : [146.388, 146.388, 146.388, 146.397, 146.386, 142.43,
#             142.442, 142.428, 145.361, 146.388, 146.388, 146.388,
#             146.397, 145.361],
#        'brightness' : [316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
#             359.6, 312.9, 100, 400, 500, 300, 250],
#        'acq_date' : ['2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
#             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
#             '2006-01-28', '2006-01-28', '2006-01-30']}
#
#        firms, description = TestReaderDF._read_firms_synth(firms)
#        firms = bf._firms_cons_days(firms)
#        firms = bf._firms_clustering(firms)
#        firms = bf._firms_event(firms)
#        centroids = bf._centroids_creation(firms)
#        # add brightness matrix
#
#        brightness = bf._calc_brightness(firms, centroids)
#
#        bf.centroids = centroids
#        bf.event_id = np.array(np.unique(firms['event_id'].values))
#        bf.event_name = np.array(np.unique(firms['event_id']))
#        bf.intensity = brightness
#
#        bf.plot_intensity(event=0)
#        bf.plot_intensity(event=1)
#        bf.plot_intensity(event=2)
#        bf.plot_intensity(event=3)
#        bf.plot_intensity(event=4)
#        bf.plot_intensity(event=5)
#
#
#
## Execute Tests
#TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderDF)
#unittest.TextTestRunner(verbosity=2).run(TESTS)
