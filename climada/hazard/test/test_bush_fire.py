"""
Test Bush fire class
"""

import os
import unittest
import pandas as pd
from datetime import datetime
import numpy as np


from climada.hazard.bush_fire import BushFire

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_FIRMS = os.path.join(DATA_DIR, "Victoria_firms.csv")
#TEST_FIRMS = os.path.join(DATA_DIR, "Spain_2013-2018_firms.csv")
description = ''

class TestReaderDF(unittest.TestCase):
    """Test loading functions from the BushFire class for synthetic data"""

    def _read_firms_synth(firms, description=''):
        """Read synthetic files from bushfire data.

        Parameters:
            firms: dataframe with latitude, longitude, acquisition date (acq_date)
            description (str, optional): description of the events

        Returns:
            firms, description
        """
        firms = pd.DataFrame.from_dict(firms)
        for index, acq_date in enumerate(firms['acq_date'].values):
            d = datetime.strptime(acq_date, '%Y-%m-%d').date()
            datenum = datetime.strptime(acq_date, '%Y-%M-%d').toordinal()
            firms.at[index, 'datenum'] = datenum
        return firms, description
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

    def test_brightness_one_pass(self):
        bf = BushFire()
        firms = {'latitude' : [-38.104, -38.104, -38.104, -38.104, -38.104, -38.093, -38.095,
             -37.433, -37.421, -37.423, -37.45, -38.104, -38.104, -38.104,
             -38.095, -37.45, -38.045, -38.047, -38.036, -37.983,
             -37.978, -37.979, -37.981, -37.45, -37.431,
             -37.421, -37.423, -38.104],
        'longitude' : [146.388, 146.388,146.388, 146.388, 146.388, 146.397, 
                       146.386, 142.43,142.442, 142.428, 145.361, 146.388, 
                       146.388, 146.388,
             146.397, 145.361, 146.416, 146.404, 146.413, 146.33,
             146.311, 146.299, 146.288, 145.361, 142.445,
             142.442, 142.428, 146.388],
        'brightness' : [400, 10, 316.5, 150, 500, 312.6, 312.7, 324.4, 373.6,
             359.6, 312.9, 100, 400, 500, 300, 250, 100, 150, 300, 400,
             250, 300, 332, 450, 200, 150, 400, 100],
        'acq_date' : ['2008-03-08', '2008-03-08', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24',
             '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-24', '2006-01-25', '2006-01-25',
             '2006-01-28', '2006-01-28', '2006-01-30', '2006-03-06', '2006-03-06', '2006-03-06', 
             '2006-03-06', '2007-01-24', '2007-01-24', '2007-01-24', '2007-01-24', '2007-01-24',
             '2008-03-08', '2008-03-08', '2008-03-08']}

        firms, description = TestReaderDF._read_firms_synth(firms)
        firms = bf._firms_cons_days(firms)
        firms = bf._firms_clustering(firms)
        firms = bf._firms_event(firms)
        centroids = bf._centroids_creation(firms)
         # add brightness matrix
        brightness, num_centr, latlon = bf._calc_brightness(firms, centroids)
        
        #check event 1
        self.assertEqual(brightness[0, 493], 500)
        self.assertEqual(brightness[0, 1996], 500)
        self.assertEqual(brightness[0, 1493], 312.7)
        self.assertEqual(brightness[0, 5993], 312.7)
        self.assertEqual(brightness[0, 4496], 312.6)
        self.assertEqual(brightness[0, 10997], 312.6)
        #check event 2
        self.assertEqual(brightness[1, 232867], 312.9)
#        
    



# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderDF)
unittest.TextTestRunner(verbosity=2).run(TESTS)
