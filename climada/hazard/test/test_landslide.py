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

Unit test landslide module.
"""
import unittest
import numpy as np

from climada import CONFIG
from climada.hazard.landslide import Landslide

DATA_DIR = CONFIG.hazard.test_data.dir()
LS_HIST_FILE = DATA_DIR / 'test_ls_hist.shp'
LS_PROB_FILE = DATA_DIR / 'test_ls_prob.tif'

class TestLandslideModule(unittest.TestCase):     

    def test_set_ls_hist(self):
        """ Test function set_ls_hist()"""
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=[48, 23, 40, 20], 
                                  path_sourcefile=LS_HIST_FILE, 
                                  check_plots=0)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.tag.haz_type, 'LS')
        self.assertEqual(min(LS_hist.intensity.data),1)
        self.assertEqual(max(LS_hist.intensity.data),1)

        
    def test_set_ls_prob(self):
        """ Test the function set_ls_prob()"""
        LS_prob = Landslide()
        LS_prob.set_ls_prob(bbox=[46, 11, 45, 8], 
                            path_sourcefile=LS_PROB_FILE, 
                            check_plots=0)
        LS_prob.fraction = LS_prob.fraction/10e6

        self.assertEqual(LS_prob.tag.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(1, 43200))
        self.assertEqual(LS_prob.fraction.shape,(1, 43200))
        self.assertEqual(max(LS_prob.intensity.data),1)
        self.assertEqual(min(LS_prob.intensity.data),1)
        self.assertEqual(LS_prob.intensity.todense().min(),0)
        self.assertEqual(max(LS_prob.fraction.data),2.1e-05)
        self.assertEqual(min(LS_prob.fraction.data),5e-07)
        self.assertEqual(LS_prob.fraction.todense().min(),0)
        self.assertEqual(LS_prob.frequency, np.array([1]))
        
        self.assertEqual(LS_prob.centroids.crs.data, {'init': 'epsg:4326'})
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)
    
    def test_sample_events_from_probs(self):
        LS_sampled_evs = Landslide()
        LS_sampled_evs.set_ls_prob(bbox=[46, 11, 45, 8], 
                            path_sourcefile=LS_PROB_FILE, 
                            check_plots=0)
        LS_sampled_evs.fraction = LS_sampled_evs.fraction/10e6
        LS_sampled_evs.sample_events_from_probs(n_years=100)
        self.assertTrue(max(LS_sampled_evs.fraction_prob.data) == 2.1e-05)
        self.assertTrue(min(LS_sampled_evs.fraction_prob.data) == 5e-07)
        self.assertTrue(max(LS_sampled_evs.fraction.data) <= 100)
        self.assertTrue(min(LS_sampled_evs.fraction.data) == 0)
      
      
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)           
