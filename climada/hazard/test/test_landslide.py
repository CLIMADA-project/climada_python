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
import os
import numpy as np
from climada.hazard.landslide import Landslide

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestLandslideModule(unittest.TestCase):     
             
    def test_incl_affected_surroundings(self):
        #TODO: write test once function is done
        """test function _incl_affected_surroundings()"""
        pass
    
    def test_get_hist_events(self):
        """test function _get_hist_events()"""
        empty_LS = Landslide()
        bbox = [48,23,40,20]
        path_sourcefile = os.path.join(TESTDATA_DIR,'test_ls_hist.shp')
        ls_subgdf = empty_LS._get_hist_events(bbox, path_sourcefile)
        self.assertEqual(len(ls_subgdf), 272)
        self.assertTrue(max(ls_subgdf.geometry.y)<=bbox[0])
        self.assertTrue(max(ls_subgdf.geometry.x)<=bbox[1])
        self.assertTrue(min(ls_subgdf.geometry.y)>=bbox[2])
        self.assertTrue(min(ls_subgdf.geometry.x)>=bbox[3])

    def test_set_ls_hist(self):
        """ Test function set_ls_hist()"""
        LS_hist = Landslide()
        COOLR_path = os.path.join(TESTDATA_DIR,'test_ls_hist.shp')
        LS_hist.set_ls_hist(bbox=[48, 23, 40, 20], 
                                  path_sourcefile=COOLR_path, 
                                  check_plots=0)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.tag.haz_type, 'LS')
        self.assertEqual(min(LS_hist.intensity.data),1)
        self.assertEqual(max(LS_hist.intensity.data),1)

        
    def test_set_ls_prob(self):
        """ Test the function set_ls_prob()"""
        LS_prob = Landslide()
        LS_prob.set_ls_prob(bbox=[46, 11, 45, 8], 
                                  path_sourcefile=os.path.join(TESTDATA_DIR, 'test_ls_prob.tif'), 
                                  check_plots=0)

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
        
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)           
