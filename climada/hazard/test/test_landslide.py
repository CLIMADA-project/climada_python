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
import math
from rasterio.windows import Window
from climada.hazard.landslide import Landslide

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


class TestLandslideModule(unittest.TestCase):     
    
    def test_get_window_from_coords(self):
        """Test function _get_window_from_coords"""
        empty_LS = Landslide()
        window_array = empty_LS._get_window_from_coords(
            path_sourcefile=os.path.join(TESTDATA_DIR, 'test_ls_prob.tif'), 
            bbox=[45.5,9.5,45.1,8.5])
        self.assertEqual(window_array[0], 60)
        self.assertEqual(window_array[1], 60)
        self.assertEqual(window_array[2], 120)
        self.assertEqual(window_array[3], 48)
        
    def test_get_raster_meta(self):
        """Test function _get_raster_meta"""
        empty_LS = Landslide()
        pixel_width, pixel_height = empty_LS._get_raster_meta(
            path_sourcefile = os.path.join(TESTDATA_DIR, 'test_ls_prob.tif'), 
            window_array = [60, 60, 120, 48])
        self.assertTrue(math.isclose(pixel_width, -0.00833, rel_tol=1e-03))
        self.assertTrue(math.isclose(pixel_height, 0.00833, rel_tol=1e-03))
             
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
        # self.assertEqual(LS_prob.intensity_prob.shape,(1, 129600))
        # self.assertEqual(max(LS_prob.intensity.data),1)
        # self.assertEqual(min(LS_prob.intensity.data),0)
        # self.assertEqual(LS_prob.intensity.shape,(1, 129600))
        # self.assertAlmostEqual(max(LS_prob.intensity_prob.data),8.999999999e-05)
        # self.assertEqual(min(LS_prob.intensity_prob.data),5e-07)
        # self.assertEqual(LS_prob.centroids.size, 129600)    
        
        # LS_prob_nb = Landslide()
        # LS_prob_nb.set_ls_prob(bbox=[48, 10, 45, 7], 
        #                           path_sourcefile=os.path.join(SYSTEM_DIR, 'ls_pr/ls_pr.tif'), 
        #                           incl_neighbour=False, check_plots=0)
        # self.assertEqual(LS_prob_nb.tag.haz_type, 'LS')
        # self.assertEqual(LS_prob_nb.intensity_prob.shape,(1, 129600))
        # self.assertEqual(max(LS_prob_nb.intensity.data),1)
        # self.assertEqual(min(LS_prob_nb.intensity.data),0)
        # self.assertEqual(LS_prob_nb.intensity.shape,(1, 129600))
        # self.assertAlmostEqual(max(LS_prob_nb.intensity_prob.data),8.999999999e-05)
        # self.assertEqual(min(LS_prob_nb.intensity_prob.data),5e-07)
        # self.assertEqual(LS_prob_nb.centroids.size, 129600) 
        
        # self.assertTrue(sum(LS_prob.intensity.data)<sum(LS_prob_nb.intensity.data))
    
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)           