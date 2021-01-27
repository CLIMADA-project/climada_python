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
import math
from rasterio.windows import Window

from climada import CONFIG
from climada.hazard.landslide import Landslide
from climada.util.constants import SYSTEM_DIR as LS_FILE_DIR

DATA_DIR = CONFIG.hazard.test_data.dir()


class TestLandslideModule(unittest.TestCase):     
             
    def test_incl_affected_surroundings(self):
        #TODO: write test once function is done
        """test function _incl_affected_surroundings()"""
        pass
    
def test_get_window_from_coords(self):
        empty_LS = Landslide()
        window_array = empty_LS._get_window_from_coords(
            path_sourcefile=os.path.join(LS_FILE_DIR, 'ls_pr_NGI_UNEP/ls_pr.tif'),
            bbox=[47, 8, 46, 7])
        self.assertEqual(window_array[0], 22440)
        self.assertEqual(window_array[1], 5159)
        self.assertEqual(window_array[2], 120)
        self.assertEqual(window_array[3], 120)

    def test_get_raster_meta(self):
        empty_LS = Landslide()
        pixel_width, pixel_height = empty_LS._get_raster_meta(
            path_sourcefile=os.path.join(LS_FILE_DIR, 'ls_pr_NGI_UNEP/ls_pr.tif'),
            window_array=[865, 840, 120, 120])
        self.assertTrue(math.isclose(pixel_width, -0.00833, rel_tol=1e-03))
        self.assertTrue(math.isclose(pixel_height, 0.00833, rel_tol=1e-03))

    def test_intensity_cat_to_prob(self):
        empty_LS = Landslide()
        window_array = empty_LS._get_window_from_coords(
            path_sourcefile=os.path.join(DATA_DIR_TEST,
                                         'test_global_landslide_nowcast_20190501.tif'),
            bbox=[47, 23, 46, 22])
        empty_LS.set_raster(
            [os.path.join(DATA_DIR_TEST, 'test_global_landslide_nowcast_20190501.tif')],
            window=Window(window_array[0], window_array[1], window_array[3], window_array[2]))
        empty_LS._intensity_cat_to_prob(max_prob=0.0001)
        self.assertTrue(max(empty_LS.intensity_cat.data) == 2)
        self.assertTrue(min(empty_LS.intensity_cat.data) == 1)
        self.assertTrue(max(empty_LS.intensity.data) == 0.0001)
        self.assertTrue(min(empty_LS.intensity.data) == 0)

    def test_intensity_prob_to_binom(self):
        empty_LS = Landslide()
        window_array = empty_LS._get_window_from_coords(
            path_sourcefile=os.path.join(DATA_DIR_TEST,
                                         'test_global_landslide_nowcast_20190501.tif'),
            bbox=[47, 23, 46, 22])
        empty_LS.set_raster(
            [os.path.join(DATA_DIR_TEST, 'test_global_landslide_nowcast_20190501.tif')],
            window=Window(window_array[0], window_array[1], window_array[3], window_array[2]))
        empty_LS._intensity_cat_to_prob(max_prob=0.0001)
        empty_LS._intensity_prob_to_binom(100)
        self.assertTrue(max(empty_LS.intensity_prob.data) == 0.0001)
        self.assertTrue(min(empty_LS.intensity_prob.data) == 0)
        self.assertTrue(max(empty_LS.intensity.data) == 1)
        self.assertTrue(min(empty_LS.intensity.data) == 0)

    def test_intensity_binom_to_range(self):
        empty_LS = Landslide()
        window_array = empty_LS._get_window_from_coords(
            path_sourcefile=os.path.join(DATA_DIR_TEST,
                                         'test_global_landslide_nowcast_20190501.tif'),
            bbox=[47, 23, 46, 22])
        empty_LS.set_raster(
            [os.path.join(DATA_DIR_TEST, 'test_global_landslide_nowcast_20190501.tif')],
            window=Window(window_array[0], window_array[1], window_array[3], window_array[2]))
        empty_LS._intensity_cat_to_prob(max_prob=0.0001)
        empty_LS._intensity_prob_to_binom(100)
        empty_LS.check()
        empty_LS.centroids.set_meta_to_lat_lon()
        empty_LS.centroids.set_geometry_points()
        empty_LS._intensity_binom_to_range(max_dist=1000)
        self.assertTrue(
            len(empty_LS.intensity.data[(empty_LS.intensity.data > 0)
                                        & (empty_LS.intensity.data < 1)]) > 0)

    def test_gdf_from_bbox(self):
        """test function _gdf_from_bbox()"""
        empty_LS = Landslide()
        bbox = [48,23,40,20]
        path_sourcefile = os.path.join(TESTDATA_DIR,'test_ls_hist.shp')
        ls_subgdf = empty_LS._gdf_from_bbox(bbox, path_sourcefile)
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
