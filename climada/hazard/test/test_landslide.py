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
# import datetime as dt
# from datetime import timedelta
# import numpy as np
# import glob
from rasterio.windows import Window
# from climada.hazard import landslide
from climada.hazard.landslide import Landslide
import math
from climada.util.constants import DATA_DIR
LS_FILE_DIR = os.path.join(DATA_DIR, 'system')

DATA_DIR_TEST = os.path.join(os.path.dirname(__file__), 'data')

# class TestTiffFcts(unittest.TestCase):
#    """Test functions for getting input tiffs in landslide module, outside Landslide() instance"""
#    def test_get_nowcast_tiff(self):
#        start_date = dt.datetime.strftime(dt.datetime.now() - timedelta(2), '%Y-%m-%d')
#        end_date = dt.datetime.strftime(dt.datetime.now() - timedelta(1), '%Y-%m-%d')
#        tif_type= ["monthly","daily"]
#
#        for item in tif_type:
#            landslide.get_nowcast_tiff(tif_type=item, startTime=start_date, endTime=end_date, save_path=DATA_DIR)
#
#        search_criteria = "LS*.tif"
#        LS_files_daily = glob.glob(os.path.join(DATA_DIR, search_criteria))
#        search_criteria = "*5400.tif"
#        LS_files_monthly = glob.glob(os.path.join(os.getcwd(), search_criteria))
#
#        self.assertTrue(len(LS_files_daily)>0)
#        self.assertTrue(len(LS_files_monthly)==12)
#
#        for item in LS_files_daily:
#            os.remove(item)
#
#        for item in LS_files_monthly:
#            os.remove(item)

#    def test_combine_nowcast_tiff(self):
#        landslide.combine_nowcast_tiff(DATA_DIR, search_criteria='test_global*.tif', operator="maximum")
#        search_criteria = "combined*.tif"
#        combined_daily = glob.glob(os.path.join(DATA_DIR, search_criteria))
#        self.assertEqual(len(combined_daily), 1)
#        for item in combined_daily:
#            os.remove(item)
#
#        landslide.combine_nowcast_tiff(DATA_DIR, search_criteria='*5400_test.tif', operator="sum")
#        search_criteria = "combined*.tif"
#        combined_monthly = glob.glob(os.path.join(DATA_DIR, search_criteria))
#        self.assertEqual(len(combined_monthly),1)
#        for item in combined_monthly:
#            os.remove(item)

class TestLandslideModule(unittest.TestCase):

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

    def test_get_hist_events(self):
        empty_LS = Landslide()
        bbox = [48, 23, 40, 20]
        COOLR_path = os.path.join(DATA_DIR_TEST, 'nasa_global_landslide_catalog_point.shp')
        LS_catalogue_part = empty_LS._get_hist_events(bbox, COOLR_path)
        self.assertTrue(max(LS_catalogue_part.latitude) <= bbox[0])
        self.assertTrue(min(LS_catalogue_part.latitude) >= bbox[2])
        self.assertTrue(max(LS_catalogue_part.longitude) <= bbox[1])
        self.assertTrue(min(LS_catalogue_part.longitude) >= bbox[3])


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
