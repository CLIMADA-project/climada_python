#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

Integration tests & long unit test landslide module.
"""
import unittest
import os
import datetime as dt
from datetime import timedelta
import glob
from climada.hazard import landslide
from climada.hazard.landslide import Landslide
from climada.util.constants import DATA_DIR

LS_FILE_DIR = os.path.join(DATA_DIR, 'system')

DATA_DIR_TEST = os.path.join(os.path.dirname(__file__), 'data')

class TestTiffFcts(unittest.TestCase):
    """Unit tests for parts of the LS hazard module, but moved to integration tests
    for reasons of runtime: Test functions for getting input tiffs in landslide module,
    outside Landslide() instance"""
    def test_get_nowcast_tiff(self):
        start_date = dt.datetime.strftime(dt.datetime.now() - timedelta(5), '%Y-%m-%d')
        end_date = dt.datetime.strftime(dt.datetime.now() - timedelta(1), '%Y-%m-%d')
        tif_type = ["monthly", "daily"]

        for item in tif_type:
            landslide.get_nowcast_tiff(tif_type=item, starttime=start_date, endtime=end_date,
                                       save_path=DATA_DIR_TEST)

        search_criteria = "LS*.tif"
        LS_files_daily = glob.glob(os.path.join(DATA_DIR_TEST, search_criteria))
        search_criteria = "*5400.tif"
        LS_files_monthly = glob.glob(os.path.join(os.getcwd(), search_criteria))

        self.assertTrue(len(LS_files_daily) > 0)
        self.assertTrue(len(LS_files_monthly) == 12)

        for item in LS_files_daily:
            os.remove(item)

        for item in LS_files_monthly:
            os.remove(item)

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

class TestLSHazard(unittest.TestCase):
    """Integration test for LS hazard sets build in Landslide module"""
    def test_set_ls_model_hist(self):
        """Test the function set_LS_model for model 0 (historic hazard set)"""
        LS_hist = Landslide()
        LS_hist.set_ls_model_hist(
            bbox=[48, 10, 45, 7],
            path_sourcefile=os.path.join(DATA_DIR_TEST,
                                         'nasa_global_landslide_catalog_point.shp'),
            check_plots=0)
        self.assertEqual(LS_hist.size, 49)
        self.assertEqual(LS_hist.tag.haz_type, 'LS')
        self.assertEqual(min(LS_hist.intensity.data), 1)
        self.assertEqual(max(LS_hist.intensity.data), 1)


    def test_set_ls_model_prob(self):
        """Test the function set_LS_model for model versio UNEP_NGI, with and without neighbours"""
        LS_prob = Landslide()
        LS_prob.set_ls_model_prob(ls_model="UNEP_NGI", n_years=500, bbox=[48, 10, 45, 7],
                                  incl_neighbour=False, check_plots=0)
        self.assertEqual(LS_prob.tag.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity_prob.shape, (1, 129600))
        self.assertEqual(max(LS_prob.intensity.data), 1)
        self.assertEqual(min(LS_prob.intensity.data), 0)
        self.assertEqual(LS_prob.intensity.shape, (1, 129600))
        self.assertAlmostEqual(max(LS_prob.intensity_prob.data), 8.999999999e-05)
        self.assertEqual(min(LS_prob.intensity_prob.data), 5e-07)
        self.assertEqual(LS_prob.centroids.size, 129600)

        LS_prob_nb = Landslide()
        LS_prob_nb.set_ls_model_prob(ls_model="UNEP_NGI", n_years=500, bbox=[48, 10, 45, 7],
                                     incl_neighbour=True, check_plots=0)
        self.assertEqual(LS_prob_nb.tag.haz_type, 'LS')
        self.assertEqual(LS_prob_nb.intensity_prob.shape, (1, 129600))
        self.assertEqual(max(LS_prob_nb.intensity.data), 1)
        self.assertEqual(min(LS_prob_nb.intensity.data), 0)
        self.assertEqual(LS_prob_nb.intensity.shape, (1, 129600))
        self.assertAlmostEqual(max(LS_prob_nb.intensity_prob.data), 8.999999999e-05)
        self.assertEqual(min(LS_prob_nb.intensity_prob.data), 5e-07)
        self.assertEqual(LS_prob_nb.centroids.size, 129600)

        self.assertTrue(sum(LS_prob.intensity.data) < sum(LS_prob_nb.intensity.data))

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTiffFcts)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLSHazard))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
