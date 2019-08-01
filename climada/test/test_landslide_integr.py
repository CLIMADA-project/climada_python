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

Unit test landslide module.
"""
import unittest
import os
import datetime as dt
from datetime import timedelta
import numpy as np
import glob
from rasterio.windows import Window
from climada.hazard import landslide
from climada.hazard.landslide import Landslide
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

""" Unit tests for parts of the LS hazard module, but moved to integration tests 
for reasons of runtime"""

class TestTiffFcts(unittest.TestCase):
    """Test functions for getting input tiffs in landslide module, outside Landslide() instance"""
    def test_get_nowcast_tiff(self):
        start_date = dt.datetime.strftime(dt.datetime.now() - timedelta(2), '%Y-%m-%d')
        end_date = dt.datetime.strftime(dt.datetime.now() - timedelta(1), '%Y-%m-%d')
        tif_type= ["monthly","daily"]
        
        for item in tif_type:
            landslide.get_nowcast_tiff(tif_type=item, startTime=start_date, endTime=end_date, save_path=DATA_DIR)
            
        search_criteria = "LS*.tif"
        LS_files_daily = glob.glob(os.path.join(DATA_DIR, search_criteria))
        search_criteria = "*5400.tif"
        LS_files_monthly = glob.glob(os.path.join(os.getcwd(), search_criteria))
        
        self.assertTrue(len(LS_files_daily)>0)
        self.assertTrue(len(LS_files_monthly)==12)
        
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
        
  
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTiffFcts)
    unittest.TextTestRunner(verbosity=2).run(TESTS)           