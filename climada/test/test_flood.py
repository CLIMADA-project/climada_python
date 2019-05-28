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

Test tc_tracks module.
"""
import os
import unittest
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from climada.entity.impact_funcs.flood import IFRiverFlood
from climada.hazard.flood import RiverFlood, select_exact_area
from climada.hazard.centroids import Centroids
from climada.entity import ImpactFuncSet
from climada.util.constants import SYSTEM_DIR, NAT_REG_ID

        
class TestRiverFlood(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""
    
    def test_exact_area_selection(self):
        
        testCentroids = select_exact_area(['LIE'])
        
        
        
        

    def test_flood_year(self):
        """ read_flood_"""
        testRF = RiverFlood()
        testRF.set_from_nc()
        
        
        
    


# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiverFlood)
unittest.TextTestRunner(verbosity=2).run(TESTS)
