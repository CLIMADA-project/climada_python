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

Unit Tests on LitPop exposures.
"""
import os
import numpy as np
import unittest
from climada.entity.exposures.cropyield_isimip import CropyieldIsimip
from climada.util.constants import DATA_DIR

INPUT_DIR = os.path.join(DATA_DIR, 'demo/')
FILENAME = 'histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'
FILENAME_MEAN = 'hist_mean_mai-firr_1976-2005_DE_FR.h5'

class TestCropyieldIsimip(unittest.TestCase):
    """Test Cropyield_Isimip Class methods"""
    def test_load_central_EU(self):
        """Test defining cropyield_isimip Exposure from complete demo file (Central Europe)"""
        exp = CropyieldIsimip()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN, \
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]), \
                                      scenario='flexible', unit='t')

        self.assertEqual(exp.longitude.min(), -4.75)
        self.assertEqual(exp.longitude.max(), 15.75)
        self.assertEqual(exp.latitude.min(), 42.25)
        self.assertEqual(exp.latitude.max(), 54.75)
        self.assertEqual(exp.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 't / y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.value.max(), 200112.53407640097, places=5)

    def test_set_to_usd(self):
        """Test calculating cropyield_isimip Exposure in [USD / y]"""
        exp = CropyieldIsimip()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN, \
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]), \
                                      scenario='flexible', unit='t')
        exp.set_to_usd(INPUT_DIR)
        self.assertEqual(exp.longitude.min(), -4.75)
        self.assertEqual(exp.longitude.max(), 15.75)
        self.assertEqual(exp.latitude.min(), 42.25)
        self.assertEqual(exp.latitude.max(), 54.75)
        self.assertEqual(exp.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 'USD / y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.value.max(), 36329903.949638814, places=5)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCropyieldIsimip)
unittest.TextTestRunner(verbosity=2).run(TESTS)
