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
from climada.entity.exposures.crop_production import CropProduction, normalize_with_fao_cp
from climada.util.constants import DATA_DIR

INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FILENAME = 'histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'
FILENAME_MEAN = 'hist_mean_mai-firr_1976-2005_DE_FR.hdf5'

class TestCropProduction(unittest.TestCase):
    """Test Cropyield_Isimip Class methods"""
    def test_load_central_EU(self):
        """Test defining crop_production Exposure from complete demo file (Central Europe)"""
        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', unit='t', crop = 'mai', irr='firr')

        self.assertEqual(exp.longitude.min(), -4.75)
        self.assertEqual(exp.longitude.max(), 15.75)
        self.assertEqual(exp.latitude.min(), 42.25)
        self.assertEqual(exp.latitude.max(), 54.75)
        self.assertEqual(exp.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 't / y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.value.max(), 284244.81023404596, places=5)

    def test_set_to_usd(self):
        """Test calculating crop_production Exposure in [USD / y]"""
        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', unit='t', crop = 'mai', irr='firr')
        exp.set_to_usd(INPUT_DIR)
        self.assertEqual(exp.longitude.min(), -4.75)
        self.assertEqual(exp.longitude.max(), 15.75)
        self.assertEqual(exp.latitude.min(), 42.25)
        self.assertEqual(exp.latitude.max(), 54.75)
        self.assertEqual(exp.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 'USD / y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.tonnes_per_year[28], 1998.3634803238633)
        self.assertAlmostEqual(exp.value.max(), 51603897.28533253, places=5)
        self.assertAlmostEqual(exp.value.mean(), 907401.9933073953, places=5)
        self.assertEqual(exp.value.min(), 0.0)
        
    
    def test_set_to_usd_unnecessary(self):
        """Test calculating cropyield_isimip Exposure in [USD / y]"""
        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', crop = 'mai', irr='firr')
        self.assertEqual(exp.longitude.min(), -4.75)
        self.assertEqual(exp.longitude.max(), 15.75)
        self.assertEqual(exp.latitude.min(), 42.25)
        self.assertEqual(exp.latitude.max(), 54.75)
        self.assertEqual(exp.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 'USD / y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.value.max(), 51603897.28533253, places=6)

    def test_normalize_with_fao_cp(self):
        """ Test normalizing of two given exposures countrywise (usually firr + norr)
        with the mean crop production quantity"""
        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                          bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                          scenario='flexible', crop = 'mai', unit='t', irr='firr')
        country_list, ratio, exp_firr_norm, exp_noirr_norm, fao_crop_production, exp_tot_production = \
             normalize_with_fao_cp(exp, exp, input_dir=INPUT_DIR,
                              yearrange=np.array([2009, 2018]), unit='t', return_data=True)
        self.assertAlmostEqual(ratio[2], 17.671166854032993)
        self.assertAlmostEqual(ratio[11], .86250775)
        self.assertAlmostEqual(fao_crop_production[2], 673416.4)
        self.assertAlmostEqual(fao_crop_production[11], 160328.7)
        self.assertAlmostEqual(np.nanmax(exp_firr_norm.value.values), 220735.69212710857)
        self.assertAlmostEqual(np.nanmax(exp_firr_norm.value.values), np.nanmax(exp_noirr_norm.value.values))
        self.assertAlmostEqual(np.nanmax(exp.value.values), 284244.81023404596)
        self.assertAlmostEqual(np.nansum(exp_noirr_norm.value.values) + np.nansum(exp_firr_norm.value.values), np.nansum(fao_crop_production), places=1)

        self.assertListEqual(list(country_list), [0, 40, 56, 70, 191, 203, 208, 250,
                                                  276, 380, 442, 528, 616, 705, 724, 756, 826])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCropProduction)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
