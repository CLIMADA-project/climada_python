"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Unit Tests on LitPop exposures.
"""
import unittest
import numpy as np
from climada.entity.exposures.crop_production import CropProduction, normalize_with_fao_cp
from climada.util.constants import DEMO_DIR

INPUT_DIR = DEMO_DIR
FILENAME = 'histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'
FILENAME_MEAN = 'hist_mean_mai-firr_1976-2005_DE_FR.hdf5'
FILENAME_AREA = 'crop_production_demo_data_cultivated_area_CHE.nc4'
FILENAME_YIELD = 'crop_production_demo_data_yields_CHE.nc4'

class TestCropProduction(unittest.TestCase):
    """Test CropProduction Class methods"""
    def test_set_from_area_and_yield_nc4(self):
        """Test defining crop_production Exposure from area and yield
        data extracted from netcdf test data for Switzerland"""
        exp = CropProduction()
        exp.set_from_area_and_yield_nc4('whe', 2, 2,
                                        FILENAME_YIELD, FILENAME_AREA,
                                        'yield.tot', 'cultivated area all',
                                        input_dir=INPUT_DIR)

        self.assertEqual(exp.crop, 'whe')
        self.assertEqual(exp.gdf.shape[0], 55)
        self.assertEqual(exp.meta['width'] * exp.meta['height'], 55)
        self.assertIn(756, exp.gdf.region_id.values)
        self.assertIn(380, exp.gdf.region_id.values)
        self.assertAlmostEqual(exp.gdf['value'].max(), 253225.66611428373)

    def test_isimip_load_central_EU(self):
        """Test defining crop_production Exposure from complete demo file
        (Central Europe), isimip approach"""
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', unit='t/y', crop = 'mai', irr='firr')

        self.assertEqual(exp.gdf.longitude.min(), -4.75)
        self.assertEqual(exp.gdf.longitude.max(), 15.75)
        self.assertEqual(exp.gdf.latitude.min(), 42.25)
        self.assertEqual(exp.gdf.latitude.max(), 54.75)
        self.assertEqual(exp.gdf.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 't/y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.gdf.value.max(), 284244.81023404596, places=5)

    def test_set_value_to_usd(self):
        """Test calculating crop_production Exposure in [USD/y]"""
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', unit='t/y', crop = 'mai', irr='firr')
        exp.set_value_to_usd(INPUT_DIR, yearrange=(2000, 2018))
        self.assertEqual(exp.gdf.longitude.min(), -4.75)
        self.assertEqual(exp.gdf.longitude.max(), 15.75)
        self.assertEqual(exp.gdf.latitude.min(), 42.25)
        self.assertEqual(exp.gdf.latitude.max(), 54.75)
        self.assertEqual(exp.gdf.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 'USD/y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.gdf.tonnes_per_year[28], 1998.3634803238633)
        self.assertAlmostEqual(exp.gdf.value.max(), 51603897.28533253, places=5)
        self.assertAlmostEqual(exp.gdf.value.mean(), 907401.9933073953, places=5)
        self.assertEqual(exp.gdf.value.min(), 0.0)

    def test_set_value_to_kcal(self):
        """Test calculating crop_production Exposure in [kcal/y]"""

        # (1) biomass = True
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 45, 10, 50], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', unit='t/y', crop = 'mai', irr='firr')
        max_tonnes = exp.gdf.value.max()
        exp.set_value_to_kcal()


        self.assertEqual(exp.gdf.latitude.min(), 45.25)
        self.assertEqual(exp.gdf.latitude.max(), 49.75)
        self.assertEqual(exp.gdf.value.shape, (300,))
        self.assertAlmostEqual(exp.gdf.value.max(), 3.56e6 * max_tonnes, places=3)
        self.assertAlmostEqual(exp.gdf.value.max(), 852926234509.3002, places=3)
        self.assertAlmostEqual(exp.gdf.value.mean(), 19419372198.727455, places=4)
        self.assertEqual(exp.gdf.value.min(), 0.0)

        # (2) biomass = False
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                   bbox=[-5, 45, 10, 50], yearrange=np.array([2001, 2005]),
                                   scenario='flexible', unit='t/y', crop = 'mai', irr='firr')
        max_tonnes = exp.gdf.value.max()
        exp.set_value_to_kcal(biomass=False)
        self.assertEqual(exp.gdf.latitude.min(), 45.25)
        self.assertEqual(exp.gdf.value.shape, (300,))
        self.assertAlmostEqual(exp.gdf.value.max(), 3.56e6 * max_tonnes /(1-.12),
                               places=3)
        self.assertAlmostEqual(exp.gdf.value.mean(), 22067468407.644833, places=4)

    def set_value_to_usd(self):
        """Test calculating cropyield_isimip Exposure in [USD/y]"""
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                      bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                      scenario='flexible', crop = 'mai', irr='firr', unit='USD/y')
        self.assertEqual(exp.gdf.longitude.min(), -4.75)
        self.assertEqual(exp.gdf.longitude.max(), 15.75)
        self.assertEqual(exp.gdf.latitude.min(), 42.25)
        self.assertEqual(exp.gdf.latitude.max(), 54.75)
        self.assertEqual(exp.gdf.value.shape, (1092,))
        self.assertEqual(exp.value_unit, 'USD/y')
        self.assertEqual(exp.crop, 'mai')
        self.assertAlmostEqual(exp.gdf.value.max(), 51603897.28533253, places=6)

    def test_normalize_with_fao_cp(self):
        """ Test normalizing of two given exposures countrywise (usually firr + norr)
        with the mean crop production quantity"""
        exp = CropProduction()
        exp.set_from_isimip_netcdf(input_dir=INPUT_DIR, filename=FILENAME, hist_mean=FILENAME_MEAN,
                                          bbox=[-5, 42, 16, 55], yearrange=np.array([2001, 2005]),
                                          scenario='flexible', crop = 'mai', unit='t/y', irr='firr')
        country_list, ratio, exp_firr_norm, exp_noirr_norm, fao_crop_production, _exp_tot_production = \
             normalize_with_fao_cp(exp, exp, input_dir=INPUT_DIR,
                              yearrange=np.array([2009, 2018]), unit='t/y', return_data=True)
        self.assertAlmostEqual(ratio[2], 17.671166854032993)
        self.assertAlmostEqual(ratio[11], .86250775)
        self.assertAlmostEqual(fao_crop_production[2], 673416.4)
        self.assertAlmostEqual(fao_crop_production[11], 160328.7)
        self.assertAlmostEqual(np.nanmax(exp_firr_norm.gdf.value.values), 220735.69212710857)
        self.assertAlmostEqual(np.nanmax(exp_firr_norm.gdf.value.values), np.nanmax(exp_noirr_norm.gdf.value.values))
        self.assertAlmostEqual(np.nanmax(exp.gdf.value.values), 284244.81023404596)
        self.assertAlmostEqual(np.nansum(exp_noirr_norm.gdf.value.values) + np.nansum(exp_firr_norm.gdf.value.values), np.nansum(fao_crop_production), places=1)

        self.assertListEqual(list(country_list), [0, 40, 56, 70, 191, 203, 208, 250,
                                                  276, 380, 442, 528, 616, 705, 724, 756, 826])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCropProduction)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
