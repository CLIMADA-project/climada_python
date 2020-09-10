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

Tests on Drought Hazard exposure and Impact function.
"""

import unittest
import os
import numpy as np
from climada.util.constants import DATA_DIR
from climada.hazard.relative_cropyield import (RelativeCropyield, init_hazard_set, 
                                               calc_his_haz)
from climada.entity.exposures.crop_production import CropProduction
from climada.entity import ImpactFuncSet, IFRelativeCropyield
from climada.engine import Impact


INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FN_STR_DEMO = 'annual_FR_DE_DEMO'
FILENAME_LU = 'histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'
FILENAME_MEAN = 'hist_mean_mai-firr_1976-2005_DE_FR.hdf5'


class TestIntegr(unittest.TestCase):
    """Test loading functions from the ISIMIP Agricultural Drought class and
        computing impact on crop production"""
    def test_EU(self):
        """test with demo data containing France and Germany"""
        bbox = [-5, 42, 16, 55]
        haz = RelativeCropyield()
        haz.set_from_single_run(input_dir=INPUT_DIR, yearrange=(2001, 2005), bbox=bbox,
                                ag_model='lpjml', cl_model='ipsl-cm5a-lr', scenario='historical',
                                soc='2005soc', co2='co2', crop='whe', irr='noirr',
                                fn_str_var=FN_STR_DEMO)
        hist_mean = haz.calc_mean(yearrange_mean=(2001, 2005))
        haz.set_rel_yield_to_int(hist_mean)
        haz.centroids.set_region_id()

        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME_LU, hist_mean=FILENAME_MEAN,
                                              bbox=bbox, yearrange=(2001, 2005),
                                              scenario='flexible', unit='t', crop='whe', irr='firr')

        exp.set_to_usd(INPUT_DIR)
        exp.assign_centroids(haz, threshold=20)

        if_cp = ImpactFuncSet()
        if_def = IFRelativeCropyield()
        if_def.set_relativeyield()
        if_cp.append(if_def)
        if_cp.check()

        impact = Impact()
        impact.calc(exp.loc[exp.region_id == 276], if_cp, haz.select(['2002']), save_mat=True)

        exp_manual = exp.value.loc[exp.region_id == 276].values
        impact_manual = haz.select(event_names=['2002'], reg_id=276).intensity.multiply(exp_manual)
        dif = (impact_manual - impact.imp_mat).data

        self.assertEqual(haz.tag.haz_type, 'RC')
        self.assertEqual(haz.size, 5)
        self.assertEqual(haz.centroids.size, 1092)
        self.assertAlmostEqual(haz.intensity.mean(), -2.0489097e-08)
        self.assertAlmostEqual(exp.value.max(), 53074789.755290434)
        self.assertEqual(exp.latitude.values.size, 1092)
        self.assertAlmostEqual(exp.value[3], 0.0)
        self.assertAlmostEqual(exp.value[1077], 405026.6857207429)
        self.assertAlmostEqual(impact.imp_mat.data[3], -176102.5359452465 )
        self.assertEqual(len(dif), 0)

    def test_EU_nan(self):
        """Test whether setting the zeros in exp.value to NaN changes the impact"""
        bbox=[0, 42, 10, 52]
        haz = RelativeCropyield()
        haz.set_from_single_run(input_dir=INPUT_DIR, yearrange=(2001, 2005), bbox=bbox,
                                ag_model='lpjml', cl_model='ipsl-cm5a-lr', scenario='historical',
                                soc='2005soc', co2='co2', crop='whe', irr='noirr',
                                fn_str_var=FN_STR_DEMO)
        hist_mean = haz.calc_mean(yearrange_mean=(2001, 2005))
        haz.set_rel_yield_to_int(hist_mean)
        haz.centroids.set_region_id()

        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME_LU, hist_mean=FILENAME_MEAN,
                                              bbox=bbox, yearrange=(2001, 2005),
                                              scenario='flexible', unit='t', crop='whe', irr='firr')
        exp.assign_centroids(haz, threshold=20)

        if_cp = ImpactFuncSet()
        if_def = IFRelativeCropyield()
        if_def.set_relativeyield()
        if_cp.append(if_def)
        if_cp.check()

        impact = Impact()
        impact.calc(exp, if_cp, haz, save_mat=True)

        exp_nan = CropProduction()
        exp_nan.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME_LU, hist_mean=FILENAME_MEAN,
                                              bbox=[0, 42, 10, 52], yearrange=(2001, 2005),
                                              scenario='flexible', unit='t', crop='whe', irr='firr')
        exp_nan.value[exp_nan.value==0] = np.nan
        exp_nan.assign_centroids(haz, threshold=20)

        impact_nan = Impact()
        impact_nan.calc(exp_nan, if_cp, haz, save_mat=True)
        self.assertListEqual(list(impact.at_event), list(impact_nan.at_event))
        self.assertAlmostEqual(12.056545220060798, impact_nan.aai_agg)
        self.assertAlmostEqual(12.056545220060798 , impact.aai_agg)

    def test_hist_mean_of_full_haz_set(self):
        """Test creation of full hazard set"""

        output_list = list()
        bbox = [116.25, 38.75,  117.75, 39.75]
        files_his = ['gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_global_DEMO_TJANJIN_annual_1861_2005.nc',
                     'pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-firr_global_annual_DEMO_TJANJIN_1861_2005.nc',
                     'pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-noirr_global_annual_DEMO_TJANJIN_1861_2005.nc']
        
        (his_file_list, file_props, hist_mean_per_crop, 
          scenario_list, crop_list) = init_hazard_set(files_his, input_dir=INPUT_DIR,
                                                      bbox=bbox, isimip_run = 'test_file', 
                                                      yearrange_his = np.array([1980,2005]))
        yearrange_mean = np.array([1980,2005])
        for his_file in his_file_list:
            haz_his, filename, hist_mean = calc_his_haz(his_file, file_props, input_dir=INPUT_DIR, 
                                                        bbox=bbox, yearrange_mean=yearrange_mean)

            hist_mean_per_crop[(file_props[his_file])['crop_irr']]['value'][ 
                hist_mean_per_crop[(file_props[his_file])['crop_irr']]['idx'], :] = hist_mean
            hist_mean_per_crop[file_props[his_file]['crop_irr']]['idx'] += 1
        
        
        self.assertEqual(np.shape(hist_mean_per_crop['whe-firr']['value'])[0], 1)
        self.assertEqual(np.shape(hist_mean_per_crop['whe-noirr']['value'])[0], 2) 
        
        # calculate mean hist_mean for each crop-irrigation
        for crop_irr in crop_list:
            mean = np.mean((hist_mean_per_crop[crop_irr])['value'], 0)
            output_list.append(mean)
        
        self.assertEqual('whe-noirr', crop_list[0])
        self.assertEqual(np.mean(hist_mean_per_crop['whe-noirr']['value']), np.mean(output_list[0]))
        self.assertEqual(np.mean(hist_mean_per_crop['whe-noirr']['value'][:,1]), output_list[0][1])
        
                                                      
# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIntegr)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
