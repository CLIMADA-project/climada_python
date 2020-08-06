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
from climada.util.constants import DATA_DIR
from climada.hazard.relative_cropyield import RelativeCropyield
from climada.entity.exposures.crop_production import CropProduction, normalize_with_fao_cp
from climada.entity import ImpactFuncSet, IFRelativeCropyield
from climada.engine import Impact


INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FN_STR_DEMO = 'annual_FR_DE_DEMO'
FILENAME_LU = 'histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'
FILENAME_MEAN = 'hist_mean_mai-firr_1976-2005_DE_FR.hdf5'


class TestIntegr(unittest.TestCase):
    """Test loading functions from the ISIMIP Agricultural Drought class"""
    def test_EU(self):

        haz = RelativeCropyield()
        haz.set_from_single_run(input_dir=INPUT_DIR, yearrange=(2001, 2005),
                                ag_model='lpjml', cl_model='ipsl-cm5a-lr', scenario='historical',
                                soc='2005soc', co2='co2', crop='whe', irr='noirr',
                                fn_str_var=FN_STR_DEMO)
        hist_mean = haz.calc_mean(yearrange=(2001, 2005))
        haz.set_rel_yield_to_int(hist_mean)
        haz.centroids.set_region_id()
        
        exp = CropProduction()
        exp.set_from_single_run(input_dir=INPUT_DIR, filename=FILENAME_LU, hist_mean=FILENAME_MEAN,
                                              bbox=[-5, 42, 16, 55], yearrange=(2001, 2005),
                                              scenario='flexible', unit='t', irr='firr')

        exp.set_to_usd(INPUT_DIR)
        exp.assign_centroids(haz, threshold=20)
        # country_list, ratio,  \
        # exp_firr_norm, exp_noirr_norm = normalize_with_fao_cp(exp, exp, 
        #                                                       input_dir=INPUT_DIR, 
        #                                                       return_data=False)

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
        self.assertEqual(exp.latitude.values.size, 1092)
        self.assertEqual(exp.value[3], 0.0)
        self.assertAlmostEqual(impact.imp_mat.data[3], -25802.37206534073)
        self.assertEqual(len(dif), 0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIntegr)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
