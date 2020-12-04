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

Tests on LitPop exposures.
"""

import unittest
import numpy as np
import pandas as pd

from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures import litpop as lp
from climada.entity.exposures import gpw_import
from climada.util.finance import world_bank_wealth_account
from climada.util.coordinates import equal_crs

# ---------------------
class TestLitPopExposure(unittest.TestCase):
    """Test LitPop exposure data model:"""

    def test_switzerland300_pass(self):
        """Create LitPop entity for Switzerland on 300 arcsec:"""
        country_name = ['CHE']
        resolution = 300
        fin_mode = 'income_group'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, fin_mode=fin_mode)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 300 arcsec', cm.output[0])
        self.assertEqual(ent.region_id.min(), 756)
        self.assertEqual(ent.region_id.max(), 756)
        self.assertEqual(np.int(ent.value.sum().round()), 3356545986884)
        self.assertIn('LitPop for Switzerland at 300 as, year=2016', ent.tag.description)
        self.assertIn('financial mode=income_group', ent.tag.description)
        self.assertIn('GPW-year=2015', ent.tag.description)
        self.assertIn('BM-year=2016', ent.tag.description)
        self.assertIn('exp=[1, 1]', ent.tag.description)
        self.assertTrue(equal_crs(ent.crs['init'], {'init': 'epsg:4326'}))
        self.assertEqual(ent.meta['width'], 54)
        self.assertEqual(ent.meta['height'], 23)
        self.assertTrue(equal_crs(ent.meta['crs'], {'init': 'epsg:4326'}))
        self.assertAlmostEqual(ent.meta['transform'][0], 0.08333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][1], 0)
        self.assertAlmostEqual(ent.meta['transform'][2], 5.9166666666666)
        self.assertAlmostEqual(ent.meta['transform'][3], 0)
        self.assertAlmostEqual(ent.meta['transform'][4], -0.08333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][5], 47.7499999999999)

    def test_switzerland30normPop_pass(self):
        """Create LitPop entity for Switzerland on 30 arcsec:"""
        country_name = ['CHE']
        resolution = 30
        exp = [0, 1]
        fin_mode = 'norm'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, exponent=exp,
                            fin_mode=fin_mode, reference_year=2015)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 30 arcsec', cm.output[0])
        self.assertEqual(ent.region_id.min(), 756)
        self.assertEqual(ent.region_id.max(), 756)
        self.assertEqual(np.int((1000 * ent.value.sum()).round()), 1000)

    def test_suriname30_nfw_pass(self):
        """Create LitPop entity for Suriname for non-finanical wealth:"""
        country_name = ['SUR']
        fin_mode = 'nfw'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, reference_year=2016, fin_mode=fin_mode)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 30.0 arcsec', cm.output[0])
        self.assertEqual(ent.region_id.min(), 740)
        self.assertEqual(ent.region_id.max(), 740)
        self.assertEqual(np.int(ent.value.sum().round()), 2304662017)

    def test_switzerland300_pc2016_pass(self):
        """Create LitPop entity for Switzerland 2016 with admin1 and produced capital:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2016
        adm1 = True
        cons = True
        comparison_total_val = world_bank_wealth_account(country_name[0], ref_year, no_land=1)[1]
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution,
                            reference_year=ref_year, fin_mode=fin_mode,
                            conserve_cntrytotal=cons, calc_admin1=adm1)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 300 arcsec', cm.output[0])
        self.assertEqual(np.around(ent.value.sum(), 0), np.around(comparison_total_val, 0))
        self.assertEqual(np.int(ent.value.sum().round()), 2225854927260)

    def test_switzerland300_pc2013_pass(self):
        """Create LitPop entity for Switzerland 2013 for produced capital:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2013
        comparison_total_val = world_bank_wealth_account(country_name[0],
                                                         ref_year, no_land=1)[1]
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution,
                            reference_year=ref_year, fin_mode=fin_mode)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 300 arcsec', cm.output[0])
        self.assertEqual(ent.value.sum(), comparison_total_val)
        self.assertEqual(np.int(ent.value.sum().round()), 2296358085749)

class TestFunctionIntegration(unittest.TestCase):
    """Test the integration of major functions within the LitPop module"""

    def test_get_litpop_box(self):
        """test functions _litpop_box2coords and _get_litpop_box for Taiwan"""
        curr_country = 'TWN'
        resolution = 3000
        cut_bbox = lp._get_country_shape(curr_country, 1)[0]
        all_coords = lp._litpop_box2coords(cut_bbox, resolution, 1)
        self.assertEqual(len(all_coords), 25)
        self.assertTrue(22.08333333333333 in min(all_coords)) 
        self.assertTrue(117.91666666666666 in min(all_coords))
        litpop_data = lp._get_litpop_box(cut_bbox, resolution, 0, 2016, [1, 1])
        self.assertEqual(len(litpop_data), 25)
        self.assertIn(max(litpop_data), [544316890, 594091108.0, 594091108])

    def test_calc_admin1(self):
        """test function _calc_admin1 for Switzerland.
        All required functions are tested in unit tests"""
        resolution = 300
        curr_country = 'CHE'
        country_info = dict()
        admin1_info = dict()
        country_info[curr_country], admin1_info[curr_country] = \
            lp._get_country_info(curr_country)
        curr_shp = lp._get_country_shape(curr_country, 0)
        for cntry_iso, cntry_val in country_info.items():
            _, total_asset_val = lp.gdp(cntry_iso, 2016, curr_shp)
            cntry_val.append(total_asset_val)
        lp._get_gdp2asset_factor(country_info, 2016, curr_shp, fin_mode='gdp')
        cut_bbox = lp._get_country_shape(curr_country, 1)[0]
        all_coords = lp._litpop_box2coords(cut_bbox, resolution, 1)
        mask = lp._mask_from_shape(curr_shp, resolution=resolution,
                                   points2check=all_coords)
        litpop_data = lp._get_litpop_box(cut_bbox, resolution, 0, 2016, [3, 0])
        litpop_curr = litpop_data[mask.sp_index.indices]
        lon, lat = zip(*np.array(all_coords)[mask.sp_index.indices])
        litpop_curr = lp._calc_admin1(curr_country, country_info[curr_country],
                                      admin1_info[curr_country], litpop_curr,
                                      list(zip(lon, lat)), resolution, 0, conserve_cntrytotal=0,
                                      check_plot=0, masks_adm1=[], return_data=1)
        self.assertEqual(len(litpop_curr), 699)
        self.assertAlmostEqual(max(litpop_curr), 80313679854.39496, places=2)

    def test_gpw_import(self):
        """test import of population data (Gridded Population of the World GWP)
        via function gpw_import.get_box_gpw() for Swaziland"""
        bbox = [30.78291, -27.3164, 32.11741, -25.73600]
        gpw, lon, lat = gpw_import.get_box_gpw(cut_bbox=bbox, resolution=300,
                                               return_coords=1, reference_year=2015)
        self.assertEqual(len(gpw), 323)
        self.assertIn(np.around(max(gpw)), [103070.0, 137840.0])
        self.assertEqual(type(gpw),
                         type(pd.arrays.SparseArray(data=1, fill_value=0)))
        self.assertAlmostEqual(lat[0], -27.3164)
        self.assertAlmostEqual(lat[1], 0.083333333)
        self.assertAlmostEqual(lon[0], 30.78291)
        self.assertAlmostEqual(lon[1], 0.083333333)

class TestValidation(unittest.TestCase):
    """Test LitPop exposure data model:"""

    def test_validation_switzerland30(self):
        """Validation for Switzerland: two combinations of Lit and Pop,
            checking Pearson correlation coefficient and RMSF"""
        rho = lp.admin1_validation('CHE', ['LitPop', 'Lit5'], [[1, 1], [5, 0]],
                                   res_arcsec=30, check_plot=False)[0]
        self.assertEqual(np.int(round(rho[0] * 1e12)), 945416798729)
        self.assertEqual(np.int(round(rho[-1] * 1e12)), 3246081648798)

class TestSetAdmin1(unittest.TestCase):
    """Test adding name and ID of Admin1-region to exposure"""

    def test_exposure_set_admin1_BGD(self):
        """Test function exposure_set_admin1 of litpop for Bangladesh"""
        country_name = ['BGD']
        resolution = 600
        ent = LitPop()
        ent.set_country(country_name, res_arcsec=resolution)
        ent = lp.exposure_set_admin1(ent, 600)
        self.assertIn(5492, ent.admin1_ID.values)
        self.assertIn('Chittagong', ent.admin1.values)
        self.assertEqual(len(np.unique(ent.admin1_ID)), 7)
        self.assertEqual(np.min(np.unique(ent.admin1_ID)), 1806.0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestValidation)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunctionIntegration))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLitPopExposure))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSetAdmin1))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
