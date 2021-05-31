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

Tests on LitPop exposures.
"""

import unittest
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from climada.entity.exposures.litpop import litpop as lp
from climada.entity.exposures import gpw_import
from climada.util.finance import world_bank_wealth_account, gdp
import climada.util.coordinates as u_coord


class TestLitPopExposure(unittest.TestCase):
    """Test LitPop exposure data model:"""

    def test_switzerland300_pass(self):
        """Create LitPop entity for Switzerland on 300 arcsec:"""
        country_name = ['CHE']
        resolution = 300
        fin_mode = 'income_group'
        ent = lp.LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, fin_mode=fin_mode,
                            reference_year=2016)
        # print(cm)
        self.assertIn('LitPop: Init Exposure for country: CHE, 756', cm.output[0])
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        self.assertAlmostEqual(ent.gdf.value.sum()/3356545987390.9, 1.0)
        self.assertIn("LitPop Exposure for ['CHE'] at 300 as, year: 2016", ent.tag.description)
        self.assertIn('financial mode=income_group', ent.tag.description)
        self.assertIn('exp=[1, 1]', ent.tag.description)
        self.assertTrue(u_coord.equal_crs(ent.crs, {'init': 'epsg:4326'}))
        self.assertEqual(ent.meta['width'], 54)
        self.assertEqual(ent.meta['height'], 23)
        self.assertTrue(u_coord.equal_crs(ent.meta['crs'], {'init': 'epsg:4326'}))
        self.assertAlmostEqual(ent.meta['transform'][0], 0.08333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][1], 0)
        self.assertAlmostEqual(ent.meta['transform'][2], 5.9166666666666)
        self.assertAlmostEqual(ent.meta['transform'][3], 0)
        self.assertAlmostEqual(ent.meta['transform'][4], -0.08333333333333333)
        self.assertAlmostEqual(ent.meta['transform'][5], 47.75)

    def test_switzerland30normPop_pass(self):
        """Create LitPop entity for Switzerland on 30 arcsec:"""
        country_name = ['CHE']
        resolution = 30
        exp = [0, 1]
        fin_mode = 'norm'
        ent = lp.LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, exponents=exp,
                            fin_mode=fin_mode, reference_year=2015)
        # print(cm)
        self.assertIn('LitPop: Init Exposure for country: CHE, 756', cm.output[0])
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        self.assertEqual(ent.gdf.value.sum(), 1.0)

    def test_suriname30_nfw_pass(self):
        """Create LitPop entity for Suriname for non-finanical wealth:"""
        country_name = ['SUR']
        fin_mode = 'nfw'
        ent = lp.LitPop()
        ent.set_country(country_name, reference_year=2016, fin_mode=fin_mode)

        self.assertEqual(ent.gdf.region_id.min(), 740)
        self.assertEqual(ent.gdf.region_id.max(), 740)
        self.assertEqual(np.int(ent.gdf.value.sum().round()), 2304662017)

    def test_switzerland300_pc2016_pass(self):
        """Create LitPop entity for Switzerland 2016 with admin1 and produced capital:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2016
        adm1 = True
        comparison_total_val = world_bank_wealth_account(country_name[0], ref_year, no_land=1)[1]
        ent = lp.LitPop()
        ent.set_country(country_name, res_arcsec=resolution,
                        reference_year=ref_year, fin_mode=fin_mode,
                        admin1_calc=adm1)

        self.assertEqual(np.around(ent.gdf.value.sum(), 0), np.around(comparison_total_val, 0))
        self.assertEqual(ent.value_unit, 'USD')


    def test_switzerland300_resample_first_false_pass(self):
        """Create LitPop entity for Switzerland 2013 for produced capital
        and resampling after combining Lit and Pop:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2016
        comparison_total_val = world_bank_wealth_account(country_name[0],
                                                         ref_year, no_land=1)[1]
        ent = lp.LitPop()
        ent.set_countries(country_name, res_arcsec=resolution,
                        reference_year=ref_year, fin_mode=fin_mode,
                        resample_first=False)

        self.assertEqual(ent.gdf.value.sum(), comparison_total_val)
        self.assertEqual(ent.value_unit, 'USD')

    def test_set_custom_shape_zurich_pass(self):
        """test initiating LitPop for custom shape (square around Zurich City)
        Distributing an imaginary total value of 1000 USD"""
        bounds = (8.41, 47.2, 8.70, 47.45) # (min_lon, max_lon, min_lat, max_lat)
        # bounds = (-85, -11, 5, 40)
        shape = Polygon([
            (bounds[0], bounds[3]),
            (bounds[2], bounds[3]),
            (bounds[2], bounds[1]),
            (bounds[0], bounds[1])
            ])
        ent = lp.LitPop()
        ent.set_custom_shape(shape, res_arcsec=30, total_value_abs=1000)
        self.assertEqual(ent.gdf.value.sum(), 1000.0)
        self.assertEqual(ent.gdf.value.min(), 0.0)
        # index of largest value:
        self.assertEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].index[0], 482)
        self.assertAlmostEqual(ent.gdf.latitude.min(), 47.20416666666661)

    def test_Liechtenstein_15_lit_pass(self):
        """Create Nightlights entity for Liechtenstein 2016:"""
        country_name = 'Liechtenstein'
        ref_year = 2016
        ent = lp.LitPop()
        ent.set_lit(country_name, reference_year=ref_year)

        self.assertEqual(ent.gdf.value.sum(), 36469.0)
        self.assertEqual(ent.gdf.region_id[1], 438)
        self.assertEqual(ent.value_unit, '')
        self.assertAlmostEqual(ent.gdf.latitude.max(), 47.260416666666664)
        self.assertAlmostEqual(ent.meta['transform'][4], -15/3600)

    def test_Liechtenstein_30_pop_pass(self):
        """Create population count entity for Liechtenstein 2015:"""
        country_name = 'Liechtenstein'
        ref_year = 2015
        ent = lp.LitPop()
        ent.set_pop(country_name, reference_year=ref_year)

        self.assertEqual(ent.gdf.value.sum(), 30068.970703125)
        self.assertEqual(ent.gdf.region_id[1], 438)
        self.assertEqual(ent.value_unit, 'people')
        self.assertAlmostEqual(ent.gdf.latitude.max(), 47.2541666666666)
        self.assertAlmostEqual(ent.meta['transform'][0], 30/3600)

class TestFunctionIntegration(unittest.TestCase):
    """Test the integration of major functions within the LitPop module"""

    def test_set_countries_calc_admin1_pass(self):
        """test method set_countries with admin1_calc=True for Switzerland"""
        country_name = "Switzerland"
        resolution = 90
        fin_mode = 'gdp'

        ent = lp.LitPop()
        ent.set_countries(country_name, res_arcsec=resolution, fin_mode=fin_mode,
                        reference_year=2016, admin1_calc=True)

        self.assertAlmostEqual(ent.gdf.value.sum(), gdp('CHE', 2016)[1])
        self.assertEqual(ent.gdf.shape[0], 7949)



    def test_calc_admin1(self):
        """test function _calc_admin1 for Switzerland."""
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
        self.assertAlmostEqual(max(litpop_curr)/80313679854.39496, 1.0)



# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLitPopExposure)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunctionIntegration))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
