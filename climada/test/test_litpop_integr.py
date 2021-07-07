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
from shapely.geometry import Polygon

from climada.entity.exposures.litpop import litpop as lp
from climada.entity.exposures.litpop import gpw_population
from climada.util.finance import world_bank_wealth_account, gdp, income_group
import climada.util.coordinates as u_coord
from climada.util.constants import SYSTEM_DIR
from climada import CONFIG


class TestLitPopExposure(unittest.TestCase):
    """Test LitPop exposure data model:"""

    def test_netherlands150_pass(self):
        """Test set_countries for Netherlands at 150 arcsec, first shape is empty"""
        ent = lp.LitPop()
        ent.set_countries('Netherlands', res_arcsec=150, reference_year=2016)
        self.assertEqual(ent.gdf.shape[0], 2829)

    def test_BLM150_pass(self):
        """Test set_countries for BLM at 150 arcsec, 2 data points"""
        ent = lp.LitPop()
        ent.set_countries('BLM', res_arcsec=150, reference_year=2016)
        self.assertEqual(ent.gdf.shape[0], 2)

    def test_Monaco150_pass(self):
        """Test set_countries for Moncao at 150 arcsec, 1 data point"""
        ent = lp.LitPop()
        ent.set_countries('Monaco', res_arcsec=150, reference_year=2016)
        self.assertEqual(ent.gdf.shape[0], 1)

    def test_switzerland300_pass(self):
        """Create LitPop entity for Switzerland on 300 arcsec:"""
        country_name = ['CHE']
        resolution = 300
        fin_mode = 'income_group'
        ent = lp.LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, fin_mode=fin_mode,
                            reference_year=2016)

        self.assertIn('LitPop: Init Exposure for country: CHE', cm.output[0])
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        # confirm that the total value is equal to GDP * (income_group+1):
        self.assertAlmostEqual(ent.gdf.value.sum()/gdp('CHE', 2016)[1],
                               (income_group('CHE', 2016)[1] + 1))
        self.assertIn("LitPop Exposure for ['CHE'] at 300 as, year: 2016", ent.tag.description)
        self.assertIn('income_group', ent.tag.description)
        self.assertIn('[1, 1]', ent.tag.description)
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
        self.assertIn('LitPop: Init Exposure for country: CHE', cm.output[0])
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        self.assertEqual(ent.gdf.value.sum(), 1.0)
        self.assertEqual(ent.ref_year, 2015)

    def test_suriname30_nfw_pass(self):
        """Create LitPop entity for Suriname for non-finanical wealth in 2016:"""
        country_name = ['SUR']
        fin_mode = 'nfw'
        ent = lp.LitPop()
        ent.set_country(country_name, reference_year=2016, fin_mode=fin_mode)

        self.assertEqual(ent.gdf.region_id.min(), 740)
        self.assertEqual(ent.gdf.region_id.max(), 740)
        self.assertEqual(ent.ref_year, 2016)

    def test_switzerland300_admin1_pc2016_pass(self):
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

        self.assertAlmostEqual(np.around(ent.gdf.value.sum()*1e-9, 0),
                         np.around(comparison_total_val*1e-9, 0), places=0)
        self.assertEqual(ent.value_unit, 'USD')


    def test_switzerland300_reproject_first_false_pass(self):
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
                        reproject_first=False)

        self.assertEqual(ent.gdf.value.sum(), comparison_total_val)
        self.assertEqual(ent.value_unit, 'USD')

    def test_set_custom_shape_zurich_pass(self):
        """test initiating LitPop for custom shape (square around Zurich City)
        Distributing an imaginary total value of 1000 USD"""
        bounds = (8.41, 47.2, 8.70, 47.45) # (min_lon, max_lon, min_lat, max_lat)
        total_value=1000
        shape = Polygon([
            (bounds[0], bounds[3]),
            (bounds[2], bounds[3]),
            (bounds[2], bounds[1]),
            (bounds[0], bounds[1])
            ])
        ent = lp.LitPop()
        ent.set_custom_shape(shape, total_value, res_arcsec=30,
                             reference_year=2016)
        self.assertEqual(ent.gdf.value.sum(), 1000.0)
        self.assertEqual(ent.gdf.value.min(), 0.0)
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        self.assertAlmostEqual(ent.gdf.latitude.min(), 47.20416666666661)
        # index and coord. of largest value:
        self.assertEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].index[0], 482)
        self.assertAlmostEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].latitude.values[0], 47.34583333333325)
        self.assertAlmostEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].longitude.values[0], 8.529166666666658)

    def test_set_custom_shape_from_countries_zurich_pass(self):
        """test initiating LitPop for custom shape (square around Zurich City)
        with set_custom_shape_from_countries()"""
        bounds = (8.41, 47.2, 8.70, 47.45) # (min_lon, max_lon, min_lat, max_lat)
        shape = Polygon([
            (bounds[0], bounds[3]),
            (bounds[2], bounds[3]),
            (bounds[2], bounds[1]),
            (bounds[0], bounds[1])
            ])
        ent = lp.LitPop()
        ent.set_custom_shape_from_countries(shape, 'Switzerland', res_arcsec=30,
                                            reference_year=2016)
        self.assertEqual(ent.gdf.value.min(), 0.0)
        self.assertEqual(ent.gdf.region_id.min(), 756)
        self.assertEqual(ent.gdf.region_id.max(), 756)
        self.assertAlmostEqual(ent.gdf.latitude.min(), 47.20416666666661)
        # coord of largest value:
        self.assertEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].index[0], 434)
        self.assertAlmostEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].latitude.values[0], 47.34583333333325)
        self.assertAlmostEqual(ent.gdf.loc[ent.gdf.value == ent.gdf.value.max()].longitude.values[0], 8.529166666666658)

    def test_Liechtenstein_15_lit_pass(self):
        """Create Nightlights entity for Liechtenstein 2016:"""
        country_name = 'Liechtenstein'
        ref_year = 2016
        ent = lp.LitPop()
        ent.set_nightlights(country_name, reference_year=ref_year)

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
        ent.set_population(country_name, reference_year=ref_year)

        self.assertEqual(ent.gdf.value.sum(), 30068.970703125)
        self.assertEqual(ent.gdf.region_id[1], 438)
        self.assertEqual(ent.value_unit, 'people')
        self.assertAlmostEqual(ent.gdf.latitude.max(), 47.2541666666666)
        self.assertAlmostEqual(ent.meta['transform'][0], 30/3600)

class TestAdmin1(unittest.TestCase):
    """Test the admin1 functionalities within the LitPop module"""

    def test_set_countries_calc_admin1_pass(self):
        """test method set_countries with admin1_calc=True for Switzerland"""
        country_name = "Switzerland"
        resolution = 90
        fin_mode = 'gdp'

        ent = lp.LitPop()
        ent.set_countries(country_name, res_arcsec=resolution, fin_mode=fin_mode,
                        reference_year=2016, admin1_calc=True)

        self.assertEqual(ent.gdf.shape[0], 7964)

    def test_calc_admin1(self):
        """test function _calc_admin1_one_country for Switzerland."""
        resolution = 300
        country = 'CHE'
        ent = lp._calc_admin1_one_country(country, resolution, (2,1), 'pc', None,
                 2016, 11, SYSTEM_DIR, False)
        self.assertEqual(ent.gdf.shape[0], 717)
        self.assertEqual(ent.gdf.region_id[88], 756)
        self.assertAlmostEqual(ent.gdf.latitude.max(), 47.708333333333336)

class TestGPWPopulation(unittest.TestCase):
    """Test gpw_population submodule"""

    def test_get_gpw_file_path_pass(self):
        """test method gpw_population.get_gpw_file_path"""
        gpw_version = CONFIG.exposures.litpop.gpw_population.gpw_version.int()
        try:
            path = gpw_population.get_gpw_file_path(gpw_version, 2020, verbatim=False)
            self.assertIn('gpw_v4_population', str(path))
        except FileExistsError as err:
            self.assertIn('lease download', err.args[0])
            self.skipTest('GPW input data for GPW v4.%i not found.' %(gpw_version))

    def test_load_gpw_pop_shape_pass(self):
        """test method gpw_population.load_gpw_pop_shape"""
        gpw_version = CONFIG.exposures.litpop.gpw_population.gpw_version.int()
        bounds = (8.41, 47.2, 8.70, 47.45) # (min_lon, max_lon, min_lat, max_lat)
        shape = Polygon([
            (bounds[0], bounds[3]),
            (bounds[2], bounds[3]),
            (bounds[2], bounds[1]),
            (bounds[0], bounds[1])
            ])
        try:
            data, meta, glb_transform = \
                gpw_population.load_gpw_pop_shape(shape, 2020, gpw_version,
                                                  verbatim=False)
            self.assertEqual(data.shape, (31, 36))
            self.assertAlmostEqual(meta['transform'][0], 0.00833333333333333)
            self.assertAlmostEqual(meta['transform'][0], glb_transform[0])

        except FileExistsError as err:
            self.assertIn('lease download', err.args[0])
            self.skipTest('GPW input data for GPW v4.%i not found.' %(gpw_version))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLitPopExposure)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAdmin1))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
