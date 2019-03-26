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

from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures import litpop as lp
from climada.util.finance import world_bank_wealth_account

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
        self.assertTrue(ent.region_id.min() == 756)
        self.assertTrue(ent.region_id.max() == 756)
        self.assertTrue(np.int(ent.value.sum().round()) == 3343726398023)

    def test_switzerland30_pass(self):
        """Create LitPop entity for Switzerland on 30 arcsec:"""
        country_name = ['CHE']
        resolution = 30
        fin_mode = 'income_group'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, \
                            fin_mode=fin_mode, reference_year=2016)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 30 arcsec', cm.output[0])
        self.assertTrue(ent.region_id.min() == 756)
        self.assertTrue(ent.region_id.max() == 756)
        self.assertTrue(np.int(ent.value.sum().round()) == 3343726398023)

    def test_switzerland30_normPopulation_pass(self):
        """Create LitPop entity for Switzerland on 30 arcsec:"""
        country_name = ['CHE']
        resolution = 30
        exp = [0, 1]
        fin_mode = 'norm'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, exponent=exp,\
                            fin_mode=fin_mode, reference_year=2015)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 30 arcsec', cm.output[0])
        self.assertTrue(ent.region_id.min() == 756)
        self.assertTrue(ent.region_id.max() == 756)
        self.assertTrue(np.int((1000*ent.value.sum()).round()) == 1000)

    def test_suriname30_nfw_pass(self):
        """Create LitPop entity for Suriname for non-finanical wealth:"""
        country_name = ['SUR']
        fin_mode = 'nfw'
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, reference_year=2016, fin_mode=fin_mode)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 30.0 arcsec', cm.output[0])
        self.assertTrue(ent.region_id.min() == 740)
        self.assertTrue(ent.region_id.max() == 740)
        self.assertTrue(np.int(ent.value.sum().round()) == 2321765217)

    def test_switzerland300_pc2016_pass(self):
        """Create LitPop entity for Switzerland 2016 with admin1 and produced capital:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2016
        adm1 = True
        cons = True
        comparison_total_val = world_bank_wealth_account(country_name[0], ref_year, \
                                                                no_land=1)[1]
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, \
                            reference_year=ref_year, fin_mode=fin_mode, \
                            conserve_cntrytotal=cons, calc_admin1=adm1)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 300 arcsec', cm.output[0])
        self.assertTrue(np.around(ent.value.sum(), 0) == np.around(comparison_total_val, 0))
        self.assertTrue(np.int(ent.value.sum().round()) == 2217353764118)

    def test_switzerland300_pc2013_pass(self):
        """Create LitPop entity for Switzerland 2013 for produced capital:"""
        country_name = ['CHE']
        fin_mode = 'pc'
        resolution = 300
        ref_year = 2013
        comparison_total_val = world_bank_wealth_account(country_name[0], \
                                                         ref_year, no_land=1)[1]
        ent = LitPop()
        with self.assertLogs('climada.entity.exposures.litpop', level='INFO') as cm:
            ent.set_country(country_name, res_arcsec=resolution, \
                            reference_year=ref_year, fin_mode=fin_mode)
        # print(cm)
        self.assertIn('Generating LitPop data at a resolution of 300 arcsec', cm.output[0])
        self.assertTrue(ent.value.sum() == comparison_total_val)
        self.assertTrue(np.int(ent.value.sum().round()) == 2296358085749)

class TestValidation(unittest.TestCase):
    """Test LitPop exposure data model:"""

    def test_validation_switzerland30(self):
        """Validation for Switzerland: two combinations of Lit and Pop,
            checking Pearson correlation coefficient and RMSF"""
        rho = lp.admin1_validation('CHE', ['LitPop', 'Lit5'], [[1, 1], [5, 0]],\
                                    res_arcsec=30, check_plot=False)[0]
        self.assertTrue(np.int(round(rho[0]*1e12)) == 945416798729)
        self.assertTrue(np.int(round(rho[-1]*1e12)) == 3246081648798)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestValidation)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLitPopExposure))
unittest.TextTestRunner(verbosity=2).run(TESTS)
