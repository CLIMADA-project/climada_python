"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test of finance module
"""
import unittest
import numpy as np
from cartopy.io import shapereader

from climada.util.finance import net_present_value, gdp, income_group, \
nat_earth_adm0, world_bank

SHP_FN = shapereader.natural_earth(resolution='10m', \
    category='cultural', name='admin_0_countries')
SHP_FILE = shapereader.Reader(SHP_FN)

class TestNetpresValue(unittest.TestCase):
    """Test date functions """
    def test_net_pres_val_pass(self):
        """ Test net_present_value against MATLAB reference"""
        years = np.arange(2018, 2041)
        disc_rates = np.ones(years.size)*0.02
        val_years = np.ones(years.size)*6.512201157564418e9
        res = net_present_value(years, disc_rates, val_years)

        self.assertEqual(1.215049630691397e+11, res)

class TestWBData(unittest.TestCase):
    def test_ne_income_grp_aia_pass(self):
        """ Test nat_earth_adm0 function Anguilla."""
        ref_year = 2012
        res_year, res_val = nat_earth_adm0('AIA', ref_year, 'INCOME_GRP',
                                           shp_file=SHP_FILE)

        ref_year = 0
        ref_val = 3
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

    def test_wb_income_grp_sxm_pass(self):
        """ Test world_bank function Sint Maarten."""
        ref_year = 2012
        res_year, res_val = world_bank('SXM', ref_year, 'INC_GRP')

        ref_year = 2012
        ref_val = 4
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

    def test_income_grp_sxm_1999_pass(self):
        """ Test income_group function Sint Maarten."""
        ref_year = 1999
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            res_year, res_val = income_group('SXM', ref_year, SHP_FILE)

        ref_year = 2010
        ref_val = 4
        self.assertIn('Income group SXM 2010: 4.', cm.output[0])
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

    def test_ne_gdp_aia_2012_pass(self):
        """ Test nat_earth_adm0 function Anguilla."""
        ref_year = 2012
        res_year, res_val = nat_earth_adm0('AIA', ref_year, 'GDP_MD_EST',
                                           'GDP_YEAR', SHP_FILE)

        ref_year = 2009
        ref_val = 1.754e+08
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

    def test_gdp_sxm_2012_pass(self):
        """ Test gdp function Sint Maarten."""
        ref_year = 2012
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            res_year, res_val = gdp('SXM', ref_year)

        ref_val = 3.658e+08
        ref_year = 2014
        self.assertIn('GDP SXM 2014: 3.658e+08', cm.output[0])
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

    def test_wb_esp_1950_pass(self):
        """ Test world_bank function Sint Maarten."""
        ref_year = 1950
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            res_year, res_val = world_bank('ESP', ref_year, 'NY.GDP.MKTP.CD')

        ref_year = 1960
        ref_val = 12072126075.397
        self.assertIn('GDP ESP 1960: 1.207e+10', cm.output[0])
        self.assertEqual(res_year, ref_year)
        self.assertEqual(res_val, ref_val)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNetpresValue)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWBData))
unittest.TextTestRunner(verbosity=2).run(TESTS)
