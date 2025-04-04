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

Test of finance module
"""

import unittest

import numpy as np
from cartopy.io import shapereader

from climada.util.finance import (
    _gdp_twn,
    gdp,
    income_group,
    nat_earth_adm0,
    net_present_value,
    wealth2gdp,
    world_bank,
    world_bank_wealth_account,
)

SHP_FN = shapereader.natural_earth(
    resolution="10m", category="cultural", name="admin_0_countries"
)
SHP_FILE = shapereader.Reader(SHP_FN)


class TestNetpresValue(unittest.TestCase):
    """Test date functions"""

    def test_net_pres_val_pass(self):
        """Test net_present_value against MATLAB reference"""
        years = np.arange(2018, 2041)
        disc_rates = np.ones(years.size) * 0.02
        val_years = np.ones(years.size) * 6.512201157564418e9
        res = net_present_value(years, disc_rates, val_years)

        self.assertEqual(1.215049630691397e11, res)


class TestWBData(unittest.TestCase):
    """Test World Bank data"""

    def test_ne_income_grp_aia_pass(self):
        """Test nat_earth_adm0 function Anguilla."""
        ref_year = 2012
        ne_year, ne_val = nat_earth_adm0("AIA", "INCOME_GRP", shp_file=SHP_FILE)

        ref_year = 0
        ref_val = 3
        self.assertEqual(ne_year, ref_year)
        self.assertEqual(ne_val, ref_val)

    def test_wb_income_grp_sxm_pass(self):
        """Test world_bank function Sint Maarten."""
        ref_year = 2012
        wb_year, wb_val = world_bank("SXM", ref_year, "INC_GRP")

        ref_year = 2012
        ref_val = 4
        self.assertEqual(wb_year, ref_year)
        self.assertEqual(wb_val, ref_val)

    def test_income_grp_sxm_1999_pass(self):
        """Test income_group function Sint Maarten."""
        ref_year = 1999
        with self.assertLogs("climada.util.finance", level="INFO") as cm:
            ig_year, ig_val = income_group("SXM", ref_year, SHP_FILE)

        ref_year = 2010
        ref_val = 4
        self.assertIn("Income group SXM 2010: 4.", cm.output[0])
        self.assertEqual(ig_year, ref_year)
        self.assertEqual(ig_val, ref_val)

    def test_ne_gdp_aia_2012_pass(self):
        """Test nat_earth_adm0 function Anguilla."""
        ref_year = 2012
        ne_year, ne_val = nat_earth_adm0("AIA", "GDP_MD", "GDP_YEAR", SHP_FILE)

        ref_year = 2009
        ref_val = 1.75e08
        self.assertEqual(ne_year, ref_year)
        self.assertEqual(ne_val, ref_val)

    def test_gdp_sxm_2010_pass(self):
        """Test gdp function Sint Maarten."""
        # If World Bank input data changes, make sure to set ref_year to a year where
        # no data is available so that the next available data point has to be selected.
        ref_year = 2010
        with self.assertLogs("climada.util.finance", level="INFO") as cm:
            gdp_year, gdp_val = gdp("SXM", ref_year)

        ref_val = 892290502.793296  # reference GDP value
        ref_year = 2010  # nearest year with data available (might change)
        # GDP and years with data available might change if worldbank input
        # data changes, check magnitude and adjust ref_val and/or ref_year
        # if test fails:
        self.assertIn("GDP SXM %i: %1.3e" % (ref_year, ref_val), cm.output[0])
        self.assertEqual(gdp_year, ref_year)
        self.assertAlmostEqual(gdp_val, ref_val, places=0)

    def test_gdp_twn_2012_pass(self):
        """Test gdp function TWN."""
        ref_year = 2014
        gdp_year, gdp_val = gdp("TWN", ref_year)
        _, gdp_val_direct = _gdp_twn(ref_year)
        ref_val = 530515000000.0
        ref_year = 2014
        self.assertEqual(gdp_year, ref_year)
        self.assertEqual(gdp_val, ref_val)
        self.assertEqual(gdp_val_direct, ref_val)

    def test_wb_esp_1950_pass(self):
        """Test world_bank function Sint Maarten."""
        ref_year = 1950
        wb_year, wb_val = world_bank("ESP", ref_year, "NY.GDP.MKTP.CD")

        ref_year = 1960
        ref_val = 12424514013.7604
        self.assertEqual(wb_year, ref_year)
        self.assertAlmostEqual(wb_val, ref_val)


class TestWealth2GDP(unittest.TestCase):
    """Test Wealth to GDP factor extraction"""

    def test_nfw_SUR_pass(self):
        """Test non-financial wealth-to-gdp factor with Suriname."""
        w2g_year, w2g_val = wealth2gdp("SUR")

        ref_year = 2016
        ref_val = 0.73656
        self.assertEqual(w2g_year, ref_year)
        self.assertEqual(w2g_val, ref_val)

    def test_nfw_BEL_pass(self):
        """Test total wealth-to-gdp factor with Belgium."""
        w2g_year, w2g_val = wealth2gdp("BEL", False)

        ref_year = 2016
        ref_val = 4.88758
        self.assertEqual(w2g_year, ref_year)
        self.assertEqual(w2g_val, ref_val)

    def test_nfw_LBY_pass(self):
        """Test missing factor with Libya."""
        _, w2g_val = wealth2gdp("LBY")

        self.assertTrue(np.isnan(w2g_val))


class TestWBWealthAccount(unittest.TestCase):
    """Test Wealth Indicator extraction from World Bank provided CSV"""

    def test_pca_DEU_2010_pass(self):
        """Test Processed Capital value Germany 2010."""
        ref_year = 2010
        cntry_iso = "DEU"
        wb_year, wb_val, q = world_bank_wealth_account(cntry_iso, ref_year, no_land=0)
        wb_year_noland, wb_val_noland, q = world_bank_wealth_account(
            cntry_iso, ref_year, no_land=1
        )
        ref_val = [
            17675048450284.9,
            19767982562092.2,
        ]  # second value as updated by worldbank on
        # October 27 2021
        ref_val_noland = [14254071330874.9, 15941921421042.1]  # dito
        self.assertEqual(wb_year, ref_year)
        self.assertEqual(q, 1)
        self.assertIn(wb_val, ref_val)
        self.assertEqual(wb_year_noland, ref_year)
        self.assertIn(wb_val_noland, ref_val_noland)

    def test_pca_CHE_2008_pass(self):
        """Test Prcoessed Capital per capita Switzerland 2008 (interp.)."""
        ref_year = 2008
        cntry_iso = "CHE"
        var_name = "NW.PCA.PC"
        wb_year, wb_val, _ = world_bank_wealth_account(
            cntry_iso, ref_year, variable_name=var_name, no_land=0
        )
        ref_val = [
            328398.7,  # values sporadically updated by worldbank
            369081.0,
        ]  # <- October 27 2021
        self.assertEqual(wb_year, ref_year)
        self.assertIn(wb_val, ref_val)

    def test_tow_IND_1985_pass(self):
        """Test Total Wealth value India 1985 (outside year range)."""
        ref_year = 1985
        cntry_iso = "IND"
        var_name = "NW.TOW.TO"
        wb_year, wb_val, _ = world_bank_wealth_account(
            cntry_iso, ref_year, variable_name=var_name
        )
        ref_val = [
            5415188681934.5,  # values sporadically updated by worldbank
            5861193808779.6,  # <- October 27 2021
            5861186556152.8,  # <- June 29 2023
            5861186367245.2,  # <- December 20 2023
        ]
        self.assertEqual(wb_year, ref_year)
        self.assertIn(wb_val, ref_val)

    def test_pca_CUB_2015_pass(self):
        """Test Processed Capital value Cuba 2015 (missing value)."""
        ref_year = 2015
        cntry_iso = "CUB"
        wb_year, wb_val, q = world_bank_wealth_account(cntry_iso, ref_year, no_land=1)
        ref_val = [
            108675762920.0,  # values sporadically updated by worldbank
            108675513472.0,  # <- Dezember 20 2023
        ]
        self.assertEqual(q, 0)
        self.assertEqual(wb_year, ref_year)
        self.assertIn(wb_val, ref_val)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNetpresValue)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWBData))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWealth2GDP))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWBWealthAccount))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
