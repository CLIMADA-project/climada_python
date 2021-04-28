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

Test tc_clim_change module
"""

import unittest

import climada.hazard.tc_clim_change as tc_cc

class TestKnutson(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_get_pass(self):
        """Test get_knutson_criterion function."""
        criterion = tc_cc.get_knutson_criterion()
        self.assertTrue(len(criterion), 20)
        for crit_val in criterion:
            self.assertTrue('year' in crit_val)
            self.assertTrue('change' in crit_val)
            self.assertTrue('variable' in crit_val)
        self.assertEqual(criterion[0]['variable'], "frequency")
        self.assertEqual(criterion[0]['change'], 1)
        self.assertEqual(criterion[4]['variable'], "intensity")
        self.assertEqual(criterion[4]['change'], 1.045)
        self.assertEqual(criterion[-10]['basin'], "SP")
        self.assertEqual(criterion[-10]['variable'], "frequency")
        self.assertEqual(criterion[-10]['change'], 1 - 0.583)

    def test_scale_pass(self):
        """Test calc_scale_knutson function."""
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2050, rcp_scenario=45),
                               0.759630751756698)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2070, rcp_scenario=45),
                               0.958978483788876)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2060, rcp_scenario=60),
                               0.825572149523299)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2080, rcp_scenario=60),
                               1.309882943406079)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2090, rcp_scenario=85),
                               2.635069196605717)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2100, rcp_scenario=85),
                               2.940055236533517)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2066, rcp_scenario=26),
                               0.341930203294547)
        self.assertAlmostEqual(tc_cc.calc_scale_knutson(ref_year=2078, rcp_scenario=26),
                               0.312383928930456)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestKnutson)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
