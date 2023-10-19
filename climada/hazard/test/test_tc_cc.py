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

    def test_get_knutson_scaling_pass(self):
        """Test get_knutson_criterion function."""
        criterion = tc_cc.get_knutson_scaling_factor()
        self.assertEqual(criterion.shape, (21, 4))

        self.assertEqual(criterion.columns[0], '2.6')
        self.assertEqual(criterion.columns[1], '4.5')
        self.assertEqual(criterion.columns[2], '6.0')
        self.assertEqual(criterion.columns[3], '8.5')

        self.assertAlmostEqual(criterion.loc[2030, '2.6'], -16.13547, 4)
        self.assertAlmostEqual(criterion.loc[2050, '4.5'], -25.19448, 4)
        self.assertAlmostEqual(criterion.loc[2070, '6.0'], -31.06633, 4)
        self.assertAlmostEqual(criterion.loc[2100, '8.5'], -58.98637, 4)

    def test_get_gmst_pass(self):
        """Test get_gmst_info function."""
        gmst_data, gmst_start_year, gmst_end_year, rcps = tc_cc.get_gmst_info()

        self.assertAlmostEqual(gmst_data.shape,
                               (len(rcps),
                                 gmst_end_year-gmst_start_year+1))
        self.assertAlmostEqual(gmst_data[0,0], -0.16)
        self.assertAlmostEqual(gmst_data[0,-1], 1.27641, 4)
        self.assertAlmostEqual(gmst_data[-1,0], -0.16)
        self.assertAlmostEqual(gmst_data[-1,-1], 4.477764, 4)

    def test_get_knutson_data_pass(self):
        """Test get_knutson_data function."""

        data_knutson = tc_cc.get_knutson_data()

        self.assertAlmostEqual(data_knutson.shape, (4,6,5))
        self.assertAlmostEqual(data_knutson[0,0,0], -34.49)
        self.assertAlmostEqual(data_knutson[-1,-1,-1], 15.419)
        self.assertAlmostEqual(data_knutson[0,-1,-1], 4.689)
        self.assertAlmostEqual(data_knutson[-1,0,0], 5.848)
        self.assertAlmostEqual(data_knutson[-1,0,-1], 22.803)
        self.assertAlmostEqual(data_knutson[2,3,2], 4.324)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestKnutson)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
