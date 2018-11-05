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

import climada.util.finance as u_fin

class TestNetpresValue(unittest.TestCase):
    """Test date functions """
    def test_net_pres_val_pass(self):
        """ Test net_present_value against MATLAB reference"""
        years = np.arange(2018, 2041)
        disc_rates = np.ones(years.size)*0.02
        val_years = np.ones(years.size)*6.512201157564418e9
        res = u_fin.net_present_value(years, disc_rates, val_years)
        
        self.assertEqual(1.215049630691397e+11, res)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNetpresValue)
unittest.TextTestRunner(verbosity=2).run(TESTS)
