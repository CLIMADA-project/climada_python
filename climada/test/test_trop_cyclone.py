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

Test TropCyclone class
"""

import unittest
import numpy as np
from scipy import sparse

from climada.hazard.trop_cyclone import TropCyclone

class TestClimateSce(unittest.TestCase):

    def test_apply_criterion_track(self):
        """Test _apply_criterion function."""
        tc = TropCyclone()
        tc.intensity = np.zeros((4, 10))
        tc.intensity[0, :] = np.arange(10)
        tc.intensity[1, 5] = 10
        tc.intensity[2, :] = np.arange(10, 20)
        tc.intensity[3, 3] = 3
        tc.intensity = sparse.csr_matrix(tc.intensity)
        tc.basin = ['NA'] * 4
        tc.basin[3] = 'NO'
        tc.category = np.array([2, 0, 4, 1])
        tc.event_id = np.arange(4)
        tc.frequency = np.ones(4) * 0.5

        tc_cc = tc.set_climate_scenario_knu(ref_year=2050, rcp_scenario=45)
        self.assertTrue(np.allclose(tc.intensity[1, :].toarray(), tc_cc.intensity[1, :].toarray()))
        self.assertTrue(np.allclose(tc.intensity[3, :].toarray(), tc_cc.intensity[3, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[0, :].toarray(), tc_cc.intensity[0, :].toarray()))
        self.assertFalse(
            np.allclose(tc.intensity[2, :].toarray(), tc_cc.intensity[2, :].toarray()))
        self.assertTrue(np.allclose(tc.frequency, tc_cc.frequency))
        self.assertEqual(
            tc_cc.tag.description,
            'climate change scenario for year 2050 and RCP 45 from Knutson et al 2015.')

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestClimateSce)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
