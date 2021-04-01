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

Test random sampling functionalities
"""

import unittest

import numpy as np
import pandas as pd

from climada.hazard.emulator import random


class TestRandom(unittest.TestCase):
    """Test random sampling functionalities"""

    def test_estimate_drop(self):
        """Test estimate_drop function"""
        events = pd.DataFrame({
            "yr": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "dummy": [4, 6, 11, 3, 5, 8, 9, 10],
            "inten": [6, 1, 3, 2, 4, 0, 5, 7],
        })
        expr, frac = random.estimate_drop(events, "yr", "inten",
                                          (2001, 2001), norm_mean=4)
        self.assertEqual(expr, 'inten < 3.0')
        self.assertLessEqual(frac, 1.0)
        self.assertGreater(frac, 0.0)


    def test_draw_poisson_events(self):
        """Test draw_poisson_events function"""
        events = pd.DataFrame({
            "dummy1": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "dummy2": [4, 6, 11, 3, 5, 8, 9, 10],
            "height": [6, 1, 3, 2, 4, 0, 5, 7],
        })

        # reasonable draw:
        draws = random.draw_poisson_events(1.5, events, "height", (2, 4))
        self.assertTrue(draws.size > 1)

        # impossible draw:
        draws = random.draw_poisson_events(1.5, events, "height", (10, 12))
        self.assertTrue(draws is None)

        # drop all but entries 0 and 7:
        draws = random.draw_poisson_events(2, events, "height", (5, 8),
                                           drop=("height < 6", 1.0))
        self.assertEqual(np.count_nonzero(~np.isin(draws, [0, 7])), 0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRandom)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
