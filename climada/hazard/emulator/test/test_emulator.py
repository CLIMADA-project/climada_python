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

Test the hazard event emulator
"""

import unittest

import numpy as np
import pandas as pd

from climada.hazard.emulator import emulator
from climada.hazard.emulator import geo


class TestEmulator(unittest.TestCase):
    """Test the hazard event emulator"""

    def test_hazard_emulator(self):
        """Test HazardEmulator class"""
        events = pd.DataFrame({
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "name": [0, 1, 2, 3, 4, 5, 6, 7],
            "year": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "month": [7, 6, 11, 6, 8, 8, 9, 10],
            "intensity": [6, 1, 3, 2, 4, 0, 5, 7],
        })
        events_obs = pd.DataFrame({
            "year": [2001, 2001, 2001, 2002, 2002, 2002],
            "month": [7, 6, 8, 7, 7, 9],
            "intensity": [3, 4, 4, 5, 3, 2],
        })
        reg = geo.TCRegion(extent=[0, 1, 0, 1])
        freq = pd.DataFrame({
            "year": [2000, 2001, 2002],
            "freq": [0.4, 0.5, 0.5],
        })
        cidx = pd.DataFrame({
            "year": [1999, 2000, 2001, 2002, 2003],
            "month": [7, 7, 7, 7, 7],
            "gmt": [10, 11, 10, 14, 13],
        })

        emu = emulator.HazardEmulator(events, events_obs, reg, freq)

        # predict only within time horizon of calibration:
        emu.predict_statistics([cidx])
        self.assertEqual(emu.stats_pred.shape[0], emu.stats.shape[0])

        # predict within time horizon of index data:
        emu.predict_statistics([cidx])
        self.assertEqual(emu.stats_pred.shape[0], cidx.shape[0])

        draws = emu.draw_realizations(10, (1999, 2001))
        self.assertTrue(np.all(np.unique(draws['real_id']) == np.arange(10)))
        self.assertTrue(np.all(np.unique(draws['year']) == np.arange(1999, 2002)))


    def test_event_pool(self):
        """Test EventPool class"""
        events = pd.DataFrame({
            "year": [2000, 2000, 2000, 2001, 2001, 2002, 2002, 2002],
            "month": [4, 6, 11, 3, 5, 8, 9, 10],
            "intensity": [6, 1, 3, 2, 4, 0, 5, 7],
        })
        pool = emulator.EventPool(events)
        draws = pool.draw_realizations(10, 1.5, 3, 1)
        self.assertEqual(len(draws), 10)
        self.assertTrue(all(2 <= d['intensity'].mean() <= 4 for d in draws))
        self.assertTrue(all(1 <= d.shape[0] <= 5 for d in draws))

        pool.drop = ("intensity < 3.0", 1.0)
        draws = pool.draw_realizations(30, 1.5, 3, 1)
        self.assertEqual(len(draws), 30)
        self.assertTrue(all(np.all(d['intensity'] >= 3) for d in draws))



# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmulator)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
