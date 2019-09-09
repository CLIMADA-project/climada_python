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

Test TropCyclone class: surges
"""

import os
import unittest
import elevation

class TestDEM(unittest.TestCase):
    """ Test use of DEM """

    def test_elevation_call_pass(self):
        """ Test elevation call """
        self.assertFalse(os.system('eio'))

    def test_rome_pass(self):
        """ Test DEM of Rome with elevation """
        elevation.clip(bounds=(12.35, 41.8, 12.65, 42), output='Rome-DEM.tif')

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDEM)
    unittest.TextTestRunner(verbosity=2).run(TESTS)

