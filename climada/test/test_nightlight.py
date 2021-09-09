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

Tests on Black marble.
"""

import unittest
import numpy as np

from shapely.geometry import Polygon

from climada.entity.exposures.litpop import nightlight as nl_utils

from climada.util.constants import SYSTEM_DIR

def init_test_shape():
    bounds = (14.18, 35.78, 14.58, 36.09)
    # (min_lon, max_lon, min_lat, max_lat)

    return bounds, Polygon([
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[1])
        ])

class TestNightlight(unittest.TestCase):
    """Test litpop.nightlight"""

    def test_load_nasa_nl_shape_single_tile_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()

    def test_load_nasa_nl_2016_shape_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()
        nl_utils.load_nasa_nl_shape(shape, 2016, data_dir=SYSTEM_DIR, dtype=None)

        # data, meta = nl_utils.load_nasa_nl_shape_single_tile(shape, path, layer=0):


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightlight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
