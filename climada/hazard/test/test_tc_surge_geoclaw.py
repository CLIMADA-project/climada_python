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

Test tc_surge_geoclaw module
"""

import datetime as dt
import unittest

import numpy as np
import pandas as pd

from climada.hazard.tc_surge_geoclaw import (boxcover_points_along_axis,
                                             bounds_to_str,
                                             dt64_to_pydt)


class TestFuncs(unittest.TestCase):
    """Test helper functions"""

    def test_boxcover(self):
        """Test boxcovering function"""
        nsplits = 4
        # sorted list of 1d-points
        points = np.array([-3., -1.3, 1.5, 1.7, 4.6, 5.4, 6.2, 6.8, 7.])
        # shuffle list of points
        points = points[[4, 7, 3, 1, 2, 5, 8, 1, 6, 0]].reshape(-1, 1)
        # this is easy to see from the sorted list of points
        boxes_correct = [[-3.0, -1.3], [1.5, 1.7], [4.6, 7.0]]
        boxes, size = boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum(b[1] - b[0] for b in boxes))

        nsplits = 3
        points = np.array([
            [0.0, 0.2], [1.3, 0.1], [2.5, 0.0],
            [3.0, 1.5], [0.2, 1.2],
            [0.4, 2.3], [0.5, 3.0],
        ])
        boxes_correct = [
            [0.0, 0.0, 2.5, 0.2],
            [0.2, 1.2, 3.0, 1.5],
            [0.4, 2.3, 0.5, 3.0],
        ]
        boxes, size = boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
        boxes, size = boxcover_points_along_axis(points[:,::-1], nsplits)
        self.assertEqual(boxes, [[b[1], b[0], b[3], b[2]] for b in boxes_correct])


    def test_bounds_to_str(self):
        """Test conversion from lon-lat-bounds tuple to lat-lon string"""
        bounds_str = [
            [(-4.2, 1.0, -3.05, 2.125), '1N-2.125N_4.2W-3.05W'],
            [(106.9, -7, 111.6875, 25.1), '7S-25.1N_106.9E-111.7E'],
            [(-6.9, -7.8334, 11, 25.1), '7.833S-25.1N_6.9W-11E'],
        ]
        for bounds, string in bounds_str:
            str_out = bounds_to_str(bounds)
            self.assertEqual(str_out, string)


    def test_dt64_to_pydt(self):
        """Test conversion from datetime64 to python datetime objects"""
        # generate test data
        pydt = [
            dt.datetime(1865, 3, 7, 20, 41, 2),
            dt.datetime(2008, 2, 29, 0, 5, 30),
            dt.datetime(2013, 12, 2),
        ]
        dt64 = pd.Series(pydt).values

        # test conversion of numpy array of dates
        pydt_conv = dt64_to_pydt(dt64)
        self.assertIsInstance(pydt_conv, list)
        self.assertEqual(len(pydt_conv), dt64.size)
        self.assertIsInstance(pydt_conv[0], dt.datetime)
        self.assertEqual(pydt_conv[1], pydt[1])

        # test conversion of single object
        pydt_conv = dt64_to_pydt(dt64[2])
        self.assertIsInstance(pydt_conv, dt.datetime)
        self.assertEqual(pydt_conv, pydt[2])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    unittest.TextTestRunner(verbosity=2).run(TESTS)