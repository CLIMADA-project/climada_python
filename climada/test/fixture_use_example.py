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

This files shows a few example of how to use the fixtures defined in common_test_fixtures.py

"""

import numpy as np

from climada.engine import ImpactCalc


class TestImpactCalc:
    def test_impact(self, exposures, hazard, impfset):
        imp = ImpactCalc(exposures, impfset, hazard).impact()
        assert imp.aai_agg == (1 / 2 * 1000) * 0.006 + (1 / 4 * 15000) * 0.004
        np.testing.assert_array_equal(
            imp.eai_exp,
            np.array(
                [
                    0.0,
                    (1000 * 0 * 0.03)
                    + (1000 * 0 * 0.01)
                    + ((1000 * 1 / 2) * 0.006)
                    + ((1000 * 1 / 4) * 0.004)
                    + ((1000 * 1) * 0.0),
                    (2000 * 0 * 0.03)
                    + (2000 * 0 * 0.01)
                    + ((2000 * 0) * 0.006)
                    + ((2000 * 1 / 4) * 0.004)
                    + ((2000 * 1) * 0.0),
                    (3000 * 0 * 0.03)
                    + (3000 * 0 * 0.01)
                    + ((3000 * 0) * 0.006)
                    + ((3000 * 1 / 4) * 0.004)
                    + ((3000 * 1) * 0.0),
                    (4000 * 0 * 0.03)
                    + (4000 * 0 * 0.01)
                    + ((4000 * 0) * 0.006)
                    + ((4000 * 1 / 4) * 0.004)
                    + ((4000 * 1) * 0.0),
                    (5000 * 0 * 0.03)
                    + (5000 * 0 * 0.01)
                    + ((5000 * 0) * 0.006)
                    + ((5000 * 1 / 4) * 0.004)
                    + ((5000 * 1) * 0.0),
                    # (Value * Int * Freq)
                ]
            ),
            err_msg="eai_exp impacts invalid",
        )
        np.testing.assert_array_equal(
            imp.at_event,
            np.array(
                [
                    0.0,
                    0.0,
                    1000 * 1 / 2,
                    (1000 + 2000 + 3000 + 4000 + 5000) * 1 / 4,
                    (1000 + 2000 + 3000 + 4000 + 5000),
                ]
            ),
            err_msg="at_event impacts invalid",
        )
        np.testing.assert_array_equal(
            imp.calc_freq_curve([20, 50, 100, 500]).impact,
            np.array([0, 0, 500, 3750]),
            err_msg="return period impacts invalid",
        )
