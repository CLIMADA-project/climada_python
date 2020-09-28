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

Test non-trivial runs of TCSurgeGeoClaw class
"""

import os
import unittest

import numpy as np
import xarray as xr

from climada.hazard.tc_surge_geoclaw import setup_clawpack, geoclaw_surge_from_track


DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'hazard/test/data')
ZOS_PATH = os.path.join(DATA_DIR, "zos_monthly.nc")
TOPO_PATH = os.path.join(DATA_DIR, "surge_topo.tif")


class TestGeoclawRun(unittest.TestCase):
    """Test functions that set up and run GeoClaw instances"""

    def test_surge_from_track(self):
        """Test geoclaw_surge_from_track function"""
        # similar to IBTrACS 2010029S12177 (OLI, 2010) hitting Tubuai
        track = xr.Dataset({
            'radius_max_wind': ('time', [15., 15, 15, 15, 15, 17, 20, 20]),
            'radius_oci': ('time', [202., 202, 202, 202, 202, 202, 202, 202]),
            'max_sustained_wind': ('time', [105., 97, 90, 85, 80, 72, 65, 66]),
            'central_pressure': ('time', [944., 950, 956, 959, 963, 968, 974, 975]),
            'time_step': ('time', np.full((8,), np.timedelta64(3, 'h'))),
        }, coords={
            'time': np.arange('2010-02-05T09:00', '2010-02-06T09:00',
                              np.timedelta64(3, 'h'), dtype='datetime64[h]'),
            'lat': ('time', [-26.33, -25.54, -24.79, -24.05,
                             -23.35, -22.7, -22.07, -21.50]),
            'lon': ('time', [-147.27, -148.0, -148.51, -148.95,
                             -149.41, -149.85, -150.27, -150.56]),
        }, attrs={
            'sid': '2010029S12177_test',
        })
        centroids = np.array([
            # points along coastline:
            [-23.44084378, -149.45562336], [-23.43322580, -149.44678650],
            [-23.42347479, -149.42088538], [-23.42377951, -149.41418156],
            [-23.41494266, -149.39742201], [-23.41494266, -149.38919460],
            [-23.38233772, -149.38949932],
            # points inland at higher altitude:
            [-23.37505943, -149.46882493], [-23.36615826, -149.45798872],
        ])
        setup_clawpack()
        intensity = geoclaw_surge_from_track(track, centroids, ZOS_PATH, TOPO_PATH)
        self.assertEqual(intensity.shape, (centroids.shape[0],))
        self.assertTrue(np.all(intensity[:7] > 0))
        self.assertTrue(np.all(intensity[7:] == 0))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGeoclawRun)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
