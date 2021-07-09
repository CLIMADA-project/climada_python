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

Test tc_tracks_forecast module.
"""

import unittest
import numpy as np

from climada import CONFIG
from climada.hazard.tc_tracks_forecast import TCForecast

DATA_DIR = CONFIG.hazard.test_data.dir()
TEST_BUFR_FILES_PATH = [
    DATA_DIR.joinpath(tbf) for tbf in [
        'tracks_22S_HEROLD_2020031912.det.bufr4',
        'tracks_22S_HEROLD_2020031912.eps.bufr4',
    ]
]
TEST_BUFR_FILES = [open(f, 'rb') for f in TEST_BUFR_FILES_PATH]
"""TC tracks in four BUFR formats as provided by ECMWF. Sourced from
https://confluence.ecmwf.int/display/FCST/New+Tropical+Cyclone+Wind+Radii+product
"""


class TestECMWF(unittest.TestCase):
    """Test reading of BUFR TC track forecasts"""

    def test_fetch_ecmwf(self):
        """Test ECMWF reader with static files"""
        forecast = TCForecast()
        forecast.fetch_ecmwf(files=TEST_BUFR_FILES)

        self.assertEqual(forecast.data[0].time.size, 2)
        self.assertEqual(forecast.data[1].lat[2], -27.)
        self.assertEqual(forecast.data[0].lon[2], 73.5)
        self.assertEqual(forecast.data[1].time_step[2], 6)
        self.assertEqual(forecast.data[1].max_sustained_wind[2], 14.9)
        self.assertEqual(forecast.data[0].central_pressure[1], 1000.)
        self.assertEqual(forecast.data[0]['time.year'][1], 2020)
        self.assertEqual(forecast.data[17]['time.month'][7], 3)
        self.assertEqual(forecast.data[17]['time.day'][7], 21)
        self.assertEqual(forecast.data[0].max_sustained_wind_unit, 'm/s')
        self.assertEqual(forecast.data[0].central_pressure_unit, 'mb')
        self.assertEqual(forecast.data[1].sid, '22S')
        self.assertEqual(forecast.data[1].name, 'HEROLD')
        np.testing.assert_array_equal(forecast.data[0].basin, 'S - South-West Indian Ocean')
        self.assertEqual(forecast.data[0].category, 'Tropical Depression')
        self.assertEqual(forecast.data[0].forecast_time,
                         np.datetime64('2020-03-19T12:00:00.000000'))
        self.assertEqual(forecast.data[1].is_ensemble, True)

    def test_equal_timestep(self):
        """Test equal timestep"""
        forecast = TCForecast()
        forecast.fetch_ecmwf(files=TEST_BUFR_FILES)
        forecast.equal_timestep(1)

        self.assertEqual(forecast.data[1].time.size, 13)
        self.assertEqual(forecast.data[1].lat.size, 13)
        self.assertEqual(forecast.data[1].lon.size, 13)
        self.assertEqual(forecast.data[1].max_sustained_wind.size, 13)
        self.assertEqual(forecast.data[1].central_pressure.size, 13)
        self.assertEqual(forecast.data[1].radius_max_wind.size, 13)
        self.assertEqual(forecast.data[1].environmental_pressure.size, 13)
        self.assertEqual(forecast.data[1].time_step[2], 1.)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestECMWF)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
