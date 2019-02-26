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

Test tc_tracks module.
"""
import os
import unittest
import datetime as dt
import numpy as np
from netCDF4 import Dataset

from climada.hazard.tc_tracks import TCTracks
from climada.util.constants import SYSTEM_DIR

class TestIBTracs(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""

    class TestResult(unittest.TestResult):
        def addError(self, test, err):
            print('ERROR while downloading Ibtracs file.')

    def test_raw_ibtracs_empty(self):
        """ read_ibtracs_netcdf"""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299')
        self.assertEqual(tc_track.get_track(), [])

    def test_read_raw_pass(self):
        """Read a tropical cyclone."""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='2017242N16333')
        self.assertEqual(len(tc_track.data), 1)
        self.assertEqual(tc_track.get_track().name, '2017242N16333')
        self.assertEqual(tc_track.get_track().time.dt.year.values[0], 2017)
        self.assertEqual(tc_track.get_track().time.dt.month.values[0], 8)
        self.assertEqual(tc_track.get_track().time.dt.day.values[0], 30)
        self.assertEqual(tc_track.get_track().time.dt.hour.values[0], 0)
        self.assertAlmostEqual(tc_track.get_track().lat.values[0], 16.1 + 3.8146972514141453e-07)
        self.assertAlmostEqual(tc_track.get_track().lon.values[0], -26.9 + 3.8146972514141453e-07)
        self.assertAlmostEqual(tc_track.get_track().max_sustained_wind.values[0], 30)
        self.assertAlmostEqual(tc_track.get_track().central_pressure.values[0], 1008)
        self.assertAlmostEqual(tc_track.get_track().environmental_pressure.values[0], 1012)
        self.assertAlmostEqual(tc_track.get_track().radius_max_wind.values[0], 0)
        self.assertEqual(tc_track.get_track().time.size, 123)

        self.assertAlmostEqual(tc_track.get_track().lat.values[-1], 36.8 - 7.629394502828291e-07)
        self.assertAlmostEqual(tc_track.get_track().lon.values[-1], -90.1 + 1.5258789005656581e-06)
        self.assertAlmostEqual(tc_track.get_track().central_pressure.values[-1], 1005)
        self.assertAlmostEqual(tc_track.get_track().max_sustained_wind.values[-1], 15)
        self.assertAlmostEqual(tc_track.get_track().environmental_pressure.values[-1], 1008)
        self.assertAlmostEqual(tc_track.get_track().radius_max_wind.values[-1], 60)

        self.assertFalse(np.isnan(tc_track.get_track().radius_max_wind.values).any())
        self.assertFalse(np.isnan(tc_track.get_track().environmental_pressure.values).any())
        self.assertFalse(np.isnan(tc_track.get_track().max_sustained_wind.values).any())
        self.assertFalse(np.isnan(tc_track.get_track().central_pressure.values).any())
        self.assertFalse(np.isnan(tc_track.get_track().lat.values).any())
        self.assertFalse(np.isnan(tc_track.get_track().lon.values).any())

        self.assertEqual(tc_track.get_track().basin, 'NA')
        self.assertEqual(tc_track.get_track().max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.get_track().central_pressure_unit, 'mb')
        self.assertEqual(tc_track.get_track().name, '2017242N16333')
        self.assertEqual(tc_track.get_track().orig_event_flag, True)
        self.assertEqual(tc_track.get_track().data_provider, 'usa')
        self.assertEqual(tc_track.get_track().category, 5)

    def test_read_range(self):
        """Read a several TCs."""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id=None,
                               year_range=(1915, 1916), basin='WP')
        self.assertEqual(tc_track.size, 0)
        
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', year_range=(1993, 1994), basin='EP')
        self.assertEqual(tc_track.size, 32)

    def test_filter_ibtracs_track_pass(self):
        """ Test _filter_ibtracs """
        fn_nc = os.path.join(os.path.abspath(SYSTEM_DIR), 'IBTrACS.ALL.v04r00.nc')

        storm_id='1988234N13299'
        tc_track = TCTracks()
        sel = tc_track._filter_ibtracs(fn_nc, storm_id, year_range=None, basin=None)
        self.assertTrue(sel, np.array([10000]))

    def test_filter_ibtracs_year_basin_pass(self):
        """ Test _filter_ibtracs """
        fn_nc = os.path.join(os.path.abspath(SYSTEM_DIR), 'IBTrACS.ALL.v04r00.nc')

        tc_track = TCTracks()
        sel = tc_track._filter_ibtracs(fn_nc, storm_id=None, year_range=(1915, 1916),
                                 basin='WP')

        nc_data=Dataset(fn_nc)
        for i_sel in sel:
            self.assertEqual('WP',
                             ''.join(nc_data.variables['basin'][i_sel, 0, :].data.astype(str)))
            isot = nc_data.variables['iso_time'][i_sel, :, :]
            val_len = isot.mask[isot.mask==False].shape[0]//isot.shape[1]
            date = isot.data[:val_len]
            year = dt.datetime.strptime(''.join(date[0].astype(str)), '%Y-%m-%d %H:%M:%S').year
            self.assertTrue(year <= 1915 or year >= 1916)

        self.assertEqual(sel.size, 48)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestIBTracs)
unittest.TextTestRunner(verbosity=2).run(TESTS)
