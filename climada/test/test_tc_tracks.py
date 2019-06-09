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

Test tc_tracks module.
"""
import os
import unittest
import datetime as dt
import numpy as np
from netCDF4 import Dataset

from climada.hazard.tc_tracks import TCTracks, _calc_land_geom, _apply_decay_coeffs
from climada.util.constants import SYSTEM_DIR

class TestDownload(unittest.TestCase):
    """Test reading TC from IBTrACS files"""

    def test_raw_ibtracs_empty_pass(self):
        """ read_ibtracs_netcdf"""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299')
        self.assertEqual(tc_track.get_track(), [])

class TestIBTracs(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""

    def test_penv_rmax_penv_pass(self):
        """ read_ibtracs_netcdf"""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1992230N11325')
        penv_ref = np.ones(97)*1010
        penv_ref[26] = 1011
        penv_ref[27] = 1012
        penv_ref[28] = 1013
        penv_ref[29] = 1014
        penv_ref[30] = 1015
        penv_ref[31] = 1014
        penv_ref[32] = 1014
        penv_ref[33] = 1014
        penv_ref[34] = 1014
        penv_ref[35] = 1012

        self.assertTrue(np.array_equal(tc_track.get_track().environmental_pressure.values,
                                       penv_ref))
        self.assertTrue(np.array_equal(tc_track.get_track().radius_max_wind.values,
                                       np.zeros(97)))

    def test_read_raw_pass(self):
        """Read a tropical cyclone."""
        tc_track = TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='2017242N16333')
        self.assertEqual(len(tc_track.data), 1)
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
        self.assertEqual(tc_track.get_track().sid, '2017242N16333')
        self.assertEqual(tc_track.get_track().name, 'IRMA')
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
                             ''.join(nc_data.variables['basin'][i_sel, 0, :].astype(str)))
            isot = nc_data.variables['iso_time'][i_sel, :, :]
            val_len = isot.mask[isot.mask==False].shape[0]//isot.shape[1]
            date = isot.data[:val_len]
            year = dt.datetime.strptime(''.join(date[0].astype(str)), '%Y-%m-%d %H:%M:%S').year
            self.assertTrue(year <= 1915 or year >= 1916)

        self.assertEqual(sel.size, 48)

    def test_ibtracs_correct_pass(self):
        """ Check correct_pres option """
        tc_try = TCTracks()
        tc_try.read_ibtracs_netcdf(provider='usa', storm_id='1982267N25289', correct_pres=True)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[0], 1011.2906)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[5], 1005.7046)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[-1], 1011.6556)

    def test_wrong_decay_pass(self):
        """ Test decay not implemented when coefficient < 1 """
        track = TCTracks()
        track.read_ibtracs_netcdf(provider='usa', storm_id='1975178N28281')

        track_gen = track.data[0]
        track_gen['lat'] = np.array([28.20340431, 28.7915261 , 29.38642458, 29.97836984, 30.56844404,
                           31.16265292, 31.74820301, 32.34449825, 32.92261894, 33.47430891,
                           34.01492525, 34.56789399, 35.08810845, 35.55965893, 35.94835174,
                           36.29355848, 36.45379561, 36.32473812, 36.07552209, 35.92224784,
                           35.84144186, 35.78298537, 35.86090718, 36.02440372, 36.37555559,
                           37.06207765, 37.73197352, 37.97524273, 38.05560287, 38.21901208,
                           38.31486156, 38.30813367, 38.28481808, 38.28410366, 38.25894812,
                           38.20583372, 38.22741099, 38.39970022, 38.68367797, 39.08329904,
                           39.41434629, 39.424984  , 39.31327716, 39.30336335, 39.31714429,
                           39.27031932, 39.30848775, 39.48759833, 39.73326595, 39.96187967,
                           40.26954226, 40.76882202, 41.40398607, 41.93809726, 42.60395785,
                           43.57074792, 44.63816143, 45.61450458, 46.68528511, 47.89209365,
                           49.15580502])
        track_gen['lon'] = np.array([-79.20514075, -79.25243311, -79.28393082, -79.32324646,
                                   -79.36668585, -79.41495519, -79.45198688, -79.40580325,
                                   -79.34965443, -79.36938122, -79.30294825, -79.06809546,
                                   -78.70281969, -78.29418936, -77.82170609, -77.30034709,
                                   -76.79004969, -76.37038827, -75.98641014, -75.58383356,
                                   -75.18310414, -74.7974524 , -74.3797645 , -73.86393572,
                                   -73.37910948, -73.01059003, -72.77051313, -72.68011328,
                                   -72.66864779, -72.62579773, -72.56307717, -72.46607618,
                                   -72.35871353, -72.31120649, -72.15537583, -71.75577051,
                                   -71.25287498, -70.75527907, -70.34788946, -70.17518421,
                                   -70.04446577, -69.76582749, -69.44372386, -69.15881376,
                                   -68.84351922, -68.47890287, -68.04184565, -67.53541437,
                                   -66.94008642, -66.25596075, -65.53496635, -64.83491802,
                                   -64.12962685, -63.54118808, -62.72934383, -61.34915091,
                                   -59.72580755, -58.24404252, -56.71972992, -55.0809336 ,
                                   -53.31524758])

        v_rel = {3: 0.002249541544102336, 1: 0.00046889526284203036, 4: 0.002649273787364977, 2: 0.0016426186150461349, 5: 0.00246400811445618, 7: 0.0030442198547309075, 6: 0.002346537842810565}
        p_rel = {3: (1.028420239620591, 0.003174733355067952), 1: (1.0046803184177564, 0.0007997633912500546), 4: (1.0498749735343516, 0.0034665588904747515), 2: (1.0140127424090262, 0.002131858515233042), 5: (1.0619445995372885, 0.003467268426139696), 7: (1.0894914184297835, 0.004315034379018768), 6: (1.0714354641894077, 0.002783787561718677)}
        track_gen.attrs['orig_event_flag'] = False

        cp_ref = np.array([1012., 1012.])
        land_geom = _calc_land_geom([track_gen])
        track_res = _apply_decay_coeffs(track_gen, v_rel, p_rel, land_geom, True)
        self.assertTrue(np.array_equal(cp_ref, track_res.central_pressure[9:11]))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDownload)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIBTracs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
