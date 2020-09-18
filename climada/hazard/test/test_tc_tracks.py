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
import xarray as xr
import numpy as np
import netCDF4 as nc

import climada.hazard.tc_tracks as tc
from climada.util import ureg
from climada.util.constants import TC_ANDREW_FL
from climada.util.coordinates import coord_on_land, dist_to_coast

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")
TEST_RAW_TRACK = os.path.join(DATA_DIR, 'Storm.2016075S11087.ibtracs_all.v03r10.csv')
TEST_TRACK_GETTELMAN = os.path.join(DATA_DIR, 'gettelman_test_tracks.nc')
TEST_TRACK_EMANUEL = os.path.join(DATA_DIR, 'emanuel_test_tracks.mat')
TEST_TRACK_EMANUEL_CORR = os.path.join(DATA_DIR, 'temp_mpircp85cal_full.mat')


class TestIBTracs(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""

    def test_raw_ibtracs_empty_pass(self):
        """Test reading TC from IBTrACS files"""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299')
        self.assertEqual(tc_track.get_track(), [])

    def test_write_read_pass(self):
        """Test writting and reading netcdf4 TCTracks instances"""
        path = os.path.join(DATA_DIR, "tc_tracks_nc")
        os.makedirs(path, exist_ok=True)
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1988234N13299',
                                     estimate_missing=True)
        tc_track.write_netcdf(path)

        tc_read = tc.TCTracks()
        tc_read.read_netcdf(path)

        self.assertEqual(tc_track.get_track().sid, tc_read.get_track().sid)

    def test_penv_rmax_penv_pass(self):
        """read_ibtracs_netcdf"""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id='1992230N11325')
        penv_ref = np.ones(97) * 1010
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

        self.assertTrue(np.allclose(
            tc_track.get_track().environmental_pressure.values, penv_ref))
        self.assertTrue(np.allclose(
            tc_track.get_track().radius_max_wind.values, np.zeros(97)))

    def test_read_raw_pass(self):
        """Read a tropical cyclone."""
        tc_track = tc.TCTracks()
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
        self.assertAlmostEqual(tc_track.get_track().radius_max_wind.values[0], 60)
        self.assertEqual(tc_track.get_track().time.size, 123)

        self.assertAlmostEqual(tc_track.get_track().lat.values[-1], 36.8 - 7.629394502828291e-07)
        self.assertAlmostEqual(tc_track.get_track().lon.values[-1], -90.100006, 5)
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
        """Read several TCs."""
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', storm_id=None,
                                     year_range=(1915, 1916), basin='WP')
        self.assertEqual(tc_track.size, 0)

        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', year_range=(1993, 1994),
                                     basin='EP', estimate_missing=False)
        self.assertEqual(tc_track.size, 34)

        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(provider='usa', year_range=(1993, 1994),
                                     basin='EP', estimate_missing=True)
        self.assertEqual(tc_track.size, 52)

    def test_ibtracs_correct_pass(self):
        """Check estimate_missing option"""
        tc_try = tc.TCTracks()
        tc_try.read_ibtracs_netcdf(provider='usa', storm_id='1982267N25289',
                                   estimate_missing=True)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[0], 1013.61584, 5)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[5], 1008.63837, 5)
        self.assertAlmostEqual(tc_try.data[0].central_pressure.values[-1], 1014.1515, 4)


class TestIO(unittest.TestCase):
    """Test reading of tracks from files of different formats"""

    def test_read_processed_ibtracs_csv(self):
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)

        self.assertEqual(tc_track.data[0].time.size, 38)
        self.assertEqual(tc_track.data[0].lon[11], -39.60)
        self.assertEqual(tc_track.data[0].lat[23], 14.10)
        self.assertEqual(tc_track.data[0].time_step[7], 6)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 55)
        self.assertEqual(tc_track.data[0].central_pressure.values[29], 975.9651)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0].time.dt.year[13], 1951)
        self.assertEqual(tc_track.data[0].time.dt.month[26], 9)
        self.assertEqual(tc_track.data[0].time.dt.day[7], 29)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].sid, '1951239N12334')
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.data[0].basin))
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_read_simulations_emanuel(self):
        tc_track = tc.TCTracks()

        tc_track.read_simulations_emanuel(TEST_TRACK_EMANUEL, hemisphere='N')
        self.assertEqual(len(tc_track.data), 4)
        self.assertEqual(tc_track.data[0].time.size, 93)
        self.assertEqual(tc_track.data[0].lon[11], -115.57)
        self.assertEqual(tc_track.data[0].lat[23], 10.758)
        self.assertEqual(tc_track.data[0].time_step[7], 2)
        self.assertAlmostEqual(tc_track.data[0].radius_max_wind[15], 44.27645788336934)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 27.1)
        self.assertEqual(tc_track.data[0].central_pressure[29], 995.31)
        self.assertTrue(np.all(tc_track.data[0].environmental_pressure == 1010))
        self.assertTrue(np.all(tc_track.data[0].time.dt.year == 1950))
        self.assertEqual(tc_track.data[0].time.dt.month[26], 10)
        self.assertEqual(tc_track.data[0].time.dt.day[7], 26)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].sid, '1')
        self.assertEqual(tc_track.data[0].name, '1')
        self.assertTrue(np.all([d.basin == 'N' for d in tc_track.data]))
        self.assertEqual(tc_track.data[0].category, 3)

        tc_track.read_simulations_emanuel(TEST_TRACK_EMANUEL_CORR)
        self.assertEqual(len(tc_track.data), 2)
        self.assertTrue(np.all([d.basin == 'S' for d in tc_track.data]))
        self.assertEqual(tc_track.data[0].radius_max_wind[15], 102.49460043196545)
        self.assertEqual(tc_track.data[0].time.dt.month[343], 2)
        self.assertEqual(tc_track.data[0].time.dt.day[343], 28)
        self.assertEqual(tc_track.data[0].time.dt.month[344], 3)
        self.assertEqual(tc_track.data[0].time.dt.day[344], 1)
        self.assertEqual(tc_track.data[1].time.dt.year[0], 2009)
        self.assertEqual(tc_track.data[1].time.dt.year[256], 2009)
        self.assertEqual(tc_track.data[1].time.dt.year[257], 2010)
        self.assertEqual(tc_track.data[1].time.dt.year[-1], 2010)

    def test_read_one_gettelman(self):
        """Test reading and model of TC from Gettelman track files"""
        tc_track_G = tc.TCTracks()
        # populate tracks by loading data from NetCDF:
        nc_data = nc.Dataset(TEST_TRACK_GETTELMAN)
        nstorms = nc_data.dimensions['storm'].size
        for i in range(nstorms):
            tc_track_G.read_one_gettelman(nc_data, i)

        self.assertEqual(tc_track_G.data[0].time.size, 29)
        self.assertEqual(tc_track_G.data[0].lon[11], 60.0)
        self.assertEqual(tc_track_G.data[0].lat[23], 10.20860481262207)
        self.assertEqual(tc_track_G.data[0].time_step[7], 3.)
        self.assertEqual(np.max(tc_track_G.data[0].radius_max_wind), 65)
        self.assertEqual(np.min(tc_track_G.data[0].radius_max_wind), 65)
        self.assertEqual(tc_track_G.data[0].max_sustained_wind[21], 39.91877223718089)
        self.assertEqual(tc_track_G.data[0].central_pressure[27], 1005.969482421875)
        self.assertEqual(np.max(tc_track_G.data[0].environmental_pressure), 1015)
        self.assertEqual(np.min(tc_track_G.data[0].environmental_pressure), 1015)
        self.assertEqual(tc_track_G.data[0].maximum_precipitation[14], 219.10108947753906)
        self.assertEqual(tc_track_G.data[0].average_precipitation[12], 101.43893432617188)
        self.assertEqual(tc_track_G.data[0].time.dt.year[13], 1979)
        self.assertEqual(tc_track_G.data[0].time.dt.month[26], 1)
        self.assertEqual(tc_track_G.data[0].time.dt.day[7], 2)
        self.assertEqual(tc_track_G.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track_G.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track_G.data[0].sid, '0')
        self.assertEqual(tc_track_G.data[0].name, '0')
        self.assertEqual(tc_track_G.data[0].basin, 'NI - North Indian')
        self.assertEqual(tc_track_G.data[0].category, 0)


class TestFuncs(unittest.TestCase):
    """Test functions over TC tracks"""

    def test_get_track_pass(self):
        """Test get_track."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        self.assertIsInstance(tc_track.get_track(), xr.Dataset)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

        tc_track_bis = tc.TCTracks()
        tc_track_bis.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.append(tc_track_bis)
        self.assertIsInstance(tc_track.get_track(), list)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

    def test_subset(self):
        """Test subset."""
        storms = ['1988169N14259', '2002073S16161', '2002143S07157']
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(storm_id=storms)
        self.assertEqual(tc_track.subset({'basin': 'SP'}).size, 2)

    def test_get_extent(self):
        """Test extent/bounds attributes."""
        storms = ['1988169N14259', '2002073S16161', '2002143S07157']
        tc_track = tc.TCTracks()
        tc_track.read_ibtracs_netcdf(storm_id=storms)
        bounds = (153.4752, -23.2000, 258.7132, 17.5166)
        extent = (bounds[0], bounds[2], bounds[1], bounds[3])
        bounds_buf = (153.3752, -23.3000, 258.8132, 17.6166)
        self.assertTrue(np.allclose(tc_track.bounds, bounds))
        self.assertTrue(np.allclose(tc_track.get_bounds(deg_buffer=0.1), bounds_buf))
        self.assertTrue(np.allclose(tc_track.extent, extent))

    def test_interp_track_pass(self):
        """Interpolate track to min_time_step. Compare to MATLAB reference."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[11], -27.426151640151684)
        self.assertAlmostEqual(float(tc_track.data[0].lat[23]), 12.300006169591480)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29],
                               1.0077614e+03)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.data[0].basin))
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_interp_origin_pass(self):
        """Interpolate track to min_time_step crossing lat origin"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([
            167.207761, 168.1, 168.936535, 169.728947, 170.5, 171.257176,
            171.946822, 172.5, 172.871797, 173.113396, 173.3, 173.496375,
            173.725522, 174., 174.331591, 174.728961, 175.2, 175.747632,
            176.354929, 177., 177.66677, 178.362433, 179.1, 179.885288,
            -179.304661, -178.5, -177.726442, -176.991938, -176.3, -175.653595,
            -175.053513, -174.5, -173.992511, -173.527342, -173.1, -172.705991,
            -172.340823, -172.
        ])
        tc_track.data[0].lat.values = np.array([
            40.196053, 40.6, 40.930215, 41.215674, 41.5, 41.816354, 42.156065,
            42.5, 42.833998, 43.16377, 43.5, 43.847656, 44.188854, 44.5,
            44.764269, 44.991925, 45.2, 45.402675, 45.602707, 45.8, 45.995402,
            46.193543, 46.4, 46.615718, 46.82312, 47., 47.130616, 47.225088,
            47.3, 47.369224, 47.435786, 47.5, 47.562858, 47.628064, 47.7,
            47.783047, 47.881586, 48.
        ])
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[0], 167.207761)
        self.assertAlmostEqual(tc_track.data[0].lon.values[-1], -172)
        self.assertAlmostEqual(tc_track.data[0].lon.values[137], 179.75187272)
        self.assertAlmostEqual(tc_track.data[0].lon.values[138], 179.885288)
        self.assertAlmostEqual(tc_track.data[0].lon.values[139], -179.98060885)
        self.assertAlmostEqual(tc_track.data[0].lon.values[140], -179.84595743)
        self.assertAlmostEqual(tc_track.data[0].lat.values[0], 40.196053)
        self.assertAlmostEqual(tc_track.data[0].lat.values[-1], 48.)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29],
                               1.0077614e+03)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.data[0].basin))
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_interp_origin_inv_pass(self):
        """Interpolate track to min_time_step crossing lat origin"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([
            167.207761, 168.1, 168.936535, 169.728947, 170.5, 171.257176,
            171.946822, 172.5, 172.871797, 173.113396, 173.3, 173.496375,
            173.725522, 174., 174.331591, 174.728961, 175.2, 175.747632,
            176.354929, 177., 177.66677, 178.362433, 179.1, 179.885288,
            -179.304661, -178.5, -177.726442, -176.991938, -176.3, -175.653595,
            -175.053513, -174.5, -173.992511, -173.527342, -173.1, -172.705991,
            -172.340823, -172.
        ])
        tc_track.data[0].lon.values = - tc_track.data[0].lon.values
        tc_track.data[0].lat.values = np.array([
            40.196053, 40.6, 40.930215, 41.215674, 41.5, 41.816354, 42.156065,
            42.5, 42.833998, 43.16377, 43.5, 43.847656, 44.188854, 44.5,
            44.764269, 44.991925, 45.2, 45.402675, 45.602707, 45.8, 45.995402,
            46.193543, 46.4, 46.615718, 46.82312, 47., 47.130616, 47.225088,
            47.3, 47.369224, 47.435786, 47.5, 47.562858, 47.628064, 47.7,
            47.783047, 47.881586, 48.
        ])
        tc_track.equal_timestep(time_step_h=1)

        self.assertEqual(tc_track.data[0].time.size, 223)
        self.assertAlmostEqual(tc_track.data[0].lon.values[0], -167.207761)
        self.assertAlmostEqual(tc_track.data[0].lon.values[-1], 172)
        self.assertAlmostEqual(tc_track.data[0].lon.values[137], -179.75187272)
        self.assertAlmostEqual(tc_track.data[0].lon.values[138], -179.885288)
        self.assertAlmostEqual(tc_track.data[0].lon.values[139], 179.98060885)
        self.assertAlmostEqual(tc_track.data[0].lon.values[140], 179.84595743)
        self.assertAlmostEqual(tc_track.data[0].lat.values[0], 40.196053)
        self.assertAlmostEqual(tc_track.data[0].lat.values[-1], 48.)
        self.assertEqual(tc_track.data[0].time_step[7], 1)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 25)
        self.assertAlmostEqual(tc_track.data[0].central_pressure.values[29],
                               1.0077614e+03)
        self.assertEqual(np.max(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.data[0].environmental_pressure), 1010)
        self.assertEqual(tc_track.data[0]['time.year'][13], 1951)
        self.assertEqual(tc_track.data[0]['time.month'][26], 8)
        self.assertEqual(tc_track.data[0]['time.day'][7], 27)
        self.assertEqual(tc_track.data[0].max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.data[0].central_pressure_unit, 'mb')
        self.assertEqual(tc_track.data[0].orig_event_flag, 1)
        self.assertEqual(tc_track.data[0].name, '1951239N12334')
        self.assertEqual(tc_track.data[0].data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.data[0].basin))
        self.assertEqual(tc_track.data[0].id_no, 1951239012334)
        self.assertEqual(tc_track.data[0].category, 1)

    def test_dist_since_lf_pass(self):
        """Test _dist_since_lf for andrew tropical cyclone."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        track = tc_track.get_track()
        track['on_land'] = ('time', coord_on_land(track.lat.values, track.lon.values))
        track['dist_since_lf'] = ('time', tc._dist_since_lf(track))

        msk = ~track.on_land
        self.assertTrue(np.all(np.isnan(track.dist_since_lf.values[msk])))
        self.assertEqual(track.dist_since_lf.values[msk].size, 38)

        self.assertGreater(track.dist_since_lf.values[-1],
                           dist_to_coast(track.lat.values[-1], track.lon.values[-1]) / 1000)
        self.assertEqual(1020.5431562223974, track['dist_since_lf'].values[-1])

        # check distances on land always increase, in second landfall
        dist_on_land = track.dist_since_lf.values[track.on_land]
        self.assertTrue(np.all(np.diff(dist_on_land)[1:] > 0))

    def test_category_pass(self):
        """Test category computation."""
        max_sus_wind = np.array([25, 30, 35, 40, 45, 45, 45, 45, 35, 25])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([25, 25, 25, 30, 30, 30, 30, 30, 25, 25, 20])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([80, 90, 100, 115, 120, 125, 130,
                                 120, 110, 80, 75, 80, 65])
        max_sus_wind_unit = 'kn'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

        max_sus_wind = np.array([
            28.769475, 34.52337, 40.277265, 46.03116, 51.785055, 51.785055,
            51.785055, 51.785055, 40.277265, 28.769475
        ])
        max_sus_wind_unit = 'mph'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([
            12.86111437, 12.86111437, 12.86111437, 15.43333724, 15.43333724,
            15.43333724, 15.43333724, 15.43333724, 12.86111437, 12.86111437,
            10.2888915
        ])
        max_sus_wind_unit = 'm/s'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([
            148.16, 166.68, 185.2, 212.98, 222.24, 231.5, 240.76, 222.24,
            203.72, 148.16, 138.9, 148.16, 120.38
        ])
        max_sus_wind_unit = 'km/h'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

    def test_estimate_params_pass(self):
        """Test track parameter estimation functions."""
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        v_max = np.array([45, np.nan, 50, 55, 0, 60, 75])
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, np.nan, -999])
        lon = np.array([np.nan, -52.8, -54.4, -56, -58.4, -59.7, -61.1])
        ref_pres = np.array([np.nan, 993, 990.2324, 986.6072, np.nan, 1004, np.nan])
        out_pres = tc._estimate_pressure(cen_pres, lat, lon, v_max)
        self.assertTrue(np.allclose(ref_pres, out_pres, equal_nan=True))

        v_max = np.array([45, np.nan, 50, 55, 0, 60, 75])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, np.nan, -999])
        lon = np.array([np.nan, -52.8, -54.4, -56, -58.4, -59.7, -61.1])
        ref_vmax = np.array([45, 46.38272, 50, 55, np.nan, 60, 75])
        out_vmax = tc._estimate_vmax(v_max, lat, lon, cen_pres)
        self.assertTrue(np.allclose(ref_vmax, out_vmax, equal_nan=True))

        roci = np.array([np.nan, -1, 145, 170, 180, 0, -5])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        ref_roci = np.array([np.nan, 182.792715, 145, 170, 180, 161.5231086, np.nan])
        out_roci = tc.estimate_roci(roci, cen_pres)
        self.assertTrue(np.allclose(ref_roci, out_roci, equal_nan=True))

        rmw = np.array([17, 33, -1, 25, np.nan, -5, 13])
        cen_pres = np.array([-999, 993, np.nan, -1, 0, 1004, np.nan])
        ref_rmw = np.array([17, 33, np.nan, 25, np.nan, 43.95543761, 13])
        out_rmw = tc.estimate_rmw(rmw, cen_pres)
        self.assertTrue(np.allclose(ref_rmw, out_rmw, equal_nan=True))

    def test_estimate_rmw_pass(self):
        """Test estimate_rmw function."""
        NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude

        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rad_max_wind = tc.estimate_rmw(
            tc_track.data[0].radius_max_wind.values,
            tc_track.data[0].central_pressure.values) * NM_TO_KM

        self.assertAlmostEqual(rad_max_wind[0], 86.4471340900, places=5)
        self.assertAlmostEqual(rad_max_wind[10], 86.525605570, places=5)
        self.assertAlmostEqual(rad_max_wind[128], 55.25462781, places=5)
        self.assertAlmostEqual(rad_max_wind[129], 54.40164284, places=5)
        self.assertAlmostEqual(rad_max_wind[130], 53.54865787, places=5)
        self.assertAlmostEqual(rad_max_wind[189], 52.62700450, places=5)
        self.assertAlmostEqual(rad_max_wind[190], 54.36738477, places=5)
        self.assertAlmostEqual(rad_max_wind[191], 56.10776504, places=5)
        self.assertAlmostEqual(rad_max_wind[192], 57.84814530, places=5)
        self.assertAlmostEqual(rad_max_wind[200], 70.00942075, places=5)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIBTracs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
