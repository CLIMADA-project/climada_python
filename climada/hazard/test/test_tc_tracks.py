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
import array
import xarray as xr
import numpy as np
import netCDF4 as nc

from climada.hazard.tc_tracks import TCTracks
import climada.hazard.tc_tracks as tc
from climada.util.constants import TC_ANDREW_FL
from climada.util.coordinates import coord_on_land, dist_to_coast

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")
TEST_RAW_TRACK = os.path.join(DATA_DIR, 'Storm.2016075S11087.ibtracs_all.v03r10.csv')
TEST_TRACK_GETTELMAN = os.path.join(DATA_DIR, 'gettelman_test_tracks.nc')
TEST_TRACK_EMANUEL = os.path.join(DATA_DIR, 'emanuel_test_tracks.mat')
TEST_TRACK_EMANUEL_CORR = os.path.join(DATA_DIR, 'temp_mpircp85cal_full.mat')

class TestIO(unittest.TestCase):
    """Test reading of tracks from files of different formats"""

    def test_read_processed_ibtracs_csv(self):
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)

        self.assertEqual(tc_track.data[0].time.size, 38)
        self.assertEqual(tc_track.data[0].lon[11], -39.60)
        self.assertEqual(tc_track.data[0].lat[23], 14.10)
        self.assertEqual(tc_track.data[0].time_step[7], 6)
        self.assertEqual(np.max(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.data[0].radius_max_wind), 0)
        self.assertEqual(tc_track.data[0].max_sustained_wind[21], 55)
        self.assertEqual(tc_track.data[0].central_pressure[29], 969.76880)
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
        tc_track = TCTracks()

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
        tc_track_G = TCTracks()
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

    def test_penv_pass(self):
        """ Test _set_penv method."""
        tc_track = TCTracks()
        basin = 'US'
        self.assertEqual(tc_track._set_penv(basin), 1010)
        basin = 'NA'
        self.assertEqual(tc_track._set_penv(basin), 1010)
        basin = 'SA'
        self.assertEqual(tc_track._set_penv(basin), 1010)
        basin = 'NI'
        self.assertEqual(tc_track._set_penv(basin), 1005)
        basin = 'SI'
        self.assertEqual(tc_track._set_penv(basin), 1005)
        basin = 'SP'
        self.assertEqual(tc_track._set_penv(basin), 1004)
        basin = 'WP'
        self.assertEqual(tc_track._set_penv(basin), 1005)
        basin = 'EP'
        self.assertEqual(tc_track._set_penv(basin), 1010)

    def test_get_track_pass(self):
        """ Test get_track."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        self.assertIsInstance(tc_track.get_track(), xr.Dataset)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

        tc_track_bis = TCTracks()
        tc_track_bis.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.append(tc_track_bis)
        self.assertIsInstance(tc_track.get_track(), list)
        self.assertIsInstance(tc_track.get_track('1951239N12334'), xr.Dataset)

    def test_interp_track_pass(self):
        """ Interpolate track to min_time_step. Compare to MATLAB reference."""
        tc_track = TCTracks()
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
                               1.005409300000005e+03)
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
        """ Interpolate track to min_time_step crossing lat origin """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([167.207761,  168.1     ,  168.936535,  169.728947,  170.5     ,
        171.257176,  171.946822,  172.5     ,  172.871797,  173.113396,   173.3     ,  173.496375,  173.725522,  174.      ,  174.331591,
        174.728961,  175.2     ,  175.747632,  176.354929,  177.      ,   177.66677 ,  178.362433,  179.1     ,  179.885288, -179.304661,
       -178.5     , -177.726442, -176.991938, -176.3     , -175.653595,  -175.053513, -174.5     , -173.992511, -173.527342, -173.1     ,
       -172.705991, -172.340823, -172.      ])
        tc_track.data[0].lat.values = np.array([40.196053,
        40.6     , 40.930215, 41.215674, 41.5     , 41.816354, 42.156065,  42.5     , 42.833998, 43.16377 , 43.5     , 43.847656, 44.188854,
        44.5     , 44.764269, 44.991925, 45.2     , 45.402675, 45.602707,  45.8     , 45.995402, 46.193543, 46.4     , 46.615718, 46.82312 ,
        47.      , 47.130616, 47.225088, 47.3     , 47.369224, 47.435786,  47.5     , 47.562858, 47.628064, 47.7     , 47.783047, 47.881586,
        48.      ])
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
                               1.005409300000005e+03)
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
        """ Interpolate track to min_time_step crossing lat origin """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.data[0].lon.values = np.array([167.207761,  168.1     ,  168.936535,  169.728947,  170.5     ,
        171.257176,  171.946822,  172.5     ,  172.871797,  173.113396,   173.3     ,  173.496375,  173.725522,  174.      ,  174.331591,
        174.728961,  175.2     ,  175.747632,  176.354929,  177.      ,   177.66677 ,  178.362433,  179.1     ,  179.885288, -179.304661,
       -178.5     , -177.726442, -176.991938, -176.3     , -175.653595,  -175.053513, -174.5     , -173.992511, -173.527342, -173.1     ,
       -172.705991, -172.340823, -172.      ])
        tc_track.data[0].lon.values = - tc_track.data[0].lon.values
        tc_track.data[0].lat.values = np.array([40.196053,
        40.6     , 40.930215, 41.215674, 41.5     , 41.816354, 42.156065,  42.5     , 42.833998, 43.16377 , 43.5     , 43.847656, 44.188854,
        44.5     , 44.764269, 44.991925, 45.2     , 45.402675, 45.602707,  45.8     , 45.995402, 46.193543, 46.4     , 46.615718, 46.82312 ,
        47.      , 47.130616, 47.225088, 47.3     , 47.369224, 47.435786,  47.5     , 47.562858, 47.628064, 47.7     , 47.783047, 47.881586,
        48.      ])
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
                               1.005409300000005e+03)
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


    def test_random_no_landfall_pass(self):
        """ Test calc_random_walk with decay and no historical tracks with landfall """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        with self.assertLogs('climada.hazard.tc_tracks', level='INFO') as cm:
            tc_track.calc_random_walk()
        self.assertIn('No historical track with landfall.', cm.output[1])

    def test_random_walk_ref_pass(self):
        """Test against MATLAB reference."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        ens_size=2
        tc_track.calc_random_walk(ens_size, seed=25, decay=False)

        self.assertEqual(len(tc_track.data), ens_size+1)

        self.assertFalse(tc_track.data[1].orig_event_flag)
        self.assertEqual(tc_track.data[1].name, '1951239N12334_gen1')
        self.assertEqual(tc_track.data[1].id_no, 1.951239012334010e+12)
        self.assertAlmostEqual(tc_track.data[1].lon[0].values, -25.0448138)
        self.assertAlmostEqual(tc_track.data[1].lon[1].values, -26.07400903)
        self.assertAlmostEqual(tc_track.data[1].lon[2].values, -27.09191673)
        self.assertAlmostEqual(tc_track.data[1].lon[3].values, -28.21366632)
        self.assertAlmostEqual(tc_track.data[1].lon[4].values, -29.33195465)
        self.assertAlmostEqual(tc_track.data[1].lon[8].values, -34.6016857)

        self.assertAlmostEqual(tc_track.data[1].lat[0].values, 11.96825841)
        self.assertAlmostEqual(tc_track.data[1].lat[4].values, 12.35820479)
        self.assertAlmostEqual(tc_track.data[1].lat[5].values, 12.45465)
        self.assertAlmostEqual(tc_track.data[1].lat[6].values, 12.5492937)
        self.assertAlmostEqual(tc_track.data[1].lat[7].values, 12.6333804)
        self.assertAlmostEqual(tc_track.data[1].lat[8].values, 12.71561952)

        self.assertFalse(tc_track.data[2].orig_event_flag)
        self.assertEqual(tc_track.data[2].name, '1951239N12334_gen2')
        self.assertAlmostEqual(tc_track.data[2].id_no, 1.951239012334020e+12)
        self.assertAlmostEqual(tc_track.data[2].lon[0].values, -25.47658461)
        self.assertAlmostEqual(tc_track.data[2].lon[3].values, -28.78978084)
        self.assertAlmostEqual(tc_track.data[2].lon[4].values, -29.9568406)
        self.assertAlmostEqual(tc_track.data[2].lon[8].values, -35.30222604)

        self.assertAlmostEqual(tc_track.data[2].lat[0].values, 11.82886685)
        self.assertAlmostEqual(tc_track.data[2].lat[6].values, 12.26400422)
        self.assertAlmostEqual(tc_track.data[2].lat[7].values, 12.3454308)
        self.assertAlmostEqual(tc_track.data[2].lat[8].values, 12.42745488)

    def test_random_walk_decay_pass(self):
        """Test land decay is called from calc_random_walk."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        ens_size=2
        with self.assertLogs('climada.hazard.tc_tracks', level='DEBUG') as cm:
            tc_track.calc_random_walk(ens_size, seed=25, decay=True)
        self.assertIn('No historical track of category Tropical Depression with landfall.', cm.output[1])
        self.assertIn('Decay parameters from category Hurrican Cat. 4 taken.', cm.output[2])
        self.assertIn('No historical track of category Hurrican Cat. 1 with landfall.', cm.output[3])
        self.assertIn('Decay parameters from category Hurrican Cat. 4 taken.', cm.output[4])
        self.assertIn('No historical track of category Hurrican Cat. 3 with landfall. Decay parameters from category Hurrican Cat. 4 taken.', cm.output[5])
        self.assertIn('No historical track of category Hurrican Cat. 5 with landfall.', cm.output[6])

    def test_calc_decay_no_landfall_pass(self):
        """ Test _calc_land_decay with no historical tracks with landfall """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        land_geom = tc._calc_land_geom(tc_track.data)
        tc._track_land_params(tc_track.data[0], land_geom)
        with self.assertLogs('climada.hazard.tc_tracks', level='INFO') as cm:
            tc_track._calc_land_decay(land_geom)
        self.assertIn('No historical track with landfall.', cm.output[0])

    def test_calc_land_decay_pass(self):
        """ Test _calc_land_decay with environmental pressure function."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        land_geom = tc._calc_land_geom(tc_track.data)
        tc._track_land_params(tc_track.data[0], land_geom)
        v_rel, p_rel = tc_track._calc_land_decay(land_geom)

        self.assertEqual(7, len(v_rel))
        for i, val in enumerate(v_rel.values()):
            self.assertAlmostEqual(val, 0.0038894834)
            self.assertTrue(i+1 in v_rel.keys())

        self.assertEqual(7, len(p_rel))
        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.0598491)
            self.assertAlmostEqual(val[1], 0.0041949237)
            self.assertTrue(i+1 in p_rel.keys())

    def test_decay_values_andrew_pass(self):
        """ Test _decay_values with central pressure function."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        s_rel = False
        land_geom = tc._calc_land_geom(tc_track.data)
        tc._track_land_params(tc_track.data[0], land_geom)
        v_lf, p_lf, x_val = tc._decay_values(tc_track.data[0], land_geom, s_rel)

        ss_category = 6
        s_cell_1 = 1*[1.0149413347244263]
        s_cell_2 = 8*[1.047120451927185]
        s_cell = s_cell_1 + s_cell_2
        p_vs_lf_time_relative = [1.0149413020277482, 1.018848167539267, 1.037696335078534, \
                                 1.0418848167539267, 1.043979057591623, 1.0450261780104713, \
                                 1.0460732984293193, 1.0471204188481675, 1.0471204188481675]

        self.assertEqual(list(p_lf.keys()), [ss_category])
        self.assertEqual(p_lf[ss_category][0], array.array('f', s_cell))
        self.assertEqual(p_lf[ss_category][1], array.array('f', p_vs_lf_time_relative))

        v_vs_lf_time_relative = [0.8846153846153846, 0.6666666666666666, 0.4166666666666667, \
                                 0.2916666666666667, 0.250000000000000, 0.250000000000000, \
                                 0.20833333333333334, 0.16666666666666666, 0.16666666666666666]
        self.assertEqual(list(v_lf.keys()), [ss_category])
        self.assertEqual(v_lf[ss_category], array.array('f', v_vs_lf_time_relative))

        x_val_ref = np.array([95.9512939453125, 53.624916076660156, 143.09530639648438,
                              225.0262908935547, 312.5832824707031, 427.43109130859375,
                              570.1857299804688, 750.3827514648438, 1020.5431518554688])
        self.assertEqual(list(x_val.keys()), [ss_category])
        self.assertTrue(np.allclose(x_val[ss_category], x_val_ref))

    def test_dist_since_lf_pass(self):
        """ Test _dist_since_lf for andrew tropical cyclone."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        track = tc_track.get_track()
        track['on_land'] = ('time', coord_on_land(track.lat.values,
             track.lon.values))
        track['dist_since_lf'] = ('time', tc._dist_since_lf(track))

        self.assertTrue(np.all(np.isnan(track.dist_since_lf.values[track.on_land == False])))
        self.assertEqual(track.dist_since_lf.values[track.on_land == False].size, 38)

        self.assertTrue(track.dist_since_lf.values[-1] >
                        dist_to_coast(track.lat.values[-1], track.lon.values[-1])/1000)
        self.assertEqual(1020.5431562223974, track['dist_since_lf'].values[-1])

        # check distances on land always increase, in second landfall
        dist_on_land = track.dist_since_lf.values[track.on_land]
        self.assertTrue(np.all(np.diff(dist_on_land)[1:] > 0))

    def test_calc_orig_lf(self):
        """ Test _calc_orig_lf for andrew tropical cyclone."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        track = tc_track.get_track()
        track['on_land'] = ('time', coord_on_land(track.lat.values,
             track.lon.values))
        sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0]
        orig_lf = tc._calc_orig_lf(track, sea_land_idx)

        self.assertEqual(orig_lf.shape, (sea_land_idx.size, 2))
        self.assertTrue(np.array_equal(orig_lf[0], np.array([25.5, -80.25])))
        self.assertTrue(np.array_equal(orig_lf[1], np.array([29.65, -91.5])))

    def test_decay_calc_coeff(self):
        """ Test _decay_calc_coeff against MATLAB"""
        x_val = {6: np.array([53.57314960249573, 142.97903059281566,
            224.76733726289183,  312.14621544207563,  426.6757021862584,
            568.9358305779094,  748.3713215157885, 1016.9904230811956])}

        v_lf = {6: np.array([0.6666666666666666, 0.4166666666666667, \
                             0.2916666666666667, 0.250000000000000,
                             0.250000000000000, 0.20833333333333334, \
                             0.16666666666666666, 0.16666666666666666])}

        p_lf = {6: (8*[1.0471204188481675], np.array([1.018848167539267, 1.037696335078534, \
                    1.0418848167539267, 1.043979057591623, 1.0450261780104713, 1.0460732984293193,
                    1.0471204188481675, 1.0471204188481675]))}

        v_rel, p_rel = tc._decay_calc_coeff(x_val, v_lf, p_lf)

        for i, val in enumerate(v_rel.values()):
            self.assertAlmostEqual(val, 0.004222091151737)
            self.assertTrue(i+1 in v_rel.keys())

        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.047120418848168)
            self.assertAlmostEqual(val[1], 0.008871782287614)
            self.assertTrue(i+1 in v_rel.keys())

    def test_apply_decay_no_landfall_pass(self):
        """ Test _apply_land_decay with no historical tracks with landfall """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        land_geom = tc._calc_land_geom(tc_track.data)
        tc._track_land_params(tc_track.data[0], land_geom)
        tc_track.data[0]['orig_event_flag']=False
        tc_ref = tc_track.data[0].copy()
        tc_track._apply_land_decay(dict(), dict(), land_geom)

        self.assertTrue(np.array_equal(tc_track.data[0].max_sustained_wind.values, tc_ref.max_sustained_wind.values))
        self.assertTrue(np.array_equal(tc_track.data[0].central_pressure.values, tc_ref.central_pressure.values))
        self.assertTrue(np.array_equal(tc_track.data[0].environmental_pressure.values, tc_ref.environmental_pressure.values))
        self.assertTrue(np.all(np.isnan(tc_track.data[0].dist_since_lf.values)))

    def test_apply_decay_pass(self):
        """ Test _apply_land_decay against MATLAB reference. """
        v_rel = { 6: 0.0038950967656296597,
                  1: 0.0038950967656296597,
                  2: 0.0038950967656296597,
                  3: 0.0038950967656296597,
                  4: 0.0038950967656296597,
                  5: 0.0038950967656296597,
                  7: 0.0038950967656296597}

        p_rel = {6: (1.0499941, 0.007978940084158488),
                 1: (1.0499941, 0.007978940084158488),
                 2: (1.0499941, 0.007978940084158488),
                 3: (1.0499941, 0.007978940084158488),
                 4: (1.0499941, 0.007978940084158488),
                 5: (1.0499941, 0.007978940084158488),
                 7: (1.0499941, 0.007978940084158488)}

        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        tc_track.data[0]['orig_event_flag'] = False
        land_geom = tc._calc_land_geom(tc_track.data)
        tc._track_land_params(tc_track.data[0], land_geom)
        tc_track._apply_land_decay(v_rel, p_rel, land_geom, s_rel=True, check_plot=False)

        p_ref = np.array([1.010000000000000, 1.009000000000000, 1.008000000000000,
                          1.006000000000000, 1.003000000000000, 1.002000000000000,
                          1.001000000000000, 1.000000000000000, 1.000000000000000,
                          1.001000000000000, 1.002000000000000, 1.005000000000000,
                          1.007000000000000, 1.010000000000000, 1.010000000000000,
                          1.010000000000000, 1.010000000000000, 1.010000000000000,
                          1.010000000000000, 1.007000000000000, 1.004000000000000,
                          1.000000000000000, 0.994000000000000, 0.981000000000000,
                          0.969000000000000, 0.961000000000000, 0.947000000000000,
                          0.933000000000000, 0.922000000000000, 0.930000000000000,
                          0.937000000000000, 0.951000000000000, 0.947000000000000,
                          0.943000000000000, 0.948000000000000, 0.946000000000000,
                          0.941000000000000, 0.937000000000000, 0.955000000000000,
                          0.9741457117, 0.99244068917, 1.00086729492, 1.00545853355,
                          1.00818354609, 1.00941850023, 1.00986192053, 1.00998400565])*1e3

        self.assertTrue(np.allclose(p_ref, tc_track.data[0].central_pressure.values))

        v_ref = np.array([0.250000000000000, 0.300000000000000, 0.300000000000000,
                          0.350000000000000, 0.350000000000000, 0.400000000000000,
                          0.450000000000000, 0.450000000000000, 0.450000000000000,
                          0.450000000000000, 0.450000000000000, 0.450000000000000,
                          0.450000000000000, 0.400000000000000, 0.400000000000000,
                          0.400000000000000, 0.400000000000000, 0.450000000000000,
                          0.450000000000000, 0.500000000000000, 0.500000000000000,
                          0.550000000000000, 0.650000000000000, 0.800000000000000,
                          0.950000000000000, 1.100000000000000, 1.300000000000000,
                          1.450000000000000, 1.500000000000000, 1.250000000000000,
                          1.300000000000000, 1.150000000000000, 1.150000000000000,
                          1.150000000000000, 1.150000000000000, 1.200000000000000,
                          1.250000000000000, 1.250000000000000, 1.200000000000000,
                          0.9737967353, 0.687255951, 0.4994850556, 0.3551480462,
                          0.2270548036, 0.1302099557, 0.0645385918, 0.0225325851])*1e2

        self.assertTrue(np.allclose(v_ref, tc_track.data[0].max_sustained_wind.values))

        cat_ref = tc.set_category(tc_track.data[0].max_sustained_wind.values, tc_track.data[0].max_sustained_wind_unit)
        self.assertEqual(cat_ref, tc_track.data[0].category)

    def test_func_decay_p_pass(self):
        """ Test decay function for pressure with its inverse."""
        s_coef = 1.05
        b_coef = 0.04
        x_val = np.arange(0, 100, 10)
        res = tc._decay_p_function(s_coef, b_coef, x_val)
        b_coef_res = tc._solve_decay_p_function(s_coef, res, x_val)

        self.assertTrue(np.allclose(b_coef_res[1:], np.ones((x_val.size-1,))*b_coef))
        self.assertTrue(np.isnan(b_coef_res[0]))

    def test_func_decay_v_pass(self):
        """ Test decay function for wind with its inverse."""
        a_coef = 0.04
        x_val = np.arange(0, 100, 10)
        res = tc._decay_v_function(a_coef, x_val)
        a_coef_res = tc._solve_decay_v_function(res, x_val)

        self.assertTrue(np.allclose(a_coef_res[1:], np.ones((x_val.size-1,))*a_coef))
        self.assertTrue(np.isnan(a_coef_res[0]))

    def test_decay_ps_value(self):
        """Test the calculation of S in pressure decay."""
        on_land_idx = 5
        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', np.arange(10))
        tr_ds['central_pressure'] = ('time', np.arange(10, 20))
        tr_ds['environmental_pressure'] = ('time', np.arange(20, 30))
        tr_ds['on_land'] = ('time', np.zeros((10,)).astype(bool))
        tr_ds.on_land[on_land_idx] = True
        p_landfall = 100

        res = tc._calc_decay_ps_value(tr_ds, p_landfall, on_land_idx, s_rel=True)
        self.assertEqual(res, float(tr_ds.environmental_pressure[on_land_idx]/p_landfall))
        res = tc._calc_decay_ps_value(tr_ds, p_landfall, on_land_idx, s_rel=False)
        self.assertEqual(res, float(tr_ds.central_pressure[on_land_idx]/p_landfall))

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

        max_sus_wind = np.array([28.769475, 34.52337, 40.277265,
                                 46.03116, 51.785055, 51.785055, 51.785055,
                                 51.785055, 40.277265, 28.769475])
        max_sus_wind_unit = 'mph'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([12.86111437, 12.86111437, 12.86111437,
                                 15.43333724, 15.43333724, 15.43333724,
                                 15.43333724, 15.43333724, 12.86111437,
                                 12.86111437, 10.2888915])
        max_sus_wind_unit = 'm/s'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([148.16, 166.68, 185.2, 212.98, 222.24, 231.5,
                                 240.76, 222.24, 203.72, 148.16, 138.9, 148.16,
                                 120.38])
        max_sus_wind_unit = 'km/h'
        cat = tc.set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

    def test_missing_pres_pass(self):
        """Test central pressure function."""
        cen_pres = np.array([-999, -999, -999, -999, -999, -999, -999, -999,
                             -999, 992, -999, -999, 993, -999, -999, 1004])
        v_max = np.array([45, 50, 50, 55, 60, 65, 70, 80, 75, 70, 70, 70, 70,
                          65, 55, 45])
        lat = np.array([13.8, 13.9, 14, 14.1, 14.1, 14.1, 14.1, 14.2, 14.2,
                        14.3, 14.4, 14.6, 14.8, 15, 15.1, 15.1])
        lon = np.array([-51.1, -52.8, -54.4, -56, -57.3, -58.4, -59.7, -61.1,
                        -62.7, -64.3, -65.8, -67.4, -69.4, -71.4, -73, -74.2])
        out_pres = tc._missing_pressure(cen_pres, v_max, lat, lon)

        ref_res = np.array([989.7085, 985.6725, 985.7236, 981.6847, 977.6324,
                            973.5743, 969.522, 961.3873, 965.5237, 969.6648,
                            969.713, 969.7688, 969.8362, 973.9936, 982.2247,
                            990.4395])
        np.testing.assert_array_almost_equal(ref_res, out_pres)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
