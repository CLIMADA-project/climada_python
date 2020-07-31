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

Test tc_tracks_synth module.
"""

import array
import numpy as np
import os
import unittest
import xarray as xr

import climada.hazard.tc_tracks as tc
import climada.hazard.tc_tracks_synth as tc_synth
import climada.util.coordinates
from climada.util.constants import TC_ANDREW_FL

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")

class TestDecay(unittest.TestCase):
    def test_apply_decay_no_landfall_pass(self):
        """Test _apply_land_decay with no historical tracks with landfall"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        tc_track.data[0]['orig_event_flag'] = False
        tc_ref = tc_track.data[0].copy()
        tc_synth._apply_land_decay(tc_track.data, dict(), dict(), land_geom)

        self.assertTrue(np.allclose(tc_track.data[0].max_sustained_wind.values,
                                    tc_ref.max_sustained_wind.values))
        self.assertTrue(np.allclose(tc_track.data[0].central_pressure.values,
                                    tc_ref.central_pressure.values))
        self.assertTrue(np.allclose(tc_track.data[0].environmental_pressure.values,
                                    tc_ref.environmental_pressure.values))
        self.assertTrue(np.all(np.isnan(tc_track.data[0].dist_since_lf.values)))

    def test_apply_decay_pass(self):
        """Test _apply_land_decay against MATLAB reference."""
        v_rel = {
            6: 0.0038950967656296597,
            1: 0.0038950967656296597,
            2: 0.0038950967656296597,
            3: 0.0038950967656296597,
            4: 0.0038950967656296597,
            5: 0.0038950967656296597,
            7: 0.0038950967656296597
        }

        p_rel = {
            6: (1.0499941, 0.007978940084158488),
            1: (1.0499941, 0.007978940084158488),
            2: (1.0499941, 0.007978940084158488),
            3: (1.0499941, 0.007978940084158488),
            4: (1.0499941, 0.007978940084158488),
            5: (1.0499941, 0.007978940084158488),
            7: (1.0499941, 0.007978940084158488)
        }

        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        tc_track.data[0]['orig_event_flag'] = False
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        tc_synth._apply_land_decay(tc_track.data, v_rel, p_rel, land_geom,
                                   s_rel=True, check_plot=False)

        p_ref = np.array([
            1.010000000000000, 1.009000000000000, 1.008000000000000,
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
            1.00818354609, 1.00941850023, 1.00986192053, 1.00998400565
        ]) * 1e3

        self.assertTrue(np.allclose(p_ref, tc_track.data[0].central_pressure.values))

        v_ref = np.array([
            0.250000000000000, 0.300000000000000, 0.300000000000000,
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
            0.9737967353, 0.687255951, 0.4994850556, 0.3551480462, 0.2270548036,
            0.1302099557, 0.0645385918, 0.0225325851
        ]) * 1e2

        self.assertTrue(np.allclose(v_ref, tc_track.data[0].max_sustained_wind.values))

        cat_ref = tc.set_category(tc_track.data[0].max_sustained_wind.values,
                                  tc_track.data[0].max_sustained_wind_unit)
        self.assertEqual(cat_ref, tc_track.data[0].category)

    def test_func_decay_p_pass(self):
        """Test decay function for pressure with its inverse."""
        s_coef = 1.05
        b_coef = 0.04
        x_val = np.arange(0, 100, 10)
        res = tc_synth._decay_p_function(s_coef, b_coef, x_val)
        b_coef_res = tc_synth._solve_decay_p_function(s_coef, res, x_val)

        self.assertTrue(np.allclose(b_coef_res[1:], np.ones((x_val.size - 1,)) * b_coef))
        self.assertTrue(np.isnan(b_coef_res[0]))

    def test_func_decay_v_pass(self):
        """Test decay function for wind with its inverse."""
        a_coef = 0.04
        x_val = np.arange(0, 100, 10)
        res = tc_synth._decay_v_function(a_coef, x_val)
        a_coef_res = tc_synth._solve_decay_v_function(res, x_val)

        self.assertTrue(np.allclose(a_coef_res[1:], np.ones((x_val.size - 1,)) * a_coef))
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

        res = tc_synth._calc_decay_ps_value(tr_ds, p_landfall, on_land_idx, s_rel=True)
        self.assertEqual(res, float(tr_ds.environmental_pressure[on_land_idx] / p_landfall))
        res = tc_synth._calc_decay_ps_value(tr_ds, p_landfall, on_land_idx, s_rel=False)
        self.assertEqual(res, float(tr_ds.central_pressure[on_land_idx] / p_landfall))

    def test_calc_decay_no_landfall_pass(self):
        """Test _calc_land_decay with no historical tracks with landfall"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='INFO') as cm:
            tc_synth._calc_land_decay(tc_track.data, land_geom)
        self.assertIn('No historical track with landfall.', cm.output[0])

    def test_calc_land_decay_pass(self):
        """Test _calc_land_decay with environmental pressure function."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        v_rel, p_rel = tc_synth._calc_land_decay(tc_track.data, land_geom)

        self.assertEqual(7, len(v_rel))
        for i, val in enumerate(v_rel.values()):
            self.assertAlmostEqual(val, 0.0038894834)
            self.assertTrue(i + 1 in v_rel.keys())

        self.assertEqual(7, len(p_rel))
        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.0598491)
            self.assertAlmostEqual(val[1], 0.0041949237)
            self.assertTrue(i + 1 in p_rel.keys())

    def test_decay_values_andrew_pass(self):
        """Test _decay_values with central pressure function."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        s_rel = False
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        v_lf, p_lf, x_val = tc_synth._decay_values(tc_track.data[0], land_geom, s_rel)

        ss_category = 6
        s_cell_1 = 1 * [1.0149413347244263]
        s_cell_2 = 8 * [1.047120451927185]
        s_cell = s_cell_1 + s_cell_2
        p_vs_lf_time_relative = [
            1.0149413020277482, 1.018848167539267, 1.037696335078534,
            1.0418848167539267, 1.043979057591623, 1.0450261780104713,
            1.0460732984293193, 1.0471204188481675, 1.0471204188481675
        ]

        self.assertEqual(list(p_lf.keys()), [ss_category])
        self.assertEqual(p_lf[ss_category][0], array.array('f', s_cell))
        self.assertEqual(p_lf[ss_category][1], array.array('f', p_vs_lf_time_relative))

        v_vs_lf_time_relative = [
            0.8846153846153846, 0.6666666666666666, 0.4166666666666667,
            0.2916666666666667, 0.250000000000000, 0.250000000000000,
            0.20833333333333334, 0.16666666666666666, 0.16666666666666666
        ]
        self.assertEqual(list(v_lf.keys()), [ss_category])
        self.assertEqual(v_lf[ss_category], array.array('f', v_vs_lf_time_relative))

        x_val_ref = np.array([
            95.9512939453125, 53.624916076660156, 143.09530639648438,
            225.0262908935547, 312.5832824707031, 427.43109130859375,
            570.1857299804688, 750.3827514648438, 1020.5431518554688
        ])
        self.assertEqual(list(x_val.keys()), [ss_category])
        self.assertTrue(np.allclose(x_val[ss_category], x_val_ref))

    def test_decay_calc_coeff(self):
        """Test _decay_calc_coeff against MATLAB"""
        x_val = {
            6: np.array([
                53.57314960249573, 142.97903059281566, 224.76733726289183,
                312.14621544207563, 426.6757021862584, 568.9358305779094,
                748.3713215157885, 1016.9904230811956
            ])
        }

        v_lf = {
            6: np.array([
                0.6666666666666666, 0.4166666666666667, 0.2916666666666667,
                0.250000000000000, 0.250000000000000, 0.20833333333333334,
                0.16666666666666666, 0.16666666666666666
            ])
        }

        p_lf = {
            6: (8 * [1.0471204188481675],
                np.array([
                    1.018848167539267, 1.037696335078534, 1.0418848167539267,
                    1.043979057591623, 1.0450261780104713, 1.0460732984293193,
                    1.0471204188481675, 1.0471204188481675
                ])
            )
        }

        v_rel, p_rel = tc_synth._decay_calc_coeff(x_val, v_lf, p_lf)

        for i, val in enumerate(v_rel.values()):
            self.assertAlmostEqual(val, 0.004222091151737)
            self.assertTrue(i + 1 in v_rel.keys())

        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.047120418848168)
            self.assertAlmostEqual(val[1], 0.008871782287614)
            self.assertTrue(i + 1 in v_rel.keys())

    def test_wrong_decay_pass(self):
        """Test decay not implemented when coefficient < 1"""
        track = tc.TCTracks()
        track.read_ibtracs_netcdf(provider='usa', storm_id='1975178N28281')

        track_gen = track.data[0]
        track_gen['lat'] = np.array([
            28.20340431, 28.7915261, 29.38642458, 29.97836984, 30.56844404,
            31.16265292, 31.74820301, 32.34449825, 32.92261894, 33.47430891,
            34.01492525, 34.56789399, 35.08810845, 35.55965893, 35.94835174,
            36.29355848, 36.45379561, 36.32473812, 36.07552209, 35.92224784,
            35.84144186, 35.78298537, 35.86090718, 36.02440372, 36.37555559,
            37.06207765, 37.73197352, 37.97524273, 38.05560287, 38.21901208,
            38.31486156, 38.30813367, 38.28481808, 38.28410366, 38.25894812,
            38.20583372, 38.22741099, 38.39970022, 38.68367797, 39.08329904,
            39.41434629, 39.424984, 39.31327716, 39.30336335, 39.31714429,
            39.27031932, 39.30848775, 39.48759833, 39.73326595, 39.96187967,
            40.26954226, 40.76882202, 41.40398607, 41.93809726, 42.60395785,
            43.57074792, 44.63816143, 45.61450458, 46.68528511, 47.89209365,
            49.15580502
        ])
        track_gen['lon'] = np.array([
            -79.20514075, -79.25243311, -79.28393082, -79.32324646,
            -79.36668585, -79.41495519, -79.45198688, -79.40580325,
            -79.34965443, -79.36938122, -79.30294825, -79.06809546,
            -78.70281969, -78.29418936, -77.82170609, -77.30034709,
            -76.79004969, -76.37038827, -75.98641014, -75.58383356,
            -75.18310414, -74.7974524, -74.3797645, -73.86393572, -73.37910948,
            -73.01059003, -72.77051313, -72.68011328, -72.66864779,
            -72.62579773, -72.56307717, -72.46607618, -72.35871353,
            -72.31120649, -72.15537583, -71.75577051, -71.25287498,
            -70.75527907, -70.34788946, -70.17518421, -70.04446577,
            -69.76582749, -69.44372386, -69.15881376, -68.84351922,
            -68.47890287, -68.04184565, -67.53541437, -66.94008642,
            -66.25596075, -65.53496635, -64.83491802, -64.12962685,
            -63.54118808, -62.72934383, -61.34915091, -59.72580755,
            -58.24404252, -56.71972992, -55.0809336, -53.31524758
        ])

        v_rel = {
            3: 0.002249541544102336,
            1: 0.00046889526284203036,
            4: 0.002649273787364977,
            2: 0.0016426186150461349,
            5: 0.00246400811445618,
            7: 0.0030442198547309075,
            6: 0.002346537842810565,
        }
        p_rel = {
            3: (1.028420239620591, 0.003174733355067952),
            1: (1.0046803184177564, 0.0007997633912500546),
            4: (1.0498749735343516, 0.0034665588904747515),
            2: (1.0140127424090262, 0.002131858515233042),
            5: (1.0619445995372885, 0.003467268426139696),
            7: (1.0894914184297835, 0.004315034379018768),
            6: (1.0714354641894077, 0.002783787561718677),
        }
        track_gen.attrs['orig_event_flag'] = False

        cp_ref = np.array([1012., 1012.])
        single_track = tc.TCTracks()
        single_track.data = [track_gen]
        extent = single_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        track_res = tc_synth._apply_decay_coeffs(track_gen, v_rel, p_rel, land_geom, True)
        self.assertTrue(np.array_equal(cp_ref, track_res.central_pressure[9:11]))

class TestSynth(unittest.TestCase):
    def test_random_no_landfall_pass(self):
        """Test calc_random_walk with decay and no historical tracks with landfall"""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='INFO') as cm:
            tc_track.calc_random_walk()
        self.assertIn('No historical track with landfall.', cm.output[1])

    def test_random_walk_ref_pass(self):
        """Test against MATLAB reference."""
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        ens_size = 2
        tc_track.calc_random_walk(ens_size=ens_size, seed=25, decay=False)

        self.assertEqual(len(tc_track.data), ens_size + 1)

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
        tc_track = tc.TCTracks()
        tc_track.read_processed_ibtracs_csv(TC_ANDREW_FL)
        ens_size = 2
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='DEBUG') as cm:
            tc_track.calc_random_walk(ens_size=ens_size, seed=25, decay=True)
        self.assertIn('No historical track of category Tropical Depression '
                      'with landfall.', cm.output[1])
        self.assertIn('Decay parameters from category Hurricane Cat. 4 taken.',
                      cm.output[2])
        self.assertIn('No historical track of category Hurricane Cat. 1 with '
                      'landfall.', cm.output[3])
        self.assertIn('Decay parameters from category Hurricane Cat. 4 taken.',
                      cm.output[4])
        self.assertIn('No historical track of category Hurricane Cat. 3 with '
                      'landfall. Decay parameters from category Hurricane Cat. '
                      '4 taken.', cm.output[5])
        self.assertIn('No historical track of category Hurricane Cat. 5 with '
                      'landfall.', cm.output[6])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDecay)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSynth))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
