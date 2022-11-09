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

Test tc_tracks_synth module.
"""

import array
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

import climada.hazard.tc_tracks as tc
import climada.hazard.tc_tracks_synth as tc_synth
import climada.util.coordinates
from climada.util.constants import TC_ANDREW_FL

DATA_DIR = Path(__file__).parent.joinpath('data')
TEST_TRACK = DATA_DIR.joinpath("trac_brb_test.csv")
TEST_TRACK_SHORT = DATA_DIR.joinpath("trac_short_test.csv")
TEST_TRACK_DECAY_END_OCEAN = DATA_DIR.joinpath('1997018S11059_gen3.nc')
TEST_TRACK_DECAY_END_OCEAN_HIST = DATA_DIR.joinpath('1997018S11059.nc')
TEST_TRACK_DECAY_PENV_GT_PCEN = DATA_DIR.joinpath('1988021S12080_gen2.nc')
TEST_TRACK_DECAY_PENV_GT_PCEN_HIST = DATA_DIR.joinpath('1988021S12080.nc')

class TestDecay(unittest.TestCase):
    def test_apply_decay_no_landfall_pass(self):
        """Test _apply_land_decay with no historical tracks with landfall"""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
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
            4: 0.0038950967656296597,
            -1: 0.0038950967656296597,
            0: 0.0038950967656296597,
            1: 0.0038950967656296597,
            2: 0.0038950967656296597,
            3: 0.0038950967656296597,
            5: 0.0038950967656296597
        }

        p_rel = {
            4: (1.0499941, 0.007978940084158488),
            -1: (1.0499941, 0.007978940084158488),
            0: (1.0499941, 0.007978940084158488),
            1: (1.0499941, 0.007978940084158488),
            2: (1.0499941, 0.007978940084158488),
            3: (1.0499941, 0.007978940084158488),
            5: (1.0499941, 0.007978940084158488)
        }

        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
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
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        expected_warning = 'only %s historical tracks were provided. ' % len(tc_track.data)
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='INFO') as cm:
            tc_synth._calc_land_decay(tc_track.data, land_geom)
        self.assertIn(expected_warning, cm.output[0])
        self.assertIn('No historical track with landfall.', cm.output[1])

    def test_calc_land_decay_pass(self):
        """Test _calc_land_decay with environmental pressure function."""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        v_rel, p_rel = tc_synth._calc_land_decay(tc_track.data, land_geom)

        self.assertEqual(7, len(v_rel))
        for i, val in enumerate(v_rel.values()):
            self.assertAlmostEqual(val, 0.0038894834)
            self.assertTrue(i - 1 in v_rel.keys())

        self.assertEqual(7, len(p_rel))
        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.0598491)
            self.assertAlmostEqual(val[1], 0.0041949237)
            self.assertTrue(i - 1 in p_rel.keys())

    def test_decay_values_andrew_pass(self):
        """Test _decay_values with central pressure function."""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
        s_rel = False
        extent = tc_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tc.track_land_params(tc_track.data[0], land_geom)
        v_lf, p_lf, x_val = tc_synth._decay_values(tc_track.data[0], land_geom, s_rel)

        ss_category = 4
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
            4: np.array([
                53.57314960249573, 142.97903059281566, 224.76733726289183,
                312.14621544207563, 426.6757021862584, 568.9358305779094,
                748.3713215157885, 1016.9904230811956
            ])
        }

        v_lf = {
            4: np.array([
                0.6666666666666666, 0.4166666666666667, 0.2916666666666667,
                0.250000000000000, 0.250000000000000, 0.20833333333333334,
                0.16666666666666666, 0.16666666666666666
            ])
        }

        p_lf = {
            4: (8 * [1.0471204188481675],
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
            self.assertTrue(i - 1 in v_rel.keys())

        for i, val in enumerate(p_rel.values()):
            self.assertAlmostEqual(val[0], 1.047120418848168)
            self.assertAlmostEqual(val[1], 0.008871782287614)
            self.assertTrue(i - 1 in v_rel.keys())

    def test_wrong_decay_pass(self):
        """Test decay not implemented when coefficient < 1"""
        track = tc.TCTracks.from_ibtracs_netcdf(provider='usa', storm_id='1975178N28281')

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
            1: 0.002249541544102336,
            -1: 0.00046889526284203036,
            2: 0.002649273787364977,
            0: 0.0016426186150461349,
            3: 0.00246400811445618,
            5: 0.0030442198547309075,
            4: 0.002346537842810565,
        }
        p_rel = {
            1: (1.028420239620591, 0.003174733355067952),
            -1: (1.0046803184177564, 0.0007997633912500546),
            2: (1.0498749735343516, 0.0034665588904747515),
            0: (1.0140127424090262, 0.002131858515233042),
            3: (1.0619445995372885, 0.003467268426139696),
            5: (1.0894914184297835, 0.004315034379018768),
            4: (1.0714354641894077, 0.002783787561718677),
        }
        track_gen.attrs['orig_event_flag'] = False

        cp_ref = np.array([1012., 1012.])
        single_track = tc.TCTracks([track_gen])
        extent = single_track.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        track_res = tc_synth._apply_decay_coeffs(track_gen, v_rel, p_rel, land_geom, True)
        self.assertTrue(np.array_equal(cp_ref, track_res.central_pressure[9:11]))

    def test_decay_end_ocean(self):
        """Test decay is applied after landfall if the track ends over the ocean"""
        # this track was generated without applying landfall decay
        # (i.e. with decay=False in tc_synth.calc_perturbed_trajectories)
        tracks_synth_nodecay_example = tc.TCTracks.from_netcdf(TEST_TRACK_DECAY_END_OCEAN)

        # apply landfall decay
        extent = tracks_synth_nodecay_example.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tracks_synth_nodecay_example.data = tc_synth._apply_land_decay(
            tracks_synth_nodecay_example.data,
            tc_synth.LANDFALL_DECAY_V,
            tc_synth.LANDFALL_DECAY_P,
            land_geom)
        track = tracks_synth_nodecay_example.data[0]

        # read its corresponding historical track
        track_hist = tc.TCTracks.from_netcdf(TEST_TRACK_DECAY_END_OCEAN_HIST)
        track_hist = track_hist.data[0]

        # Part 1: is landfall applied after going back to the ocean?
        # get that last strip over the ocean
        lf_idx = tc._get_landfall_idx(track)
        last_lf_idx = lf_idx[-1][1]
        # only suitable if track ends over the ocean
        self.assertTrue(last_lf_idx < track.time.size-2,
                         'This test should be re-written, data not suitable')
        # check pressure and wind values
        p_hist_end = track_hist.central_pressure.values[last_lf_idx:]
        p_synth_end = track.central_pressure.values[last_lf_idx:]
        self.assertTrue(np.all(p_synth_end > p_hist_end))
        v_hist_end = track_hist.max_sustained_wind.values[last_lf_idx:]
        v_synth_end = track.max_sustained_wind.values[last_lf_idx:]
        self.assertTrue(np.all(v_synth_end < v_hist_end))

        # Part 2: is landfall applied in all landfalls?
        p_hist_lf = np.concatenate([track_hist.central_pressure.values[lfs:lfe]
                                    for lfs,lfe in zip(*lf_idx)])
        p_synth_lf = np.concatenate([track.central_pressure.values[lfs:lfe]
                                     for lfs,lfe in zip(*lf_idx)])
        v_hist_lf = np.concatenate([track_hist.max_sustained_wind.values[lfs:lfe]
                                    for lfs,lfe in zip(*lf_idx)])
        v_synth_lf = np.concatenate([track.max_sustained_wind.values[lfs:lfe]
                                     for lfs,lfe in zip(*lf_idx)])
        self.assertTrue(np.all(p_synth_lf > p_hist_lf))
        self.assertTrue(np.all(v_synth_lf < v_hist_lf))
        self.assertTrue(np.all(track.central_pressure.values <= track.environmental_pressure.values))

    def test_decay_penv_gt_pcen(self):
        """Test decay is applied if penv at end of landfall < pcen just before landfall"""
        # this track was generated without applying landfall decay
        # (i.e. with decay=False in tc_synth.calc_perturbed_trajectories)
        tracks_synth_nodecay_example = tc.TCTracks.from_netcdf(TEST_TRACK_DECAY_PENV_GT_PCEN)

        # apply landfall decay
        extent = tracks_synth_nodecay_example.get_extent()
        land_geom = climada.util.coordinates.get_land_geometry(
            extent=extent, resolution=10
        )
        tracks_synth_nodecay_example.data = tc_synth._apply_land_decay(
            tracks_synth_nodecay_example.data,
            tc_synth.LANDFALL_DECAY_V,
            tc_synth.LANDFALL_DECAY_P,
            land_geom)
        track = tracks_synth_nodecay_example.data[0]

        # read its corresponding historical track
        track_hist = tc.TCTracks.from_netcdf(TEST_TRACK_DECAY_PENV_GT_PCEN_HIST)
        track_hist = track_hist.data[0]

        # Part 1: is landfall applied after going back to the ocean?
        # get that last strip over the ocean
        lf_idx = tc._get_landfall_idx(track)
        start_lf_idx, end_lf_idx = lf_idx[0][0], lf_idx[1][0]

        # check pressure and wind values
        p_hist_end = track_hist.central_pressure.values[end_lf_idx:]
        p_synth_end = track.central_pressure.values[end_lf_idx:]
        self.assertTrue(np.all(p_synth_end > p_hist_end))
        v_hist_end = track_hist.max_sustained_wind.values[end_lf_idx:]
        v_synth_end = track.max_sustained_wind.values[end_lf_idx:]
        self.assertTrue(np.all(v_synth_end < v_hist_end))

        # Part 2: is landfall applied in all landfalls?

        # central pressure
        p_hist_lf = track_hist.central_pressure.values[start_lf_idx:end_lf_idx]
        p_synth_lf = track.central_pressure.values[start_lf_idx:end_lf_idx]
        # central pressure should be higher in synth than hist; unless it was set to p_env
        self.assertTrue(np.all(
            np.logical_or(p_synth_lf > p_hist_lf,
                          p_synth_lf == track.environmental_pressure.values[start_lf_idx:end_lf_idx])
            ))
        # but for this track is should be higher towards the end
        self.assertTrue(np.any(p_synth_lf > p_hist_lf))
        self.assertTrue(np.all(p_synth_lf >= p_hist_lf))

        # wind speed
        v_hist_lf = track_hist.max_sustained_wind.values[start_lf_idx:end_lf_idx]
        v_synth_lf = track.max_sustained_wind.values[start_lf_idx:end_lf_idx]
        # wind should decrease over time for that landfall
        v_before_lf = track_hist.max_sustained_wind.values[start_lf_idx-1]
        self.assertTrue(np.all(v_synth_lf[1:] < v_before_lf))
        # and wind speed should be lower in synth than hist at the end of and after this landfall
        self.assertTrue(np.all(
            track.max_sustained_wind.values[end_lf_idx:] < track_hist.max_sustained_wind.values[end_lf_idx:]
            ))
        # finally, central minus env pressure cannot increase during this landfall
        p_env_lf = track.central_pressure.values[start_lf_idx:end_lf_idx]
        self.assertTrue(np.all(np.diff(p_env_lf - p_synth_lf) <= 0))

class TestSynth(unittest.TestCase):
    def test_angle_funs_pass(self):
        """Test functions used by random walk code"""
        self.assertAlmostEqual(tc_synth._get_bearing_angle(np.array([15, 20]),
                                                           np.array([0, 0]))[0], 90.0)
        self.assertAlmostEqual(tc_synth._get_bearing_angle(np.array([20, 20]),
                                                           np.array([0, 5]))[0], 0.0)
        self.assertAlmostEqual(tc_synth._get_bearing_angle(np.array([0, 0.00001]),
                                                           np.array([0, 0.00001]))[0], 45)
        pt_north = tc_synth._get_destination_points(0, 0, 0, 1)
        self.assertAlmostEqual(pt_north[0], 0.0)
        self.assertAlmostEqual(pt_north[1], 1.0)
        pt_west = tc_synth._get_destination_points(0, 0, -90, 3)
        self.assertAlmostEqual(pt_west[0], -3.0)
        self.assertAlmostEqual(pt_west[1], 0.0)
        pt_test = tc_synth._get_destination_points(8.523224, 47.371102,
                                                   151.14161003, 52.80812463)
        self.assertAlmostEqual(pt_test[0], 31.144113)
        self.assertAlmostEqual(pt_test[1], -1.590347)

    def test_random_no_landfall_pass(self):
        """Test calc_perturbed_trajectories with decay and no historical tracks with landfall"""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        expected_warning = 'only %s historical tracks were provided. ' % len(tc_track.data)
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='INFO') as cm:
            tc_track.calc_perturbed_trajectories(use_global_decay_params=False)
        self.assertIn(expected_warning, cm.output[1])
        self.assertIn('No historical track with landfall.', cm.output[2])

    def test_random_walk_ref_pass(self):
        """Test against MATLAB reference."""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TEST_TRACK_SHORT)
        nb_synth_tracks = 2
        tc_track.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks, seed=25, decay=False)

        self.assertEqual(len(tc_track.data), nb_synth_tracks + 1)

        self.assertFalse(tc_track.data[1].orig_event_flag)
        self.assertEqual(tc_track.data[1].name, '1951239N12334_gen1')
        self.assertEqual(tc_track.data[1].id_no, 1.951239012334010e+12)
        self.assertAlmostEqual(tc_track.data[1].lon[0].values, -25.0448138)
        self.assertAlmostEqual(tc_track.data[1].lon[1].values, -25.74439739)
        self.assertAlmostEqual(tc_track.data[1].lon[2].values, -26.54491644)
        self.assertAlmostEqual(tc_track.data[1].lon[3].values, -27.73156829)
        self.assertAlmostEqual(tc_track.data[1].lon[4].values, -28.63175987)
        self.assertAlmostEqual(tc_track.data[1].lon[8].values, -34.05293373)

        self.assertAlmostEqual(tc_track.data[1].lat[0].values, 11.96825841)
        self.assertAlmostEqual(tc_track.data[1].lat[4].values, 11.86769405)
        self.assertAlmostEqual(tc_track.data[1].lat[5].values, 11.84378139)
        self.assertAlmostEqual(tc_track.data[1].lat[6].values, 11.85957282)
        self.assertAlmostEqual(tc_track.data[1].lat[7].values, 11.84555291)
        self.assertAlmostEqual(tc_track.data[1].lat[8].values, 11.8065998)

        self.assertFalse(tc_track.data[2].orig_event_flag)
        self.assertEqual(tc_track.data[2].name, '1951239N12334_gen2')
        self.assertAlmostEqual(tc_track.data[2].id_no, 1.951239012334020e+12)
        self.assertAlmostEqual(tc_track.data[2].lon[0].values, -25.47658461)
        self.assertAlmostEqual(tc_track.data[2].lon[3].values, -28.08465841)
        self.assertAlmostEqual(tc_track.data[2].lon[4].values, -28.85901852)
        self.assertAlmostEqual(tc_track.data[2].lon[8].values, -33.62144837)

        self.assertAlmostEqual(tc_track.data[2].lat[0].values, 11.82886685)
        self.assertAlmostEqual(tc_track.data[2].lat[6].values, 11.71068012)
        self.assertAlmostEqual(tc_track.data[2].lat[7].values, 11.69832976)
        self.assertAlmostEqual(tc_track.data[2].lat[8].values, 11.64145734)

    def test_random_walk_decay_pass(self):
        """Test land decay is called from calc_perturbed_trajectories."""
        assert TC_ANDREW_FL.is_file()
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
        nb_synth_tracks = 2
        # should work if using global parameters
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='DEBUG') as cm0:
            tc_track.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks, seed=25, decay=True,
                                                 use_global_decay_params=True)
        self.assertEqual(len(cm0), 2)
        self.assertEqual(tc_track.size, 3)
        # but alert the user otherwise
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='DEBUG') as cm:
            tc_track.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks, seed=25, decay=True,
                                                 use_global_decay_params=False)
        self.assertIn('No historical track of category Tropical Depression '
                      'with landfall.', cm.output[2])
        self.assertIn('Decay parameters from category Hurricane Cat. 4 taken.',
                      cm.output[3])
        self.assertIn('No historical track of category Hurricane Cat. 1 with '
                      'landfall.', cm.output[4])
        self.assertIn('Decay parameters from category Hurricane Cat. 4 taken.',
                      cm.output[5])
        self.assertIn('No historical track of category Hurricane Cat. 3 with '
                      'landfall. Decay parameters from category Hurricane Cat. '
                      '4 taken.', cm.output[6])
        self.assertIn('No historical track of category Hurricane Cat. 5 with '
                      'landfall.', cm.output[7])

    def test_random_walk_identical_pass(self):
        """Test 0 perturbation leads to identical tracks."""
        tc_track = tc.TCTracks.from_processed_ibtracs_csv(TC_ANDREW_FL)
        nb_synth_tracks = 2
        tc_track.calc_perturbed_trajectories(nb_synth_tracks=nb_synth_tracks,
                                  max_shift_ini=0, max_dspeed_rel=0, max_ddirection=0, decay=False)
        orig_track = tc_track.data[0]
        for syn_track in tc_track.data[1:]:
            np.testing.assert_allclose(orig_track.lon.values, syn_track.lon.values, atol=1e-4)
            np.testing.assert_allclose(orig_track.lat.values, syn_track.lat.values, atol=1e-4)
            for varname in ["time", "time_step", "radius_max_wind", "max_sustained_wind",
                            "central_pressure", "environmental_pressure"]:
                np.testing.assert_array_equal(orig_track[varname].values,
                                              syn_track[varname].values)

    def test_random_walk_single_point(self):
        found = False
        for year in range(1951, 1981):
            tc_track = tc.TCTracks.from_ibtracs_netcdf(provider='usa',
                                         year_range=(year,year),
                                         discard_single_points=False)
            singlept = np.where([x.time.size == 1 for x in tc_track.data])[0]
            found = len(singlept) > 0
            if found:
                # found a case with a single-point track, keep max three tracks for efficiency
                tc_track.data = tc_track.data[max(0, singlept[0]-1):singlept[0]+2]
                n_tr = tc_track.size
                tc_track.equal_timestep()
                tc_track.calc_perturbed_trajectories(nb_synth_tracks=2)
                self.assertEqual((2+1)*n_tr, tc_track.size)
                break
        self.assertTrue(found)

    def test_cutoff_tracks(self):
        tc_track = tc.TCTracks.from_ibtracs_netcdf(storm_id='1986226N30276')
        tc_track.equal_timestep()
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='DEBUG') as cm:
            tc_track.calc_perturbed_trajectories(nb_synth_tracks=10)
        self.assertIn('The following generated synthetic tracks moved beyond '
                      'the range of [-70, 70] degrees latitude', cm.output[1])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDecay)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSynth))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
