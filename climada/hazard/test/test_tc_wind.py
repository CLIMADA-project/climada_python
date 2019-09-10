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

Test TropCyclone class
"""

import os
import unittest
import numpy as np
from pint import UnitRegistry

from climada.hazard.tc_tracks import TCTracks
import climada.hazard.trop_cyclone as tc
import climada.hazard.tc_wind as tc_wd
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")

CENT_CLB = Centroids()
CENT_CLB.read_mat(GLB_CENTROIDS_MAT)
CENT_CLB.dist_coast *= 1000 # set dist_coast to m

class TestModel(unittest.TestCase):
    """Test modelling of tropical cyclone"""

    def test_extra_rad_max_wind_pass(self):
        """ Test _extra_rad_max_wind function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rad_max_wind = tc_wd._extra_rad_max_wind(tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg)

        self.assertEqual(rad_max_wind[0], 75.536713749999905)
        self.assertAlmostEqual(rad_max_wind[10], 75.592659583328057)
        self.assertAlmostEqual(rad_max_wind[128], 46.686527832605236)
        self.assertEqual(rad_max_wind[129], 46.089211533333405)
        self.assertAlmostEqual(rad_max_wind[130], 45.672274889277276)
        self.assertEqual(rad_max_wind[189], 45.132715266666672)
        self.assertAlmostEqual(rad_max_wind[190], 45.979603999211285)
        self.assertAlmostEqual(rad_max_wind[191], 47.287173876478825)
        self.assertEqual(rad_max_wind[192], 48.875090249999985)
        self.assertAlmostEqual(rad_max_wind[200], 59.975901084074955)

    def test_bs_hol08_pass(self):
        """" Test _bs_hol08 function. Compare to MATLAB reference."""
        v_trans = 5.241999541820597
        penv = 1010
        pcen = 1005.263333333329
        prepcen = 1005.258500000000
        lat = 12.299999504631343
        xx = 0.586781395348824
        tint = 1
        _bs_res = tc_wd._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.270856908796045)

        v_trans = 5.123882725120426
        penv = 1010
        pcen = 1005.268166666671
        prepcen = 1005.263333333329
        lat = 12.299999279463769
        xx = 0.586794883720942
        tint = 1
        _bs_res = tc_wd._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.265551666104679)

    def test_stat_holland(self):
        """ Test _stat_holland function. Compare to MATLAB reference."""
        r_arr = np.array([293.6067129546862, 298.2652319413182])
        r_max = 75.547902916671745
        hol_b = 1.265551666104679
        penv = 1010
        pcen = 1005.268166666671
        ycoord = 12.299999279463769

        _v_arr = tc_wd._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 5.384115724400597)
        self.assertAlmostEqual(_v_arr[1], 5.281356766052531)

        r_arr = np.array([])
        _v_arr = tc_wd._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertTrue(np.array_equal(_v_arr, np.array([])))

        r_arr = np.array([299.4501244109841,
                          291.0737897183741,
                          292.5441003235722])
        r_max = 40.665454622610511
        hol_b = 1.486076257880692
        penv = 1010
        pcen = 970.8727666672957
        ycoord = 14.089110370469488

        _v_arr = tc_wd._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 11.279764005440288)
        self.assertAlmostEqual(_v_arr[1], 11.682978583939310)
        self.assertAlmostEqual(_v_arr[2], 11.610940769149384)

    def test_vtrans_correct(self):
        """ Test _vtrans_correct function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc_wd._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        coast_centr = tc.coastal_centr_idx(CENT_CLB)
        new_centr = CENT_CLB.coord[coast_centr]
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        close_centr = np.array([400381, 400382, 400383, 400384, 401110,
                                1019665]) - 1

        v_trans_corr = tc_wd._vtrans_correct(
            tc_track.data[0].lat.values[i_node:i_node+2], 
            tc_track.data[0].lon.values[i_node:i_node+2],
            tc_track.data[0].radius_max_wind.values[i_node],
            new_centr[close_centr, :], r_arr)

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        
        v_trans = 10.191466256012880 / to_kn
        v_trans_corr *= v_trans
        self.assertEqual(v_trans_corr.size, 6)
        self.assertAlmostEqual(v_trans_corr[0] * to_kn, 2.490082696506720)
        self.assertAlmostEqual(v_trans_corr[1] * to_kn, 2.418324821762491)
        self.assertAlmostEqual(v_trans_corr[2] * to_kn, 2.344175399115656)
        self.assertAlmostEqual(v_trans_corr[3] * to_kn, 2.268434724527058)
        self.assertAlmostEqual(v_trans_corr[4] * to_kn, 2.416654031976129)
        self.assertAlmostEqual(v_trans_corr[5] * to_kn, -1.394485527059995)

    def test_vtrans_pass(self):
        """ Test _vtrans function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()

        v_trans = tc_wd._vtrans(tc_track.data[0].lat.values, tc_track.data[0].lon.values, 
                tc_track.data[0].time_step.values, ureg)

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        
        self.assertEqual(v_trans.size, tc_track.data[0].time.size-1)
        self.assertAlmostEqual(v_trans[i_node-1]*to_kn, 10.191466256012880)

    def test_vang_sym(self):
        """ Test _vang_sym function. Compare to MATLAB reference. """
        ureg = UnitRegistry()
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc_wd._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        v_trans = 5.2429431910897559
        v_ang = tc_wd._vang_sym(tc_track.data[0].environmental_pressure.values[i_node], 
            tc_track.data[0].central_pressure.values[i_node-1:i_node+1], 
            tc_track.data[0].lat.values[i_node], 
            tc_track.data[0].time_step.values[i_node], 
            tc_track.data[0].radius_max_wind.values[i_node], 
            r_arr, v_trans, model=0)

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertEqual(v_ang.size, 6)
        self.assertAlmostEqual(v_ang[0] * to_kn, 10.774196807905097)
        self.assertAlmostEqual(v_ang[1] * to_kn, 10.591725180482094)
        self.assertAlmostEqual(v_ang[2] * to_kn, 10.398212766600055)
        self.assertAlmostEqual(v_ang[3] * to_kn, 10.195108683240084)
        self.assertAlmostEqual(v_ang[4] * to_kn, 10.319869893291429)
        self.assertAlmostEqual(v_ang[5] * to_kn, 11.305188714213809)

    def test_windfield(self):
        """ Test _windfield function. Compare to MATLAB reference. """
        ureg = UnitRegistry()
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc_wd._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        int_track = tc_track.data[0].sel(time=slice('1951-08-27', '1951-08-28'))
        coast_centr = tc.coastal_centr_idx(CENT_CLB)

        wind = tc_wd.windfield(int_track, CENT_CLB.coord, coast_centr, model=0, 
                               inten_thresh=17.5)

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertEqual(wind.shape, (CENT_CLB.size,))
        
        wind = wind[coast_centr]
        self.assertEqual(np.nonzero(wind)[0].size, 5)
        self.assertTrue(np.array_equal(wind.nonzero()[0],
            np.array([1019062, 1019183, 1019304, 1019425, 1019667]) - 1))

        self.assertAlmostEqual(wind[1019061] * to_kn, 35.961499748377776, 4)
        self.assertTrue(np.isclose(35.961499748377776,
                                   wind[1019061] * to_kn))

        self.assertAlmostEqual(wind[1019182] * to_kn, 35.985591640301138, 6)
        self.assertTrue(np.isclose(35.985591640301138,
                                   wind[1019182] * to_kn))

        self.assertAlmostEqual(wind[1019303] * to_kn, 35.567653569424614, 5)
        self.assertTrue(np.isclose(35.567653569424614,
                                   wind[1019303] * to_kn))

        self.assertAlmostEqual(wind[1019424] * to_kn, 34.470214174079388, 6)
        self.assertTrue(np.isclose(34.470214174079388,
                                   wind[1019424] * to_kn))

        self.assertAlmostEqual(wind[1019666] * to_kn, 34.067538078331282, 6)
        self.assertTrue(np.isclose(34.067538078331282,
                                   wind[1019666] * to_kn))

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
