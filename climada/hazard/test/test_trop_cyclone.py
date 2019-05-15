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
from scipy import sparse
import datetime as dt

import climada.hazard.trop_cyclone as tc
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.trop_cyclone import TropCyclone
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HAZ_TEST_MAT = os.path.join(DATA_DIR, 'atl_prob_no_name.mat')
TEST_TRACK = os.path.join(DATA_DIR, "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, "trac_short_test.csv")

CENTR_DIR = os.path.join(os.path.dirname(__file__), 'data/')
CENTR_TEST_BRB = Centroids()
CENTR_TEST_BRB.read_mat(os.path.join(CENTR_DIR, 'centr_brb_test.mat'))

CENT_CLB = Centroids()
CENT_CLB.read_mat(GLB_CENTROIDS_MAT)
CENT_CLB.dist_coast *= 1000 # set dist_coast to m

class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _hazard_from_track function."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.equal_timestep()
        coastal_centr = tc.coastal_centr_idx(CENT_CLB)
        tc_haz = TropCyclone._tc_from_track(tc_track.data[0], CENT_CLB, coastal_centr)

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 1656093)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.date.size, 1)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).year, 1951)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).month, 8)
        self.assertEqual(dt.datetime.fromordinal(tc_haz.date[0]).day, 27)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 1656093))
        self.assertEqual(tc_haz.fraction.shape, (1, 1656093))

        self.assertAlmostEqual(tc_haz.intensity[0, 1630393],
                               18.511077471450232, 6)
        self.assertEqual(tc_haz.intensity[0, 1630394], 0)
        self.assertEqual(tc_haz.fraction[0, 1630393], 1)
        self.assertEqual(tc_haz.fraction[0, 1630394], 0)

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 7)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 7)

    def test_set_one_file_pass(self):
        """ Test set function set_from_tracks with one input."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, 'IBTrACS: 1951239N12334')
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_two_files_pass(self):
        """ Test set function set_from_tracks with two ibtracs."""
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv([TEST_TRACK_SHORT, TEST_TRACK_SHORT])
        tc_haz = TropCyclone()
        tc_haz.set_from_tracks(tc_track, CENTR_TEST_BRB)
        tc_haz.remove_duplicates()
        tc_haz.check()

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, ['IBTrACS: 1951239N12334', 'IBTrACS: 1951239N12334'])
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))

        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

class TestModel(unittest.TestCase):
    """Test modelling of tropical cyclone"""

    def test_extra_rad_max_wind_pass(self):
        """ Test _extra_rad_max_wind function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        rad_max_wind = tc._extra_rad_max_wind(tc_track.data[0].central_pressure.values, 
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
        _bs_res = tc._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.270856908796045)

        v_trans = 5.123882725120426
        penv = 1010
        pcen = 1005.268166666671
        prepcen = 1005.263333333329
        lat = 12.299999279463769
        xx = 0.586794883720942
        tint = 1
        _bs_res = tc._bs_hol08(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.265551666104679)

    def test_stat_holland(self):
        """ Test _stat_holland function. Compare to MATLAB reference."""
        r_arr = np.array([293.6067129546862, 298.2652319413182])
        r_max = 75.547902916671745
        hol_b = 1.265551666104679
        penv = 1010
        pcen = 1005.268166666671
        ycoord = 12.299999279463769

        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 5.384115724400597)
        self.assertAlmostEqual(_v_arr[1], 5.281356766052531)

        r_arr = np.array([])
        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertTrue(np.array_equal(_v_arr, np.array([])))

        r_arr = np.array([299.4501244109841,
                          291.0737897183741,
                          292.5441003235722])
        r_max = 40.665454622610511
        hol_b = 1.486076257880692
        penv = 1010
        pcen = 970.8727666672957
        ycoord = 14.089110370469488

        _v_arr = tc._stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord)
        self.assertAlmostEqual(_v_arr[0], 11.279764005440288)
        self.assertAlmostEqual(_v_arr[1], 11.682978583939310)
        self.assertAlmostEqual(_v_arr[2], 11.610940769149384)

    def test_coastal_centroids_pass(self):
        """ Test selection of centroids close to coast. MATLAB reference. """
        coastal = tc.coastal_centr_idx(CENT_CLB)

        self.assertEqual(coastal.size, 1044882)
        self.assertEqual(CENT_CLB.lat[coastal[0]], -55.800000000000004)
        self.assertEqual(CENT_CLB.lon[coastal[0]],  -68.200000000000003)

        self.assertEqual(CENT_CLB.lat[coastal[5]], -55.700000000000003)
        self.assertEqual(CENT_CLB.lat[coastal[10]],  -55.700000000000003)
        self.assertEqual(CENT_CLB.lat[coastal[100]], -55.300000000000004)
        self.assertEqual(CENT_CLB.lat[coastal[1000]], -53.900000000000006)
        self.assertEqual(CENT_CLB.lat[coastal[10000]], -44.400000000000006)
        self.assertEqual(CENT_CLB.lat[coastal[100000]], -25)
        self.assertEqual(CENT_CLB.lat[coastal[1000000]], 60.900000000000006)

        self.assertEqual(CENT_CLB.lat[coastal[1044881]], 60.049999999999997)
        self.assertEqual(CENT_CLB.lon[coastal[1044881]],  180.0000000000000)

    def test_vtrans_correct(self):
        """ Test _vtrans_correct function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        i_node = 1
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK)
        tc_track.equal_timestep()
        tc_track.data[0]['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        coast_centr = tc.coastal_centr_idx(CENT_CLB)
        new_centr = CENT_CLB.coord[coast_centr]
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        close_centr = np.array([400381, 400382, 400383, 400384, 401110,
                                1019665]) - 1

        v_trans_corr = tc._vtrans_correct(
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

        v_trans = tc._vtrans(tc_track.data[0].lat.values, tc_track.data[0].lon.values, 
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
        tc_track.data[0]['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        v_trans = 5.2429431910897559
        v_ang = tc._vang_sym(tc_track.data[0].environmental_pressure.values[i_node], 
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
        tc_track.data[0]['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
            tc_track.data[0].central_pressure.values, 
            tc_track.data[0].radius_max_wind.values, ureg))
        int_track = tc_track.data[0].sel(time=slice('1951-08-27', '1951-08-28'))
        coast_centr = tc.coastal_centr_idx(CENT_CLB)

        wind = tc._windfield(int_track, CENT_CLB.coord, coast_centr, model=0)

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

    def test_gust_from_track(self):
        """ Test gust_from_track function. Compare to MATLAB reference. """
        tc_track = TCTracks()
        tc_track.read_processed_ibtracs_csv(TEST_TRACK_SHORT)
        tc_track.equal_timestep()
        intensity = tc.gust_from_track(tc_track.data[0], CENT_CLB, model='H08')

        self.assertTrue(isinstance(intensity, sparse.csr.csr_matrix))
        self.assertEqual(intensity.shape, (1, 1656093))
        self.assertEqual(np.nonzero(intensity)[0].size, 7)

        self.assertEqual(intensity[0, 1630273], 0)
        self.assertAlmostEqual(intensity[0, 1630272], 18.505998796740347, 5)
        self.assertTrue(np.isclose(18.505998796740347,
                                   intensity[0, 1630272]))

        self.assertAlmostEqual(intensity[0, 1630393], 18.511077471450232, 6)
        self.assertTrue(np.isclose(18.511077471450232,
                                   intensity[0, 1630393]))

        self.assertAlmostEqual(intensity[0, 1630514], 18.297250626663271, 5)
        self.assertTrue(np.isclose(18.297250626663271,
                                   intensity[0, 1630514]))

        self.assertAlmostEqual(intensity[0, 1630635], 17.733240401598668, 6)
        self.assertTrue(np.isclose(17.733240401598668,
                                   intensity[0, 1630635]))

        self.assertAlmostEqual(intensity[0, 1630877], 17.525880201507256, 6)
        self.assertTrue(np.isclose(17.525880201507256,
                                   intensity[0, 1630877]))

# Execute Tests
#TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
#    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
#unittest.TextTestRunner(verbosity=2).run(TESTS)
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
