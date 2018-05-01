"""
Test TropCyclone class
"""

import unittest
import os
import numpy as np
from pint import UnitRegistry
from scipy import sparse
import datetime as dt

import climada.hazard.trop_cyclone as tc
from climada.hazard.trop_cyclone import TropCyclone
from climada.hazard.centroids.base import Centroids
from climada.util.constants import DATA_DIR, HAZ_TEST_MAT, GLB_CENTROIDS_MAT, CENTR_TEST_BRB

TEST_TRACK = os.path.join(DATA_DIR, 'test', "trac_brb_test.csv")
TEST_TRACK_SHORT = os.path.join(DATA_DIR, 'test', "trac_short_test.csv")

class TestReader(unittest.TestCase):
    """Test loading funcions from the TropCyclone class"""

    def test_set_one_pass(self):
        """Test _set_one function."""
        centroids = None
        tc_haz = TropCyclone._hazard_from_track(TEST_TRACK_SHORT, '', 
                                                centroids)

        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, TEST_TRACK_SHORT)
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.tag.file_name, GLB_CENTROIDS_MAT)
        self.assertEqual(tc_haz.centroids.id.size, 1656093)
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
        """ Test set function with one input."""
        tc_haz = TropCyclone()
        centr = Centroids(CENTR_TEST_BRB)
        tc_haz.set_from_tracks(TEST_TRACK_SHORT, centroids=centr)
        tc_haz.check()
        
        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, TEST_TRACK_SHORT)
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.tag.file_name, CENTR_TEST_BRB)
        self.assertEqual(tc_haz.centroids.id.size, 296)
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
        """ Test construct tropical cyclone from two IbTracs."""
        tc_haz = TropCyclone()
        centr = Centroids(CENTR_TEST_BRB)
        tc_haz.set_from_tracks([TEST_TRACK_SHORT, TEST_TRACK_SHORT], 
                               centroids=centr)
        tc_haz.check()
        
        self.assertEqual(tc_haz.tag.haz_type, 'TC')
        self.assertEqual(tc_haz.tag.description, '')
        self.assertEqual(tc_haz.tag.file_name, TEST_TRACK_SHORT)
        self.assertEqual(tc_haz.units, 'm/s')
        self.assertEqual(tc_haz.centroids.tag.file_name, CENTR_TEST_BRB)
        self.assertEqual(tc_haz.centroids.id.size, 296)
        self.assertEqual(tc_haz.event_id.size, 1)
        self.assertEqual(tc_haz.event_id[0], 1)
        self.assertEqual(tc_haz.event_name, ['1951239N12334'])
        self.assertTrue(np.array_equal(tc_haz.frequency, np.array([1])))
        self.assertTrue(np.array_equal(tc_haz.orig, np.array([True])))
        self.assertTrue(isinstance(tc_haz.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(tc_haz.fraction, sparse.csr.csr_matrix))
        self.assertEqual(tc_haz.intensity.shape, (1, 296))
        self.assertEqual(tc_haz.fraction.shape, (1, 296))
        self.assertEqual(len(tc_haz.tracks), 1)
        
        self.assertEqual(tc_haz.fraction.nonzero()[0].size, 0)
        self.assertEqual(tc_haz.intensity.nonzero()[0].size, 0)

    def test_read_haz_and_tc_pass(self):
        """ Read a hazard file and a IbTrac in parallel. """
        centr = Centroids(CENTR_TEST_BRB)
        tc_haz1 = TropCyclone()
        tc_haz1.read(HAZ_TEST_MAT)
        tc_haz2 = TropCyclone()
        tc_haz2.set_from_tracks(TEST_TRACK_SHORT, centroids=centr)
        tc_haz2.append(tc_haz1)
        tc_haz2.check()
        self.assertEqual(tc_haz2.intensity.shape, (14451, 396))

class TestIBTracs(unittest.TestCase):
    """Test reading and model of TC from IBTrACS files"""

    def test_category_pass(self):
        """Test category computation."""
        max_sus_wind = np.array([25, 30, 35, 40, 45, 45, 45, 45, 35, 25])
        max_sus_wind_unit = 'kn'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([25, 25, 25, 30, 30, 30, 30, 30, 25, 25, 20])
        max_sus_wind_unit = 'kn'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([80, 90, 100, 115, 120, 125, 130,
                                 120, 110, 80, 75, 80, 65])
        max_sus_wind_unit = 'kn'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(4, cat)

        max_sus_wind = np.array([28.769475, 34.52337, 40.277265,
                                 46.03116, 51.785055, 51.785055, 51.785055,
                                 51.785055, 40.277265, 28.769475])
        max_sus_wind_unit = 'mph'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(0, cat)

        max_sus_wind = np.array([12.86111437, 12.86111437, 12.86111437,
                                 15.43333724, 15.43333724, 15.43333724,
                                 15.43333724, 15.43333724, 12.86111437,
                                 12.86111437, 10.2888915])
        max_sus_wind_unit = 'm/s'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
        self.assertEqual(-1, cat)

        max_sus_wind = np.array([148.16, 166.68, 185.2, 212.98, 222.24, 231.5,
                                 240.76, 222.24, 203.72, 148.16, 138.9, 148.16,
                                 120.38])
        max_sus_wind_unit = 'km/h'
        cat = tc._set_category(max_sus_wind, max_sus_wind_unit)
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

    def test_read_pass(self):
        """Read a tropical cyclone."""
        tc_track = tc.read_ibtracs(TEST_TRACK)

        self.assertEqual(tc_track.time.size, 38)
        self.assertEqual(tc_track.lon[11], -39.60)
        self.assertEqual(tc_track.lat[23], 14.10)
        self.assertEqual(tc_track.time_step[7], 6)
        self.assertEqual(np.max(tc_track.radius_max_wind), 0)
        self.assertEqual(np.min(tc_track.radius_max_wind), 0)
        self.assertEqual(tc_track.max_sustained_wind[21], 55)
        self.assertEqual(tc_track.central_pressure[29], 969.76880)
        self.assertEqual(np.max(tc_track.environmental_pressure), 1010)
        self.assertEqual(np.min(tc_track.environmental_pressure), 1010)
        self.assertEqual(tc_track.time.dt.year[13], 1951)
        self.assertEqual(tc_track.time.dt.month[26], 9)
        self.assertEqual(tc_track.time.dt.day[7], 29)
        self.assertEqual(tc_track.max_sustained_wind_unit, 'kn')
        self.assertEqual(tc_track.central_pressure_unit, 'mb')
        self.assertEqual(tc_track.orig_event_flag, 1)
        self.assertEqual(tc_track.name, '1951239N12334')
        self.assertEqual(tc_track.data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(tc_track.basin))
        self.assertEqual(tc_track.id_no, 1951239012334)
        self.assertEqual(tc_track.category, 1)

    def test_interp_track_pass(self):
        """ Interpolate track to min_time_step. Compare to MATLAB reference."""
        tc_track = tc.read_ibtracs(TEST_TRACK)
        int_track = tc.interp_track(tc_track)

        self.assertEqual(int_track.time.size, 223)
        self.assertAlmostEqual(int_track.lon.values[11], -27.426151640151684)
        self.assertEqual(int_track.lat[23], 12.300006169591480)
        self.assertEqual(int_track.time_step[7], 1)
        self.assertEqual(np.max(int_track.radius_max_wind), 0)
        self.assertEqual(np.min(int_track.radius_max_wind), 0)
        self.assertEqual(int_track.max_sustained_wind[21], 25)
        self.assertAlmostEqual(int_track.central_pressure.values[29],
                               1.005409300000005e+03)
        self.assertEqual(np.max(int_track.environmental_pressure), 1010)
        self.assertEqual(np.min(int_track.environmental_pressure), 1010)
        self.assertEqual(int_track['time.year'][13], 1951)
        self.assertEqual(int_track['time.month'][26], 8)
        self.assertEqual(int_track['time.day'][7], 27)
        self.assertEqual(int_track.max_sustained_wind_unit, 'kn')
        self.assertEqual(int_track.central_pressure_unit, 'mb')
        self.assertEqual(int_track.orig_event_flag, 1)
        self.assertEqual(int_track.name, '1951239N12334')
        self.assertEqual(int_track.data_provider, 'hurdat_atl')
        self.assertTrue(np.isnan(int_track.basin))
        self.assertEqual(int_track.id_no, 1951239012334)
        self.assertEqual(int_track.category, 1)

    def test_extra_rad_max_wind_pass(self):
        """ Test _extra_rad_max_wind function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        tc_track = tc.read_ibtracs(TEST_TRACK)
        int_track = tc.interp_track(tc_track)
        rad_max_wind = tc._extra_rad_max_wind(int_track, ureg)

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

    def test_bs_value_pass(self):
        """" Test _bs_value function. Compare to MATLAB reference."""
        v_trans = 5.241999541820597
        penv = 1010
        pcen = 1005.263333333329
        prepcen = 1005.258500000000
        lat = 12.299999504631343
        xx = 0.586781395348824
        tint = 1
        _bs_res = tc._bs_value(v_trans, penv, pcen, prepcen, lat, xx, tint)
        self.assertAlmostEqual(_bs_res, 1.270856908796045)

        v_trans = 5.123882725120426
        penv = 1010
        pcen = 1005.268166666671
        prepcen = 1005.263333333329
        lat = 12.299999279463769
        xx = 0.586794883720942
        tint = 1
        _bs_res = tc._bs_value(v_trans, penv, pcen, prepcen, lat, xx, tint)
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
        centroids = Centroids(GLB_CENTROIDS_MAT)
        coastal = tc._coastal_centr_idx(centroids)

        self.assertEqual(coastal.size, 1044882)
        self.assertEqual(centroids.lat[coastal[0]], -55.800000000000004)
        self.assertEqual(centroids.lon[coastal[0]],  -68.200000000000003)
        self.assertEqual(centroids.id[coastal[0]], 1)

        self.assertEqual(centroids.lat[coastal[5]], -55.700000000000003)
        self.assertEqual(centroids.lat[coastal[10]],  -55.700000000000003)
        self.assertEqual(centroids.lat[coastal[100]], -55.300000000000004)
        self.assertEqual(centroids.lat[coastal[1000]], -53.900000000000006)
        self.assertEqual(centroids.lat[coastal[10000]], -44.400000000000006)
        self.assertEqual(centroids.lat[coastal[100000]], -25)
        self.assertEqual(centroids.lat[coastal[1000000]], 60.900000000000006)

        self.assertEqual(centroids.lat[coastal[1044881]], 60.049999999999997)
        self.assertEqual(centroids.lon[coastal[1044881]],  180.0000000000000)
        self.assertEqual(centroids.id[coastal[1044881]], 3043681)

    def test_vtrans_holland(self):
        """ Test _vtrans_holland function. Compare to MATLAB reference."""
        ureg = UnitRegistry()
        i_node = 1
        track = tc.read_ibtracs(TEST_TRACK)
        int_track = tc.interp_track(track)
        int_track['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
                int_track, ureg))
        centroids = Centroids(GLB_CENTROIDS_MAT)
        coast_centr = tc._coastal_centr_idx(centroids)
        new_centr = centroids.coord[coast_centr]
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        close_centr = np.array([400381, 400382, 400383, 400384, 401110,
                                1019665]) - 1

        v_trans, v_trans_corr = tc._vtrans_holland(int_track, i_node,
            new_centr, close_centr, r_arr, ureg)

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertAlmostEqual(v_trans * to_kn, 10.191466256012880)
        self.assertEqual(v_trans_corr.size, 6)
        self.assertAlmostEqual(v_trans_corr[0] * to_kn, 2.490082696506720)
        self.assertAlmostEqual(v_trans_corr[1] * to_kn, 2.418324821762491)
        self.assertAlmostEqual(v_trans_corr[2] * to_kn, 2.344175399115656)
        self.assertAlmostEqual(v_trans_corr[3] * to_kn, 2.268434724527058)
        self.assertAlmostEqual(v_trans_corr[4] * to_kn, 2.416654031976129)
        self.assertAlmostEqual(v_trans_corr[5] * to_kn, -1.394485527059995)

    def test_vang_holland(self):
        """ Test _vang_holland function. Compare to MATLAB reference. """
        ureg = UnitRegistry()
        i_node = 1
        track = tc.read_ibtracs(TEST_TRACK)
        int_track = tc.interp_track(track)
        int_track['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
                int_track, ureg))
        r_arr = np.array([286.4938638337190, 290.5930935802884,
                          295.0271327746536, 299.7811253637995,
                          296.8484825705515, 274.9892882245964])
        v_trans = 5.2429431910897559
        v_ang = tc._vang_holland(int_track, i_node, r_arr, v_trans, model='H08')

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertEqual(v_ang.size, 6)
        self.assertAlmostEqual(v_ang[0] * to_kn, 10.774196807905097)
        self.assertAlmostEqual(v_ang[1] * to_kn, 10.591725180482094)
        self.assertAlmostEqual(v_ang[2] * to_kn, 10.398212766600055)
        self.assertAlmostEqual(v_ang[3] * to_kn, 10.195108683240084)
        self.assertAlmostEqual(v_ang[4] * to_kn, 10.319869893291429)
        self.assertAlmostEqual(v_ang[5] * to_kn, 11.305188714213809)

    def test_windfield_holland(self):
        """ Test _windfield_holland function. Compare to MATLAB reference. """
        ureg = UnitRegistry()
        track = tc.read_ibtracs(TEST_TRACK)
        int_track = tc.interp_track(track)
        int_track['radius_max_wind'] = ('time', tc._extra_rad_max_wind(
                int_track, ureg))
        int_track = int_track.sel(time=slice('1951-08-27', '1951-08-28'))
        centroids = Centroids(GLB_CENTROIDS_MAT)
        coast_centr = tc._coastal_centr_idx(centroids)
        new_centr = centroids.coord[coast_centr]

        wind = tc._windfield_holland(int_track, new_centr, model='H08')

        to_kn = (1* ureg.meter / ureg.second).to(ureg.knot).magnitude
        self.assertTrue(isinstance(wind, sparse.lil.lil_matrix))
        self.assertEqual(wind.shape, (1, 1044882))
        self.assertEqual(np.nonzero(wind)[0].size, 5)
        self.assertTrue(np.array_equal(wind.nonzero()[1],
            np.array([1019062, 1019183, 1019304, 1019425, 1019667]) - 1))
        self.assertTrue(np.array_equal(wind.nonzero()[0], np.zeros((5,))))

        self.assertAlmostEqual(wind[0, 1019061] * to_kn, 35.961499748377776, 4)
        self.assertTrue(np.isclose(35.961499748377776,
                                   wind[0, 1019061] * to_kn))

        self.assertAlmostEqual(wind[0, 1019182] * to_kn, 35.985591640301138, 6)
        self.assertTrue(np.isclose(35.985591640301138,
                                   wind[0, 1019182] * to_kn))

        self.assertAlmostEqual(wind[0, 1019303] * to_kn, 35.567653569424614, 5)
        self.assertTrue(np.isclose(35.567653569424614,
                                   wind[0, 1019303] * to_kn))

        self.assertAlmostEqual(wind[0, 1019424] * to_kn, 34.470214174079388, 6)
        self.assertTrue(np.isclose(34.470214174079388,
                                   wind[0, 1019424] * to_kn))

        self.assertAlmostEqual(wind[0, 1019666] * to_kn, 34.067538078331282, 6)
        self.assertTrue(np.isclose(34.067538078331282,
                                   wind[0, 1019666] * to_kn))

    def test_gust_from_track(self):
        """ Test gust_from_track function. Compare to MATLAB reference. """
        track = tc.read_ibtracs(TEST_TRACK_SHORT)
        centroids = Centroids(GLB_CENTROIDS_MAT)
        intensity = tc.gust_from_track(track, centroids, model='H08')

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

class TestRndWalk(unittest.TestCase):
    """Test random walk for probabilistic tropical cyclone generation"""

    def test_ref_pass(self):
        """Test against MATLAB reference."""
        track = tc.read_ibtracs(TEST_TRACK_SHORT)
        rnd_ini = np.array([[0.9649, 0.1576], [0.7922, 0.9595]])
        rnd_ang = np.array([0.3922, 0.6555, 0.1712, 0.7060, 0.0318, 0.2769, \
                            0.0462, 0.0971, 0.8235, 0.6948, 0.3171, 0.9502, \
                            0.0344, 0.4387, 0.3816, 0.7655, 0.7952, 0.1869])
        ens_size=2
        track_ens = tc.calc_random_walk(track, ens_size, rnd_ini, rnd_ang)

        self.assertEqual(len(track_ens), ens_size)
        
        self.assertFalse(track_ens[0].orig_event_flag)
        self.assertEqual(track_ens[0].name, '1951239N12334_gen1')
        self.assertEqual(track_ens[0].id_no, 1.951239012334010e+12)
        self.assertEqual(track_ens[0].lon[0], -24.90265000000000)
        self.assertEqual(track_ens[0].lon[1], -25.899653369275331)
        self.assertEqual(track_ens[0].lon[2], -26.917223719188879)
        self.assertEqual(track_ens[0].lon[3], -28.021940640460727)
        self.assertEqual(track_ens[0].lon[4], -29.155418047711304)
        self.assertEqual(track_ens[0].lon[8], -34.529188419229598)
        
        self.assertEqual(track_ens[0].lat[0], 12.73830000000000)
        self.assertEqual(track_ens[0].lat[4], 13.130817937897319)
        self.assertEqual(track_ens[0].lat[5], 13.219446057176036)
        self.assertEqual(track_ens[0].lat[6], 13.291468242391597)
        self.assertEqual(track_ens[0].lat[7], 13.343819850233439)
        self.assertEqual(track_ens[0].lat[8], 13.412292879644005)

        self.assertFalse(track_ens[1].orig_event_flag)
        self.assertEqual(track_ens[1].name, '1951239N12334_gen2')
        self.assertEqual(track_ens[1].id_no, 1.951239012334020e+12)
        self.assertEqual(track_ens[1].lon[0], -26.11360000000000)
        self.assertEqual(track_ens[1].lon[3], -29.409222264217661)
        self.assertEqual(track_ens[1].lon[4], -30.584828633621079)
        self.assertEqual(track_ens[1].lon[8], -35.959133410163332)
        
        self.assertEqual(track_ens[1].lat[0], 12.989250000000000)
        self.assertEqual(track_ens[1].lat[6], 13.410297633704376)
        self.assertEqual(track_ens[1].lat[7], 13.493978269787220)
        self.assertEqual(track_ens[1].lat[8], 13.565343427825237)
    
    def test_from_class_pass(self):
        """ Test call from class."""
        centr_brb = Centroids(CENTR_TEST_BRB)
        ens_size=3
        tc_haz = tc.TropCyclone()
        tc_haz.set_from_tracks(TEST_TRACK_SHORT, centroids=centr_brb)
        tc_haz.set_random_walk(ens_size, centroids=centr_brb)

        self.assertEqual(len(tc_haz.tracks), ens_size+1)
        self.assertEqual(tc_haz.event_id.size, ens_size+1)
        tc_haz.check()

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIBTracs))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRndWalk))
unittest.TextTestRunner(verbosity=2).run(TESTS)
