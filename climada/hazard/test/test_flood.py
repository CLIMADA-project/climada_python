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

Test flood module.
"""
import unittest
import datetime as dt
import numpy as np
import pandas as pd
from climada.hazard.flood import RiverFlood
from climada.util.constants import HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC


class TestRiverFlood(unittest.TestCase):
    """Test for reading flood event from file"""
    def test_wrong_iso3_fail(self):

        emptyFlood = RiverFlood()
        with self.assertRaises(KeyError):
            RiverFlood._select_exact_area(['OYY'])
        with self.assertRaises(AttributeError):
            emptyFlood.set_from_nc(years=[2600], dph_path=HAZ_DEMO_FLDDPH,
                                   frc_path=HAZ_DEMO_FLDFRC)
        with self.assertRaises(KeyError):
            emptyFlood.set_from_nc(reg=['OYY'], dph_path=HAZ_DEMO_FLDDPH,
                                   frc_path=HAZ_DEMO_FLDFRC)

    def test_exact_area_selection_country(self):

        testCentroids, isos, natIDs = RiverFlood._select_exact_area(['LIE'])

        self.assertEqual(isos[0], 'LIE')
        self.assertEqual(natIDs.iloc[0], 118)

        self.assertEqual(testCentroids.shape, (5, 3))
        self.assertEqual(testCentroids.lon.shape[0], 13)
        self.assertAlmostEqual(testCentroids.lon[0], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[1], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[2], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[3], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[4], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[5], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[6], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[7], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[8], 9.60403)
        self.assertAlmostEqual(testCentroids.lon[9], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[10], 9.5623634)
        self.assertAlmostEqual(testCentroids.lon[11], 9.5206968)
        self.assertAlmostEqual(testCentroids.lon[12], 9.5623634)

        self.assertAlmostEqual(testCentroids.lat[0], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[1], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[2], 47.0622474)
        self.assertAlmostEqual(testCentroids.lat[3], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[4], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[5], 47.103914)
        self.assertAlmostEqual(testCentroids.lat[6], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[7], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[8], 47.1455806)
        self.assertAlmostEqual(testCentroids.lat[9], 47.1872472)
        self.assertAlmostEqual(testCentroids.lat[10], 47.1872472)
        self.assertAlmostEqual(testCentroids.lat[11], 47.2289138)
        self.assertAlmostEqual(testCentroids.lat[12], 47.2289138)
    
    def test_exact_area_selection_region(self):

        testCentroids, isos, natIDs = RiverFlood._select_exact_area(reg=['SWA'])
        self.assertEqual(isos[0], 'AFG')
        self.assertEqual(isos[3], 'IND')
        self.assertEqual(isos[8], 'PAK')
        
        self.assertEqual(natIDs.iloc[1], 20)
        self.assertEqual(natIDs.iloc[3], 94)
        self.assertEqual(natIDs.iloc[7], 155)

        self.assertEqual(testCentroids.shape, (940, 976))
        self.assertAlmostEqual(np.min(testCentroids.lat), -0.68767620000001, 4)
        self.assertAlmostEqual(np.max(testCentroids.lat), 38.43726119999998, 4)
        self.assertAlmostEqual(np.min(testCentroids.lon), 60.52061519999998, 4)
        self.assertAlmostEqual(np.max(testCentroids.lon), 101.1455501999999, 4)
        self.assertAlmostEqual(testCentroids.lon[10000], 98.27055479999999, 4)
        self.assertAlmostEqual(testCentroids.lat[10000], 11.47897099999998, 4)


    def test_isimip_country_flood(self):
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       countries=['DEU'])
        tag = ('/home/insauer/Climada/climada_python/data/demo/' +
               'flddph_WaterGAP2_miroc5_historical_flopros_gev_' +
               'picontrol_2000_0.1.nc;/home/insauer/Climada/cli' +
               'mada_python/data/demo/fldfrc_WaterGAP2_miroc5_h' +
               'istorical_flopros_gev_picontrol_2000_0.1.nc')
        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.tag.file_name, tag)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)

        self.assertEqual(rf.centroids.shape, (187, 220))
        self.assertAlmostEqual(np.min(rf.centroids.lat), 47.312247000002785, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lat), 55.0622346, 4)
        self.assertAlmostEqual(np.min(rf.centroids.lon), 5.895702599999964, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lon), 15.020687999996682, 4)
        self.assertAlmostEqual(rf.centroids.lon[1000], 8.895697799998885, 4)
        self.assertAlmostEqual(rf.centroids.lat[1000], 54.18723600000031, 4)

        self.assertEqual(rf.intensity.shape, (1, 26878))
        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 10.547529220581055, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 26021, 4)

        self.assertEqual(rf.fraction.shape, (1, 26878))
        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 0.9968000054359436, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 25875, 4)

        return
    
    def test_isimip_reg_flood(self):
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       reg=['SWA'])
        tag = ('/home/insauer/Climada/climada_python/data/demo/' +
               'flddph_WaterGAP2_miroc5_historical_flopros_gev_' +
               'picontrol_2000_0.1.nc;/home/insauer/Climada/cli' +
               'mada_python/data/demo/fldfrc_WaterGAP2_miroc5_h' +
               'istorical_flopros_gev_picontrol_2000_0.1.nc')
        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.tag.file_name, tag)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)

        self.assertEqual(rf.centroids.shape, (877, 976))
        self.assertAlmostEqual(np.min(rf.centroids.lat), -0.687676199985944, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lat), 38.43726119999998, 4)
        self.assertAlmostEqual(np.min(rf.centroids.lon), 60.52061519999998, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lon), 101.14555019998537, 4)
        self.assertAlmostEqual(rf.centroids.lon[10000], 70.31226619999646, 4)
        self.assertAlmostEqual(rf.centroids.lat[10000], 35.812265400000925, 4)

        self.assertEqual(rf.intensity.shape, (1, 301181))
        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 16.69780921936035, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 258661, 4)

        self.assertEqual(rf.fraction.shape, (1, 301181))
        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 1.0, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 3461, 4)
        
        return
    
    def test_NATearth_country_flood(self):
#        rf = RiverFlood()
#        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
#                       countries=['DEU'], ISINatIDGrid=False)
#        tag = ('/home/insauer/Climada/climada_python/data/demo/' +
#               'flddph_WaterGAP2_miroc5_historical_flopros_gev_' +
#               'picontrol_2000_0.1.nc;/home/insauer/Climada/cli' +
#               'mada_python/data/demo/fldfrc_WaterGAP2_miroc5_h' +
#               'istorical_flopros_gev_picontrol_2000_0.1.nc')
#        self.assertEqual(rf.date[0], 730303)
#        self.assertEqual(rf.tag.file_name, tag)
#        self.assertEqual(rf.event_id[0], 0)
#        self.assertEqual(rf.event_name[0], '2000')
#        self.assertEqual(rf.orig[0], False)
#        self.assertAlmostEqual(rf.frequency[0], 1.)
#
#        self.assertEqual(rf.centroids.shape, (877, 976))
        
        return
    
    def test_NATearth_reg_flood(self):
        return
    
    def test_global_flood(self):
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC)
        tag = ('/home/insauer/Climada/climada_python/data/demo/' +
               'flddph_WaterGAP2_miroc5_historical_flopros_gev_' +
               'picontrol_2000_0.1.nc;/home/insauer/Climada/cli' +
               'mada_python/data/demo/fldfrc_WaterGAP2_miroc5_h' +
               'istorical_flopros_gev_picontrol_2000_0.1.nc')
        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.tag.file_name, tag)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)
        
        self.assertEqual(rf.centroids.shape, (4320, 8640))
        self.assertAlmostEqual(np.min(rf.centroids.lat), -89.9791666666871, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lat), 89.97916666666667, 4)
        self.assertAlmostEqual(np.min(rf.centroids.lon), -179.9791666666666, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lon), 179.97916666658486, 4)
        self.assertAlmostEqual(rf.centroids.lon[10000], -123.31250000001288, 4)
        self.assertAlmostEqual(rf.centroids.lat[10000], 89.9375, 4)

        self.assertEqual(rf.intensity.shape, (1, 37324800))
        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 19.276295, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 26190437, 4)

        self.assertEqual(rf.fraction.shape, (1, 37324800))
        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 1.0, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 3341440, 4)
        return
    
    def test_centroids_flood(self):
        testCentroids, isos, natIDs = RiverFlood._select_exact_area(['AFG'])
        
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       centroids=testCentroids)
        tag = ('/home/insauer/Climada/climada_python/data/demo/' +
               'flddph_WaterGAP2_miroc5_historical_flopros_gev_' +
               'picontrol_2000_0.1.nc;/home/insauer/Climada/cli' +
               'mada_python/data/demo/fldfrc_WaterGAP2_miroc5_h' +
               'istorical_flopros_gev_picontrol_2000_0.1.nc')
#        self.assertEqual(rf.date[0], 730303)
#        self.assertEqual(rf.tag.file_name, tag)
#        self.assertEqual(rf.event_id[0], 0)
#        self.assertEqual(rf.event_name[0], '2000')
#        self.assertEqual(rf.orig[0], False)
#        self.assertAlmostEqual(rf.frequency[0], 1.)
#        
        return
    
    def test_meta_centroids_flood(self):
        return
    def test_regularGrid_centroids_flood(self):
        return

    def test_flooded_area(self):

        testRFset = RiverFlood()
        testRFset.set_from_nc(countries=['AFG'], dph_path=HAZ_DEMO_FLDDPH,
                              frc_path=HAZ_DEMO_FLDFRC)
        years = [2000, 2001, 2002]
        manipulated_dates = [730303, 730669, 731034]
        for i in range(len(years)):
            testRFaddset = RiverFlood()
            testRFaddset.set_from_nc(countries=['AFG'],
                                     dph_path=HAZ_DEMO_FLDDPH,
                                     frc_path=HAZ_DEMO_FLDFRC)
            testRFaddset.date = [manipulated_dates[i]]
            if i == 0:
                testRFaddset.event_name = ['2000_2']
            else:
                testRFaddset.event_name = [str(years[i])]
            testRFset.append(testRFaddset)

        testRFset.set_flooded_area(save_centr=True)
        self.assertEqual(testRFset.units, 'm')

        self.assertEqual(testRFset.fla_event.shape[0], 4)
        self.assertEqual(testRFset.fla_annual.shape[0], 3)
        self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[0]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[0]),
                         3252)
        self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[2]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[2]),
                         3252)

        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[0]),
                               34400996.45855092, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[0]),
                         3252)
        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[2]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[2]),
                         3252)

        self.assertAlmostEqual(testRFset.fla_event[0],
                               6244242013.5826435, 3)
        self.assertAlmostEqual(testRFset.fla_annual[0],
                               12488484027.165287, 3)
        self.assertAlmostEqual(testRFset.fla_ann_av,
                               8325656018.110191, 3)
        self.assertAlmostEqual(testRFset.fla_ev_av,
                               6244242013.5826435, 3)

    def test_select_events(self):
        testRFTime = RiverFlood()
        tt1 = dt.datetime.strptime('1988-07-02', '%Y-%m-%d')
        tt2 = dt.datetime.strptime('2010-04-01', '%Y-%m-%d')
        tt3 = dt.datetime.strptime('1997-07-02', '%Y-%m-%d')
        tt4 = dt.datetime.strptime('1990-07-02', '%Y-%m-%d')
        years = [2010, 1997]
        test_time = np.array([tt1, tt2, tt3, tt4])
        self.assertTrue(np.array_equal(
                        testRFTime._select_event(test_time, years), [1, 2]))
        years = [1988, 1990]
        self.assertTrue(np.array_equal(
                        testRFTime._select_event(test_time, years), [0, 3]))

    def test_intersect_area(self):
        rf = RiverFlood()
        testCentroids, isos, natIDs = RiverFlood._select_exact_area(['LIE'])
        rf.centroids = testCentroids
        mask = rf._intersect_area(natIDs)
        self.assertEqual(testCentroids.shape, mask.shape)
        testmask = np.array([[True, True, True],
                            [True, True, True],
                            [True, True, True],
                            [True, True, False],
                            [True, True, False]])
        self.assertTrue(np.array_equal(mask, testmask))

#
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiverFlood)
unittest.TextTestRunner(verbosity=2).run(TESTS)
