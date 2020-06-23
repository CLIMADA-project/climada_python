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
from datetime import date
from climada.hazard.river_flood import RiverFlood
from climada.util.constants import HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC


class TestRiverFlood(unittest.TestCase):
    """Test for reading flood event from file"""

    def test_exact_area_selection(self):
        testCentroids, iso, natID = RiverFlood._select_exact_area(['LIE'])

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
        self.assertEqual(iso[0], 'LIE')

        testCentroids, iso, natID = RiverFlood._select_exact_area(['DEU'])
        self.assertAlmostEqual(np.max(testCentroids.lon), 15.020687999999979)
        self.assertAlmostEqual(np.min(testCentroids.lon), 5.895702599999964)
        self.assertAlmostEqual(np.max(testCentroids.lat), 55.0622346)
        self.assertAlmostEqual(np.min(testCentroids.lat), 47.312247)
        self.assertEqual(iso[0], 'DEU')

    def test_full_flood(self):
        """ read_flood"""
        testRF = RiverFlood()
        testRF.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC)
        self.assertEqual(testRF.centroids.size, 37324800)
        testRF.centroids.set_meta_to_lat_lon()
        self.assertAlmostEqual(np.max(testRF.centroids.lon),
                               179.97916666666663)
        self.assertAlmostEqual(np.min(testRF.centroids.lon),
                               -179.97916666666666)
        self.assertAlmostEqual(np.max(testRF.centroids.lat),
                               89.97916666666667)
        self.assertAlmostEqual(np.min(testRF.centroids.lat),
                               -89.97916666666666)

        self.assertAlmostEqual(testRF.centroids.lon[20000000],
                               113.35416666666663)
        self.assertAlmostEqual(testRF.centroids.lat[20000000],
                               -6.4375)
        self.assertAlmostEqual(testRF.centroids.lon[10000000],
                               -33.3125)
        self.assertAlmostEqual(testRF.centroids.lat[10000000],
                               41.770833333333336)

        self.assertEqual(testRF.date[0], 730303)
        self.assertEqual((date.fromordinal(testRF.date[0]).year), 2000)

        self.assertEqual(testRF.orig[0], 0)
        self.assertEqual(np.argmax(testRF.intensity), 26190437)
        self.assertAlmostEqual(np.max(testRF.fraction), 1.0)
        self.assertAlmostEqual(testRF.intensity[0, 3341441], 0.9583404)
        self.assertAlmostEqual(testRF.intensity[0, 3341442], 0.9583404)
        self.assertAlmostEqual(testRF.fraction[0, 3341441], 0.41666666)
        self.assertAlmostEqual(testRF.fraction[0, 3341442], 0.375)
        self.assertEqual(np.argmax(testRF.fraction[0]), 3341440)

    def test_region_selection(self):

        testRFReg = RiverFlood()
        testRFReg.set_from_nc(reg=['SWA'], dph_path=HAZ_DEMO_FLDDPH,
                              frc_path=HAZ_DEMO_FLDFRC, ISINatIDGrid=True)
        self.assertEqual(testRFReg.centroids.lat.size, 301181)

        centr_ori, iso, natID = RiverFlood._select_exact_area(reg=['SWA'])
        centr_ori.set_lat_lon_to_meta()

        self.assertAlmostEqual(np.max(testRFReg.centroids.lon),
                               101.14555019999997)
        self.assertAlmostEqual(np.min(testRFReg.centroids.lon),
                               60.52061519999998)
        self.assertAlmostEqual(np.max(testRFReg.centroids.lat),
                               38.43726119999998)
        self.assertAlmostEqual(np.min(testRFReg.centroids.lat),
                               -0.6876762000000127)
        self.assertEqual(testRFReg.date[0], 730303)
        self.assertEqual((date.fromordinal(testRFReg.date[0]).year), 2000)
        self.assertEqual(testRFReg.orig[0], 0)
        self.assertAlmostEqual(np.max(testRFReg.intensity), 16.69780921936035)

    def test_flooded_area(self):

        testRFArea = RiverFlood()

        testRFArea.set_from_nc(countries=['AUT'], dph_path=HAZ_DEMO_FLDDPH,
                               frc_path=HAZ_DEMO_FLDFRC, ISINatIDGrid=True)
        self.assertEqual(testRFArea.centroids.lat.size, 5782)
        self.assertAlmostEqual(np.max(testRFArea.centroids.lon),
                               17.104017999999968)
        self.assertAlmostEqual(np.min(testRFArea.centroids.lon),
                               9.562363399999981)
        self.assertAlmostEqual(np.max(testRFArea.centroids.lat),
                               48.978911)
        self.assertAlmostEqual(np.min(testRFArea.centroids.lat),
                               46.39558179999999)
        self.assertEqual(testRFArea.orig[0], 0)
        self.assertEqual(np.max(testRFArea.intensity),
                         9.613386154174805)
        self.assertAlmostEqual(np.max(testRFArea.fraction),
                               0.5103999972343445)

        testRFArea.set_flooded_area(save_centr=True)
        self.assertAlmostEqual(np.max(testRFArea.fla_ev_centr),
                               7396511.421906647, 3)
        self.assertAlmostEqual(np.max(testRFArea.fla_ann_centr),
                               7396511.421906647, 3)

        self.assertAlmostEqual(testRFArea.fla_event[0],
                               229956891.5531019, 3)
        self.assertAlmostEqual(testRFArea.fla_annual[0],
                               229956891.5531019, 3)

        testRFset = RiverFlood()
        testRFset.set_from_nc(countries=['AFG'], dph_path=HAZ_DEMO_FLDDPH,
                              frc_path=HAZ_DEMO_FLDFRC, ISINatIDGrid=True)
        years = [2000, 2001, 2002]
        manipulated_dates = [730303, 730669, 731034]
        for i in range(len(years)):
            testRFaddset = RiverFlood()
            testRFaddset.set_from_nc(countries=['AFG'])
            testRFaddset.date = np.array([manipulated_dates[i]])
            if i == 0:
                testRFaddset.event_name = ['2000_2']
            else:
                testRFaddset.event_name = [str(years[i])]
            testRFset.append(testRFaddset)

        testRFset.set_flooded_area(save_centr=True)

        self.assertEqual(testRFset.fla_event.shape[0], 4)
        self.assertEqual(testRFset.fla_annual.shape[0], 3)
        self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[0]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[0]),
                         32610)
        self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[2]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[2]),
                         32610)

        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[0]),
                               34400996.45855092, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[0]),
                         32610)
        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[2]),
                               17200498.22927546, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[2]),
                         32610)

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
        self.assertTrue(np.array_equal(testRFTime._select_event(test_time,
                                                                years),
                        [1, 2]))
        years = [1988, 1990]
        self.assertTrue(np.array_equal(testRFTime._select_event(test_time,
                                                                years),
                        [0, 3]))


#
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiverFlood)
unittest.TextTestRunner(verbosity=2).run(TESTS)
