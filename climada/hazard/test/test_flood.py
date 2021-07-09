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

Test flood module.
"""
import unittest
import datetime as dt
import numpy as np
from climada.hazard.river_flood import RiverFlood
from climada.util.constants import HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC
from climada.hazard.centroids import Centroids


class TestRiverFlood(unittest.TestCase):
    """Test for reading flood event from file"""

    def test_wrong_iso3_fail(self):

        emptyFlood = RiverFlood()
        with self.assertRaises(LookupError):
            RiverFlood._select_exact_area(['OYY'])
        with self.assertRaises(AttributeError):
            emptyFlood.set_from_nc(years=[2600], dph_path=HAZ_DEMO_FLDDPH,
                                   frc_path=HAZ_DEMO_FLDFRC)
        with self.assertRaises(KeyError):
            emptyFlood.set_from_nc(reg=['OYY'], dph_path=HAZ_DEMO_FLDDPH,
                                   frc_path=HAZ_DEMO_FLDFRC, ISINatIDGrid=True)

    def test_exact_area_selection_country(self):

        testCentroids, isos, natIDs = RiverFlood._select_exact_area(['LIE'])

        self.assertEqual(isos[0], 'LIE')
        self.assertEqual(natIDs[0], 118)

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

        testCentr, isos, natIDs = RiverFlood._select_exact_area(reg=['SWA'])

        self.assertEqual(testCentr.shape, (877, 976))
        self.assertAlmostEqual(np.min(testCentr.lat), -0.68767620000001, 4)
        self.assertAlmostEqual(np.max(testCentr.lat), 38.43726119999998, 4)
        self.assertAlmostEqual(np.min(testCentr.lon), 60.52061519999998, 4)
        self.assertAlmostEqual(np.max(testCentr.lon), 101.1455501999999, 4)
        self.assertAlmostEqual(testCentr.lon[10000], 98.27055479999999, 4)
        self.assertAlmostEqual(testCentr.lat[10000], 11.47897099999998, 4)

    def test_isimip_country_flood(self):
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       countries=['DEU'], ISINatIDGrid=True)
        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)

        self.assertAlmostEqual(np.min(rf.centroids.lat), 47.312247000002785, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lat), 55.0622346, 4)
        self.assertAlmostEqual(np.min(rf.centroids.lon), 5.895702599999964, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lon), 15.020687999996682, 4)
        self.assertAlmostEqual(rf.centroids.lon[1000], 9.145697399999989, 4)
        self.assertAlmostEqual(rf.centroids.lat[1000], 47.89557939999999, 4)

        self.assertEqual(rf.intensity.shape, (1, 26878))
        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 10.547529220581055, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 938, 4)

        self.assertEqual(rf.fraction.shape, (1, 26878))
        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 0.9968000054359436, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 1052, 4)
        return

    def test_NATearth_country_flood(self):
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       countries=['DEU'])

        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)

        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 10.547529, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 38380, 4)

        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 0.9968000054359436, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 38143, 4)

    def test_centroids_flood(self):

        # this is going to go through the meta part
        rand_centroids = Centroids()
        lat = np.arange(47, 56, 0.2)
        lon = np.arange(5, 15, 0.2)
        lon, lat = np.meshgrid(lon, lat)
        rand_centroids.set_lat_lon(lat.flatten(), lon.flatten())
        rf = RiverFlood()
        rf.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                       centroids=rand_centroids, ISINatIDGrid=False)

        self.assertEqual(rf.date[0], 730303)
        self.assertEqual(rf.event_id[0], 0)
        self.assertEqual(rf.event_name[0], '2000')
        self.assertEqual(rf.orig[0], False)
        self.assertAlmostEqual(rf.frequency[0], 1.)

        self.assertEqual(rf.centroids.shape, (45, 50))
        self.assertAlmostEqual(np.min(rf.centroids.lat), 47.0, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lat), 55.8, 4)
        self.assertAlmostEqual(np.min(rf.centroids.lon), 5.0, 4)
        self.assertAlmostEqual(np.max(rf.centroids.lon), 14.8, 4)
        self.assertAlmostEqual(rf.centroids.lon[90], 13.0, 4)
        self.assertAlmostEqual(rf.centroids.lat[90], 47.2, 4)

        self.assertEqual(rf.intensity.shape, (1, 2250))
        self.assertAlmostEqual(np.min(rf.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.intensity), 8.921593, 4)
        self.assertEqual(np.argmin(rf.intensity), 0, 4)
        self.assertEqual(np.argmax(rf.intensity), 191, 4)

        self.assertEqual(rf.fraction.shape, (1, 2250))
        self.assertAlmostEqual(np.min(rf.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf.fraction), 0.92, 4)
        self.assertEqual(np.argmin(rf.fraction), 0, 4)
        self.assertEqual(np.argmax(rf.fraction), 1438, 4)

    def test_meta_centroids_flood(self):
        min_lat, max_lat, min_lon, max_lon = 45.7, 47.8, 7.5, 10.5
        cent = Centroids()
        cent.set_raster_from_pnt_bounds((min_lon, min_lat, max_lon, max_lat),
                                        res=0.05)
        rf_rast = RiverFlood()
        rf_rast.set_from_nc(dph_path=HAZ_DEMO_FLDDPH, frc_path=HAZ_DEMO_FLDFRC,
                            centroids=cent)
        self.assertEqual(rf_rast.centroids.shape, (43, 61))
        self.assertAlmostEqual(np.min(rf_rast.centroids.lat),
                               45.70000000000012, 4)
        self.assertAlmostEqual(np.max(rf_rast.centroids.lat), 47.8, 4)
        self.assertAlmostEqual(np.min(rf_rast.centroids.lon), 7.5, 4)
        self.assertAlmostEqual(np.max(rf_rast.centroids.lon),
                               10.49999999999999, 4)
        self.assertAlmostEqual(rf_rast.centroids.lon[90],
                               8.949999999999996, 4)
        self.assertAlmostEqual(rf_rast.centroids.lat[90], 47.75, 4)

        self.assertEqual(rf_rast.intensity.shape, (1, 2623))
        self.assertAlmostEqual(np.min(rf_rast.intensity), 0.0, 4)
        self.assertAlmostEqual(np.max(rf_rast.intensity), 5.8037286, 4)
        self.assertEqual(np.argmin(rf_rast.intensity), 0, 4)
        self.assertEqual(np.argmax(rf_rast.intensity), 55, 4)

        self.assertEqual(rf_rast.fraction.shape, (1, 2623))
        self.assertAlmostEqual(np.min(rf_rast.fraction), 0.0, 4)
        self.assertAlmostEqual(np.max(rf_rast.fraction), 0.4896, 4)
        self.assertEqual(np.argmin(rf_rast.fraction), 0, 4)
        self.assertEqual(np.argmax(rf_rast.fraction), 360, 4)

#    def test_regularGrid_centroids_flood(self):
#        return
#
    def test_flooded_area(self):

        testRFset = RiverFlood()
        testRFset.set_from_nc(countries=['DEU', 'CHE'], dph_path=HAZ_DEMO_FLDDPH,
                              frc_path=HAZ_DEMO_FLDFRC, ISINatIDGrid=True)
        years = [2000, 2001, 2002]
        manipulated_dates = [730303, 730669, 731034]
        for i in range(len(years)):
            testRFaddset = RiverFlood()
            testRFaddset.set_from_nc(countries=['DEU', 'CHE'],
                                     dph_path=HAZ_DEMO_FLDDPH,
                                     frc_path=HAZ_DEMO_FLDFRC,
                                     ISINatIDGrid=True)
            testRFaddset.date = np.array([manipulated_dates[i]])
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
                               14388131.402572632, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[0]),
                         3812)
        self.assertAlmostEqual(np.max(testRFset.fla_ev_centr[2]),
                               14388131.402572632, 3)
        self.assertEqual(np.argmax(testRFset.fla_ev_centr[2]),
                         3812)

        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[0]),
                               28776262.805145264, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[0]),
                         3812)
        self.assertAlmostEqual(np.max(testRFset.fla_ann_centr[2]),
                               14388131.402572632, 3)
        self.assertEqual(np.argmax(testRFset.fla_ann_centr[2]),
                         3812)

        self.assertAlmostEqual(testRFset.fla_event[0],
                               2463979258.8144045, 3)
        self.assertAlmostEqual(testRFset.fla_annual[0],
                               4927958517.628809, 3)
        self.assertAlmostEqual(testRFset.fla_ann_av,
                               3285305678.419206, 3)
        self.assertAlmostEqual(testRFset.fla_ev_av,
                               2463979258.8144045, 3)

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


if __name__ == "__main__":
    # Execute Tests
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiverFlood)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
