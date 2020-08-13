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

Test tc_surge_geoclaw module
"""

import datetime as dt
import logging
import os
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from climada.hazard import TCTracks
from climada.hazard.tc_surge_geoclaw import (boxcover_points_along_axis,
                                             bounds_to_str,
                                             clawpack_info,
                                             dt64_to_pydt,
                                             load_topography,
                                             mean_max_sea_level,
                                             setup_clawpack,
                                             TCSurgeEvents,
                                             TCSurgeGeoClaw)


LOGGER = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
ZOS_PATH = os.path.join(DATA_DIR, "zos_monthly.nc")
TOPO_PATH = os.path.join(DATA_DIR, "surge_topo.tif")


class TestFuncs(unittest.TestCase):
    """Test helper functions"""

    def test_boxcover(self):
        """Test boxcovering function"""
        nsplits = 4
        # sorted list of 1d-points
        points = np.array([-3., -1.3, 1.5, 1.7, 4.6, 5.4, 6.2, 6.8, 7.])
        # shuffle list of points
        points = points[[4, 7, 3, 1, 2, 5, 8, 1, 6, 0]].reshape(-1, 1)
        # this is easy to see from the sorted list of points
        boxes_correct = [[-3.0, -1.3], [1.5, 1.7], [4.6, 7.0]]
        boxes, size = boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum(b[1] - b[0] for b in boxes))

        nsplits = 3
        points = np.array([
            [0.0, 0.2], [1.3, 0.1], [2.5, 0.0],
            [3.0, 1.5], [0.2, 1.2],
            [0.4, 2.3], [0.5, 3.0],
        ])
        boxes_correct = [
            [0.0, 0.0, 2.5, 0.2],
            [0.2, 1.2, 3.0, 1.5],
            [0.4, 2.3, 0.5, 3.0],
        ]
        boxes, size = boxcover_points_along_axis(points, nsplits)
        self.assertEqual(boxes, boxes_correct)
        self.assertEqual(size, sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
        boxes, size = boxcover_points_along_axis(points[:,::-1], nsplits)
        self.assertEqual(boxes, [[b[1], b[0], b[3], b[2]] for b in boxes_correct])


    def test_bounds_to_str(self):
        """Test conversion from lon-lat-bounds tuple to lat-lon string"""
        bounds_str = [
            [(-4.2, 1.0, -3.05, 2.125), '1N-2.125N_4.2W-3.05W'],
            [(106.9, -7, 111.6875, 25.1), '7S-25.1N_106.9E-111.7E'],
            [(-6.9, -7.8334, 11, 25.1), '7.833S-25.1N_6.9W-11E'],
        ]
        for bounds, string in bounds_str:
            str_out = bounds_to_str(bounds)
            self.assertEqual(str_out, string)


    def test_dt64_to_pydt(self):
        """Test conversion from datetime64 to python datetime objects"""
        # generate test data
        pydt = [
            dt.datetime(1865, 3, 7, 20, 41, 2),
            dt.datetime(2008, 2, 29, 0, 5, 30),
            dt.datetime(2013, 12, 2),
        ]
        dt64 = pd.Series(pydt).values

        # test conversion of numpy array of dates
        pydt_conv = dt64_to_pydt(dt64)
        self.assertIsInstance(pydt_conv, list)
        self.assertEqual(len(pydt_conv), dt64.size)
        self.assertIsInstance(pydt_conv[0], dt.datetime)
        self.assertEqual(pydt_conv[1], pydt[1])

        # test conversion of single object
        pydt_conv = dt64_to_pydt(dt64[2])
        self.assertIsInstance(pydt_conv, dt.datetime)
        self.assertEqual(pydt_conv, pydt[2])


    def test_clawpack_setup(self):
        """Test setup_clawpack function"""
        LOGGER.disabled = False
        setup_clawpack()
        import clawpack.pyclaw
        self.assertFalse(LOGGER.disabled)
        path, decorators = clawpack_info()
        self.assertTrue(path is not None)


    def test_surge_events(self):
        """Test TCSurgeEvents object"""
        # Artificial track with two "landfall" events
        radii = np.array([40, 40, 40, 30, 30, 30, 40, 30, 30, 30,
                          40, 40, 40, 30, 30, 30, 30, 30, 40, 60])
        track = xr.Dataset({
            'radius_max_wind': ('time', radii),
            'radius_oci': ('time', 4.1 * radii),
            'max_sustained_wind': ('time', [10, 10, 40, 40, 40, 40, 10, 40, 40, 40,
                                            10, 10, 10, 40, 40, 40, 40, 40, 10, 10]),
            'central_pressure': ('time', np.full((20,), 970)),
            'time_step': ('time', np.full((20,), np.timedelta64(6, 'h'))),
        }, coords={
            'time': np.arange('2000-01-01T00:00', '2000-01-06T00:00',
                              np.timedelta64(6, 'h'), dtype='datetime64[h]'),
            'lat': ('time', np.linspace(5, 30, 20)),
            'lon': ('time', np.zeros(20)),
        })

        # centroids clearly too far away from track
        centroids = np.array([ar.ravel() for ar in np.meshgrid(np.linspace(-20, -10, 10),
                                                               np.linspace(50, 80, 10))]).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 0)

        # one comapct set of centroids in the middle of the track
        centroids = np.array([ar.ravel() for ar in np.meshgrid(np.linspace(15, 17, 100),
                                                               np.linspace(-2, 2, 100))]).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 1)
        self.assertTrue(np.all(s_events.time_mask_buffered[0][s_events.time_mask[0]]))
        self.assertTrue(np.all(~s_events.time_mask[0][:7]))
        self.assertTrue(np.all(s_events.time_mask[0][7:10]))
        self.assertTrue(np.all(~s_events.time_mask[0][10:]))

        # three sets of centroids
        centroids = np.concatenate([
            # first half and close to the track
            [ar.ravel() for ar in np.meshgrid(np.linspace(6, 8, 50), np.linspace(-2, -0.5, 50))],
            # second half on both sides of the track
            [ar.ravel() for ar in np.meshgrid(np.linspace(19, 22, 50), np.linspace(0.5, 1.5, 50))],
            [ar.ravel() for ar in np.meshgrid(np.linspace(24, 25, 50), np.linspace(-1.0, 0.3, 50))],
            # at the end, where storm is too weak to create surge
            [ar.ravel() for ar in np.meshgrid(np.linspace(29, 32, 50), np.linspace(0, 1, 50))],
        ], axis=1).T
        s_events = TCSurgeEvents(track, centroids)
        self.assertEqual(s_events.nevents, 2)
        self.assertEqual(len(list(s_events)), 2)
        for i in range(2):
            self.assertTrue(np.all(s_events.time_mask_buffered[i][s_events.time_mask[i]]))
        self.assertTrue(np.all(~s_events.time_mask_buffered[0][:1]))
        self.assertTrue(np.all(s_events.time_mask_buffered[0][1:4]))
        self.assertTrue(np.all(~s_events.time_mask_buffered[0][4:]))
        self.assertTrue(np.all(~s_events.time_mask_buffered[1][:12]))
        self.assertTrue(np.all(s_events.time_mask_buffered[1][12:17]))
        self.assertTrue(np.all(~s_events.time_mask_buffered[1][17:]))
        # for the double set in second half, it's advantageous to split surge area in two:
        self.assertEqual(len(s_events.surge_areas[1]), 2)


    def test_load_sea_level(self):
        """Test mean_max_sea_level function"""
        months = [np.array([[2010, 1]]), np.array([[2010, 2]]), np.array([[2010, 1], [2010, 2]])]
        bounds = [
            (-153.62, -28.79, -144.75, -18.44),
            (-153, -20, -150, -19),
            (-152, -28.5, -145, -27.5),
        ]
        sea_level = []
        for mon in months:
            for bnd in bounds:
                sea_level.append(mean_max_sea_level(ZOS_PATH, mon, bnd))
        sea_level = np.array(sea_level).reshape(len(months), len(bounds))
        self.assertTrue(np.allclose(sea_level[:-1].mean(axis=0), sea_level[-1]))
        self.assertTrue(np.all(sea_level[:,0] > sea_level[:,1]))
        self.assertTrue(np.all(sea_level[:,0] > sea_level[:,2]))
        self.assertAlmostEqual(sea_level[0,0], 1.332, places=3)
        self.assertAlmostEqual(sea_level[1,0], 1.270, places=3)
        self.assertAlmostEqual(sea_level[2,0], 1.301, places=3)
        self.assertTrue(np.all((sea_level[:,1] < 1.31) & (sea_level[:,1] > 1.24)))
        self.assertTrue(np.all((sea_level[:,2] < 0.93) & (sea_level[:,2] > 0.88)))


    def test_load_topography(self):
        """Test load_topography function"""
        resolutions = [15, 30, 41, 90]
        bounds = [
            (-153.62, -28.79, -144.75, -18.44),
            (-153, -20, -150, -19),
            (-152, -28.5, -145, -27.5),
            (-149.54, -23.47, -149.37, -23.33)
        ]
        zvalues = []
        for res_as in resolutions:
            for bnd in bounds:
                result = load_topography(TOPO_PATH, bnd, res_as)
                topo_bounds = result[0]
                self.assertLessEqual(topo_bounds[0], bnd[0])
                self.assertLessEqual(topo_bounds[1], bnd[1])
                self.assertGreaterEqual(topo_bounds[2], bnd[2])
                self.assertGreaterEqual(topo_bounds[3], bnd[3])
                xcoords, ycoords = result[1:3]
                self.assertTrue(np.all((xcoords >= topo_bounds[0]) & (xcoords <= topo_bounds[2])))
                self.assertTrue(np.all((ycoords >= topo_bounds[1]) & (ycoords <= topo_bounds[3])))
                zvalues.append(result[3])
            self.assertTrue(np.all(zvalues[-2] == -1000))
            self.assertTrue(np.all(zvalues[-3] == -1000))
            self.assertGreater(zvalues[-1].max(), 100)
            if len(zvalues) > 4:
                self.assertLess(zvalues[-1].size, zvalues[-5].size)
                self.assertLess(zvalues[-2].size, zvalues[-6].size)
                self.assertLess(zvalues[-3].size, zvalues[-7].size)
                self.assertLess(zvalues[-4].size, zvalues[-8].size)



class TestHazardInit(unittest.TestCase):
    """Test init and properties of TCSurgeGeoClaw class"""

    def test_init(self):
        """Test TCSurgeGeoClaw basic object properties"""
        # use dummy track that is too weak to actually produce surge
        track = xr.Dataset({
            'radius_max_wind': ('time', np.full((8,), 10.)),
            'radius_oci': ('time', np.full((8,), 200.)),
            'max_sustained_wind': ('time', np.full((8,), 30.)),
            'central_pressure': ('time', np.full((8,), 990.)),
            'time_step': ('time', np.full((8,), np.timedelta64(3, 'h'))),
        }, coords={
            'time': np.arange('2010-02-05', '2010-02-06',
                              np.timedelta64(3, 'h'), dtype='datetime64[h]'),
            'lat': ('time', np.linspace(-26.3, -21.5, 8)),
            'lon': ('time', np.linspace(-147.3, -150.6, 8)),
        }, attrs={
            'sid': '2010029S12177_test_dummy',
            'name': 'Dummy',
            'orig_event_flag': True,
            'category': 0,
            'basin': "SPW",
        })
        tracks = TCTracks()
        tracks.data = [track, track]
        haz = TCSurgeGeoClaw.from_tc_tracks(tracks, ZOS_PATH, TOPO_PATH)
        self.assertIsInstance(haz, TCSurgeGeoClaw)
        self.assertEqual(haz.intensity.shape[0], 2)
        self.assertTrue(np.all(haz.intensity.toarray() == 0))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHazardInit))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
