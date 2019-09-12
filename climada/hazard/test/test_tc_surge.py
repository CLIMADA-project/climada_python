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

Test TropCyclone class: surges
"""

import numpy as np
import unittest
from scipy import sparse

from climada.hazard.tc_surge import _wind_to_surge, _set_centroids_att, \
_surge_decay, DECAY_MAX_ELEVATION, DECAY_INLAND_DIST_KM
from climada.hazard.centroids.centr import Centroids, DEM_NODATA

class TestModel(unittest.TestCase):
    """ Test model """

    CENTR_BANG = Centroids()
    CENTR_BANG.set_raster_from_pnt_bounds((89, 21.5, 90, 23), 0.01)
    _set_centroids_att(CENTR_BANG, dist_coast_decay=True, dem_product='SRTM3')

    def test_formula_pass(self):
        inten_mat = sparse.lil_matrix((10, 12))
        inten_mat[0, 1] = 50
        inten_mat[0, 9] = 55
        inten_mat[3, 9] = 28
        inten_mat[6, 3] = 32
        inten_mat[7, 2] = 10
        inten_mat = inten_mat.tocsr()

        inten_surge = _wind_to_surge(inten_mat)
        self.assertEqual(inten_surge.shape, inten_mat.shape)
        self.assertTrue(np.allclose(inten_surge.indices, inten_mat.indices))
        self.assertTrue(np.allclose(inten_surge.indptr, inten_mat.indptr))
        self.assertEqual((inten_mat[0, 1]-26.8224)*0.1023+1.8288, inten_surge[0, 1])
        self.assertEqual((inten_mat[0, 9]-26.8224)*0.1023+1.8288, inten_surge[0, 9])
        self.assertEqual((inten_mat[3, 9]-26.8224)*0.1023+1.8288, inten_surge[3, 9])
        self.assertEqual((inten_mat[6, 3]-26.8224)*0.1023+1.8288, inten_surge[6, 3])
        self.assertEqual(0+1.8288, inten_surge[7, 2])

    def test_centroids_att_two_pass(self):
        """ Test set all attributes centroids """
        centr_bang = Centroids()
        centr_bang.set_raster_from_pnt_bounds((89, 21.5, 90, 23), 0.01)
        _set_centroids_att(centr_bang, dist_coast_decay=True, dem_product='SRTM3')
        self.assertEqual(centr_bang.dist_coast.size, centr_bang.size)
        self.assertEqual(centr_bang.elevation.size, centr_bang.size)
        self.assertAlmostEqual(centr_bang.elevation.max(), 22.743055555555557)

        centr_bang.set_meta_to_lat_lon()
        centr_bang.meta = dict()
        centr_bang.elevation = np.array([])
        centr_bang.dist_coast = np.array([])
        _set_centroids_att(centr_bang, dist_coast_decay=True, dem_product='SRTM3')
        self.assertEqual(centr_bang.dist_coast.size, centr_bang.size)
        self.assertEqual(centr_bang.elevation.size, centr_bang.size)
        self.assertAlmostEqual(centr_bang.elevation.max(), 28.0)

    def test_centroids_with_att_pass(self):
        """ Dont fill attributes if present in centroids """
        ini_elev = self.CENTR_BANG.elevation
        ini_dist = self.CENTR_BANG.dist_coast
        _set_centroids_att(self.CENTR_BANG, dist_coast_decay=True, dem_product='SRTM1')
        self.assertTrue(ini_elev is self.CENTR_BANG.elevation)
        self.assertTrue(ini_dist is self.CENTR_BANG.dist_coast)

    def test_centroids_att_one_pass(self):
        """ Test set one attribute centroids """
        centr_bang = Centroids()
        centr_bang.set_raster_from_pnt_bounds((89, 21.5, 90, 23), 0.01)
        _set_centroids_att(centr_bang, dist_coast_decay=False, dem_product='SRTM3')
        self.assertEqual(centr_bang.dist_coast.size, 0)
        self.assertEqual(centr_bang.elevation.size, centr_bang.size)

    def test_decay_points_pass(self):
        """ Test _surge_decay with centroids as points """
        save_meta = self.CENTR_BANG.meta
        self.CENTR_BANG.meta = dict()
        inten_surge = sparse.csr_matrix([np.linspace(0, 20, self.CENTR_BANG.size),
                                         np.linspace(0, 5, self.CENTR_BANG.size)])
        inten, fract = _surge_decay(inten_surge, self.CENTR_BANG, 'SRTM3')

        self.assertEqual(inten_surge.shape, inten.shape)
        self.assertEqual(inten_surge.shape, fract.shape)
        self.assertEqual(self.CENTR_BANG.size, fract.shape[1])

        to_zero = np.logical_or(self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION,
                                self.CENTR_BANG.dist_coast > DECAY_INLAND_DIST_KM*1000)
        self.assertFalse(inten[0, to_zero].data.size)
        self.assertAlmostEqual(inten[0, 13261], 18.12257715)
        self.assertAlmostEqual(inten[1, 13261], 5.07897059)

        # negative elevation centroids close to coast increase surge height
        neg_elev = np.argwhere(np.logical_and(np.logical_and(self.CENTR_BANG.elevation<0,
                                              self.CENTR_BANG.elevation!=DEM_NODATA),
                                              self.CENTR_BANG.dist_coast < 1400)).squeeze()
        self.assertTrue((inten[0, neg_elev]>inten_surge[0, neg_elev]).todense().all())
        self.assertTrue((inten[1, neg_elev]>inten_surge[1, neg_elev]).todense().all())

        # always decay surge height in positive elvation places
        pos_elev = self.CENTR_BANG.elevation>0
        self.assertFalse((inten[0, pos_elev]>inten_surge[0, pos_elev]).todense().any())
        self.assertFalse((inten[1, pos_elev]>inten_surge[1, pos_elev]).todense().any())

        # fraction for each event is the same. values between 0 and 1
        self.assertEqual(fract.min(), 0)
        self.assertEqual(fract.max(), 1)
        self.assertEqual(np.unique(fract.data).size, 1)

        self.CENTR_BANG.meta = save_meta

    def test_decay_raster_pass(self):
        """ Test _surge_decay with centroids as raster """
        inten_surge = sparse.csr_matrix([np.linspace(0, 20, self.CENTR_BANG.size),
                                         np.linspace(0, 5, self.CENTR_BANG.size)])
        inten, fract = _surge_decay(inten_surge, self.CENTR_BANG, 'SRTM3')

        self.assertEqual(inten_surge.shape, inten.shape)
        self.assertEqual(inten_surge.shape, fract.shape)
        self.assertEqual(self.CENTR_BANG.size, fract.shape[1])

        to_zero = np.logical_or(self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION,
                                self.CENTR_BANG.dist_coast > DECAY_INLAND_DIST_KM*1000)
        self.assertFalse(inten[0, to_zero].data.size)
        self.assertAlmostEqual(inten[0, 13261], 18.12257715)
        self.assertAlmostEqual(inten[1, 13261], 5.07897059)

        # negative elevation centroids close to coast increase surge height
        neg_elev = np.argwhere(np.logical_and(np.logical_and(self.CENTR_BANG.elevation<0,
                                              self.CENTR_BANG.elevation!=DEM_NODATA),
                                              self.CENTR_BANG.dist_coast < 1400)).squeeze()
        self.assertTrue((inten[0, neg_elev]>inten_surge[0, neg_elev]).todense().all())
        self.assertTrue((inten[1, neg_elev]>inten_surge[1, neg_elev]).todense().all())

        # always decay surge height in positive elvation places
        pos_elev = self.CENTR_BANG.elevation>0
        self.assertFalse((inten[0, pos_elev]>inten_surge[0, pos_elev]).todense().any())
        self.assertFalse((inten[1, pos_elev]>inten_surge[1, pos_elev]).todense().any())

        # fraction for each event is the same. values between 0 and 1
        self.assertEqual(fract.min(), 0)
        self.assertEqual(fract.max(), 1)
        self.assertTrue(np.unique(fract.data).size > 2)

    def test_decay_without_dist_pass(self):
        """ Test _surge_decay without distance decay and points """
        save_meta = self.CENTR_BANG.meta
        dist_coast = self.CENTR_BANG.dist_coast

        self.CENTR_BANG.meta = dict()
        self.CENTR_BANG.dist_coast = np.array([])
        inten_surge = sparse.csr_matrix([np.linspace(0, 20, self.CENTR_BANG.size),
                                         np.linspace(0, 5, self.CENTR_BANG.size)])
        inten, fract = _surge_decay(inten_surge, self.CENTR_BANG, 'SRTM3')

        self.assertEqual(inten_surge.shape, inten.shape)
        self.assertEqual(inten_surge.shape, fract.shape)
        self.assertEqual(self.CENTR_BANG.size, fract.shape[1])

        to_zero = self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION
        self.assertFalse(inten[0, to_zero].data.size)

        equal = np.logical_not(to_zero)
        sub_elev = self.CENTR_BANG.elevation[equal].copy()
        sub_elev[sub_elev==DEM_NODATA] = 0
        self.assertTrue(np.allclose(np.maximum(inten_surge[0, equal] - sub_elev - inten[0, equal], 0),
                                    np.zeros(equal.astype(int).sum())))
        self.assertTrue(np.allclose(np.maximum(inten_surge[1, equal] - sub_elev - inten[1, equal], 0),
                                    np.zeros(equal.astype(int).sum())))

        # fraction for each event is the same. values between 0 and 1
        self.assertEqual(fract.min(), 0)
        self.assertEqual(fract.max(), 1)
        self.assertEqual(np.unique(fract.data).size, 1)

        self.CENTR_BANG.meta = save_meta
        self.CENTR_BANG.dist_coast = dist_coast

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
