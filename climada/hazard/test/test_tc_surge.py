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
_surge_decay, DECAY_MAX_ELEVATION, DECAY_INLAND_DIST_KM, _substract_sparse_surge, \
_calc_inland_decay, DECAY_RATE_M_KM
from climada.hazard.centroids.centr import Centroids, DEM_NODATA

class TestModel(unittest.TestCase):
    """ Test model """

    CENTR_BANG = Centroids()
    CENTR_BANG.set_raster_from_pnt_bounds((89, 21.5, 90, 23), 0.01)
    _set_centroids_att(CENTR_BANG, dist_coast_decay=True, dem_product='SRTM3',
                       as_pixel=True, min_resol=1.0e-8)

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
        _set_centroids_att(centr_bang, dist_coast_decay=True, dem_product='SRTM3',
                           as_pixel=True)
        self.assertEqual(centr_bang.dist_coast.size, centr_bang.size)
        self.assertEqual(centr_bang.elevation.size, centr_bang.size)
        self.assertAlmostEqual(centr_bang.elevation.max(), 22.743055555555557)

        centr_bang.elevation = np.array([])
        centr_bang.dist_coast = np.array([])
        _set_centroids_att(centr_bang, dist_coast_decay=True, dem_product='SRTM3',
                           as_pixel=False)
        self.assertEqual(centr_bang.dist_coast.size, centr_bang.size)
        self.assertEqual(centr_bang.elevation.size, centr_bang.size)
        self.assertAlmostEqual(centr_bang.elevation.max(), 28.0)

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

    def test_decay_dist_pass(self):
        """ Test _calc_inland_decay with distance """     
        in_decay = _calc_inland_decay(self.CENTR_BANG)
        
        to_zero = np.logical_or(self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION, \
            np.logical_and(self.CENTR_BANG.elevation > 0, \
            self.CENTR_BANG.dist_coast > DECAY_INLAND_DIST_KM*1000))
        self.assertTrue(np.allclose(in_decay[to_zero], np.ones(to_zero.astype(int).sum())*1000))
        
        zero = (self.CENTR_BANG.elevation <= 0)
        self.assertTrue(np.allclose(in_decay[zero], np.zeros(zero.astype(int).sum())))

        value = np.maximum(self.CENTR_BANG.dist_coast/1000*DECAY_RATE_M_KM, 0)
        position = np.logical_not(np.logical_or(to_zero, zero))
        self.assertTrue(np.allclose(in_decay[position], value[position]))

    def test_decay_all_pass(self):
        ini_dist = self.CENTR_BANG.dist_coast
        self.CENTR_BANG.dist_coast = np.array([])
        in_decay = _calc_inland_decay(self.CENTR_BANG)

        to_zero = self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION
        self.assertTrue(np.allclose(in_decay[to_zero], np.ones(to_zero.astype(int).sum())*1000))
        
        zero = (self.CENTR_BANG.elevation <= 0)
        self.assertTrue(np.allclose(in_decay[zero], np.zeros(zero.astype(int).sum())))
        
        self.CENTR_BANG.dist_coast = ini_dist

    def test_substract_sparse_pass(self):
        """ Test _substract_sparse_surge """
        
        elevation = np.array([DEM_NODATA, DEM_NODATA, -1, 1, 2, -3, 5, 10, 15, 20])
        decay = np.ones(10)
        decay[np.logical_or(elevation <= 0, elevation == DEM_NODATA)] = 0
        inten_surge = sparse.csr_matrix(np.arange(0, 20).reshape(2, 10))
        inten_out = _substract_sparse_surge(inten_surge, elevation, decay, 0)

        # no modification where no surge
        self.assertEqual(inten_out[0, 0], 0)
        # no modification where no elevation
        no_el = np.argwhere(np.logical_or(elevation <= 0, elevation == DEM_NODATA)).squeeze()
        self.assertTrue(np.allclose(np.array(inten_out[:, no_el].todense()), np.array(inten_surge[:, no_el].todense())))
        # modification
        self.assertAlmostEqual(inten_out[0, 3], inten_surge[0, 3]-elevation[3]-decay[3])
        self.assertAlmostEqual(inten_out[1, 3], inten_surge[1, 3]-elevation[3]-decay[3])
        self.assertAlmostEqual(inten_out[0, 4], inten_surge[0, 4]-elevation[4]-decay[4])
        self.assertAlmostEqual(inten_out[1, 4], inten_surge[1, 4]-elevation[4]-decay[4])
        self.assertAlmostEqual(inten_out[0, -1], max(inten_surge[0, -1]-elevation[-1]-decay[-1], 0))
        self.assertAlmostEqual(inten_out[1, -1], max(inten_surge[1, -1]-elevation[-1]-decay[-1], 0))

    def test_substract_sparse_SLR_pass(self):
        """ Test _substract_sparse_surge with add_sea_level_rise>0 """
        add_sea_level_rise = np.random.randint(3)+1
        elevation = np.array([DEM_NODATA, DEM_NODATA, -1, 1, 2, -3, 5, 10, 15, 20])
        decay = np.ones(10)
        decay[np.logical_or(elevation <= 0, elevation == DEM_NODATA)] = 0
        inten_surge = sparse.csr_matrix(np.arange(0, 20).reshape(2, 10))
        inten_out = _substract_sparse_surge(inten_surge, elevation, decay, add_sea_level_rise)

        # no modification where no surge
        self.assertEqual(inten_out[0, 0], 0)
        # modification
        self.assertAlmostEqual(inten_out[0, 3], inten_surge[0, 3]-elevation[3]-decay[3]+add_sea_level_rise)
        self.assertAlmostEqual(inten_out[1, 3], inten_surge[1, 3]-elevation[3]-decay[3]+add_sea_level_rise)
        self.assertAlmostEqual(inten_out[0, 4], inten_surge[0, 4]-elevation[4]-decay[4]+add_sea_level_rise)
        self.assertAlmostEqual(inten_out[1, 4], inten_surge[1, 4]-elevation[4]-decay[4]+add_sea_level_rise)
        self.assertAlmostEqual(inten_out[0, -1], max(inten_surge[0, -1]-elevation[-1]-decay[-1]+add_sea_level_rise, 0))
        self.assertAlmostEqual(inten_out[1, -1], max(inten_surge[1, -1]-elevation[-1]-decay[-1]+add_sea_level_rise, 0))


    def test_decay_no_fract_pass(self):
        """ Test _surge_decay with set_fraction False """
        inten_surge = sparse.csr_matrix([np.linspace(0, 20, self.CENTR_BANG.size),
                                         np.linspace(0, 5, self.CENTR_BANG.size)])
        inten, fract = _surge_decay(inten_surge, self.CENTR_BANG, 'SRTM3', set_fraction=False,
                                    min_resol=1.0e-8, add_sea_level_rise=0)

        self.assertEqual(inten_surge.shape, inten.shape)
        self.assertEqual(inten_surge.shape, fract.shape)
        self.assertIsInstance(fract, sparse.csr_matrix)
        self.assertIsInstance(inten, sparse.csr_matrix)
        self.assertEqual(self.CENTR_BANG.size, fract.shape[1])

        to_zero = np.logical_or(self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION,
                                np.logical_and(self.CENTR_BANG.elevation > 0,
                                               self.CENTR_BANG.dist_coast > DECAY_INLAND_DIST_KM*1000))
        self.assertEqual(inten[0, to_zero].data.size, 0)
        # not modified
        self.assertAlmostEqual(inten[0, 13261], inten_surge[0, 13261])
        self.assertAlmostEqual(inten[1, 13261], inten_surge[1, 13261])

        # fraction for each event is the same. values between 0 and 1
        self.assertEqual(fract.min(), 0)
        self.assertEqual(fract.max(), 1)
        self.assertEqual(np.unique(fract.data).size, 1)

    def test_decay_raster_pass(self):
        """ Test _surge_decay with set_fraction True """
        inten_surge = sparse.csr_matrix([np.linspace(0, 20, self.CENTR_BANG.size),
                                         np.linspace(0, 5, self.CENTR_BANG.size)])
        inten, fract = _surge_decay(inten_surge, self.CENTR_BANG, 'SRTM3', set_fraction=True,
                                    min_resol=1.0e-8, add_sea_level_rise=0)

        self.assertEqual(inten_surge.shape, inten.shape)
        self.assertEqual(inten_surge.shape, fract.shape)
        self.assertIsInstance(fract, sparse.csr_matrix)
        self.assertIsInstance(inten, sparse.csr_matrix)
        self.assertEqual(self.CENTR_BANG.size, fract.shape[1])

        to_zero = np.logical_or(self.CENTR_BANG.elevation > DECAY_MAX_ELEVATION,
                                np.logical_and(self.CENTR_BANG.elevation > 0,
                                               self.CENTR_BANG.dist_coast > DECAY_INLAND_DIST_KM*1000))
        self.assertEqual(inten[0, to_zero].data.size, 0)
        # not modified
        self.assertAlmostEqual(inten[0, 13261], inten_surge[0, 13261])
        self.assertAlmostEqual(inten[1, 13261], inten_surge[1, 13261])

        # fraction for each event is the same. values between 0 and 1
        self.assertEqual(fract.min(), 0)
        self.assertEqual(fract.max(), 1)
        self.assertTrue(np.unique(fract.data).size > 2)

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestModel)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
