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

Test interpolation module.
"""
import unittest
import numpy as np

import climada.util.interpolation as interp
from climada.util.constants import ONE_LAT_KM

def def_input_values():
    """Default input coordinates and centroids values"""
    # Load exposures coordinates from demo entity file
    exposures = np.array([
        [26.933899, -80.128799],
        [26.957203, -80.098284],
        [26.783846, -80.748947],
        [26.645524, -80.550704],
        [26.897796, -80.596929],
        [26.925359, -80.220966],
        [26.914768, -80.07466],
        [26.853491, -80.190281],
        [26.845099, -80.083904],
        [26.82651, -80.213493],
        [26.842772, -80.0591],
        [26.825905, -80.630096],
        [26.80465, -80.075301],
        [26.788649, -80.069885],
        [26.704277, -80.656841],
        [26.71005, -80.190085],
        [26.755412, -80.08955],
        [26.678449, -80.041179],
        [26.725649, -80.1324],
        [26.720599, -80.091746],
        [26.71255, -80.068579],
        [26.6649, -80.090698],
        [26.664699, -80.1254],
        [26.663149, -80.151401],
        [26.66875, -80.058749],
        [26.638517, -80.283371],
        [26.59309, -80.206901],
        [26.617449, -80.090649],
        [26.620079, -80.055001],
        [26.596795, -80.128711],
        [26.577049, -80.076435],
        [26.524585, -80.080105],
        [26.524158, -80.06398],
        [26.523737, -80.178973],
        [26.520284, -80.110519],
        [26.547349, -80.057701],
        [26.463399, -80.064251],
        [26.45905, -80.07875],
        [26.45558, -80.139247],
        [26.453699, -80.104316],
        [26.449999, -80.188545],
        [26.397299, -80.21902],
        [26.4084, -80.092391],
        [26.40875, -80.1575],
        [26.379113, -80.102028],
        [26.3809, -80.16885],
        [26.349068, -80.116401],
        [26.346349, -80.08385],
        [26.348015, -80.241305],
        [26.347957, -80.158855]
    ])

    # Define centroids
    centroids = np.zeros((100, 2))
    inext = 0
    for ilon in range(10):
        for ilat in range(10):
            centroids[inext][0] = 20 + ilat + 1
            centroids[inext][1] = -85 + ilon + 1

            inext = inext + 1

    return exposures, centroids

def def_ref():
    """Default output reference"""
    return np.array([46, 46, 36, 36, 36, 46, 46, 46, 46, 46, 46,
                     36, 46, 46, 36, 46, 46, 46, 46, 46, 46, 46,
                     46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                     46, 46, 46, 45, 45, 45, 45, 45, 45, 45, 45,
                     45, 45, 45, 45, 45, 45])

def def_ref_50():
    """Default output reference for maximum distance threshold 50km"""
    return np.array([46, 46, 36, -1, 36, 46, 46, 46, 46, 46, 46, 36, 46, 46,
                     36, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
                     46, 46, 46, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 45,
                     45, 45, 45, 45, 45, 45, 45, 45])

class TestDistance(unittest.TestCase):
    """Test distance functions."""
    def test_dist_approx_pass(self):
        """Test against matlab reference."""
        lats1 = 45.5
        lons1 = -32.2
        cos_lats1 = np.cos(np.radians(lats1))
        lats2 = 14
        lons2 = 56
        self.assertAlmostEqual(7709.827814738594,
                               interp.dist_approx(lats1, lons1, cos_lats1, lats2, lons2))

    def test_dist_sqr_approx_pass(self):
        """Test against matlab reference."""
        lats1 = 45.5
        lons1 = -32.2
        cos_lats1 = np.cos(np.radians(lats1))
        lats2 = 14
        lons2 = 56
        self.assertAlmostEqual(
            7709.827814738594,
            np.sqrt(interp.dist_sqr_approx(lats1, lons1, cos_lats1, lats2, lons2)) * ONE_LAT_KM)

class TestInterpIndex(unittest.TestCase):
    """Test interpol_index function's interface"""

    def test_wrong_method_fail(self):
        """Check exception is thrown when wrong method is given"""
        with self.assertLogs('climada.util.interpolation', level='ERROR') as cm:
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 2)), 'method')
        self.assertIn('Interpolation using method with distance haversine is not supported.',
                      cm.output[0])

    def test_wrong_distance_fail(self):
        """Check exception is thrown when wrong distance is given"""
        with self.assertLogs('climada.util.interpolation', level='ERROR') as cm:
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 2)),
                                  distance='distance')
        self.assertIn('Interpolation using NN with distance distance is not supported.',
                      cm.output[0])

    def test_wrong_centroid_fail(self):
        """Check exception is thrown when centroids missing one dimension"""
        with self.assertRaises(IndexError):
            interp.interpol_index(np.ones((10, 1)), np.ones((7, 2)),
                                  distance='approx')
        with self.assertRaises(ValueError):
            interp.interpol_index(np.ones((10, 1)), np.ones((7, 2)),
                                  distance='haversine')

    def test_wrong_coord_fail(self):
        """Check exception is thrown when coordinates missing one dimension"""
        with self.assertRaises(IndexError):
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 1)),
                                  distance='approx')
        with self.assertRaises(ValueError):
            interp.interpol_index(np.ones((10, 2)), np.ones((7, 1)),
                                  distance='haversine')

class TestNN(unittest.TestCase):
    """Test interpolator neareast neighbor with approximate distance"""

    def tearDown(self):
        interp.THRESHOLD = 100

    def normal_pass(self, dist):
        """Checking result against matlab climada_demo_step_by_step"""
        # Load input
        exposures, centroids = def_input_values()

        # Interpolate with default threshold
        neighbors = interp.interpol_index(centroids, exposures,
                                          'NN', dist)
        # Reference output
        ref_neighbors = def_ref()
        # Check results
        self.assertEqual(exposures.shape[0], len(neighbors))
        self.assertTrue(np.array_equal(neighbors, ref_neighbors))

    def normal_warning(self, dist):
        """Checking that a warning is raised when minimum distance greater
        than threshold"""
        # Load input
        exposures, centroids = def_input_values()

        # Interpolate with lower threshold to raise warnings
        threshold = 50
        with self.assertLogs('climada.util.interpolation', level='INFO') as cm:
            neighbors = interp.interpol_index(centroids, exposures, 'NN',
                                              dist, threshold=threshold)
        self.assertIn("Distance to closest centroid", cm.output[0])

        ref_neighbors = def_ref_50()
        self.assertTrue(np.array_equal(neighbors, ref_neighbors))

    def repeat_coord_pass(self, dist):
        """Check that exposures with the same coordinates have same
        neighbors"""

        # Load input
        exposures, centroids = def_input_values()

        # Repeat a coordinate
        exposures[2, :] = exposures[0, :]

        # Interpolate with default threshold
        neighbors = interp.interpol_index(centroids, exposures, 'NN', dist)

        # Check output neighbors have same size as coordinates
        self.assertEqual(len(neighbors), exposures.shape[0])
        # Check copied coordinates have same neighbors
        self.assertEqual(neighbors[2], neighbors[0])

    def test_approx_normal_pass(self):
        """Call normal_pass test for approxiamte distance"""
        self.normal_pass('approx')

    def test_approx_normal_warning(self):
        """Call normal_warning test for approxiamte distance"""
        self.normal_warning('approx')

    def test_approx_repeat_coord_pass(self):
        """Call repeat_coord_pass test for approxiamte distance"""
        self.repeat_coord_pass('approx')

    def test_haver_normal_pass(self):
        """Call normal_pass test for haversine distance"""
        self.normal_pass('haversine')

    def test_haver_normal_warning(self):
        """Call normal_warning test for haversine distance"""
        self.normal_warning('haversine')

    def test_haver_repeat_coord_pass(self):
        """Call repeat_coord_pass test for haversine distance"""
        self.repeat_coord_pass('haversine')

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNN)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInterpIndex))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDistance))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
