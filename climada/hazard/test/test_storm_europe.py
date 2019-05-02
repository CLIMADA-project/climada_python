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

Test StormEurope class
"""

import os
import unittest
import datetime as dt
import numpy as np
from scipy import sparse

from climada.hazard import StormEurope, Centroids

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

fn = [
    'fp_lothar_crop-test.nc',
    'fp_xynthia_crop-test.nc',
]
TEST_NCS = [os.path.join(DATA_DIR, f) for f in fn]
""" 
These test files have been generated using the netCDF kitchen sink:
ncks -d latitude,50.5,54.0 -d longitude,3.0,7.5 ./file_in.nc ./file_out.nc
"""

TEST_CENTROIDS = Centroids(os.path.join(DATA_DIR, 'fp_centroids-test.csv'))


class TestReader(unittest.TestCase):
    """ Test loading functions from the StormEurope class """

    def test_centroids_from_nc(self):
        """ Test if centroids can be constructed correctly """
        cent = StormEurope._centroids_from_nc(TEST_NCS[0])

        self.assertTrue(isinstance(cent, Centroids))
        self.assertTrue(isinstance(cent.coord, np.ndarray))
        self.assertEqual(cent.size, 9944)
        self.assertEqual(cent.coord.shape[0], cent.id.shape[0])

    def test_read_footprints(self):
        """ Test read_footprints function, using two small test files"""
        storms = StormEurope()
        storms.read_footprints(TEST_NCS, description='test_description')

        self.assertEqual(storms.tag.haz_type, 'WS')
        self.assertEqual(storms.units, 'm/s')
        self.assertEqual(storms.event_id.size, 2)
        self.assertEqual(storms.date.size, 2)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).year, 1999)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).month, 12)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).day, 26)
        self.assertEqual(storms.event_id[0], 1)
        self.assertEqual(storms.event_name[0], 'Lothar')
        self.assertIsInstance(storms.intensity, 
                              sparse.csr.csr_matrix)
        self.assertIsInstance(storms.fraction, 
                              sparse.csr.csr_matrix)
        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(storms.fraction.shape, (2, 9944))

    def test_read_with_ref(self):
        """ Test read_footprints while passing in a reference raster. """
        storms = StormEurope()
        storms.read_footprints(TEST_NCS, ref_raster=TEST_NCS[1])

        self.assertEqual(storms.tag.haz_type, 'WS')
        self.assertEqual(storms.units, 'm/s')
        self.assertEqual(storms.event_id.size, 2)
        self.assertEqual(storms.date.size, 2)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).year, 1999)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).month, 12)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).day, 26)
        self.assertEqual(storms.event_id[0], 1)
        self.assertEqual(storms.event_name[0], 'Lothar')
        self.assertTrue(isinstance(storms.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(storms.fraction, sparse.csr.csr_matrix))
        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(storms.fraction.shape, (2, 9944))

    def test_read_with_cent(self):
        """ Test read_footprints while passing in a Centroids object """
        storms = StormEurope()
        storms.read_footprints(TEST_NCS, centroids=TEST_CENTROIDS)

        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(
            np.count_nonzero(
                ~np.isnan(storms.centroids.region_id)
            ),
            6401
        )

    def test_set_ssi(self):
        """ Test set_ssi with both dawkins and wisc_gust methodology. """
        storms = StormEurope()
        storms.read_footprints(TEST_NCS)
        
        storms.set_ssi(method='dawkins')
        ssi_dawg = np.asarray([1.51114627e+09, 6.44053524e+08])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_dawg)
        )

        storms.set_ssi(method='wisc_gust')
        ssi_gusty = np.asarray([1.48558417e+09, 6.13437760e+08])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_gusty)
        )

        storms.set_ssi(threshold=20, on_land=False)
        ssi_special = np.asarray([3.09951236e+09, 1.29563312e+09])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_special)
        )

    def test_generate_prob_storms(self):
        """ Test the probabilistic storm generator; calls _hist2prob as well as
        Centroids.set_region_id() """
        storms = StormEurope()
        storms.read_footprints(TEST_NCS)
        storms_prob = storms.generate_prob_storms()

        self.assertEqual(
            np.count_nonzero(storms.centroids.region_id),
            6190 
            # here, we don't rasterise; we check if the centroids lie in a 
            # polygon. that is to say, it's not the majority of a raster pixel,
            # but the centroid's location that is decisive
        )
        self.assertEqual(storms_prob.size, 60)
        self.assertEqual(np.count_nonzero(storms_prob.orig), 2)
        self.assertEqual(storms_prob.centroids.size, 3054)
        self.assertIsInstance(storms_prob.intensity, 
                              sparse.csr.csr_matrix)


# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(TESTS)
