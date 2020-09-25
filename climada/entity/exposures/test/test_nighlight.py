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

Test Nightlight module.
"""
import unittest
import numpy as np

from climada.entity.exposures import nightlight
from climada.util.constants import SYSTEM_DIR

BM_FILENAMES = nightlight.BM_FILENAMES

class TestNightLight(unittest.TestCase):
    """Test nightlight functions."""

    def test_required_files(self):
        """Test check_required_nl_files function with various countries."""
        # Switzerland
        bbox = [5.954809204000128, 45.82071848599999, 10.466626831000013, 47.801166077000076]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),
                                      [0., 0., 0., 0., 1., 0., 0., 0.])
        np.testing.assert_array_equal(
            nightlight.check_required_nl_files(min_lon, min_lat, max_lon, max_lat),
            [0., 0., 0., 0., 1., 0., 0., 0.])

        # UK
        bbox = [-13.69131425699993, 49.90961334800005, 1.7711694670000497, 60.84788646000004]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),
                                      [0., 0., 1., 0., 1., 0., 0., 0.])
        np.testing.assert_array_equal(
            nightlight.check_required_nl_files(min_lon, min_lat, max_lon, max_lat),
            [0., 0., 1., 0., 1., 0., 0., 0.])

        # entire world
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),
                                      [1., 1., 1., 1., 1., 1., 1., 1.])
        np.testing.assert_array_equal(
            nightlight.check_required_nl_files(min_lon, min_lat, max_lon, max_lat),
            [1., 1., 1., 1., 1., 1., 1., 1.])

        # Not enough coordinates
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        self.assertRaises(ValueError, nightlight.check_required_nl_files,
                          min_lon, min_lat, max_lon)

        # Invalid coordinate order
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        self.assertRaises(ValueError, nightlight.check_required_nl_files,
                          max_lon, min_lat, min_lon, max_lat)
        self.assertRaises(ValueError, nightlight.check_required_nl_files,
                          min_lon, max_lat, max_lon, min_lat)

    def test_check_files_exist(self):
        """Test check_nightlight_local_file_exists"""
        # If invalid path is supplied it has to fall back to DATA_DIR
        np.testing.assert_array_equal(
            nightlight.check_nl_local_file_exists(
                np.ones(np.count_nonzero(BM_FILENAMES)), 'Invalid/path')[0],
            nightlight.check_nl_local_file_exists(
                np.ones(np.count_nonzero(BM_FILENAMES)), SYSTEM_DIR)[0])

    def test_download_nightlight_files(self):
        """Test check_nightlight_local_file_exists"""
        # Not the same length of arguments
        self.assertRaises(ValueError, nightlight.download_nl_files, (1, 0, 1), (1, 1))

        # The same length but not the correct length
        self.assertRaises(ValueError, nightlight.download_nl_files, (1, 0, 1), (1, 1, 1))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightLight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
