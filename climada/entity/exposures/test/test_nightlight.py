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

Test Nightlight module.
"""
import unittest
import numpy as np

from climada.entity.exposures.litpop import nightlight
from climada.util.constants import SYSTEM_DIR
from pathlib import Path

BM_FILENAMES = nightlight.BM_FILENAMES

class TestNightLight(unittest.TestCase):
    """Test nightlight functions."""

    def test_required_files(self):
        """Test get_required_nl_files function with various countries."""
        # Switzerland
        bbox = (5.954809204000128, 45.82071848599999, 10.466626831000013, 47.801166077000076)
        # min_lon, min_lat, max_lon, max_lat = bbox
        np.testing.assert_array_equal(nightlight.get_required_nl_files(bbox),
                                      [0., 0., 0., 0., 1., 0., 0., 0.])

        # UK
        bbox = (-13.69131425699993, 49.90961334800005, 1.7711694670000497, 60.84788646000004)
        np.testing.assert_array_equal(nightlight.get_required_nl_files(bbox),
                                      [0., 0., 1., 0., 1., 0., 0., 0.])

        # entire world
        bbox = (-180, -90, 180, 90)
        np.testing.assert_array_equal(nightlight.get_required_nl_files(bbox),
                                      [1., 1., 1., 1., 1., 1., 1., 1.])

        # Invalid coordinate order or bbox length
        self.assertRaises(ValueError, nightlight.get_required_nl_files,
                          (-180, 90, 180, -90))
        self.assertRaises(ValueError, nightlight.get_required_nl_files,
                          (180, -90, -180, 90))
        self.assertRaises(ValueError, nightlight.get_required_nl_files,
                          (-90, 90))

    def test_check_files_exist(self):
        """Test check_nightlight_local_file_exists"""
        # If invalid directory is supplied it has to fail
        try:
            nightlight.check_nl_local_file_exists(
                np.ones(np.count_nonzero(BM_FILENAMES)), 'Invalid/path')[0]
            raise Exception("if the path is not valid, check_nl_local_file_exists should fail")
        except ValueError:
            pass
        files_exist = nightlight.check_nl_local_file_exists(
            np.ones(np.count_nonzero(BM_FILENAMES)), SYSTEM_DIR)
        self.assertTrue(
            files_exist.sum() > 0,
            f'{files_exist} {BM_FILENAMES}'
        )

    def test_download_nightlight_files(self):
        """Test check_nightlight_local_file_exists"""
        # Not the same length of arguments
        self.assertRaises(ValueError, nightlight.download_nl_files, (1, 0, 1), (1, 1))

        # The same length but not the correct length
        self.assertRaises(ValueError, nightlight.download_nl_files, (1, 0, 1), (1, 1, 1))

    def test_get_required_nl_files(self):
        """ get_required_nl_files return a boolean matrix of 0 and 1
            indicating which tile of NASA nighlight files are needed giving
            a bounding box. This test check a few configuration of tiles
            and check that a value error is raised if the bounding box are 
            incorrect """

        # incorrect bounds: bounds size =!  4, min lon > max lon, min lat > min lat
        BOUNDS = [(20, 30, 40), 
                  (120, -20, 110, 30), 
                  (-120, 50, 130, 10)]
        # correct bounds
        bounds_c1 = (-120, -20, 0, 40)
        bounds_c2 = (-70, -20, 10, 40)
        bounds_c3 = (160, 10, 180, 40)

        for bounds in BOUNDS:
            with self.assertRaises(ValueError) as cm:
    
                nightlight.get_required_nl_files(bounds = bounds)

                self.assertEqual('Invalid bounds supplied. `bounds` must be tuple'
                                ' with (min_lon, min_lat, max_lon, max_lat).',
                                str(cm.exception))
        
        # test first correct bounds configurations
        req_files = nightlight.get_required_nl_files(bounds = bounds_c1)
        bool = np.array_equal(np.array([1, 1, 1, 1, 1, 1, 0, 0]), req_files)
        self.assertTrue(bool)
        # second correct configuration
        req_files = nightlight.get_required_nl_files(bounds = bounds_c2)
        bool = np.array_equal(np.array([0, 0, 1, 1, 1, 1, 0, 0]), req_files)
        self.assertTrue(bool)
        # third correct configuration
        req_files = nightlight.get_required_nl_files(bounds = bounds_c3)
        bool = np.array_equal(np.array([0, 0, 0, 0, 0, 0, 1, 0]), req_files)
        self.assertTrue(bool)
    
    def test_check_nl_local_file_exists(self):
        """ Test that an array with the correct number of already existing files
            is produced, the LOGGER messages logged and the ValueError raised. """

        # check logger messages by giving a to short req_file
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'WARNING') as cm:
            nightlight.check_nl_local_file_exists(required_files = np.array([0, 0, 1, 1]))
        self.assertIn('The parameter \'required_files\' was too short and '
                       'is ignored.', cm.output[0])
             
        # check logger message: not all files are available 
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'DEBUG') as cm:
            nightlight.check_nl_local_file_exists()
        self.assertIn('Not all satellite files available. '
                     f'Found 5 out of 8 required files in {Path(SYSTEM_DIR)}', cm.output[0])
            
        # check logger message: no files found in checkpath 
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'INFO') as cm:
            # using a random path where no files are stored
            nightlight.check_nl_local_file_exists(check_path = Path('climada/entity/exposures'))
        self.assertIn('No satellite files found locally in climada/entity/exposures', cm.output[0])
      
        # test raises with wrong path
        with self.assertRaises(ValueError) as cm:
            nightlight.check_nl_local_file_exists(check_path = '/random/wrong/path')
        self.assertEqual('The given path does not exist: /random/wrong/path', str(cm.exception))

        # test that files_exist is correct 
        files_exist = nightlight.check_nl_local_file_exists()
        self.assertEqual(int(sum(files_exist)), 5)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightLight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
