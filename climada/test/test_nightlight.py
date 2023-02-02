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

Tests on Black marble.
"""

import unittest
import numpy as np

from shapely.geometry import Polygon
from climada.entity.exposures.litpop import nightlight 
from climada.util.constants import (SYSTEM_DIR, CONFIG)
from climada.util import files_handler 
from osgeo import gdal
from pathlib import Path
from tempfile import TemporaryDirectory

BM_FILENAMES = nightlight.BM_FILENAMES

def init_test_shape():
    bounds = (14.18, 35.78, 14.58, 36.09)
    # (min_lon, max_lon, min_lat, max_lat)

    return bounds, Polygon([
        (bounds[0], bounds[3]),
        (bounds[2], bounds[3]),
        (bounds[2], bounds[1]),
        (bounds[0], bounds[1])
        ])

class TestNightlight(unittest.TestCase):
    """Test litpop.nightlight"""

    def test_load_nasa_nl_shape_single_tile_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()

    def test_load_nasa_nl_2016_shape_pass(self):
        """load_nasa_nl_shape_single_tile pass"""
        bounds, shape = init_test_shape()
        nightlight.load_nasa_nl_shape(shape, 2016, data_dir=SYSTEM_DIR, dtype=None)

        # data, meta = nightlight.load_nasa_nl_shape_single_tile(shape, path, layer=0):

    def test_read_bm_files(self):
        """" Test that read_bm_files function read NASA BlackMarble GeoTiff and output
             an array and a gdal DataSet."""
        # Create a temporary directory and the associated path
        TEMPDIR = TemporaryDirectory()
        tempdir_path = str(Path(TEMPDIR.name))
        # Download 'BlackMarble_2016_A1_geo_gray.tif' in the temporary directory and create a path 
        urls = CONFIG.exposures.litpop.nightlights.nasa_sites.list()
        url = str(urls[0]) + 'BlackMarble_2016_A1_geo_gray.tif'
        files_handler.download_file(url = url, download_dir = tempdir_path)
        file_path = str(Path(TEMPDIR.name, 'BlackMarble_2016_A1_geo_gray.tif'))
        # Check that the array and gdal DataSet are returned 
        arr1, curr_file = nightlight.read_bm_file(file_path)
        # Check that the outputs are a numpy array and a gdal DataSet
        self.assertIsInstance(arr1, np.ndarray) 
        self.assertIsInstance(curr_file, gdal.Dataset) 
        # Check that the correct band is selected 
        self.assertEqual(curr_file.GetRasterBand(1).DataType, 1) 
        # Check that the right exception is raised 
        with self.assertRaises(FileNotFoundError) as cm:
            nightlight.read_bm_file(file_path = '/Wrong/path/file.tif')   
        self.assertEqual('Invalid path: check that the path to BlackMarble file is correct.',
                      str(cm.exception))
        # Check if the logger message is correct
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'DEBUG') as cm:
            nightlight.read_bm_file(file_path)
            self.assertIn('Importing' + file_path, cm.output[0])
        # delate the temporary repository
        TEMPDIR.cleanup()



    def test_download_nl_files(self):
        """ Test that BlackMarble GeoTiff files are downloaded. """
        # Create a temporary directory and the associated path
        TEMPDIR = TemporaryDirectory()
        tempdir_path = str(Path(TEMPDIR.name))
        # Test Raises
        with self.assertRaises(ValueError) as cm:
            nightlight.download_nl_files(req_files=np.ones(5),
                                        files_exist=np.zeros(4),
                                        dwnl_path = tempdir_path)
        self.assertEqual('The given arguments are invalid. req_files and '
                         'files_exist must both be as long as there are files to download '
                         '(8).', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            nightlight.download_nl_files(dwnl_path = 'not a folder')
        self.assertEqual('The folder not a folder does not exist. Operation aborted.', str(cm.exception))
        # Test logger 
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'DEBUG') as cm:
            dwl_path = nightlight.download_nl_files(req_files=np.ones(len(BM_FILENAMES),),
                                        files_exist=np.ones(len(BM_FILENAMES),),
                                        dwnl_path = tempdir_path, year = 2016)
            self.assertIn('All required files already exist. No downloads necessary.', cm.output[0])
        # Test the download 
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'DEBUG') as cm:
            dwl_path = nightlight.download_nl_files(req_files = np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                                        files_exist = np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                                        dwnl_path = tempdir_path)
            self.assertIn('Attempting to download file from '
                        'https://eoimages.gsfc.nasa.gov/images/imagerecords/'
                        '144000/144897/BlackMarble_2016_A1_geo_gray.tif', cm.output[0])
            #Test if dwl_path has been returned 
            self.assertEqual(tempdir_path == dwl_path, True) 
        # Delate the temporary repository
        TEMPDIR.cleanup()

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightlight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
