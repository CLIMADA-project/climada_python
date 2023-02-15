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
import gzip
import shutil
import os
from tempfile import TemporaryDirectory

import numpy as np
import scipy.sparse as sparse
from shapely.geometry import Polygon

from climada.entity.exposures.litpop import nightlight
from osgeo import gdal
from climada.util.constants import (SYSTEM_DIR, CONFIG)
from climada.util import files_handler

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
        temp_dir = TemporaryDirectory()
        # Download 'BlackMarble_2016_A1_geo_gray.tif' in the temporary directory and create a path
        urls = CONFIG.exposures.litpop.nightlights.nasa_sites.list()
        url = str(urls[0]) + 'BlackMarble_2016_A1_geo_gray.tif'
        files_handler.download_file(url=url, download_dir=temp_dir.name)

        filename = 'BlackMarble_2016_A1_geo_gray.tif'
        # Check that the array and gdal DataSet are returned
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            arr1, curr_file = nightlight.read_bm_file(bm_path=temp_dir.name, filename=filename)
        # Check correct logging
        self.assertIn('Importing' + temp_dir.name, cm.output[0])

        # Check that the outputs are a numpy array and a gdal DataSet
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(curr_file, gdal.Dataset)
        # Check that the correct band is selected
        self.assertEqual(curr_file.GetRasterBand(1).DataType, 1)

        # Release dataset, so the GC can close the file
        curr_file = None

        # Check that the right exception is raised
        with self.assertRaises(FileNotFoundError) as cm:
            nightlight.read_bm_file(bm_path='/Wrong/path/file.tif', filename='file.tif')
        self.assertEqual('Invalid path: check that the path to BlackMarble file is correct.',
                      str(cm.exception))

        # delete the temporary repository
        temp_dir.cleanup()

    def test_download_nl_files(self):
        """ Test that BlackMarble GeoTiff files are downloaded. """
        # Create a temporary directory and the associated path
        temp_dir = TemporaryDirectory()
        # Test Raises
        with self.assertRaises(ValueError) as cm:
            nightlight.download_nl_files(req_files=np.ones(5),
                                        files_exist=np.zeros(4),
                                        dwnl_path=temp_dir.name)
        self.assertEqual('The given arguments are invalid. req_files and '
                         'files_exist must both be as long as there are files to download '
                         '(8).', str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            nightlight.download_nl_files(dwnl_path='not a folder')
        self.assertEqual('The folder not a folder does not exist. Operation aborted.',
                         str(cm.exception))
        # Test logger
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            dwl_path = nightlight.download_nl_files(req_files=np.ones(len(BM_FILENAMES),),
                                        files_exist=np.ones(len(BM_FILENAMES),),
                                        dwnl_path=temp_dir.name, year=2016)
            self.assertIn('All required files already exist. No downloads necessary.', cm.output[0])
        # Test the download
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            dwl_path = nightlight.download_nl_files(req_files=np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                                        files_exist=np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                                        dwnl_path=temp_dir.name)
            self.assertIn('Attempting to download file from '
                        'https://eoimages.gsfc.nasa.gov/images/imagerecords/'
                        '144000/144897/BlackMarble_2016_A1_geo_gray.tif', cm.output[0])
            #Test if dwl_path has been returned
            self.assertEqual(temp_dir.name, dwl_path)
        # Delete the temporary repository
        temp_dir.cleanup()

    def test_unzip_tif_to_py(self):
        """ Test that .gz files are unzipped and read as a sparse matrix """
        # compress a demo .tif file to .gz
        path_file_tif = 'data/demo/earth_engine/dresden.tif'
        with open(path_file_tif, 'rb') as f_in:
            with gzip.open('dresden.tif.gz','wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # test LOGGER
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='INFO') as cm:
            file_name, night=nightlight.unzip_tif_to_py('dresden.tif.gz')
            self.assertIn('Unzipping file dresden.tif', cm.output[0])
        # test file_name
            self.assertEqual(str(file_name), 'dresden.tif')
        # test the sparse matrix
            self.assertIsInstance(night, sparse._csr.csr_matrix)
        # delate the created demo .gz file
        os.remove('dresden.tif.gz')

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightlight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
