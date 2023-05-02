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
import os
import tarfile
import affine
import numpy as np
import scipy.sparse as sparse

from shapely.geometry import Polygon
from pathlib import Path
from tempfile import TemporaryDirectory
from osgeo import gdal
from climada.entity.exposures.litpop import nightlight
from climada.util.constants import (SYSTEM_DIR, CONFIG)
from climada.util import (files_handler, ureg)

BM_FILENAMES = nightlight.BM_FILENAMES
NOAA_RESOLUTION_DEG = (30 * ureg.arc_second).to(ureg.deg).magnitude

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

    def test_load_nasa_nl_shape_single_tile(self):
        """ Test that the function returns a np.ndarray containing 
            the cropped .tif image values. Test that
            just one layer is returned. """

        # Initialization
        path = Path(SYSTEM_DIR, 'BlackMarble_2016_C1_geo_gray.tif')
        _, shape = init_test_shape()

        # Test cropped output
        out_image, meta = nightlight.load_nasa_nl_shape_single_tile(geometry = shape, path = path)
        self.assertIsInstance(out_image, np.ndarray)
        self.assertEqual(len(out_image.shape), 2) 

        # Test meta ouput
        self.assertEqual(meta['height'],out_image.shape[0])
        self.assertEqual(meta['width'],out_image.shape[1])
        self.assertEqual(meta['driver'], 'GTiff')
        self.assertEqual(meta['transform'], affine.Affine(0.004166666666666667, 0.0, 14.179166666666667,
                                                            0.0, -0.004166666666666667, 36.09166666666667))
        # Test raises
        with self.assertRaises(IndexError) as cm:
            nightlight.load_nasa_nl_shape_single_tile(geometry = shape, path = path, layer = 4)
        self.assertEqual("BlackMarble_2016_C1_geo_gray.tif has only 3 layers, layer 4 can't be accessed.",
                        str(cm.exception))
        # Test logger
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            nightlight.load_nasa_nl_shape_single_tile(geometry = shape, path = path)
        self.assertIn('Read cropped BlackMarble_2016_C1_geo_gray.tif as np.ndarray.', cm.output[0])
        
    def test_read_bm_files(self):
        """" Test that read_bm_files function read NASA BlackMarble GeoTiff and output
             an array and a gdal DataSet."""
        
        # Download 'BlackMarble_2016_A1_geo_gray.tif' in the temporary directory and create a path
        temp_dir = TemporaryDirectory()
        urls = CONFIG.exposures.litpop.nightlights.nasa_sites.list()
        url = str(urls[0]) + 'BlackMarble_2016_A1_geo_gray.tif'
        files_handler.download_file(url = url, download_dir = temp_dir.name)
        filename = 'BlackMarble_2016_A1_geo_gray.tif'

        # Test logger
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            arr1, curr_file = nightlight.read_bm_file(bm_path=temp_dir.name, filename=filename)
        self.assertIn('Importing' + temp_dir.name, cm.output[0])

        # Check outputs are a np.array and a gdal DataSet and band 1 is selected
        self.assertIsInstance(arr1, np.ndarray)
        self.assertIsInstance(curr_file, gdal.Dataset)
        self.assertEqual(curr_file.GetRasterBand(1).DataType, 1)

        # Release dataset, so the GC can close the file
        curr_file = None

        # Check that the right exception is raised
        with self.assertRaises(FileNotFoundError) as cm:
            nightlight.read_bm_file(bm_path='/Wrong/path/file.tif', filename='file.tif')
        self.assertEqual('Invalid path: check that the path to BlackMarble file is correct.',
                      str(cm.exception))
        temp_dir.cleanup()

    def test_download_nl_files(self):
        """ Test that BlackMarble GeoTiff files are downloaded. """
        
        # Test Raises
        temp_dir = TemporaryDirectory()
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

        # Test download
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='DEBUG') as cm:
            dwl_path = nightlight.download_nl_files(req_files=np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                                        files_exist=np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                                        dwnl_path=temp_dir.name)
        self.assertIn('Attempting to download file from '
                        'https://eoimages.gsfc.nasa.gov/images/imagerecords/'
                        '144000/144897/BlackMarble_2016_A1_geo_gray.tif', cm.output[0])
        # Test if dwl_path has been returned
        self.assertEqual(temp_dir.name, dwl_path)
        temp_dir.cleanup()

    def test_unzip_tif_to_py(self):
        """ Test that .gz files are unzipped and read as a sparse matrix, 
            file_name is correct and logger message recorded. """

        path_file_tif_gz = str(Path(SYSTEM_DIR, 'F182013.v4c_web.stable_lights.avg_vis.tif.gz'))
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level='INFO') as cm:
            file_name, night = nightlight.unzip_tif_to_py(path_file_tif_gz)
        self.assertIn(f'Unzipping file {path_file_tif_gz}', cm.output[0])
        self.assertEqual(str(file_name), 'F182013.v4c_web.stable_lights.avg_vis.tif')
        self.assertIsInstance(night, sparse._csr.csr_matrix)
        os.remove(Path(SYSTEM_DIR, 'F182013.v4c_web.stable_lights.avg_vis.p'))

    def test_load_nightlight_noaa(self):
        """ Test that data is downloaded if not present in SYSTEM_DIR,
            or not downloaded if present. Test the three outputs of the
            function."""

        # Using an already existing file and without providing arguments
        night, coord_nl, fn_light = nightlight.load_nightlight_noaa()
        self.assertIsInstance(night, sparse._csr.csr_matrix)
        self.assertIn('F182013.v4c_web.stable_lights.avg_vis.tif',str(fn_light))
        self.assertTrue(np.array_equal(np.array([[-65,NOAA_RESOLUTION_DEG],
                                    [-180, NOAA_RESOLUTION_DEG]]),coord_nl))
        os.remove(SYSTEM_DIR.joinpath('F182013.v4c_web.stable_lights.avg_vis.p'))
        # With arguments
        night, coord_nl, fn_light = nightlight.load_nightlight_noaa(ref_year = 2010, sat_name = 'F18')
        self.assertIsInstance(night, sparse._csr.csr_matrix)
        self.assertIn('F182010.v4d_web.stable_lights.avg_vis', str(fn_light))
        file = str(SYSTEM_DIR.joinpath('F182010.v4d_web.stable_lights.avg_vis.p')) 
        os.remove(file)

        # Test raises from wrong input agruments
        with self.assertRaises(ValueError) as cm:
            night, coord_nl, fn_light = nightlight.load_nightlight_noaa(
                                            ref_year = 2050, sat_name = 'F150')
        self.assertEqual('Nightlight intensities for year 2050 and satellite F150 do not exist.', 
                            str(cm.exception))

    def test_untar_noaa_stable_nighlight(self):
        """ Testing that input .tar file is moved into SYSTEM_DIR, 
            tif.gz file is extracted from .tar file and moved into SYSTEM_DIR, 
            exception are raised when no .tif.gz file is present in the tar file,
            and the logger message is recorded if more then one .tif.gz is present in 
            .tar file. """

        # Create path to .tif.gz and .csv files already existing in SYSTEM_DIR
        path_tif_gz_1 = Path(SYSTEM_DIR, 'F182013.v4c_web.stable_lights.avg_vis.tif.gz') 
        path_csv = Path(SYSTEM_DIR, 'GDP_TWN_IMF_WEO_data.csv')  
        path_tar = Path(SYSTEM_DIR, 'sample.tar')  

        # Create .tar file and add .tif.gz and .csv 
        file_tar = tarfile.open(path_tar, "w") #create the tar file
        file_tar.add(name = path_tif_gz_1, recursive = False, arcname = 'F182013.v4c_web.stable_lights.avg_vis.tif.gz') 
        file_tar.close()

        # Test that the files has been moved
        path_to_test = nightlight.untar_noaa_stable_nightlight(path_tar)
        self.assertTrue(path_to_test.exists())
        self.assertTrue(path_tar .exists())
        os.remove(path_tar)

        # Put no .tif.gz file in .tar file and check raises
        path_tar = Path(SYSTEM_DIR, 'sample.tar') 
        file_tar = tarfile.open(path_tar, "w") #create the tar file
        file_tar.add(name = path_csv, recursive = False, arcname ='GDP_TWN_IMF_WEO_data.csv' ) 
        file_tar.close()
        with self.assertRaises(ValueError) as cm:
            nightlight.untar_noaa_stable_nightlight(path_tar)
        self.assertEqual('No stable light intensities for selected year and satellite '
                            f'in file {path_tar}',str(cm.exception))
        os.remove(path_tar)

        # Test logger with having two .tif.gz file in .tar file  
        file_tar = tarfile.open(path_tar, "w") #create the tar file
        file_tar.add(name = path_tif_gz_1, recursive = False, arcname = 'F182013.v4c_web.stable_lights.avg_vis.tif.gz' ) 
        file_tar.add(name = path_tif_gz_1, recursive = False, arcname = 'F182013.v4c_web.stable_lights.avg_vis.tif.gz' ) 
        file_tar.close()
        with self.assertLogs('climada.entity.exposures.litpop.nightlight', level = 'WARNING') as cm:
            nightlight.untar_noaa_stable_nightlight(path_tar)
        self.assertIn('found more than one potential intensity file in', cm.output[0])
        os.remove(path_tar)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightlight)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
