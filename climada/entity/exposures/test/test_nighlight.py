"""
Test Nightlight module.
"""
import unittest
import numpy as np
from scipy import sparse

from climada.entity.exposures import nightlight
from climada.util.constants import SYSTEM_DIR

BM_FILENAMES = nightlight.BM_FILENAMES

class TestNightLight(unittest.TestCase):
    """Test nightlight functions."""

    def test_required_files(self):
        """ Test check_required_nl_files function with various countries."""
        #Switzerland
        bbox = [5.954809204000128, 45.82071848599999, 10.466626831000013, 47.801166077000076]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),\
                         [0., 0., 0., 0., 1., 0., 0., 0.])
        np.testing.assert_array_equal(nightlight.check_required_nl_files(min_lon, min_lat,\
                                max_lon, max_lat), [0., 0., 0., 0., 1., 0., 0., 0.])

        #UK
        bbox = [-13.69131425699993, 49.90961334800005, 1.7711694670000497, 60.84788646000004]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),\
                         [0., 0., 1., 0., 1., 0., 0., 0.])
        np.testing.assert_array_equal(nightlight.check_required_nl_files(min_lon,\
                        min_lat, max_lon, max_lat), [0., 0., 1., 0., 1., 0., 0., 0.])

        #entire world
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        np.testing.assert_array_equal(nightlight.check_required_nl_files(bbox),\
                         [1., 1., 1., 1., 1., 1., 1., 1.])
        np.testing.assert_array_equal(nightlight.check_required_nl_files(min_lon,\
                        min_lat, max_lon, max_lat), [1., 1., 1., 1., 1., 1., 1., 1.])

        #Not enough coordinates
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        self.assertRaises(ValueError, nightlight.check_required_nl_files,\
                          min_lon, min_lat, max_lon)

        #Invalid coordinate order
        bbox = [-180, -90, 180, 90]
        min_lon, min_lat, max_lon, max_lat = bbox

        self.assertRaises(ValueError, nightlight.check_required_nl_files,\
                          max_lon, min_lat, min_lon, max_lat)
        self.assertRaises(ValueError, nightlight.check_required_nl_files,\
                          min_lon, max_lat, max_lon, min_lat)

    def test_check_files_exist(self):
        """ Test check_nightlight_local_file_exists"""
        # If invalid path is supplied it has to fall back to DATA_DIR
        np.testing.assert_array_equal(nightlight.check_nl_local_file_exists(np.ones\
                       (np.count_nonzero(BM_FILENAMES)), 'Invalid/path')[0],\
                        nightlight.check_nl_local_file_exists(np.ones\
                      (np.count_nonzero(BM_FILENAMES)), SYSTEM_DIR)[0])

    def test_download_nightlight_files(self):
        """ Test check_nightlight_local_file_exists"""
        # Not the same length of arguments
        self.assertRaises(ValueError, nightlight.download_nl_files ,(1, 0, 1), (1, 1))

        # The same length but not the correct length
        self.assertRaises(ValueError, nightlight.download_nl_files, (1, 0, 1), (1, 1, 1))
        
    def test_cut_nl_nasa_1_pass(self):
        """Test cut_nl_nasa situation 2->3->4->5."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21599, 21600)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (1, 0)
        in_lon_nb = (1, 2)
        
        idx_info = [2, -1, False]
        aux_nl = np.zeros((21600, 21600))
        aux_nl[21599, 21599] = 100
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (1, 1))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        
        idx_info[0] = 3
        idx_info[1] = 2
        aux_nl = np.zeros((21600, 21600))
        aux_nl[0, 21599] = 101
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (2, 1))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        self.assertEqual(nl_mat.tocsr()[1, 0], 101.0)
        
        idx_info[0] = 4
        idx_info[1] = 3
        aux_nl = np.zeros((21600, 21600))
        aux_nl[21599, 0] = 102
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (2, 2))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        self.assertEqual(nl_mat.tocsr()[1, 0], 101.0)
        self.assertEqual(nl_mat.tocsr()[0, 1], 102.0)

        idx_info[0] = 5
        idx_info[1] = 4
        aux_nl = np.zeros((21600, 21600))
        aux_nl[0, 0] = 103
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (2, 2))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        self.assertEqual(nl_mat.tocsr()[1, 0], 101.0)
        self.assertEqual(nl_mat.tocsr()[0, 1], 102.0)
        self.assertEqual(nl_mat.tocsr()[1, 1], 103.0)
        
    def test_cut_nl_nasa_2_pass(self):
        """Test cut_nl_nasa situation 3->5."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21599, 21599)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (1, 1)
        in_lon_nb = (1, 2)
        
        idx_info = [3, -1, False]
        aux_nl = np.zeros((21600, 21600))
        aux_nl[0, 21599] = 100
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (1, 1))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        
        idx_info[0] = 5
        idx_info[1] = 3
        aux_nl = np.zeros((21600, 21600))
        aux_nl[0, 0] = 101
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (1, 2))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        self.assertEqual(nl_mat.tocsr()[0, 1], 101.0)

    def test_cut_nl_nasa_3_pass(self):
        """Test cut_nl_nasa situation 2->4."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21600, 21600)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (0, 0)
        in_lon_nb = (1, 2)
    
        idx_info = [2, -1, False]
        aux_nl = np.zeros((21600, 21600))
        aux_nl[21599, 21599] = 100
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (1, 1))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        
        idx_info[0] = 4
        idx_info[1] = 2
        aux_nl = np.zeros((21600, 21600))
        aux_nl[21599, 0] = 101
        nightlight.cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                        in_lon, in_lat_nb, in_lon_nb)
        
        self.assertEqual(nl_mat.shape, (1, 2))
        self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
        self.assertEqual(nl_mat.tocsr()[0, 1], 101.0)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestNightLight)
unittest.TextTestRunner(verbosity=2).run(TESTS)
