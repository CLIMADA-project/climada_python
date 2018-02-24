"""
Test hdf5_handler module.
"""

import unittest
import numpy as np
import h5py

import climada.util.hdf5_handler as hdf5
from climada.util.constants import HAZ_DEMO_MAT, HAZ_TEST_MAT, ENT_DEMO_MAT

class TestFunc(unittest.TestCase):
    '''Test the auxiliary functions used to retrieve variables from HDF5'''

    def test_get_string_pass(self):
        '''Check function to get a string from input integer array'''

        # Load input
        contents = hdf5.read(HAZ_TEST_MAT)

        # Convert several strings
        str_date = hdf5.get_string(contents['hazard']['date'])
        str_comment = hdf5.get_string(contents['hazard']['comment'])
        str_wf = hdf5.get_string(contents['hazard']['windfield_comment'])
        str_fn = hdf5.get_string(contents['hazard']['filename'])

        # Check results
        self.assertEqual('14-Nov-2017 10:09:05', str_date)
        self.assertEqual(
            'TC hazard event set, generated 14-Nov-2017 10:09:05', \
                str_comment)
        self.assertEqual(
            'generating 14450 windfields took 0.25 min ' + \
            '(0.0010 sec/event)', str_wf)
        self.assertEqual('/Users/aznarsig/Documents/MATLAB/climada_data/' + \
                         'hazards/atl_prob.mat', str_fn)

    def test_get_sparse_mat_pass(self):
        '''Check contents of imported sparse matrix, using the function \
        to build a sparse matrix from the read HDF5 variable'''

        # Load input
        contents = hdf5.read(HAZ_TEST_MAT)

        # get matrix size
        mat_shape = (len(contents['hazard']['event_ID']), \
                     len(contents['hazard']['centroid_ID']))
        spr_mat = hdf5.get_sparse_csr_mat(contents['hazard']['intensity'], \
                                      mat_shape)

        self.assertEqual(mat_shape[0], spr_mat.shape[0])
        self.assertEqual(mat_shape[1], spr_mat.shape[1])
        self.assertEqual(0, spr_mat[0, 0])
        self.assertEqual(9.2029355562400745, spr_mat[7916, 98])
        self.assertEqual(34.426829019435729, spr_mat[12839, 96])
        self.assertEqual(61.81217342446773, spr_mat[1557, 97])
        self.assertEqual(6.4470644550625753, spr_mat[6658, 97])
        self.assertEqual(0, spr_mat[298, 9])
        self.assertEqual(19.385821399329995, spr_mat[15, 0])
        self.assertEqual(52.253444444444447, spr_mat[76, 95])
        self.assertEqual(0, spr_mat[126, 86])

    def test_get_str_from_ref(self):
        """ Check import string from a HDF5 object reference"""
        file = h5py.File(ENT_DEMO_MAT, 'r')
        var = file['entity']['assets']['Value_unit'][0][0]
        res = hdf5.get_str_from_ref(ENT_DEMO_MAT, var)
        self.assertEqual('USD', res)

    def test_get_list_str_from_ref(self):
        """ Check import string from a HDF5 object reference"""
        file = h5py.File(HAZ_DEMO_MAT, 'r')
        var = file['hazard']['name']
        var_list = hdf5.get_list_str_from_ref(HAZ_DEMO_MAT, var)
        self.assertEqual('NNN_1185404_gen7', var_list[157])
        self.assertEqual('ALFA_gen8', var_list[9898])
        self.assertEqual('ALBERTO_gen6', var_list[12566])

class TestReader(unittest.TestCase):
    '''Test HDF5 reader'''

    def test_hazard_pass(self):
        '''Checking result against matlab atl_prob.mat file'''

        # Load input
        contents = hdf5.read(HAZ_TEST_MAT)

        # Check read contents
        self.assertEqual(1, len(contents))
        self.assertTrue('hazard' in contents.keys())
        self.assertEqual(False, '#refs#' in contents.keys())

        hazard = contents['hazard']
        self.assertTrue('reference_year' in hazard.keys())
        self.assertTrue('lon' in hazard.keys())
        self.assertTrue('lat' in hazard.keys())
        self.assertTrue('centroid_ID' in hazard.keys())
        self.assertTrue('orig_years' in hazard.keys())
        self.assertTrue('orig_event_count' in hazard.keys())
        self.assertTrue('event_count' in hazard.keys())
        self.assertTrue('event_ID' in hazard.keys())
        self.assertTrue('category' in hazard.keys())
        self.assertTrue('orig_event_flag' in hazard.keys())
        self.assertTrue('yyyy' in hazard.keys())
        self.assertTrue('mm' in hazard.keys())
        self.assertTrue('dd' in hazard.keys())
        self.assertTrue('datenum' in hazard.keys())
        self.assertTrue('scenario' in hazard.keys())
        self.assertTrue('intensity' in hazard.keys())
        self.assertFalse('name' in hazard.keys())
        self.assertTrue('frequency' in hazard.keys())
        self.assertTrue('matrix_density' in hazard.keys())
        self.assertTrue('windfield_comment' in hazard.keys())
        self.assertTrue('peril_ID' in hazard.keys())
        self.assertTrue('filename' in hazard.keys())
        self.assertTrue('comment' in hazard.keys())
        self.assertTrue('date' in hazard.keys())
        self.assertTrue('units' in hazard.keys())
        self.assertTrue('orig_yearset' in hazard.keys())
        self.assertTrue('fraction' in hazard.keys())
        self.assertEqual(26, len(hazard.keys()))

        # Check some random values
        mat_shape = (len(contents['hazard']['event_ID']), \
             len(contents['hazard']['centroid_ID']))
        sp_mat = hdf5.get_sparse_csr_mat(hazard['intensity'], mat_shape)

        self.assertTrue(np.array_equal(np.array([[84], [67]]), \
                                              hazard['peril_ID']))
        self.assertEqual(34.537289477809473, sp_mat[2862, 97])
        self.assertEqual(-80, hazard['lon'][46])
        self.assertEqual(28, hazard['lat'][87])
        self.assertEqual(2016, hazard['reference_year'])

    def test_with_refs_pass(self):
        '''Allow to load references of the matlab file'''

        # Load input
        refs = True
        contents = hdf5.read(HAZ_TEST_MAT, refs)

        # Check read contents
        self.assertEqual(2, len(contents))
        self.assertTrue('hazard' in contents.keys())
        self.assertTrue('#refs#' in contents.keys())

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFunc))
unittest.TextTestRunner(verbosity=2).run(TESTS)
