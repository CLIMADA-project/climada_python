"""
Test Hazard base class.
"""

import unittest
import datetime as dt
import numpy as np
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.source import READ_SET
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.util.constants import HAZ_TEST_XLS, HAZ_TEST_MAT

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    @staticmethod
    def good_hazard():
        """Define well a hazard"""
        haz = Hazard()
        haz.centroids = Centroids()
        haz.centroids.region_id = np.array([1, 2])
        haz.centroids.id = np.array([1, 2])
        haz.centroids.coord = np.array([[1, 2], [3, 3]])
        haz.event_id = np.array([1, 2, 3])
        haz.event_name = ['A', 'B', 'C']
        haz.frequency = np.array([1, 2, 3])
        # events x centroids
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])
        haz.fraction = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])

        return haz

    def test_check_wrongCentroids_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.centroids.region_id = np.array([1, 2, 3, 4])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Centroids.region_id size: 2 != 4.', cm.output[0])

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.frequency = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.frequency size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2]])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.intensity row size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrongFrac_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.fraction = sparse.csr_matrix([[1], [1], [1]])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.fraction column size: 2 != 1.', \
                         cm.output[0])

    def test_check_wrongEvName_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.event_name = ['M']

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.event_name size: 3 != 1.', \
                         cm.output[0])

    def test_check_wrongId_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.event_id = np.array([1, 2, 1])

        with self.assertLogs('climada.hazard.base', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('There are events with the same identifier.', \
                         cm.output[0])

    def test_check_wrong_date_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.date = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.date size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrong_orig_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.orig = np.array([1, 2, 3, 4])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.check()
        self.assertIn('Invalid Hazard.orig size: 3 != 4.', \
                         cm.output[0])

    def test_get_def_vars(self):
        """ Test def_source_vars function."""
        self.assertTrue(Hazard.get_def_file_var_names('xls') == 
                        READ_SET['XLS'][0])
        self.assertTrue(Hazard.get_def_file_var_names('.mat') == 
                        READ_SET['MAT'][0])

    def test_event_name_to_id_pass(self):
        """ Test event_name_to_id function."""
        haz = Hazard('TC', HAZ_TEST_XLS)
        self.assertEqual(haz.event_name_to_id('event001')[0], 1)
        self.assertEqual(haz.event_name_to_id('event084')[0], 84)

    def test_event_name_to_id_fail(self):
        """ Test event_name_to_id function."""
        haz = Hazard('TC', HAZ_TEST_XLS)
        with self.assertLogs('climada.hazard.base', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.event_name_to_id('1050')
        self.assertIn('No event with name: 1050', cm.output[0])

    def test_event_id_to_name_pass(self):
        """ Test event_id_to_name function."""
        haz = Hazard('TC', HAZ_TEST_XLS)
        self.assertEqual(haz.event_id_to_name(2), 'event002')
        self.assertEqual(haz.event_id_to_name(48), 'event048')

    def test_event_id_to_name_fail(self):
        """ Test event_id_to_name function."""
        haz = Hazard('TC', HAZ_TEST_XLS)
        with self.assertLogs('climada.hazard.base', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz.event_id_to_name(1050)
        self.assertIn('No event with id: 1050', cm.output[0])

    def test_date_to_str(self):
        """ Test _date_to_str function"""
        ordinal_date = dt.datetime.toordinal(dt.datetime(2018, 4, 6))
        self.assertEqual('2018-04-06', Hazard._date_to_str(ordinal_date))

    def test_get_date_strings_pass(self):
        haz = Hazard('TC', HAZ_TEST_MAT)
        haz.event_name[5] = 'HAZEL'
        haz.event_name[10] = 'HAZEL'

        self.assertEqual(len(haz.get_date_strings('HAZEL')), 2)
        self.assertEqual(haz.get_date_strings('HAZEL')[0], 
                         haz._date_to_str(haz.date[5]))
        self.assertEqual(haz.get_date_strings('HAZEL')[1], 
                         haz._date_to_str(haz.date[10]))
        
        self.assertEqual(haz.get_date_strings(2)[0], haz._date_to_str(haz.date[1]))

        self.assertEqual(len(haz.get_date_strings()), haz.date.size)
        self.assertEqual(haz.get_date_strings()[560], 
                         haz._date_to_str(haz.date[560]))

class TestAppend(unittest.TestCase):
    """Test append method."""

    @staticmethod
    def dummy_hazard():
        hazard = Hazard()
        hazard.tag = TagHazard('TC', 'file1.mat', 'Description 1')
        hazard.centroids = Centroids()
        hazard.centroids.tag = Tag('file_1.mat', 'description 1')
        hazard.centroids.coord = np.array([[1, 2], [3, 4], [5, 6]])
        hazard.centroids.id = np.array([5, 7, 9])
        hazard.event_id = np.array([1, 2, 3, 4])
        hazard.event_name = ['ev1', 'ev2', 'ev3', 'ev4']
        hazard.date = np.array([1, 2, 3, 4])
        hazard.orig = np.array([True, False, False, True])
        hazard.frequency = np.array([0.1, 0.5, 0.5, 0.2])
        hazard.fraction = sparse.csr_matrix([[0.02, 0.03, 0.04], \
                                              [0.01, 0.01, 0.01], \
                                              [0.3, 0.1, 0.0], \
                                              [0.3, 0.2, 0.0]])
        hazard.intensity = sparse.csr_matrix([[0.2, 0.3, 0.4], \
                                              [0.1, 0.1, 0.01], \
                                              [4.3, 2.1, 1.0], \
                                              [5.3, 0.2, 1.3]])
        hazard.units = 'm/s'
        
        return hazard

    def test_append_empty_fill(self):
        """Append an empty. Obtain initial hazard."""
        haz1 = Hazard('TC', HAZ_TEST_XLS)
        haz2 = Hazard('TC')
        haz1.append(haz2)
        haz1.check()
        
        # expected values
        haz1_orig = Hazard('TC', HAZ_TEST_XLS)  
        self.assertEqual(haz1.event_name, haz1_orig.event_name)
        self.assertTrue(np.array_equal(haz1.event_id, haz1_orig.event_id))
        self.assertTrue(np.array_equal(haz1.date, haz1_orig.date))
        self.assertTrue(np.array_equal(haz1.orig, haz1_orig.orig))
        self.assertTrue(np.array_equal(haz1.frequency, haz1_orig.frequency))
        self.assertTrue((haz1.intensity != haz1_orig.intensity).nnz == 0)
        self.assertTrue((haz1.fraction != haz1_orig.fraction).nnz == 0)
        self.assertEqual(haz1.units, haz1_orig.units)
        self.assertEqual(haz1.tag.file_name, haz1_orig.tag.file_name)
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, haz1_orig.tag.description)
        
    def test_append_to_empty_fill(self):
        """Append to an empty hazard a filled one. Obtain filled one."""
        haz1 = Hazard('TC')
        haz2 = Hazard('TC', HAZ_TEST_XLS)
        haz1.append(haz2)
        haz1.check()
        
        # expected values
        haz1_orig = Hazard('TC', HAZ_TEST_XLS)     
        self.assertEqual(haz1.event_name, haz1_orig.event_name)
        self.assertTrue(np.array_equal(haz1.event_id, haz1_orig.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz1_orig.frequency))
        self.assertTrue(np.array_equal(haz1.date, haz1_orig.date))
        self.assertTrue(np.array_equal(haz1.orig, haz1_orig.orig))
        self.assertTrue((haz1.intensity != haz1_orig.intensity).nnz == 0)
        self.assertTrue((haz1.fraction != haz1_orig.fraction).nnz == 0)
        self.assertEqual(haz1.units, haz1_orig.units)
        self.assertEqual(haz1.tag.file_name, haz1_orig.tag.file_name)
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, haz1_orig.tag.description)
        
    def test_equal_same(self):
        """Append the same hazard. Obtain initial hazard."""
        haz1 = Hazard('TC', HAZ_TEST_XLS)
        haz2 = Hazard('TC', HAZ_TEST_XLS)
        haz1.append(haz2)
        
        haz1.check()
        self.assertEqual(haz1.event_name, haz2.event_name)
        self.assertTrue(np.array_equal(haz1.event_id, haz2.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz2.frequency))
        self.assertTrue(np.array_equal(haz1.date, haz2.date))
        self.assertTrue(np.array_equal(haz1.orig, haz2.orig))
        self.assertTrue((haz1.intensity != haz2.intensity).nnz == 0)
        self.assertTrue((haz1.fraction != haz2.fraction).nnz == 0)
        self.assertEqual(haz1.units, haz2.units)
        self.assertEqual(haz1.tag.file_name, haz2.tag.file_name)
        self.assertEqual(haz1.tag.haz_type, haz2.tag.haz_type)
        self.assertEqual(haz1.tag.description, haz2.tag.description)

    def test_all_different_extend(self):
        """Append totally different hazard."""
        haz1 = self.dummy_hazard()
        haz2 = Hazard()
        haz2.tag = TagHazard('TC', 'file2.mat', 'Description 2')
        haz2.centroids = Centroids()
        haz2.centroids.tag = Tag('file_2.mat', 'description 2')
        haz2.centroids.coord = np.array([[7, 8], [9, 10], [11, 12]])
        haz2.centroids.id = np.array([10, 11, 12])
        haz2.event_id = np.array([5, 6, 7, 8])
        haz2.event_name = ['ev5', 'ev6', 'ev7', 'ev8']
        haz2.frequency = np.array([0.9, 0.75, 0.75, 0.22])
        haz2.fraction = sparse.csr_matrix([[0.2, 0.3, 0.4], \
                                           [0.1, 0.1, 0.1], \
                                           [0.3, 0.1, 0.9], \
                                           [0.3, 0.2, 0.8]])
        haz2.intensity = sparse.csr_matrix([[0.2, 3.3, 6.4], \
                                            [1.1, 0.1, 1.01], \
                                            [8.3, 4.1, 4.0], \
                                            [9.3, 9.2, 1.7]])
        haz2.units = 'm/s'
        
        haz1.append(haz2)
        haz1.check()

        # expected values
        haz1_orig = self.dummy_hazard()
        exp_inten = np.zeros((8, 6))
        exp_inten[0:4, 0:3] = haz1_orig.intensity.todense()
        exp_inten[4:8, 3:6] = haz2.intensity.todense()
        exp_frac = np.zeros((8, 6))
        exp_frac[0:4, 0:3] = haz1_orig.fraction.todense()
        exp_frac[4:8, 3:6] = haz2.fraction.todense()
        self.assertEqual(haz1.event_id.size, 8)
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        for i_ev in range(haz1.event_id.size):
            self.assertTrue(any((haz1.intensity[i_ev].todense() == exp_inten).all(1)))
            self.assertTrue(any((haz1.fraction[i_ev].todense() == exp_frac).all(1)))    
            self.assertTrue(haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name)
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id))
            self.assertTrue(haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency))

        self.assertEqual(haz1.centroids.id.size, 6)
        self.assertTrue(np.array_equal(haz1.centroids.id, np.array([5, 7, 9, 10, 11, 12])))
        self.assertEqual(haz1_orig.units, haz1.units)
        self.assertEqual(haz1.tag.file_name, \
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, \
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_same_centroids_extend(self):
        """Append hazard with same centroids, different events."""
        haz1 = self.dummy_hazard()
        haz2 = Hazard()
        haz2.tag = TagHazard('TC', 'file2.mat', 'Description 2')
        
        haz2.centroids = haz1.centroids
        haz2.event_id = np.array([5, 6, 7, 8])
        haz2.event_name = ['ev5', 'ev6', 'ev7', 'ev8']
        haz2.frequency = np.array([0.9, 0.75, 0.75, 0.22])
        haz2.fraction = sparse.csr_matrix([[0.2, 0.3, 0.4], \
                                           [0.1, 0.1, 0.1], \
                                           [0.3, 0.1, 0.9], \
                                           [0.3, 0.2, 0.8]])
        haz2.intensity = sparse.csr_matrix([[0.2, 3.3, 6.4], \
                                            [1.1, 0.1, 1.01], \
                                            [8.3, 4.1, 4.0], \
                                            [9.3, 9.2, 1.7]])
        haz2.units = 'm/s'
        
        haz1.append(haz2)
        haz1.check()

        # expected values
        haz1_orig = self.dummy_hazard()
        exp_inten = np.zeros((8, 3))
        exp_inten[0:4, 0:3] = haz1_orig.intensity.todense()
        exp_inten[4:8, 0:3] = haz2.intensity.todense()
        exp_frac = np.zeros((8, 3))
        exp_frac[0:4, 0:3] = haz1_orig.fraction.todense()
        exp_frac[4:8, 0:3] = haz2.fraction.todense()

        self.assertEqual(haz1.event_id.size, 8)
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        for i_ev in range(haz1.event_id.size):
            self.assertTrue(any((haz1.intensity[i_ev].todense() == exp_inten).all(1)))
            self.assertTrue(any((haz1.fraction[i_ev].todense() == exp_frac).all(1)))    
            self.assertTrue(haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name)
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id))
            self.assertTrue(haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency))

        self.assertEqual(haz1.centroids.id.size, 3)
        self.assertTrue(np.array_equal(haz1.centroids.coord, haz2.centroids.coord))
        self.assertEqual(haz1.tag.file_name, \
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, \
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_same_events_same(self):
        """Append hazard with same events (and diff centroids), remain the same."""
        haz1 = self.dummy_hazard()
        haz2 = Hazard()
        haz2.tag = TagHazard('TC', 'file2.mat', 'Description 2')
        haz2.centroids = Centroids()
        haz2.centroids.tag = Tag('file_1.mat', 'description 1')
        haz2.centroids.coord = np.array([[7, 8], [9, 10], [11, 12]])
        haz2.centroids.id = np.array([5, 7, 9])

        haz2.event_id = haz1.event_id
        haz2.event_name = haz1.event_name
        haz2.frequency = haz1.frequency
        haz2.date = haz1.date
        haz2.fraction = sparse.csr_matrix([[0.22, 0.32, 0.44], \
                                           [0.11, 0.11, 0.11], \
                                           [0.32, 0.11, 0.99], \
                                           [0.32, 0.22, 0.88]])
        haz2.intensity = sparse.csr_matrix([[0.22, 3.33, 6.44], \
                                            [1.11, 0.11, 1.11], \
                                            [8.33, 4.11, 4.4], \
                                            [9.33, 9.22, 1.77]])
        haz2.units = 'm/s'
        
        haz1.append(haz2)
        haz1.check()

        # expected values
        haz1_orig = self.dummy_hazard()
        self.assertTrue(np.array_equal(haz1_orig.intensity.todense(), \
                                       haz1.intensity.todense()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(np.array_equal(haz1_orig.fraction.todense(), \
                                       haz1.fraction.todense()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        self.assertEqual(haz1.event_name, haz1_orig.event_name)
        self.assertTrue(np.array_equal(haz1.date, haz1_orig.date))
        self.assertTrue(np.array_equal(haz1.orig, haz1_orig.orig))
        self.assertTrue(np.array_equal(haz1.event_id, \
                                       haz1_orig.event_id))
        self.assertTrue(np.array_equal(haz1.centroids.id, haz1_orig.centroids.id))
        self.assertTrue(np.array_equal(haz1.frequency, haz1_orig.frequency))
        self.assertEqual(haz1_orig.units, haz1.units)

        self.assertEqual(haz1.tag.file_name, \
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, \
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_some_diff_extend(self):
        """Append hazard with some equal events and centroids and some new. 
        New hazard overwrites old."""
        haz1 = self.dummy_hazard()
        haz2 = Hazard()
        haz2.tag = TagHazard('TC', 'file2.mat', 'Description 2')
        haz2.centroids = Centroids()
        haz2.centroids.tag = Tag('file_2.mat', 'description 2')
        haz2.centroids.coord = np.array([[3, 4], [9, 10], [11, 12]])
        haz2.centroids.id = np.array([10, 11, 12])
        haz2.event_id = np.array([5, 6, 7, 8])
        haz2.event_name = ['ev5', 'ev3', 'ev6', 'ev7']
        haz2.frequency = np.array([1, 2, 3, 4])
        haz2.date = np.array([1, 3, 4, 5])
        haz2.fraction = sparse.csr_matrix([[1, 2, 3], \
                                           [4, 5, 6], \
                                           [7, 8, 9], \
                                           [10, 11, 12]])
        haz2.intensity = sparse.csr_matrix([[-1, -2, -3], \
                                            [-4, -5, -6], \
                                            [-7, -8, -9], \
                                            [-10, -11, -12]])
        haz2.units = 'm/s'
    
        haz1.append(haz2)
        haz1.check()
    
        # expected values: ev3 and centroid [3,4] have been overwritten
        haz1_orig = self.dummy_hazard()
        exp_inten = np.zeros((7, 5))
        exp_inten[0:4, 0:3] = haz1_orig.intensity.todense()
        # ev5
        exp_inten[4, 1] = haz2.intensity[0, 0]
        exp_inten[4, 3:5] = haz2.intensity[0, 1:3].todense()
        # ev6
        exp_inten[5, 1] = haz2.intensity[2, 0]
        exp_inten[5, 3:5] = haz2.intensity[2, 1:3].todense()
        # ev7
        exp_inten[6, 1] = haz2.intensity[3, 0]
        exp_inten[6, 3:5] = haz2.intensity[3, 1:3].todense()

        exp_frac = np.zeros((7, 5))
        exp_frac[0:4, 0:3] = haz1_orig.fraction.todense()
        # ev5
        exp_frac[4, 1] = haz2.fraction[0, 0]
        exp_frac[4, 3:5] = haz2.fraction[0, 1:3].todense()
        # ev6
        exp_frac[5, 1] = haz2.fraction[2, 0]
        exp_frac[5, 3:5] = haz2.fraction[2, 1:3].todense()
        # ev7
        exp_frac[6, 1] = haz2.fraction[3, 0]
        exp_frac[6, 3:5] = haz2.fraction[3, 1:3].todense()
        
        exp_freq = haz1_orig.frequency
        exp_freq = np.append(exp_freq, np.array([haz2.frequency[0], \
                                      haz2.frequency[2], haz2.frequency[3]]))

        self.assertEqual(haz1.event_id.size, 7)
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        for i_ev in range(haz1.event_id.size):
            self.assertTrue(any((haz1.intensity[i_ev].todense() == exp_inten).all(1)))
            self.assertTrue(any((haz1.fraction[i_ev].todense() == exp_frac).all(1)))    
            self.assertTrue(haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name)
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id))
            self.assertTrue(haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency))

        self.assertEqual(haz1.centroids.id.size, 5)
        self.assertTrue(np.array_equal(haz1.centroids.id, np.array([5, 7, 9, 11, 12])))
        self.assertEqual(haz1_orig.units, haz1.units)
        self.assertEqual(haz1.tag.file_name, \
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description, \
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_incompatible_type_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = self.dummy_hazard()
        haz2 = self.dummy_hazard()
        haz2.tag = TagHazard('WS', 'file2.mat', 'Description 2')
        with self.assertLogs('climada.hazard.tag', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz1.append(haz2)
        self.assertIn("Hazards of different type can't be appended: "\
                         + "TC != WS.", cm.output[0])

    def test_incompatible_units_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = self.dummy_hazard()
        haz2 = self.dummy_hazard()
        haz2.units = 'km/h'
        with self.assertLogs('climada.hazard.base', level='ERROR') as cm:
            with self.assertRaises(ValueError): 
                haz1.append(haz2)
        self.assertIn("Hazards with different units can't be appended: "\
            + 'm/s != km/h.', cm.output[0])

class TestReadParallel(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def test_two_same_append(self):
        """Read in parallel two times the same file. Hazard does not contain 
        repetition."""
        haz = Hazard('TC')
        file_names = [HAZ_TEST_XLS, HAZ_TEST_XLS]
        haz.read(file_names)
        self.assertTrue(np.array_equal(haz.event_id, np.arange(1,101)))
        haz.check()
    
    def test_two_diff_append(self):
        """Read in parallel a matlab file and an excel file."""        
        haz = Hazard('TC')
        file_names = [HAZ_TEST_XLS, HAZ_TEST_MAT]
        haz.read(file_names)
        self.assertTrue(haz.event_id.size, 14550)
        haz.check()

class TestStats(unittest.TestCase):
    """Test return period statistics"""

    def test_degenerate_pass(self):
        """ Test degenerate call. """
        haz = Hazard('TC', HAZ_TEST_MAT)
        return_period = np.array([25, 50, 100, 250])
        haz.intensity = np.zeros(haz.intensity.shape)
        inten_stats = haz._compute_stats(return_period)
        self.assertTrue(np.array_equal(inten_stats, np.zeros((4, 100))))

    def test_ref_all_pass(self):
        """Compare against reference."""        
        haz = Hazard(file_name=HAZ_TEST_MAT)
        return_period = np.array([25, 50, 100, 250])
        inten_stats = haz._compute_stats(return_period)

        self.assertAlmostEqual(inten_stats[0][0], 55.424015590131290)
        self.assertAlmostEqual(inten_stats[1][0], 67.221687644669998)
        self.assertAlmostEqual(inten_stats[2][0], 79.019359699208721)
        self.assertAlmostEqual(inten_stats[3][0], 94.615033842370963)

        self.assertAlmostEqual(inten_stats[1][66], 70.608592953031405)
        self.assertAlmostEqual(inten_stats[3][33], 88.510983305123631)
        self.assertAlmostEqual(inten_stats[2][99], 79.717518054203623)

    def test_ref_orig_pass(self):
        """Compare against reference."""        
        haz = Hazard(file_name=HAZ_TEST_MAT)
        return_period = np.array([25, 50, 100, 250])
        inten_stats = haz._compute_stats(return_period, True)

        self.assertAlmostEqual(inten_stats[0][0], 55.763055630205507)
        self.assertAlmostEqual(inten_stats[0][82], 51.142476557641359)
        self.assertAlmostEqual(inten_stats[2][96], 79.651362050077225)
        self.assertAlmostEqual(inten_stats[1][20], 66.439658831126991)

class TestYearset(unittest.TestCase):
    """Test return period statistics"""

    def test_ref_pass(self):
        """ Test against matlab reference. """
        haz = Hazard('TC', HAZ_TEST_MAT)
        orig_year_set = haz.calc_year_set()
        
        self.assertTrue(np.array_equal(np.array(list(orig_year_set.keys())), 
                                       np.arange(1851, 2012)))
        self.assertTrue(np.array_equal(orig_year_set[1851], 
                                       np.array([1,11,21,31])))
        self.assertTrue(np.array_equal(orig_year_set[1958], 
                                       np.array([8421,8431,8441,8451,8461,8471,8481,8491,8501,8511])))
        self.assertTrue(np.array_equal(orig_year_set[1986], 
                                       np.array([11101,11111,11121,11131,11141,11151])))
        self.assertTrue(np.array_equal(orig_year_set[1997], 
                                       np.array([12221,12231,12241,12251,12261,12271,12281,12291])))
        self.assertTrue(np.array_equal(orig_year_set[2006], 
                                       np.array([13571,13581,13591,13601,13611,13621,13631,13641,13651,13661])))
        self.assertTrue(np.array_equal(orig_year_set[2010], 
                                       np.array([14071,14081,14091,14101,14111,14121,14131,14141,14151,14161,14171,14181,14191,14201,14211,14221,14231,14241,14251])))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestAppend)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoader))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStats))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestYearset))
unittest.TextTestRunner(verbosity=2).run(TESTS)
