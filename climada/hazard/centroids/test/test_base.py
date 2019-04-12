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

Test Centroids base class.
"""

import os
import unittest
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.hazard.centroids.source import READ_SET
from climada.hazard.centroids.tag import Tag

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
CENTR_BRB = os.path.join(DATA_DIR, 'centr_brb_test.mat')

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Centroids class"""

    @staticmethod
    def good_centroids():
        """Define well a Centroids"""
        cen = Centroids()
        cen.coord = np.array([[1, 2], [3, 4], [5, 6]])
        cen.id = np.array([1, 2, 3])
        cen.region_id = np.array([1, 2, 3])

        return cen

    def test_check_wrongCoord_fail(self):
        """Wrong centroids definition"""
        cen = self.good_centroids()
        cen.coord = np.array([[1, 2], [3, 4]])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                cen.check()
        self.assertIn('Invalid Centroids.coord row size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrongRegion_fail(self):
        """Wrong centroids definition"""
        cen = self.good_centroids()
        cen.region_id = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                cen.check()
        self.assertIn('Invalid Centroids.region_id size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrongId_fail(self):
        """Wrong centroids definition"""
        cen = self.good_centroids()
        cen.id = np.array([1, 2, 2])
        with self.assertLogs('climada.hazard.centroids.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                cen.check()
        self.assertIn('There are centroids with the same identifier.', \
                         cm.output[0])

    def test_get_def_vars(self):
        """ Test def_source_vars function."""
        self.assertTrue(Centroids.get_def_file_var_names('xls') ==
                        READ_SET['XLS'][0])
        self.assertTrue(Centroids.get_def_file_var_names('.mat') ==
                        READ_SET['MAT'][0])

    def test_get_nearest_id_pass(self):
        """ Get id of nearest centroids."""
        cen = self.good_centroids()
        lat, lon = 4.9, 6.1
        self.assertEqual(cen.get_nearest_id(lat, lon), cen.id[2])

        lat, lon = 0.1, 1.2
        self.assertEqual(cen.get_nearest_id(lat, lon), cen.id[0])

        lat, lon = 0.1, 4.1
        self.assertEqual(cen.get_nearest_id(lat, lon), cen.id[0])

class TestAppend(unittest.TestCase):
    """Test append function."""

    def test_appended_type(self):
        """Append the same centroids."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])
        centr2 = centr1

        centr1.append(centr2)
        self.assertEqual(type(centr1.tag.file_name), str)
        self.assertEqual(type(centr1.tag.description), str)
        self.assertEqual(type(centr1.coord), np.ndarray)
        self.assertEqual(type(centr1.id), np.ndarray)
        self.assertTrue(type(centr1.region_id), np.ndarray)

    def test_append_empty_fill(self):
        """Append the same centroids."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])
        centr2 = Centroids()

        centr1.append(centr2)
        self.assertEqual(centr1.tag.file_name, 'file_1.mat')
        self.assertEqual(centr1.tag.description, 'description 1')
        self.assertEqual(centr1.coord.shape, (3, 2))
        self.assertTrue(np.array_equal(centr1.coord[0, :], [1, 2]))
        self.assertTrue(np.array_equal(centr1.coord[1, :], [3, 4]))
        self.assertTrue(np.array_equal(centr1.coord[2, :], [5, 6]))
        self.assertEqual(centr1.id.shape, (3,))
        self.assertTrue(np.array_equal(centr1.id, np.array([5, 7, 9])))
        self.assertTrue(np.array_equal(centr1.region_id, np.array([], int)))

    def test_append_to_empty_fill(self):
        """Append to empty centroids."""
        centr1 = Centroids()
        centr2 = Centroids()
        centr2.tag = Tag('file_1.mat', 'description 1')
        centr2.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr2.id = np.array([5, 7, 9])

        centr1.append(centr2)
        self.assertEqual(centr1.tag.file_name, 'file_1.mat')
        self.assertEqual(centr1.tag.description, 'description 1')
        self.assertEqual(centr1.coord.shape, (3, 2))
        self.assertTrue(np.array_equal(centr1.coord[0, :], [1, 2]))
        self.assertTrue(np.array_equal(centr1.coord[1, :], [3, 4]))
        self.assertTrue(np.array_equal(centr1.coord[2, :], [5, 6]))
        self.assertEqual(centr1.id.shape, (3,))
        self.assertTrue(np.array_equal(centr1.id, np.array([5, 7, 9])))
        self.assertTrue(np.array_equal(centr1.region_id, np.array([], int)))

    def test_same_centroids_pass(self):
        """Append the same centroids."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])

        centr2 = centr1

        centr1.append(centr2)
        self.assertEqual(centr1.tag.file_name, 'file_1.mat')
        self.assertEqual(centr1.tag.description, 'description 1')
        self.assertEqual(centr1.coord.shape, (3, 2))
        self.assertTrue(np.array_equal(centr1.coord[0, :], [1, 2]))
        self.assertTrue(np.array_equal(centr1.coord[1, :], [3, 4]))
        self.assertTrue(np.array_equal(centr1.coord[2, :], [5, 6]))
        self.assertEqual(centr1.id.shape, (3,))
        self.assertTrue(np.array_equal(centr1.id, np.array([5, 7, 9])))
        self.assertTrue(np.array_equal(centr1.region_id, np.array([], int)))

    def test_new_elem_pass(self):
        """Append a centroids with a new element."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])
        centr1.region_id = np.array([1, 2, 3])
        centr1.dist_coast = np.array([1.5, 2.6, 3.5])

        centr2 = Centroids()
        centr2.tag = Tag('file_2.mat', 'description 2')
        centr2.coord = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        centr2.id = np.array([6, 7, 9, 1])
        centr2.region_id = np.array([3, 4, 5, 6])
        centr2.dist_coast = np.array([4.5, 5.6, 6.5, 7.8])

        centr1.append(centr2)
        self.assertEqual(len(centr1.tag.file_name), 2)
        self.assertEqual(len(centr1.tag.description), 2)
        self.assertEqual(centr1.coord.shape, (4, 2))
        self.assertTrue(np.array_equal(centr1.coord[0, :], [1, 2]))
        self.assertTrue(np.array_equal(centr1.coord[1, :], [3, 4]))
        self.assertTrue(np.array_equal(centr1.coord[2, :], [5, 6]))
        self.assertTrue(np.array_equal(centr1.coord[3, :], [7, 8]))
        self.assertEqual(centr1.id.shape, (4,))
        self.assertTrue(np.array_equal(centr1.id, np.array([5, 7, 9, 1])))
        self.assertTrue(np.array_equal(centr1.region_id,
                                       np.array([1, 2, 3, 6])))
        self.assertTrue(np.array_equal(centr1.dist_coast,
                                       np.array([1.5, 2.6, 3.5, 7.8])))

    def test_all_new_elem_pass(self):
        """Append a centroids with a new element."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])

        centr2 = Centroids()
        centr2.tag = Tag('file_2.mat', 'description 2')
        centr2.coord = np.array([[7, 8], [1, 5]])
        centr2.id = np.array([5, 7])

        centr1.append(centr2)
        self.assertEqual(len(centr1.tag.file_name), 2)
        self.assertEqual(len(centr1.tag.description), 2)
        self.assertEqual(centr1.coord.shape, (5, 2))
        self.assertTrue(np.array_equal(centr1.coord[0, :], [1, 2]))
        self.assertTrue(np.array_equal(centr1.coord[1, :], [3, 4]))
        self.assertTrue(np.array_equal(centr1.coord[2, :], [5, 6]))
        self.assertTrue(np.array_equal(centr1.coord[3, :], [7, 8]))
        self.assertTrue(np.array_equal(centr1.coord[4, :], [1, 5]))
        self.assertEqual(centr1.id.shape, (5,))
        self.assertTrue(np.array_equal(centr1.id, \
                                              np.array([5, 7, 9, 10, 11])))
        self.assertTrue(np.array_equal(centr1.region_id, np.array([], int)))

    def test_without_region_pass(self):
        """Append centroids without region id."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])

        centr2 = Centroids()
        centr2.tag = Tag('file_2.mat', 'description 2')
        centr2.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr2.id = np.array([5, 7, 9])
        centr2.region_id = np.array([1, 1, 1])

        centr1.append(centr2)
        self.assertTrue(np.array_equal(centr1.region_id, np.array([], int)))

        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])
        centr1.region_id = np.array([1, 1, 1])

        centr2 = Centroids()
        centr2.tag = Tag('file_2.mat', 'description 2')
        centr2.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr2.id = np.array([5, 7, 9])

        centr1.append(centr2)
        self.assertTrue(np.array_equal(centr1.region_id, np.array([1, 1, 1])))

    def test_with_region_pass(self):
        """Append the same centroids with region id."""
        centr1 = Centroids()
        centr1.tag = Tag('file_1.mat', 'description 1')
        centr1.coord = np.array([[1, 2], [3, 4], [5, 6]])
        centr1.id = np.array([5, 7, 9])
        centr1.region_id = np.array([1, 1, 1])

        centr2 = centr1

        centr1.append(centr2)
        self.assertTrue(np.array_equal(centr1.region_id, np.array([1, 1, 1])))

class TestSelect(unittest.TestCase):
    """ Test select method """

    def test_select_pass(self):
        """ Select successfully."""
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.region_id = np.arange(centr_brb.size)
        sel_brb = centr_brb.select(reg_id=5)
        self.assertEqual(sel_brb.size, 1)
        self.assertEqual(sel_brb.id.size, 1)
        self.assertEqual(sel_brb.region_id.size, 1)
        self.assertEqual(sel_brb.dist_coast.size, 1)
        self.assertEqual(sel_brb.region_id[0], 5)
        self.assertEqual(sel_brb.coord.shape, (1, 2))

    def test_select_prop_pass(self):
        """ Select successfully with a property set."""
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.region_id = np.arange(centr_brb.size)
        centr_brb.set_on_land()
        sel_brb = centr_brb.select(reg_id=5)
        self.assertEqual(sel_brb.size, 1)
        self.assertEqual(sel_brb.id.size, 1)
        self.assertEqual(sel_brb.region_id.size, 1)
        self.assertEqual(sel_brb.on_land.size, 1)
        self.assertEqual(sel_brb.dist_coast.size, 1)
        self.assertEqual(sel_brb.region_id[0], 5)
        self.assertEqual(sel_brb.coord.shape, (1, 2))

    def test_select_prop_list_pass(self):
        """ Select successfully with a property set."""
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.region_id = np.arange(centr_brb.size)
        centr_brb.set_on_land()
        sel_brb = centr_brb.select(reg_id=[4, 5])
        self.assertEqual(sel_brb.size, 2)
        self.assertEqual(sel_brb.id.size, 2)
        self.assertEqual(sel_brb.region_id.size, 2)
        self.assertEqual(sel_brb.on_land.size, 2)
        self.assertEqual(sel_brb.dist_coast.size, 2)
        self.assertTrue(np.array_equal(sel_brb.region_id, [4, 5]))
        self.assertEqual(sel_brb.coord.shape, (2, 2))

class TestMethods(unittest.TestCase):
    """Test additional methods."""

    def test_calc_dist_coast_pass(self):
        """Test against reference data."""
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.set_dist_coast()
        self.assertEqual(centr_brb.id.size, centr_brb.dist_coast.size)
        self.assertAlmostEqual(5.7988200982894105, centr_brb.dist_coast[1])
        self.assertAlmostEqual(166.36505441711506, centr_brb.dist_coast[-2])

    def test_set_region_id(self):
        """ Test that the region id setter works """
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.set_region_id()
        self.assertEqual(
            np.count_nonzero(centr_brb.region_id),
            6
        )
        self.assertEqual(centr_brb.region_id[0], 52) # 052 for barbados

    def test_remove_duplicate_coord_pass(self):
        """Test removal of duplicate coords."""
        centr_brb = Centroids(CENTR_BRB)
        centr_brb.set_dist_coast()
        # create duplicates manually:
        centr_brb.coord[100] = centr_brb.coord[101]
        centr_brb.coord[120] = centr_brb.coord[101]
        centr_brb.coord[5] = [12.5, -59.7]
        centr_brb.coord[133] = [12.5, -59.7]
        centr_brb.coord[121] = [12.5, -59.7]
        self.assertEqual(centr_brb.size, 296)
        with self.assertLogs('climada.hazard.centroids.base', level='INFO') as cm:
            centr_brb.remove_duplicate_coord()
        self.assertIn('Removing duplicate centroids', cm.output[0])
        self.assertIn('12.5 -59.7', cm.output[1])
        self.assertIn('12.54166', cm.output[2])
        self.assertEqual(centr_brb.size, 292) # 5 centroids removed...
        with self.assertLogs('climada.hazard.centroids.base', level='INFO') as cm:
            centr_brb.remove_duplicate_coord()
        self.assertIn('No centroids with duplicate coordinates', cm.output[0])
        self.assertEqual(centr_brb.dist_coast.size, 292)
        self.assertEqual(centr_brb.id.size, 292)
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSelect)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoader))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMethods))
unittest.TextTestRunner(verbosity=2).run(TESTS)
