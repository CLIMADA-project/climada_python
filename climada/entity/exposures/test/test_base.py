"""
Test Exposure base class.
"""
import os
import unittest
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.entity.exposures.source import READ_SET
from climada.hazard.base import Hazard
from climada.util.coordinates import GridPoints
from climada.util.constants import ENT_TEMPLATE_XLS

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

def good_exposures():
    """Followng values are defined for each exposure"""
    expo = Exposures()
    expo.id = np.array([1, 2, 3])
    expo.coord = np.array([[1, 2], [2, 3], [3, 4]])
    expo.value = np.array([1, 2, 3])
    expo.deductible = np.array([1, 2, 3])
    expo.cover = np.array([])
    expo.impact_id = np.array([1, 2, 3])
    expo.category_id = np.array([1, 2, 3])
    expo.region_id = np.array([1, 2, 3])
    expo.assigned['TC'] = np.array([1, 2, 3])
    return expo

class TestConstructor(unittest.TestCase):
    """Test exposures attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        expo = Exposures()

        self.assertEqual(len(expo.__dict__), 12)
        self.assertTrue(hasattr(expo, 'id'))
        self.assertTrue(hasattr(expo, 'coord'))
        self.assertTrue(hasattr(expo, 'value'))
        self.assertTrue(hasattr(expo, 'deductible'))
        self.assertTrue(hasattr(expo, 'cover'))
        self.assertTrue(hasattr(expo, 'impact_id'))
        self.assertTrue(hasattr(expo, 'category_id'))
        self.assertTrue(hasattr(expo, 'region_id'))
        self.assertTrue(hasattr(expo, 'assigned'))
        self.assertTrue(hasattr(expo, 'tag'))
        self.assertTrue(hasattr(expo, 'value_unit'))
        self.assertTrue(hasattr(expo, 'ref_year'))

    def test_get_def_vars(self):
        """ Test def_source_vars function."""
        self.assertTrue(Exposures.get_def_file_var_names('.xls') ==
                        READ_SET['XLS'][0])
        self.assertTrue(Exposures.get_def_file_var_names('.mat') ==
                        READ_SET['MAT'][0])

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_assign_diff(self):
        """Append Exposure to empty one."""
        # Fill with dummy values the GridPoints
        expo = good_exposures()

        expo_app = Exposures()
        expo_app.id = np.arange(4, 9)
        expo_app.value = np.arange(4, 9)
        expo_app.impact_id = np.arange(4, 9)
        expo_app.coord = np.ones((5, 2))
        expo_app.assigned['TC'] = np.ones((5, 2))

        expo.append(expo_app)
        self.assertTrue(len(expo.assigned['TC']), 8)

        expo_app.assigned['WS'] = np.ones((5, 2))
        expo.append(expo_app)
        self.assertTrue(len(expo.assigned['TC']), 8)
        self.assertTrue(len(expo.assigned['WS']), 5)

    def test_append_to_empty_same(self):
        """Append Exposure to empty one."""
        # Fill with dummy values the GridPoints
        expo = Exposures()
        expo_app = good_exposures()

        expo.append(expo_app)

        self.assertTrue(np.array_equal(expo.id, expo_app.id))
        self.assertTrue(np.array_equal(expo.coord, expo_app.coord))
        self.assertTrue(np.array_equal(expo.value, expo_app.value))
        self.assertTrue(np.array_equal(expo.deductible, expo_app.deductible))
        self.assertTrue(np.array_equal(expo.cover, expo_app.cover))
        self.assertTrue(np.array_equal(expo.impact_id, expo_app.impact_id))
        self.assertTrue(np.array_equal(expo.category_id, expo_app.category_id))
        self.assertTrue(np.array_equal(expo.region_id, expo_app.region_id))
        self.assertTrue(np.array_equal(expo.assigned['TC'], \
                                       expo_app.assigned['TC']))
        self.assertTrue(np.array_equal(expo.tag.description, \
                                       expo_app.tag.description))
        self.assertTrue(np.array_equal(expo.tag.file_name, \
                                       expo_app.tag.file_name))
        self.assertTrue(np.array_equal(expo.value_unit, expo_app.value_unit))
        self.assertTrue(np.array_equal(expo.ref_year, expo_app.ref_year))

    def test_append_equal_increase(self):
        """Append the same Exposure. All values are appended and new id are
        provided for the new values."""
        # Fill with dummy values the GridPoints
        expo = good_exposures()
        expo_app = good_exposures()

        expo.append(expo_app)
        expo.check()

        expo_check = good_exposures()
        self.assertTrue(np.array_equal(expo.id, \
                        np.array([1, 2, 3, 4, 5, 6])))
        self.assertTrue(np.array_equal(expo.coord, \
                        np.append(expo_check.coord, expo_app.coord, axis=0)))
        self.assertTrue(np.array_equal(expo.value, \
                        np.append(expo_check.value, expo_app.value)))
        self.assertTrue(np.array_equal(expo.deductible, \
                        np.append(expo_check.deductible, expo_app.deductible)))
        self.assertTrue(np.array_equal(expo.cover, \
                                       np.array([])))
        self.assertTrue(np.array_equal(expo.impact_id, \
                        np.append(expo_check.impact_id, expo_app.impact_id)))
        self.assertTrue(np.array_equal(expo.category_id, \
                        np.append(expo_check.category_id, \
                                  expo_app.category_id)))
        self.assertTrue(np.array_equal(expo.region_id, \
                        np.append(expo_check.region_id, expo_app.region_id)))
        self.assertTrue(np.array_equal(expo.assigned['TC'], \
                np.append(expo_check.assigned['TC'], expo_app.assigned['TC'])))
        self.assertEqual(expo.tag.description, '')
        self.assertEqual(expo.tag.file_name, '')
        self.assertTrue(np.array_equal(expo.value_unit, expo_check.value_unit))
        self.assertTrue(np.array_equal(expo.ref_year, expo_check.ref_year))

    def test_append_different_append(self):
        """Append Exposure with same and new values. All values are appended
        and same ids are substituted by new."""
        # Fill with dummy values the GridPoints
        expo = good_exposures()
        expo_app = Exposures()
        expo_app.id = np.array([1, 2, 3, 4, 5])
        expo_app.coord = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [7, 8]])
        expo_app.value = np.array([1, 2, 3, 4, 5])
        expo_app.deductible = np.array([1, 2, 3, 4, 5])
        expo_app.cover = np.array([1, 2, 3, 4, 5])
        expo_app.impact_id = np.array([1, 2, 3, 4, 5])
        expo_app.category_id = np.array([1, 2, 3, 4, 5])
        expo_app.region_id = np.array([1, 2, 3, 4, 5])
        expo_app.assigned['TC'] = np.array([1, 2, 3, 4, 5])

        expo.append(expo_app)
        expo.check()

        expo_check = good_exposures()
        self.assertTrue(np.array_equal(expo.id, \
                        np.array([1, 2, 3, 6, 7, 8, 4, 5])))
        self.assertTrue(np.array_equal(expo.coord, \
                        np.append(expo_check.coord, expo_app.coord, axis=0)))
        self.assertTrue(np.array_equal(expo.value, \
                        np.append(expo_check.value, expo_app.value)))
        self.assertTrue(np.array_equal(expo.deductible, \
                        np.append(expo_check.deductible, expo_app.deductible)))
        self.assertTrue(np.array_equal(expo.cover, \
                        np.array([])))
        self.assertTrue(np.array_equal(expo.impact_id, \
                        np.append(expo_check.impact_id, expo_app.impact_id)))
        self.assertTrue(np.array_equal(expo.category_id, \
                        np.append(expo_check.category_id, \
                        expo_app.category_id)))
        self.assertTrue(np.array_equal(expo.region_id, \
                        np.append(expo_check.region_id, expo_app.region_id)))
        self.assertTrue(np.array_equal(expo.assigned['TC'], \
                np.append(expo_check.assigned['TC'], expo_app.assigned['TC'])))
        self.assertEqual(expo.tag.description, '')
        self.assertEqual(expo.tag.file_name, '')
        self.assertTrue(np.array_equal(expo.value_unit, expo_check.value_unit))
        self.assertTrue(np.array_equal(expo.ref_year, expo_check.ref_year))

class TestAssign(unittest.TestCase):
    """Check assign function"""

    def test_assign_pass(self):
        """ Check that assigned attribute is correctly set."""
        # Fill with dummy values the GridPoints
        expo = Exposures()
        num_coord = 4
        expo.coord = np.ones((num_coord, 2))
        # Fill with dummy values the centroids
        haz = Hazard()
        haz.tag.haz_type = 'TC'
        haz.centroids.coord = GridPoints(np.ones((num_coord+6, 2)))
        # assign
        expo.assign(haz)

        # check assigned variable has been set with correct length
        self.assertEqual(num_coord, len(expo.assigned['TC']))

class TestReadParallel(unittest.TestCase):
    """Check read function with several files"""

    def test_read_two_pass(self):
        """Both files are read and appended."""
        descriptions = ['desc1', 'desc2']
        expo = Exposures([ENT_TEST_XLS, ENT_TEST_XLS], descriptions)
        self.assertEqual(expo.tag.file_name, ENT_TEST_XLS)
        self.assertEqual(expo.tag.description, 'desc1 + desc2')
        self.assertEqual(expo.id.size, 2*50)

    def test_read_incompatible_fail(self):
        """Error raised if incompatible exposures are appended."""
        expo = Exposures()
        with self.assertLogs('climada.entity.exposures', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.read([ENT_TEST_XLS, ENT_TEMPLATE_XLS])
        self.assertIn('Append not possible. Different reference years.', \
                         cm.output[0])

class TestRemove(unittest.TestCase):
    """Check read function with several files"""

    def test_read_one_pass(self):
        """Remove an exposure of existing id."""
        expo = good_exposures()
        expo.remove(1)
        expo.check()
        self.assertEqual(expo.size, 2)

    def test_read_two_pass(self):
        """Remove an exposure of existing id."""
        expo = good_exposures()
        expo.remove([1,3])
        expo.check()
        self.assertEqual(expo.size, 1)

    def test_read_three_pass(self):
        """Remove an exposure of existing id."""
        expo = good_exposures()
        expo.remove([1,3, 2])
        expo.check()
        self.assertEqual(expo.size, 0)

    def test_read_wrong_one_pass(self):
        """Remove exposure that does not exist."""
        expo = good_exposures()
        expo.remove(5)
        self.assertEqual(expo.size, 3)

class TestChecker(unittest.TestCase):
    """Test loading funcions from the Exposures class"""
    def test_check_wrongValue_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.value = np.array([1, 2])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.value size: 3 != 2.', cm.output[0])

    def test_check_wrongCoord_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.coord = np.array([[1, 2]])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.coord row size: 3 != 1.', \
                      cm.output[0])

    def test_check_wrongDeduct_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.deductible = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.deductible size: 3 != 2.', \
                         cm.output[0])

    def test_check_wrongCover_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.cover = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.cover size: 3 != 2.', cm.output[0])

    def test_check_wrongImpact_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.impact_id = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.impact_id size: 3 != 2.',
                      cm.output[0])

    def test_check_wrongCategory_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.category_id = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.category_id size: 3 != 2.',
                      cm.output[0])

    def test_check_wrongRegion_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.region_id = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.region_id size: 3 != 2.',
                      cm.output[0])

    def test_check_wrongAssigned_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.assigned['TC'] = np.array([1, 2])

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('Invalid Exposures.assigned size: 3 != 2.',
                      cm.output[0])

    def test_check_wrongId_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.id = np.array([1, 2, 1])

        with self.assertLogs('climada.entity.exposures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('There are exposures with the same identifier.',
                      cm.output[0])

class TestSelect(unittest.TestCase):
    """Test select_region from the Exposures class"""
    def test_sel_reg_pass(self):
        """Select region"""
        expo = good_exposures()
        sel_expo = expo.select(1)

        self.assertEqual(sel_expo.value.size, 1)
        self.assertEqual(sel_expo.value[0], 1)
        self.assertEqual(sel_expo.id.size, 1)
        self.assertEqual(sel_expo.id[0], 1)
        self.assertEqual(sel_expo.impact_id.size, 1)
        self.assertEqual(sel_expo.impact_id[0], 1)
        self.assertEqual(sel_expo.deductible.size, 1)
        self.assertEqual(sel_expo.deductible[0], 1)
        self.assertEqual(sel_expo.category_id.size, 1)
        self.assertEqual(sel_expo.category_id[0], 1)
        self.assertEqual(sel_expo.region_id.size, 1)
        self.assertEqual(sel_expo.region_id[0], 1)
        self.assertEqual(sel_expo.cover.size, 0)
        self.assertEqual(sel_expo.assigned['TC'].size, 1)
        self.assertEqual(sel_expo.assigned['TC'][0], 1)
        self.assertEqual(sel_expo.coord.shape[0], 1)
        self.assertEqual(sel_expo.coord[0, 0], 1)
        self.assertEqual(sel_expo.coord[0, 1], 2)
        self.assertIsInstance(sel_expo, Exposures)

    def test_sel_wrong_pass(self):
        """Select non-existent region"""
        expo = good_exposures()
        sel_expo = expo.select(5)
        self.assertEqual(sel_expo, None)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecker)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAssign))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRemove))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelect))
unittest.TextTestRunner(verbosity=2).run(TESTS)
