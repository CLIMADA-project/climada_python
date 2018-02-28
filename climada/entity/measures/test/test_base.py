"""
Test Measures class.
"""

import unittest
import numpy

from climada.entity.measures.base import Measures, Action
from climada.util.constants import ENT_TEMPLATE_XLS

class TestConstructor(unittest.TestCase):
    """Test impact function attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Seawall'
        self.assertTrue(hasattr(meas, 'tag'))
        self.assertTrue(hasattr(meas, '_data'))
        self.assertTrue(hasattr(act_1, 'name'))
        self.assertTrue(hasattr(act_1, 'color_rgb'))
        self.assertTrue(hasattr(act_1, 'cost'))
        self.assertTrue(hasattr(act_1, 'hazard_freq_cutoff'))
        self.assertTrue(hasattr(act_1, 'hazard_intensity'))
        self.assertTrue(hasattr(act_1, 'mdd_impact'))
        self.assertTrue(hasattr(act_1, 'paa_impact'))
        self.assertTrue(hasattr(act_1, 'risk_transf_attach'))
        self.assertTrue(hasattr(act_1, 'risk_transf_cover'))
        
class TestContainer(unittest.TestCase):
    """Test Measures as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong Vulnerability provided."""
        meas = Measures()
        act_1 = Action()
        with self.assertLogs('climada.entity.measures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.add_action(act_1)
        self.assertIn("Input Action's name not set.", cm.output[0])

        with self.assertLogs('climada.entity.measures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.add_action(45)
        self.assertIn("Input value is not of type Action.", cm.output[0])

    def test_remove_action_pass(self):
        """Test remove_action removes Action of Measures correcty."""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        meas.add_action(act_1)
        meas.remove_action('Mangrove')
        self.assertEqual(0, len(meas._data))

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        meas.add_action(act_1)
        with self.assertLogs('climada.entity.measures.base', level='WARNING') as cm:
            meas.remove_action('Seawall')
        self.assertIn('No Action with name Seawall.', cm.output[0])

    def test_get_names_pass(self):
        """Test get_names function."""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        meas.add_action(act_1)
        self.assertEqual(1, len(meas.get_names()))
        self.assertEqual(['Mangrove'], meas.get_names())

        act_2 = Action()
        act_2.name = 'Seawall'
        meas.add_action(act_2)
        self.assertEqual(2, len(meas.get_names()))
        self.assertIn('Mangrove', meas.get_names())
        self.assertIn('Seawall', meas.get_names())

    def test_get_action_pass(self):
        """Test normal functionality of get_action method."""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        meas.add_action(act_1)
        self.assertIs(act_1, meas.get_action('Mangrove'))

        act_2 = Action()
        act_2.name = 'Seawall'
        meas.add_action(act_2)
        self.assertIs(act_1, meas.get_action('Mangrove'))
        self.assertIs(act_2, meas.get_action('Seawall'))
        self.assertEqual(2, len(meas.get_action()))

    def test_get_action_wrong_error(self):
        """Test get_action method with wrong inputs."""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Seawall'
        meas.add_action(act_1)
        self.assertEqual([], meas.get_action('Mangrove'))

    def test_num_action_pass(self):
        """Test num_action function."""
        meas = Measures()
        self.assertEqual(0, meas.num_action())
        act_1 = Action()
        act_1.name = 'Mangrove'
        meas.add_action(act_1)
        self.assertEqual(1, meas.num_action())
        meas.add_action(act_1)
        self.assertEqual(1, meas.num_action())

        act_2 = Action()
        act_2.name = 'Seawall'
        meas.add_action(act_2)
        self.assertEqual(2, meas.num_action())

class TestChecker(unittest.TestCase):
    """Test check functionality of the Measures class"""

    def test_check_wronginten_fail(self):
        """Wrong intensity definition"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.hazard_intensity = (1, 2, 3)
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        meas.add_action(act_1)
        
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Action.hazard_intensity size: 2 != 3.', \
                         cm.output[0])

    def test_check_wrongColor_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = (1, 2)
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_intensity = (1, 2)
        meas.add_action(act_1)
        
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Action.color_rgb size: 3 != 2.', cm.output[0])

    def test_check_wrongMDD_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1)
        act_1.paa_impact = (1, 2)
        act_1.hazard_intensity = (1, 2)
        meas.add_action(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Action.mdd_impact has wrong dimensions.', cm.output[0])

    def test_check_wrongPAA_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2, 3, 4)
        act_1.hazard_intensity = (1, 2)
        meas.add_action(act_1)
        
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Action.paa_impact size: 2 != 4.', cm.output[0])

    def test_check_name_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        act_1 = Action()
        act_1.name = 'LaLa'
        meas._data['LoLo'] = act_1
        with self.assertRaises(ValueError) as error:
            meas.check()
        self.assertEqual('Wrong Action.name: LoLo != LaLa', \
                         str(error.exception))

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_append_to_empty_same(self):
        """Append Measures to empty one.""" 
        meas = Measures()
        meas_add = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_intensity = (1, 2)
        meas_add.add_action(act_1)
        
        meas.append(meas_add)
        meas.check()
        
        self.assertEqual(meas.num_action(), 1)
        self.assertEqual(meas.get_names(), [act_1.name])

    def test_append_equal_same(self):
        """Append the same Measures. The inital Measures is obtained."""     
        meas = Measures()
        meas_add = Measures()
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_intensity = (1, 2)
        meas.add_action(act_1)
        meas_add.add_action(act_1)
        
        meas.append(meas_add)
        meas.check()

        self.assertEqual(meas.num_action(), 1)
        self.assertEqual(meas.get_names(), [act_1.name])

    def test_append_different_append(self):
        """Append Measures with same and new values. The actions
        with repeated name are overwritten."""
        act_1 = Action()
        act_1.name = 'Mangrove'
        act_1.color_rgb = numpy.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_intensity = (1, 2)
        
        act_11 = Action()
        act_11.name = 'Mangrove'
        act_11.color_rgb = numpy.array([1, 1, 1])
        act_11.mdd_impact = (1, 2)
        act_11.paa_impact = (1, 3)
        act_11.hazard_intensity = (1, 2)
        
        act_2 = Action()
        act_2.name = 'Anything'
        act_2.color_rgb = numpy.array([1, 1, 1])
        act_2.mdd_impact = (1, 2)
        act_2.paa_impact = (1, 2)
        act_2.hazard_intensity = (1, 2)
        
        meas = Measures()
        meas.add_action(act_1)
        meas_add = Measures()
        meas_add.add_action(act_11)
        meas_add.add_action(act_2)
        
        meas.append(meas_add)
        meas.check()

        self.assertEqual(meas.num_action(), 2)
        self.assertEqual(meas.get_names(), [act_1.name, act_2.name])
        self.assertEqual(meas.get_action(act_1.name).paa_impact, act_11.paa_impact)

class TestReadParallel(unittest.TestCase):
    """Check read function with several files"""

    def test_read_two_pass(self):
        """Both files are readed and appended."""
        descriptions = ['desc1','desc2']
        meas = Measures([ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS], descriptions)
        self.assertEqual(meas.tag.file_name, [ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS])
        self.assertEqual(meas.tag.description, descriptions)
        self.assertEqual(meas.num_action(), 7)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
unittest.TextTestRunner(verbosity=2).run(TESTS)
