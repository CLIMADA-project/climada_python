"""
Test ImpactFuncs class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.base import ImpactFuncs, Vulnerability
from climada.util.constants import ENT_TEMPLATE_XLS

class TestConstructor(unittest.TestCase):
    """Test impact function attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'TC'
        vulner_1.id = '2'
        self.assertTrue(hasattr(imp_fun, 'tag'))
        self.assertTrue(hasattr(imp_fun, '_data'))
        self.assertTrue(hasattr(vulner_1, 'haz_type'))
        self.assertTrue(hasattr(vulner_1, 'name'))
        self.assertTrue(hasattr(vulner_1, 'id'))
        self.assertTrue(hasattr(vulner_1, 'intensity_unit'))
        self.assertTrue(hasattr(vulner_1, 'mdd'))
        self.assertTrue(hasattr(vulner_1, 'paa'))

class TestContainer(unittest.TestCase):
    """Test ImpactFuncs as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong Vulnerability provided."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        with self.assertLogs('climada.entity.impact_funcs.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_vulner(vulner_1)
        self.assertIn("Input Vulnerability's hazard type not set.", cm.output[0])

        vulner_1.haz_type = 'TC'
        with self.assertLogs('climada.entity.impact_funcs.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_vulner(vulner_1)
        self.assertIn("Input Vulnerability's id not set.", cm.output[0])

        with self.assertLogs('climada.entity.impact_funcs.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_vulner(45)
        self.assertIn("Input value is not of type Vulnerability.", cm.output[0])

    def test_remove_vulner_pass(self):
        """Test remove_vulner removes Vulnerability of ImpactFuncs correcty."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        imp_fun.remove_vulner()
        self.assertEqual(0, len(imp_fun._data))

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        with self.assertLogs('climada.entity.impact_funcs.base', level='WARNING') as cm:
            imp_fun.remove_vulner('FL')
        self.assertIn('No Vulnerability with hazard FL.', cm.output[0])
        with self.assertLogs('climada.entity.impact_funcs.base', level='WARNING') as cm:
            imp_fun.remove_vulner(vul_id=3)
        self.assertIn('No Vulnerability with id 3.', cm.output[0])

    def test_get_hazards_pass(self):
        """Test get_hazard_types function."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_2 = Vulnerability()
        vulner_2.id = 1
        vulner_2.haz_type = 'TC'
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_3 = Vulnerability()
        vulner_3.id = 1
        vulner_3.haz_type = 'FL'
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(2, len(imp_fun.get_hazard_types()))
        self.assertIn('TC', imp_fun.get_hazard_types())
        self.assertIn('FL', imp_fun.get_hazard_types())

    def test_get_ids_pass(self):
        """Test normal functionality of get_ids method."""
        imp_fun = ImpactFuncs()
        self.assertEqual({}, imp_fun.get_ids())

        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(1, len(imp_fun.get_ids('TC')))
        self.assertEqual([1], imp_fun.get_ids('TC'))

        vulner_2 = Vulnerability()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(2, len(imp_fun.get_ids('TC')))
        self.assertEqual([1, 3], imp_fun.get_ids('TC'))

        vulner_3 = Vulnerability()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(2, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertIn('FL', imp_fun.get_ids())
        self.assertEqual(2, len(imp_fun.get_ids('TC')))
        self.assertEqual([1, 3], imp_fun.get_ids('TC'))
        self.assertEqual(1, len(imp_fun.get_ids('FL')))
        self.assertEqual([3], imp_fun.get_ids('FL'))

    def test_get_ids_wrong_zero(self):
        """Test get_ids method with wrong inputs."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        self.assertEqual([], imp_fun.get_ids('TC'))

    def test_get_vulner_pass(self):
        """Test normal functionality of get_vulner method."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=56)))
        self.assertIs(vulner_1, imp_fun.get_vulner('WS', 56)[0])

        vulner_2 = Vulnerability()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 6
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(2, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=6)))
        self.assertIs(vulner_2, imp_fun.get_vulner('WS', 6)[0])

        vulner_3 = Vulnerability()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 6
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(2, len(imp_fun.get_vulner(vul_id=6)))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=56)))
        self.assertEqual(2, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner('TC')))
        self.assertIs(vulner_3, imp_fun.get_vulner('TC', 6)[0])

        self.assertEqual(2, len(imp_fun.get_vulner().keys()))
        self.assertEqual(1, len(imp_fun.get_vulner()['TC'].keys()))
        self.assertEqual(2, len(imp_fun.get_vulner()['WS'].keys()))

    def test_get_vulner_wrong_error(self):
        """Test get_vulner method with wrong inputs."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        self.assertEqual([], imp_fun.get_vulner('TC'))

    def test_num_vulner_pass(self):
        """Test num_vulner function."""
        imp_fun = ImpactFuncs()
        self.assertEqual(0, imp_fun.num_vulner())
        
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, imp_fun.num_vulner())
        self.assertEqual(1, imp_fun.num_vulner('WS', 56))
        self.assertEqual(1, imp_fun.num_vulner('WS'))
        self.assertEqual(1, imp_fun.num_vulner(vul_id=56))
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, imp_fun.num_vulner())
        self.assertEqual(1, imp_fun.num_vulner('WS', 56))
        self.assertEqual(1, imp_fun.num_vulner('WS'))
        self.assertEqual(1, imp_fun.num_vulner(vul_id=56))

        vulner_2 = Vulnerability()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 5
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(2, imp_fun.num_vulner())
        self.assertEqual(1, imp_fun.num_vulner('WS', 56))
        self.assertEqual(2, imp_fun.num_vulner('WS'))
        self.assertEqual(1, imp_fun.num_vulner(vul_id=56))
        self.assertEqual(1, imp_fun.num_vulner(vul_id=5))

        vulner_3 = Vulnerability()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 5
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(3, imp_fun.num_vulner())
        self.assertEqual(1, imp_fun.num_vulner('TC', 5))
        self.assertEqual(2, imp_fun.num_vulner('WS'))
        self.assertEqual(1, imp_fun.num_vulner('TC'))
        self.assertEqual(1, imp_fun.num_vulner(vul_id=56))
        self.assertEqual(2, imp_fun.num_vulner(vul_id=5))

    def test_num_vulner_wrong_zero(self):
        """Test num_vulner method with wrong inputs."""
        imp_fun = ImpactFuncs()
        self.assertEqual(0, imp_fun.num_vulner('TC'))
        self.assertEqual(0, imp_fun.num_vulner('TC', 3))
        self.assertEqual(0, imp_fun.num_vulner(vul_id=3))
            
    def test_add_vulner_pass(self):
        """Test add_vulner adds Vulnerability to ImpactFuncs correctly."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(1, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())

        vulner_2 = Vulnerability()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(2, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['TC'].keys())

        vulner_3 = Vulnerability()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(2, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertIn('FL', imp_fun._data.keys())
        self.assertEqual(2, len(imp_fun._data['TC']))
        self.assertEqual(1, len(imp_fun._data['FL']))
        self.assertIn(1, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['FL'].keys())

    def test_remove_add_pass(self):
        """Test vulnerability can be added after removing."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        imp_fun.remove_vulner()
        self.assertEqual(0, len(imp_fun.get_hazard_types()))
        self.assertEqual(0, len(imp_fun.get_ids()))

        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual('TC', imp_fun.get_hazard_types()[0])
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertEqual([1], imp_fun.get_ids('TC'))

class TestChecker(unittest.TestCase):
    """Test loading funcions from the ImpactFuncs class"""
    def test_check_wrongPAA_fail(self):
        """Wrong PAA definition"""
        imp_fun = ImpactFuncs()
        vulner = Vulnerability()
        vulner.id = 1
        vulner.haz_type = 'TC'
        vulner.intensity = np.array([1, 2, 3])
        vulner.mdd = np.array([1, 2, 3])
        vulner.paa = np.array([1, 2])
        imp_fun.add_vulner(vulner)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.check()
        self.assertIn('Invalid Vulnerability.paa size: 3 != 2.', cm.output[0])

    def test_check_wrongMDD_fail(self):
        """Wrong MDD definition"""
        imp_fun = ImpactFuncs()
        vulner = Vulnerability()
        vulner.id = 1
        vulner.haz_type = 'TC'
        vulner.intensity = np.array([1, 2, 3])
        vulner.mdd = np.array([1, 2])
        vulner.paa = np.array([1, 2, 3])
        imp_fun.add_vulner(vulner)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.check()
        self.assertIn('Invalid Vulnerability.mdd size: 3 != 2.', cm.output[0])

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_append_to_empty_same(self):
        """Append ImpactFuncs to empty one."""     
        imp_fun = ImpactFuncs()
        imp_fun_add = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.add_vulner(vulner_1)

        vulner_2 = Vulnerability()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun_add.add_vulner(vulner_2)

        vulner_3 = Vulnerability()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.add_vulner(vulner_3)
        
        imp_fun_add.tag.file_name = 'file1.txt'
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.num_vulner(), 3)
        self.assertEqual(imp_fun.num_vulner('TC'), 2)
        self.assertEqual(imp_fun.num_vulner('FL'), 1)
        self.assertEqual(imp_fun.tag.file_name, imp_fun_add.tag.file_name)
        self.assertEqual(imp_fun.tag.description, imp_fun_add.tag.description)

    def test_append_equal_same(self):
        """Append the same ImpactFuncs. The inital ImpactFuncs is obtained."""     
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)
        
        imp_fun_add = ImpactFuncs()
        imp_fun_add.add_vulner(vulner_1)
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.num_vulner(), 1)
        self.assertEqual(imp_fun.num_vulner('TC'), 1)

    def test_append_different_append(self):
        """Append ImpactFuncs with same and new values. The vulnerabilities
        with repeated id are overwritten."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_vulner(vulner_1)

        vulner_2 = Vulnerability()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_vulner(vulner_2)

        vulner_3 = Vulnerability()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_vulner(vulner_3)
        
        imp_fun_add = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.add_vulner(vulner_1)

        vulner_2 = Vulnerability()
        vulner_2.id = 1
        vulner_2.haz_type = 'WS'
        imp_fun_add.add_vulner(vulner_2)

        vulner_3 = Vulnerability()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.add_vulner(vulner_3)
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.num_vulner(), 4)
        self.assertEqual(imp_fun.num_vulner('TC'), 2)
        self.assertEqual(imp_fun.num_vulner('FL'), 1)
        self.assertEqual(imp_fun.num_vulner('WS'), 1)      

class TestReadParallel(unittest.TestCase):
    """Check read function with several files"""

    def test_read_two_pass(self):
        """Both files are readed and appended."""
        descriptions = ['desc1','desc2']
        imp_funcs = ImpactFuncs([ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS], descriptions)
        self.assertEqual(imp_funcs.tag.file_name, [ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS])
        self.assertEqual(imp_funcs.tag.description, descriptions)
        self.assertEqual(imp_funcs.num_vulner(), 11)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
unittest.TextTestRunner(verbosity=2).run(TESTS)
