"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along 
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test ImpactFuncSet class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet, ImpactFunc
from climada.entity.impact_funcs.source import READ_SET
from climada.util.constants import ENT_TEMPLATE_XLS

class TestConstructor(unittest.TestCase):
    """Test impact function attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
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

    def test_get_def_vars(self):
        """ Test def_source_vars function."""
        self.assertTrue(ImpactFuncSet.get_def_file_var_names('xls') == 
                        READ_SET['XLS'][0])
        self.assertTrue(ImpactFuncSet.get_def_file_var_names('mat') == 
                        READ_SET['MAT'][0])

class TestContainer(unittest.TestCase):
    """Test ImpactFuncSet as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong ImpactFunc provided."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set', 
                             level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_func(vulner_1)
        self.assertIn("Input ImpactFunc's hazard type not set.", cm.output[0])

        vulner_1.haz_type = 'TC'
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set', 
                             level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_func(vulner_1)
        self.assertIn("Input ImpactFunc's id not set.", cm.output[0])

        with self.assertLogs('climada.entity.impact_funcs.impact_func_set', 
                             level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.add_func(45)
        self.assertIn("Input value is not of type ImpactFunc.", cm.output[0])

    def test_remove_func_pass(self):
        """Test remove_func removes ImpactFunc of ImpactFuncSet correcty."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        imp_fun.remove_func()
        self.assertEqual(0, len(imp_fun._data))

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set', level='WARNING') as cm:
            imp_fun.remove_func('FL')
        self.assertIn('No ImpactFunc with hazard FL.', cm.output[0])
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set', level='WARNING') as cm:
            imp_fun.remove_func(fun_id=3)
        self.assertIn('No ImpactFunc with id 3.', cm.output[0])

    def test_get_hazards_pass(self):
        """Test get_hazard_types function."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_2 = ImpactFunc()
        vulner_2.id = 1
        vulner_2.haz_type = 'TC'
        imp_fun.add_func(vulner_2)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_3 = ImpactFunc()
        vulner_3.id = 1
        vulner_3.haz_type = 'FL'
        imp_fun.add_func(vulner_3)
        self.assertEqual(2, len(imp_fun.get_hazard_types()))
        self.assertIn('TC', imp_fun.get_hazard_types())
        self.assertIn('FL', imp_fun.get_hazard_types())

    def test_get_ids_pass(self):
        """Test normal functionality of get_ids method."""
        imp_fun = ImpactFuncSet()
        self.assertEqual({}, imp_fun.get_ids())

        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(1, len(imp_fun.get_ids('TC')))
        self.assertEqual([1], imp_fun.get_ids('TC'))

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_func(vulner_2)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(2, len(imp_fun.get_ids('TC')))
        self.assertEqual([1, 3], imp_fun.get_ids('TC'))

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_func(vulner_3)
        self.assertEqual(2, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertIn('FL', imp_fun.get_ids())
        self.assertEqual(2, len(imp_fun.get_ids('TC')))
        self.assertEqual([1, 3], imp_fun.get_ids('TC'))
        self.assertEqual(1, len(imp_fun.get_ids('FL')))
        self.assertEqual([3], imp_fun.get_ids('FL'))

    def test_get_ids_wrong_zero(self):
        """Test get_ids method with wrong inputs."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_func(vulner_1)
        self.assertEqual([], imp_fun.get_ids('TC'))

    def test_get_func_pass(self):
        """Test normal functionality of get_func method."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=56)))
        self.assertIs(vulner_1, imp_fun.get_func('WS', 56)[0])

        vulner_2 = ImpactFunc()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 6
        imp_fun.add_func(vulner_2)
        self.assertEqual(2, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=6)))
        self.assertIs(vulner_2, imp_fun.get_func('WS', 6)[0])

        vulner_3 = ImpactFunc()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 6
        imp_fun.add_func(vulner_3)
        self.assertEqual(2, len(imp_fun.get_func(fun_id=6)))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=56)))
        self.assertEqual(2, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func('TC')))
        self.assertIs(vulner_3, imp_fun.get_func('TC', 6)[0])

        self.assertEqual(2, len(imp_fun.get_func().keys()))
        self.assertEqual(1, len(imp_fun.get_func()['TC'].keys()))
        self.assertEqual(2, len(imp_fun.get_func()['WS'].keys()))

    def test_get_func_wrong_error(self):
        """Test get_func method with wrong inputs."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_func(vulner_1)
        self.assertEqual([], imp_fun.get_func('TC'))

    def test_size_pass(self):
        """Test size function."""
        imp_fun = ImpactFuncSet()
        self.assertEqual(0, imp_fun.size())
        
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(1, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(1, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))

        vulner_2 = ImpactFunc()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 5
        imp_fun.add_func(vulner_2)
        self.assertEqual(2, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(2, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))
        self.assertEqual(1, imp_fun.size(fun_id=5))

        vulner_3 = ImpactFunc()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 5
        imp_fun.add_func(vulner_3)
        self.assertEqual(3, imp_fun.size())
        self.assertEqual(1, imp_fun.size('TC', 5))
        self.assertEqual(2, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size('TC'))
        self.assertEqual(1, imp_fun.size(fun_id=56))
        self.assertEqual(2, imp_fun.size(fun_id=5))

    def test_size_wrong_zero(self):
        """Test size method with wrong inputs."""
        imp_fun = ImpactFuncSet()
        self.assertEqual(0, imp_fun.size('TC'))
        self.assertEqual(0, imp_fun.size('TC', 3))
        self.assertEqual(0, imp_fun.size(fun_id=3))
            
    def test_add_func_pass(self):
        """Test add_func adds ImpactFunc to ImpactFuncSet correctly."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(1, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_func(vulner_2)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(2, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['TC'].keys())

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_func(vulner_3)
        self.assertEqual(2, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertIn('FL', imp_fun._data.keys())
        self.assertEqual(2, len(imp_fun._data['TC']))
        self.assertEqual(1, len(imp_fun._data['FL']))
        self.assertIn(1, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['FL'].keys())

    def test_remove_add_pass(self):
        """Test ImpactFunc can be added after removing."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        imp_fun.remove_func()
        self.assertEqual(0, len(imp_fun.get_hazard_types()))
        self.assertEqual(0, len(imp_fun.get_ids()))

        imp_fun.add_func(vulner_1)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual('TC', imp_fun.get_hazard_types()[0])
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertEqual([1], imp_fun.get_ids('TC'))

class TestChecker(unittest.TestCase):
    """Test loading funcions from the ImpactFuncSet class"""
    def test_check_wrongPAA_fail(self):
        """Wrong PAA definition"""
        imp_fun = ImpactFuncSet()
        vulner = ImpactFunc()
        vulner.id = 1
        vulner.haz_type = 'TC'
        vulner.intensity = np.array([1, 2, 3])
        vulner.mdd = np.array([1, 2, 3])
        vulner.paa = np.array([1, 2])
        imp_fun.add_func(vulner)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.check()
        self.assertIn('Invalid ImpactFunc.paa size: 3 != 2.', cm.output[0])

    def test_check_wrongMDD_fail(self):
        """Wrong MDD definition"""
        imp_fun = ImpactFuncSet()
        vulner = ImpactFunc()
        vulner.id = 1
        vulner.haz_type = 'TC'
        vulner.intensity = np.array([1, 2, 3])
        vulner.mdd = np.array([1, 2])
        vulner.paa = np.array([1, 2, 3])
        imp_fun.add_func(vulner)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.check()
        self.assertIn('Invalid ImpactFunc.mdd size: 3 != 2.', cm.output[0])

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_append_to_empty_same(self):
        """Append ImpactFuncSet to empty one."""     
        imp_fun = ImpactFuncSet()
        imp_fun_add = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.add_func(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun_add.add_func(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.add_func(vulner_3)
        
        imp_fun_add.tag.file_name = 'file1.txt'
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 3)
        self.assertEqual(imp_fun.size('TC'), 2)
        self.assertEqual(imp_fun.size('FL'), 1)
        self.assertEqual(imp_fun.tag.file_name, imp_fun_add.tag.file_name)
        self.assertEqual(imp_fun.tag.description, imp_fun_add.tag.description)

    def test_append_equal_same(self):
        """Append the same ImpactFuncSet. The inital ImpactFuncSet is obtained."""     
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)
        
        imp_fun_add = ImpactFuncSet()
        imp_fun_add.add_func(vulner_1)
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 1)
        self.assertEqual(imp_fun.size('TC'), 1)

    def test_append_different_append(self):
        """Append ImpactFuncSet with same and new values. The vulnerabilities
        with repeated id are overwritten."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.add_func(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.add_func(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.add_func(vulner_3)
        
        imp_fun_add = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.add_func(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 1
        vulner_2.haz_type = 'WS'
        imp_fun_add.add_func(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.add_func(vulner_3)
        
        imp_fun.append(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 4)
        self.assertEqual(imp_fun.size('TC'), 2)
        self.assertEqual(imp_fun.size('FL'), 1)
        self.assertEqual(imp_fun.size('WS'), 1)      

class TestReadParallel(unittest.TestCase):
    """Check read function with several files"""

    def test_read_two_pass(self):
        """Both files are readed and appended."""
        descriptions = ['desc1','desc2']
        imp_funcs = ImpactFuncSet([ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS], descriptions)
        self.assertEqual(imp_funcs.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(imp_funcs.tag.description, 'desc1 + desc2')
        self.assertEqual(imp_funcs.size(), 11)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
unittest.TextTestRunner(verbosity=2).run(TESTS)
