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

Test ImpactFuncSet class.
"""
import unittest
import numpy as np

from climada import CONFIG
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet, ImpactFunc
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_TODAY

ENT_TEST_MAT = CONFIG.exposures.test_data.dir().joinpath('demo_today.mat')

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

class TestContainer(unittest.TestCase):
    """Test ImpactFuncSet as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong ImpactFunc provided."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set',
                             level='WARNING') as cm:
            imp_fun.append(vulner_1)
        self.assertIn("Input ImpactFunc's hazard type not set.", cm.output[0])

        vulner_1.haz_type = 'TC'
        with self.assertLogs('climada.entity.impact_funcs.impact_func_set',
                             level='WARNING') as cm:
            imp_fun.append(vulner_1)
        self.assertIn("Input ImpactFunc's id not set.", cm.output[0])

        with self.assertLogs('climada.entity.impact_funcs.impact_func_set',
                             level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.append(45)
        self.assertIn("Input value is not of type ImpactFunc.", cm.output[0])

    def test_remove_func_pass(self):
        """Test remove_func removes ImpactFunc of ImpactFuncSet correcty."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.append(vulner_1)
        imp_fun.remove_func()
        self.assertEqual(0, len(imp_fun._data))

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.append(vulner_1)
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
        imp_fun.append(vulner_1)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_2 = ImpactFunc()
        vulner_2.id = 1
        vulner_2.haz_type = 'TC'
        imp_fun.append(vulner_2)
        self.assertEqual(1, len(imp_fun.get_hazard_types()))
        self.assertEqual(['TC'], imp_fun.get_hazard_types())

        vulner_3 = ImpactFunc()
        vulner_3.id = 1
        vulner_3.haz_type = 'FL'
        imp_fun.append(vulner_3)
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
        imp_fun.append(vulner_1)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(1, len(imp_fun.get_ids('TC')))
        self.assertEqual([1], imp_fun.get_ids('TC'))

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.append(vulner_2)
        self.assertEqual(1, len(imp_fun.get_ids()))
        self.assertIn('TC', imp_fun.get_ids())
        self.assertEqual(2, len(imp_fun.get_ids('TC')))
        self.assertEqual([1, 3], imp_fun.get_ids('TC'))

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.append(vulner_3)
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
        imp_fun.append(vulner_1)
        self.assertEqual([], imp_fun.get_ids('TC'))

    def test_get_func_pass(self):
        """Test normal functionality of get_func method."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.append(vulner_1)
        self.assertEqual(1, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=56)))
        self.assertIs(vulner_1, imp_fun.get_func('WS', 56))

        vulner_2 = ImpactFunc()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 6
        imp_fun.append(vulner_2)
        self.assertEqual(2, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=6)))
        self.assertIs(vulner_2, imp_fun.get_func('WS', 6))

        vulner_3 = ImpactFunc()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 6
        imp_fun.append(vulner_3)
        self.assertEqual(2, len(imp_fun.get_func(fun_id=6)))
        self.assertEqual(1, len(imp_fun.get_func(fun_id=56)))
        self.assertEqual(2, len(imp_fun.get_func('WS')))
        self.assertEqual(1, len(imp_fun.get_func('TC')))
        self.assertIs(vulner_3, imp_fun.get_func('TC', 6))

        self.assertEqual(2, len(imp_fun.get_func().keys()))
        self.assertEqual(1, len(imp_fun.get_func()['TC'].keys()))
        self.assertEqual(2, len(imp_fun.get_func()['WS'].keys()))

    def test_get_func_wrong_error(self):
        """Test get_func method with wrong inputs."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.append(vulner_1)
        self.assertEqual([], imp_fun.get_func('TC'))

    def test_size_pass(self):
        """Test size function."""
        imp_fun = ImpactFuncSet()
        self.assertEqual(0, imp_fun.size())

        vulner_1 = ImpactFunc()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.append(vulner_1)
        self.assertEqual(1, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(1, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))
        imp_fun.append(vulner_1)
        self.assertEqual(1, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(1, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))

        vulner_2 = ImpactFunc()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 5
        imp_fun.append(vulner_2)
        self.assertEqual(2, imp_fun.size())
        self.assertEqual(1, imp_fun.size('WS', 56))
        self.assertEqual(2, imp_fun.size('WS'))
        self.assertEqual(1, imp_fun.size(fun_id=56))
        self.assertEqual(1, imp_fun.size(fun_id=5))

        vulner_3 = ImpactFunc()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 5
        imp_fun.append(vulner_3)
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

    def test_append_pass(self):
        """Test append adds ImpactFunc to ImpactFuncSet correctly."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.append(vulner_1)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(1, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.append(vulner_2)
        self.assertEqual(1, len(imp_fun._data))
        self.assertIn('TC', imp_fun._data.keys())
        self.assertEqual(2, len(imp_fun._data['TC']))
        self.assertIn(1, imp_fun._data['TC'].keys())
        self.assertIn(3, imp_fun._data['TC'].keys())

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.append(vulner_3)
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
        imp_fun.append(vulner_1)
        imp_fun.remove_func()
        self.assertEqual(0, len(imp_fun.get_hazard_types()))
        self.assertEqual(0, len(imp_fun.get_ids()))

        imp_fun.append(vulner_1)
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
        imp_fun.append(vulner)

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
        imp_fun.append(vulner)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                imp_fun.check()
        self.assertIn('Invalid ImpactFunc.mdd size: 3 != 2.', cm.output[0])

class TestExtend(unittest.TestCase):
    """Check extend function"""
    def test_extend_to_empty_same(self):
        """Extend ImpactFuncSet to empty one."""
        imp_fun = ImpactFuncSet()
        imp_fun_add = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.append(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun_add.append(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.append(vulner_3)

        imp_fun_add.tag.file_name = 'file1.txt'

        imp_fun.extend(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 3)
        self.assertEqual(imp_fun.size('TC'), 2)
        self.assertEqual(imp_fun.size('FL'), 1)
        self.assertEqual(imp_fun.tag.file_name, imp_fun_add.tag.file_name)
        self.assertEqual(imp_fun.tag.description, imp_fun_add.tag.description)

    def test_extend_equal_same(self):
        """Extend the same ImpactFuncSet. The inital ImpactFuncSet is obtained."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.append(vulner_1)

        imp_fun_add = ImpactFuncSet()
        imp_fun_add.append(vulner_1)

        imp_fun.extend(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 1)
        self.assertEqual(imp_fun.size('TC'), 1)

    def test_extend_different_extend(self):
        """Extend ImpactFuncSet with same and new values. The vulnerabilities
        with repeated id are overwritten."""
        imp_fun = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun.append(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 3
        vulner_2.haz_type = 'TC'
        imp_fun.append(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun.append(vulner_3)

        imp_fun_add = ImpactFuncSet()
        vulner_1 = ImpactFunc()
        vulner_1.id = 1
        vulner_1.haz_type = 'TC'
        imp_fun_add.append(vulner_1)

        vulner_2 = ImpactFunc()
        vulner_2.id = 1
        vulner_2.haz_type = 'WS'
        imp_fun_add.append(vulner_2)

        vulner_3 = ImpactFunc()
        vulner_3.id = 3
        vulner_3.haz_type = 'FL'
        imp_fun_add.append(vulner_3)

        imp_fun.extend(imp_fun_add)
        imp_fun.check()

        self.assertEqual(imp_fun.size(), 4)
        self.assertEqual(imp_fun.size('TC'), 2)
        self.assertEqual(imp_fun.size('FL'), 1)
        self.assertEqual(imp_fun.size('WS'), 1)

class TestReaderMat(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_demo_file_pass(self):
        """Read demo excel file"""
        # Read demo mat file
        imp_funcs = ImpactFuncSet()
        description = 'One single file.'
        imp_funcs.read_mat(ENT_TEST_MAT, description)

        # Check results
        n_funcs = 2
        hazard = 'TC'
        first_id = 1
        second_id = 3

        self.assertEqual(len(imp_funcs._data), 1)
        self.assertEqual(len(imp_funcs._data[hazard]), n_funcs)

        # first function
        self.assertEqual(imp_funcs._data[hazard][first_id].id, 1)
        self.assertEqual(imp_funcs._data[hazard][first_id].name,
                         'Tropical cyclone default')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit,
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][first_id].intensity.shape,
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][first_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[8], 0.41079600)

        self.assertEqual(imp_funcs._data[hazard][first_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[8], 1)

        # second function
        self.assertEqual(imp_funcs._data[hazard][second_id].id, 3)
        self.assertEqual(imp_funcs._data[hazard][second_id].name,
                         'TC Building code')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit,
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][second_id].intensity.shape,
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][second_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[8], 0.4)

        self.assertEqual(imp_funcs._data[hazard][second_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[8], 1)

        # general information
        self.assertEqual(imp_funcs.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(imp_funcs.tag.description, description)

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_demo_file_pass(self):
        """Read demo excel file"""
        # Read demo excel file
        imp_funcs = ImpactFuncSet()
        description = 'One single file.'
        imp_funcs.read_excel(ENT_DEMO_TODAY, description)

        # Check results
        n_funcs = 2
        hazard = 'TC'
        first_id = 1
        second_id = 3

        self.assertEqual(len(imp_funcs._data), 1)
        self.assertEqual(len(imp_funcs._data[hazard]), n_funcs)

        # first function
        self.assertEqual(imp_funcs._data[hazard][first_id].id, 1)
        self.assertEqual(imp_funcs._data[hazard][first_id].name,
                         'Tropical cyclone default')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit,
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][first_id].intensity.shape,
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][first_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[8], 0.41079600)

        self.assertEqual(imp_funcs._data[hazard][first_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[8], 1)

        # second function
        self.assertEqual(imp_funcs._data[hazard][second_id].id, 3)
        self.assertEqual(imp_funcs._data[hazard][second_id].name,
                         'TC Building code')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit,
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][second_id].intensity.shape,
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][second_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[8], 0.4)

        self.assertEqual(imp_funcs._data[hazard][second_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[8], 1)

        # general information
        self.assertEqual(imp_funcs.tag.file_name, ENT_DEMO_TODAY)
        self.assertEqual(imp_funcs.tag.description, description)

    def test_template_file_pass(self):
        """Read template excel file"""
        imp_funcs = ImpactFuncSet()
        imp_funcs.read_excel(ENT_TEMPLATE_XLS)
        # Check some results
        self.assertEqual(len(imp_funcs._data), 10)
        self.assertEqual(len(imp_funcs._data['TC'][3].paa), 9)
        self.assertEqual(len(imp_funcs._data['EQ'][1].intensity), 14)
        self.assertEqual(len(imp_funcs._data['HS'][1].mdd), 16)

class TestWriter(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_write_read_pass(self):
        """Write + read excel file"""

        imp_funcs = ImpactFuncSet()
        imp_funcs.tag.file_name = 'No file name'
        imp_funcs.tag.description = 'test writer'

        imp1 = ImpactFunc()
        imp1.id = 1
        imp1.name = 'code 1'
        imp1.intensity_unit = 'm/s'
        imp1.haz_type = 'TC'
        imp1.intensity = np.arange(100)
        imp1.mdd = np.arange(100) * 0.5
        imp1.paa = np.ones(100)
        imp_funcs.append(imp1)

        imp2 = ImpactFunc()
        imp2.id = 2
        imp2.name = 'code 2'
        imp2.intensity_unit = 'm/s'
        imp2.haz_type = 'TC'
        imp2.intensity = np.arange(102)
        imp2.mdd = np.arange(102) * 0.25
        imp2.paa = np.ones(102)
        imp_funcs.append(imp2)

        imp3 = ImpactFunc()
        imp3.id = 1
        imp3.name = 'code 1'
        imp3.intensity_unit = 'm'
        imp3.haz_type = 'FL'
        imp3.intensity = np.arange(86)
        imp3.mdd = np.arange(86) * 0.15
        imp3.paa = np.ones(86)
        imp_funcs.append(imp3)

        imp4 = ImpactFunc()
        imp4.id = 15
        imp4.name = 'code 15'
        imp4.intensity_unit = 'K'
        imp4.haz_type = 'DR'
        imp4.intensity = np.arange(5)
        imp4.mdd = np.arange(5)
        imp4.paa = np.ones(5)
        imp_funcs.append(imp4)

        file_name = CONFIG.impact_funcs.test_data.dir().joinpath('test_write.xlsx')
        imp_funcs.write_excel(file_name)

        imp_res = ImpactFuncSet()
        imp_res.read_excel(file_name)

        self.assertEqual(imp_res.tag.file_name, file_name)
        self.assertEqual(imp_res.tag.description, '')

        # first function
        for fun_haz, fun_dict in imp_res.get_func().items():
            for fun_id, fun in fun_dict.items():
                if fun_haz == 'TC' and fun_id == 1:
                    ref_fun = imp1
                elif fun_haz == 'TC' and fun_id == 2:
                    ref_fun = imp2
                elif fun_haz == 'FL' and fun_id == 1:
                    ref_fun = imp3
                elif fun_haz == 'DR' and fun_id == 15:
                    ref_fun = imp4
                else:
                    self.assertEqual(1, 0)

                self.assertEqual(ref_fun.haz_type, fun.haz_type)
                self.assertEqual(ref_fun.id, fun.id)
                self.assertEqual(ref_fun.name, fun.name)
                self.assertEqual(ref_fun.intensity_unit, fun.intensity_unit)
                self.assertTrue(np.allclose(ref_fun.intensity, fun.intensity))
                self.assertTrue(np.allclose(ref_fun.mdd, fun.mdd))
                self.assertTrue(np.allclose(ref_fun.paa, fun.paa))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtend))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWriter))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
