"""
Test ImpactFuncs class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.base import ImpactFuncs, Vulnerability
from climada.util.constants import ENT_DEMO_XLS

class TestContainer(unittest.TestCase):
    """Test ImpactFuncs as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong Vulnerability provided."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        with self.assertRaises(ValueError) as error:
            imp_fun.add_vulner(vulner_1)
        self.assertEqual("Input Vulnerability's hazard type not set.", \
                         str(error.exception))

        vulner_1.haz_type = 'TC'
        with self.assertRaises(ValueError) as error:
            imp_fun.add_vulner(vulner_1)
        self.assertEqual("Input Vulnerability's id not set.", \
                         str(error.exception))

        with self.assertRaises(ValueError) as error:
            imp_fun.add_vulner(45)
        self.assertEqual("Input value is not of type Vulnerability.", \
                         str(error.exception))

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
        with self.assertRaises(ValueError) as error:
            imp_fun.remove_vulner('FL')
        self.assertEqual('No Vulnerability with hazard FL.', \
                         str(error.exception))
        with self.assertRaises(ValueError) as error:
            imp_fun.remove_vulner(vul_id=3)
        self.assertEqual('No Vulnerability with id 3.', \
                         str(error.exception))

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

    def test_get_ids_wrong_error(self):
        """Test get_ids method with wrong inputs."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        with self.assertRaises(ValueError) as error:
            imp_fun.get_ids('TC')
        self.assertEqual('No Vulnerability with hazard TC.', \
                         str(error.exception))

    def test_get_vulner_pass(self):
        """Test normal functionality of get_vulner method."""
        imp_fun = ImpactFuncs()
        vulner_1 = Vulnerability()
        vulner_1.haz_type = 'WS'
        vulner_1.id = 56
        imp_fun.add_vulner(vulner_1)
        self.assertEqual(1, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=56)))
        self.assertIs(vulner_1, imp_fun.get_vulner('WS', 56))

        vulner_2 = Vulnerability()
        vulner_2.haz_type = 'WS'
        vulner_2.id = 6
        imp_fun.add_vulner(vulner_2)
        self.assertEqual(2, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=6)))
        self.assertIs(vulner_2, imp_fun.get_vulner('WS', 6))

        vulner_3 = Vulnerability()
        vulner_3.haz_type = 'TC'
        vulner_3.id = 6
        imp_fun.add_vulner(vulner_3)
        self.assertEqual(2, len(imp_fun.get_vulner(vul_id=6)))
        self.assertEqual(1, len(imp_fun.get_vulner(vul_id=56)))
        self.assertEqual(2, len(imp_fun.get_vulner('WS')))
        self.assertEqual(1, len(imp_fun.get_vulner('TC')))
        self.assertIs(vulner_3, imp_fun.get_vulner('TC', 6))

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
        with self.assertRaises(ValueError) as error:
            imp_fun.get_vulner('TC')
        self.assertEqual('No Vulnerability with hazard TC.', \
                         str(error.exception))

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

    def test_num_vulner_wrong_error(self):
        """Test num_vulner method with wrong inputs."""
        imp_fun = ImpactFuncs()
        
        try:
            imp_fun.num_vulner('TC')
        except ValueError as error:
            self.assertEqual('No Vulnerability with hazard TC.', error.args[0])

        try:
            imp_fun.num_vulner('TC', 3)
        except ValueError as error:
            self.assertEqual('No Vulnerability with hazard TC and id 3.', \
                             error.args[0])

        try:
            imp_fun.num_vulner(vul_id=3)
        except ValueError as error:
            self.assertEqual('No Vulnerability with id 3.', \
                             error.args[0])

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

class TestLoader(unittest.TestCase):
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

        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Invalid Vulnerability.paa size: 3 != 2', \
                         str(error.exception))

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

        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Invalid Vulnerability.mdd size: 3 != 2', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """"Load function not implemented"""
        imp_fun = ImpactFuncs()
        with self.assertRaises(NotImplementedError):
            imp_fun.load(ENT_DEMO_XLS)

    def test_read_notimplemented(self):
        """Read function not implemented"""
        imp_fun = ImpactFuncs()
        with self.assertRaises(NotImplementedError):
            imp_fun.read(ENT_DEMO_XLS)

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            ImpactFuncs(ENT_DEMO_XLS)

class TestInterpolation(unittest.TestCase):
    """Impact function interpolation test"""

    def test_wrongAttribute_fail(self):
        """Interpolation of wrong variable fails."""
        imp_fun = Vulnerability()
        intensity = 3
        with self.assertRaises(ValueError) as error:
            imp_fun.interpolate(intensity, 'mdg')
        self.assertEqual('Attribute of the impact function mdg not found.',\
                         str(error.exception))

    def test_mdd_pass(self):
        """Good interpolation of MDD."""
        imp_fun = Vulnerability()
        imp_fun.intensity = np.array([0,1])
        imp_fun.mdd = np.array([1,2])
        imp_fun.paa = np.array([3,4])
        intensity = 0.5
        resul = imp_fun.interpolate(intensity, 'mdd')
        self.assertEqual(1.5, resul)

    def test_paa_pass(self):
        """Good interpolation of PAA."""
        imp_fun = Vulnerability()
        imp_fun.intensity = np.array([0,1])
        imp_fun.mdd = np.array([1,2])
        imp_fun.paa = np.array([3,4])
        intensity = 0.5
        resul = imp_fun.interpolate(intensity, 'paa')
        self.assertEqual(3.5, resul)        

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLoader))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInterpolation))
unittest.TextTestRunner(verbosity=2).run(TESTS)
