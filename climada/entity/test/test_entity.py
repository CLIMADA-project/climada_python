"""
Test Entity class.
"""

import unittest
import numpy as np

from climada.entity.entity import Entity
from climada.entity.exposures.base import Exposures
from climada.entity.disc_rates.base import DiscRates
from climada.entity.impact_funcs.base import ImpactFuncs
from climada.entity.measures.base import Measures
from climada.util.constants import ENT_DEMO_XLS, ENT_DEMO_MAT, ENT_TEMPLATE_XLS

class TestReader(unittest.TestCase):
    """Test reader functionality of the Entity class"""

    def test_default_pass(self):
        """Instantiating the Entity class the default entity file is loaded"""
        # Instance entity
        # Set demo file as default
        Entity.def_file = ENT_DEMO_XLS
        def_entity = Entity()

        # Check default demo excel file has been loaded
        self.assertEqual(len(def_entity.exposures.deductible), 50)
        self.assertEqual(def_entity.exposures.value[2], 12596064143.542929)

        self.assertEqual(len(def_entity.impact_funcs.get_vulner('TC', 1).mdd),\
                         9)

        self.assertIn('Mangroves', def_entity.measures.get_names())

        self.assertEqual(def_entity.disc_rates.years[5], 2005)

        self.assertTrue(isinstance(def_entity.disc_rates, DiscRates))
        self.assertTrue(isinstance(def_entity.exposures, Exposures))
        self.assertTrue(isinstance(def_entity.impact_funcs, ImpactFuncs))
        self.assertTrue(isinstance(def_entity.measures, Measures))

    def test_read_mat(self):
        """Read entity from mat file produced by climada."""
        entity_mat = Entity(ENT_DEMO_MAT)
        self.assertEqual(entity_mat.exposures.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(entity_mat.disc_rates.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(entity_mat.measures.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(entity_mat.impact_funcs.tag.file_name, ENT_DEMO_MAT)

    def test_read_excel(self):
        """Read entity from an xls file following the template."""
        entity_xls = Entity(ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.exposures.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.disc_rates.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.measures.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.impact_funcs.tag.file_name, \
                         ENT_TEMPLATE_XLS)
    
    def test_read_parallel(self):
        """Read in parallel two entities."""
        
        with self.assertRaises(ValueError) as error:
            Entity([ENT_DEMO_XLS, ENT_TEMPLATE_XLS], ['demo', 'template'])
        self.assertEqual('Append not possible. Different reference years.', \
                         str(error.exception))
        
        ent = Entity([ENT_DEMO_XLS, ENT_DEMO_XLS], ['demo', 'demo'])
        self.assertEqual(ent.exposures.id.size, 100)
        self.assertEqual(ent.exposures.tag.file_name, \
                         [ENT_DEMO_XLS, ENT_DEMO_XLS])

class TestCheck(unittest.TestCase):
    """Test entity checker."""
    def test_wrongExpo_fail(self):
        """Wrong exposures"""
        ent = Entity()
        ent.exposures.cover = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('Exposures.cover', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.exposures = Measures()
        self.assertIn('Exposures', str(error.exception))

    def test_wrongMeas_fail(self):
        """Wrong measures"""
        ent = Entity()
        actions = ent.measures.get_action()
        actions[0].color_rgb = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('Action.color_rgb', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.measures = Exposures()
        self.assertIn('Measures', str(error.exception))

    def test_wrongImpFun_fail(self):
        """Wrong impact functions"""
        ent = Entity()
        ent.impact_funcs.get_vulner('TC', 1).paa = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('Vulnerability.paa', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.impact_funcs = Exposures()
        self.assertIn('ImpactFuncs', str(error.exception))

    def test_wrongDisc_fail(self):
        """Wrong discount rates"""
        ent = Entity()
        ent.disc_rates.rates = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('DiscRates.rates', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.disc_rates = Exposures()
        self.assertIn('DiscRates', str(error.exception))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheck))
unittest.TextTestRunner(verbosity=2).run(TESTS)
