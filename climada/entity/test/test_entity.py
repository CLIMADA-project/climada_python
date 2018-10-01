"""
Test Entity class.
"""
import os
import unittest
import numpy as np

from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures
from climada.entity.disc_rates.base import DiscRates
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.base import MeasureSet
from climada.util.constants import ENT_DEMO_MAT, ENT_TEMPLATE_XLS

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

class TestReader(unittest.TestCase):
    """Test reader functionality of the Entity class"""

    def test_default_pass(self):
        """Instantiating the Entity class the default entity file is loaded"""
        # Set demo file as default
        Entity.def_file = ENT_TEST_XLS
        def_entity = Entity()

        # Check default demo excel file has been loaded
        self.assertEqual(len(def_entity.exposures.deductible), 50)
        self.assertEqual(def_entity.exposures.value[2], 12596064143.542929)

        self.assertEqual(len(def_entity.impact_funcs.get_func('TC', 1)[0].mdd),\
                         9)

        self.assertIn('Mangroves', def_entity.measures.get_names())

        self.assertEqual(def_entity.disc_rates.years[5], 2005)

        self.assertTrue(isinstance(def_entity.disc_rates, DiscRates))
        self.assertTrue(isinstance(def_entity.exposures, Exposures))
        self.assertTrue(isinstance(def_entity.impact_funcs, ImpactFuncSet))
        self.assertTrue(isinstance(def_entity.measures, MeasureSet))

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
        with self.assertLogs('climada.entity.exposures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                Entity([ENT_TEST_XLS, ENT_TEMPLATE_XLS], ['demo', 'template'])
        self.assertIn('Append not possible. Different reference years.', \
                         cm.output[0])
        
        ent = Entity([ENT_TEST_XLS, ENT_TEST_XLS], ['demo', 'demo'])
        self.assertEqual(ent.exposures.id.size, 100)
        self.assertEqual(ent.exposures.tag.file_name, ENT_TEST_XLS)

class TestCheck(unittest.TestCase):
    """Test entity checker."""
    def test_wrongExpo_fail(self):
        """Wrong exposures"""
        ent = Entity()
        ent.exposures.cover = np.array([1, 2])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.check()
        self.assertIn('Exposures.cover', cm.output[0])

        with self.assertLogs('climada.entity.entity_def', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.exposures = MeasureSet()
        self.assertIn('Exposures', cm.output[0])

    def test_wrongMeas_fail(self):
        """Wrong measures"""
        ent = Entity()
        actions = ent.measures.get_measure()
        actions[0].color_rgb = np.array([1, 2])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.check()
        self.assertIn('Measure.color_rgb', cm.output[0])

        with self.assertLogs('climada.entity.entity_def', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.measures = Exposures()
        self.assertIn('MeasureSet', cm.output[0])

    def test_wrongImpFun_fail(self):
        """Wrong impact functions"""
        ent = Entity()
        ent.impact_funcs.get_func('TC', 1)[0].paa = np.array([1, 2])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.check()
        self.assertIn('ImpactFunc.paa', cm.output[0])

        with self.assertLogs('climada.entity.entity_def', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.impact_funcs = Exposures()
        self.assertIn('ImpactFuncSet', cm.output[0])

    def test_wrongDisc_fail(self):
        """Wrong discount rates"""
        ent = Entity()
        ent.disc_rates.rates = np.array([1, 2])
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.check()
        self.assertIn('DiscRates.rates', cm.output[0])

        with self.assertLogs('climada.entity.entity_def', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                ent.disc_rates = Exposures()
        self.assertIn('DiscRates', cm.output[0])

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheck))
unittest.TextTestRunner(verbosity=2).run(TESTS)
