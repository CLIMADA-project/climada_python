"""
Test Entity class.
"""

import unittest
import numpy as np

from climada.entity.entity import Entity
from climada.entity.measures.source_excel import MeasuresExcel
from climada.entity.exposures.source_excel import ExposuresExcel
from climada.entity.exposures.base import Exposures
from climada.entity.discounts.base import Discounts
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

        self.assertEqual(len(def_entity.impact_funcs.data['TC'][1].mdd), 9)

        self.assertEqual(def_entity.measures.data[0].name, 'Mangroves')

        self.assertEqual(def_entity.discounts.years[5], 2005)

        self.assertTrue(isinstance(def_entity.discounts, Discounts))
        self.assertTrue(isinstance(def_entity.exposures, Exposures))
        self.assertTrue(isinstance(def_entity.impact_funcs, ImpactFuncs))
        self.assertTrue(isinstance(def_entity.measures, Measures))

    def test_read_mat(self):
        """Read entity from mat file produced by climada."""
        entity_mat = Entity(ENT_DEMO_MAT)
        self.assertEqual(len(entity_mat.tags), 4)
        for idx in range(4):
            self.assertEqual(entity_mat.tags[idx].file_name, ENT_DEMO_MAT)

    def test_read_excel(self):
        """Read entity from an xls file following the template."""
        entity_xls = Entity(ENT_TEMPLATE_XLS)
        self.assertEqual(len(entity_xls.tags), 4)
        for idx in range(4):
            self.assertEqual(entity_xls.tags[idx].file_name, ENT_TEMPLATE_XLS)

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
            ent.exposures = MeasuresExcel()
        self.assertIn('Exposures', str(error.exception))

    def test_wrongMeas_fail(self):
        """Wrong measures"""
        ent = Entity()
        ent.measures.data[0].color_rgb = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('Action.color_rgb', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.measures = ExposuresExcel()
        self.assertIn('Measures', str(error.exception))

    def test_wrongImpFun_fail(self):
        """Wrong impact functions"""
        ent = Entity()
        ent.impact_funcs.data['TC'][1].paa = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('ImpactFunc.paa', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.impact_funcs = ExposuresExcel()
        self.assertIn('ImpactFuncs', str(error.exception))

    def test_wrongDisc_fail(self):
        """Wrong discount rates"""
        ent = Entity()
        ent.discounts.rates = np.array([1, 2])
        with self.assertRaises(ValueError) as error:
            ent.check()
        self.assertIn('Discounts.rates', str(error.exception))

        with self.assertRaises(ValueError) as error:
            ent.discounts = ExposuresExcel()
        self.assertIn('Discounts', str(error.exception))

if __name__ == '__main__':
    unittest.main()
