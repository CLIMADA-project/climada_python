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

Test Entity class.
"""
import os
import unittest
import numpy as np

from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures
from climada.entity.disc_rates.base import DiscRates
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.measure_set import MeasureSet
from climada.util.constants import ENT_TEMPLATE_XLS

ENT_TEST_MAT = os.path.join(os.path.dirname(__file__), 
                            '../exposures/test/data/demo_today.mat')

class TestReader(unittest.TestCase):
    """Test reader functionality of the Entity class"""

    def test_default_pass(self):
        """Instantiating the Entity class the default entity file is loaded"""
        # Set demo file as default
        def_entity = Entity()
        def_entity.read_excel(ENT_TEMPLATE_XLS)

        # Check default demo excel file has been loaded
        self.assertEqual(len(def_entity.exposures.deductible), 24)
        self.assertEqual(def_entity.exposures.value[2], 12596064143.542929)

        self.assertEqual(len(def_entity.impact_funcs.get_func('TC', 1).mdd), 25)

        self.assertIn('risk transfer', def_entity.measures.get_names('TC'))

        self.assertEqual(def_entity.disc_rates.years[5], 2005)

        self.assertTrue(isinstance(def_entity.disc_rates, DiscRates))
        self.assertTrue(isinstance(def_entity.exposures, Exposures))
        self.assertTrue(isinstance(def_entity.impact_funcs, ImpactFuncSet))
        self.assertTrue(isinstance(def_entity.measures, MeasureSet))

    def test_read_mat(self):
        """Read entity from mat file produced by climada."""
        entity_mat = Entity()
        entity_mat.read_mat(ENT_TEST_MAT)
        self.assertEqual(entity_mat.exposures.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(entity_mat.disc_rates.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(entity_mat.measures.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(entity_mat.impact_funcs.tag.file_name, ENT_TEST_MAT)

    def test_read_excel(self):
        """Read entity from an xls file following the template."""
        entity_xls = Entity()
        entity_xls.read_excel(ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.exposures.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.disc_rates.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.measures.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.impact_funcs.tag.file_name,
                         ENT_TEMPLATE_XLS)

class TestCheck(unittest.TestCase):
    """Test entity checker."""

    def test_wrongMeas_fail(self):
        """Wrong measures"""
        ent = Entity()
        ent.read_excel(ENT_TEMPLATE_XLS)
        actions = ent.measures.get_measure('TC')
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
        ent.read_excel(ENT_TEMPLATE_XLS)
        ent.impact_funcs.get_func('TC', 1).paa = np.array([1, 2])
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
        ent.read_excel(ENT_TEMPLATE_XLS)
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
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheck))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
