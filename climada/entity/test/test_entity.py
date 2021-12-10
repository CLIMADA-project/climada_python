"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test Entity class.
"""
import unittest
import numpy as np

from climada import CONFIG
from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures
from climada.entity.disc_rates.base import DiscRates
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.measure_set import MeasureSet
from climada.util.constants import ENT_TEMPLATE_XLS

ENT_TEST_MAT = CONFIG.exposures.test_data.dir().joinpath('demo_today.mat')


class TestReader(unittest.TestCase):
    """Test reader functionality of the Entity class"""

    def test_default_pass(self):
        """Instantiating the Entity class the default entity file is loaded"""
        # Set demo file as default
        def_entity = Entity.from_excel(ENT_TEMPLATE_XLS)

        # Check default demo excel file has been loaded
        self.assertEqual(len(def_entity.exposures.gdf.deductible), 24)
        self.assertEqual(def_entity.exposures.gdf.value[2], 12596064143.542929)

        self.assertEqual(len(def_entity.impact_funcs.get_func('TC', 1).mdd), 25)

        self.assertIn('risk transfer', def_entity.measures.get_names('TC'))

        self.assertEqual(def_entity.disc_rates.years[5], 2005)

        self.assertTrue(isinstance(def_entity.disc_rates, DiscRates))
        self.assertTrue(isinstance(def_entity.exposures, Exposures))
        self.assertTrue(isinstance(def_entity.impact_funcs, ImpactFuncSet))
        self.assertTrue(isinstance(def_entity.measures, MeasureSet))

    def test_from_mat(self):
        """Read entity from mat file produced by climada."""
        entity_mat = Entity.from_mat(ENT_TEST_MAT)
        self.assertEqual(entity_mat.exposures.tag.file_name, str(ENT_TEST_MAT))
        self.assertEqual(entity_mat.disc_rates.tag.file_name, str(ENT_TEST_MAT))
        self.assertEqual(entity_mat.measures.tag.file_name, str(ENT_TEST_MAT))
        self.assertEqual(entity_mat.impact_funcs.tag.file_name, str(ENT_TEST_MAT))

    def test_from_excel(self):
        """Read entity from an xls file following the template."""
        entity_xls = Entity.from_excel(ENT_TEMPLATE_XLS)
        self.assertEqual(entity_xls.exposures.tag.file_name, str(ENT_TEMPLATE_XLS))
        self.assertEqual(entity_xls.disc_rates.tag.file_name, str(ENT_TEMPLATE_XLS))
        self.assertEqual(entity_xls.measures.tag.file_name, str(ENT_TEMPLATE_XLS))
        self.assertEqual(entity_xls.impact_funcs.tag.file_name,
                         str(ENT_TEMPLATE_XLS))

class TestCheck(unittest.TestCase):
    """Test entity checker."""

    def test_wrongMeas_fail(self):
        """Wrong measures"""
        ent = Entity.from_excel(ENT_TEMPLATE_XLS)
        actions = ent.measures.get_measure('TC')
        actions[0].color_rgb = np.array([1, 2])
        with self.assertRaises(ValueError) as cm:
            ent.check()
        self.assertIn('Measure.color_rgb', str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            ent.measures = Exposures()
        self.assertIn('MeasureSet', str(cm.exception))

    def test_wrongImpFun_fail(self):
        """Wrong impact functions"""
        ent = Entity.from_excel(ENT_TEMPLATE_XLS)
        ent.impact_funcs.get_func('TC', 1).paa = np.array([1, 2])
        with self.assertRaises(ValueError) as cm:
            ent.check()
        self.assertIn('ImpactFunc.paa', str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            ent.impact_funcs = Exposures()
        self.assertIn('ImpactFuncSet', str(cm.exception))

    def test_wrongDisc_fail(self):
        """Wrong discount rates"""
        ent = Entity.from_excel(ENT_TEMPLATE_XLS)
        ent.disc_rates.rates = np.array([1, 2])
        with self.assertRaises(ValueError) as cm:
            ent.check()
        self.assertIn('DiscRates.rates', str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            ent.disc_rates = Exposures()
        self.assertIn('DiscRates', str(cm.exception))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheck))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
