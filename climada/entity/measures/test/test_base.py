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

Test MeasureSet and Measure classes.
"""

import unittest
import numpy as np

from climada.hazard.base import Hazard
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.measures.base import Measure
from climada.entity.measures.measure_set import MeasureSet
from climada.entity.measures.source import READ_SET
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_MAT


class TestImpact(unittest.TestCase):
    """Test implement measures functions."""
    def test_change_imp_func_pass(self):
        """Test change_imp_func"""
        meas = MeasureSet(ENT_DEMO_MAT)
        act_1 = meas.get_measure('Mangroves')

        imp_set = ImpactFuncSet()
        imp_tc = ImpactFunc()
        imp_tc.haz_type = 'XX'
        imp_tc.id = 1
        imp_tc.intensity = np.arange(10,100, 10)
        imp_tc.intensity[0] = 0.
        imp_tc.intensity[-1] = 100.
        imp_tc.mdd = np.array([0.0, 0.0, 0.021857142857143, 0.035887500000000,
                               0.053977415307403, 0.103534246575342, 0.180414000000000,
                               0.410796000000000, 0.410796000000000])
        imp_tc.paa = np.array([0, 0.005000000000000, 0.042000000000000, 0.160000000000000,
                               0.398500000000000, 0.657000000000000, 1.000000000000000,
                               1.000000000000000, 1.000000000000000])
        imp_set.add_func(imp_tc)
        new_imp = act_1.change_imp_func(imp_set).get_func('XX')[0]

        self.assertTrue(np.array_equal(new_imp.intensity, np.array([4., 24., 34., 44.,
            54., 64., 74., 84., 104.])))
        self.assertTrue(np.array_equal(new_imp.mdd, np.array([0, 0, 0.021857142857143, 0.035887500000000,
            0.053977415307403, 0.103534246575342, 0.180414000000000, 0.410796000000000, 0.410796000000000])))
        self.assertTrue(np.array_equal(new_imp.paa, np.array([0, 0.005000000000000, 0.042000000000000,
            0.160000000000000, 0.398500000000000, 0.657000000000000, 1.000000000000000,
            1.000000000000000, 1.000000000000000])))
        self.assertFalse(id(new_imp) == id(imp_tc))

    def test_implement_pass(self):
        """Test implement"""
        meas = MeasureSet(ENT_DEMO_MAT)
        act_1 = meas.get_measure('Mangroves')

        imp_set = ImpactFuncSet()
        imp_tc = ImpactFunc()
        imp_tc.haz_type = 'XX'
        imp_tc.id = 1
        imp_tc.intensity = np.arange(10,100, 10)
        imp_tc.intensity[0] = 0.
        imp_tc.intensity[-1] = 100.
        imp_tc.mdd = np.array([0.0, 0.0, 0.021857142857143, 0.035887500000000,
                               0.053977415307403, 0.103534246575342, 0.180414000000000,
                               0.410796000000000, 0.410796000000000])
        imp_tc.paa = np.array([0, 0.005000000000000, 0.042000000000000, 0.160000000000000,
                               0.398500000000000, 0.657000000000000, 1.000000000000000,
                               1.000000000000000, 1.000000000000000])
        imp_set.add_func(imp_tc)

        hazard = Hazard('XX')
        exposures = Exposures()
        new_exp, new_ifs, new_haz = act_1.implement(exposures, imp_set, hazard)

        self.assertFalse(id(new_ifs) == id(imp_tc))
        self.assertTrue(id(new_exp) == id(exposures))
        self.assertTrue(id(new_haz) == id(hazard))

class TestConstructor(unittest.TestCase):
    """Test impact function attributes."""
    def test_attributes_all(self):
        """All attributes are defined"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Seawall'
        self.assertTrue(hasattr(meas, 'tag'))
        self.assertTrue(hasattr(meas, '_data'))
        self.assertTrue(hasattr(act_1, 'name'))
        self.assertTrue(hasattr(act_1, 'color_rgb'))
        self.assertTrue(hasattr(act_1, 'cost'))
        self.assertTrue(hasattr(act_1, 'hazard_freq_cutoff'))
        self.assertTrue(hasattr(act_1, 'hazard_inten_imp'))
        self.assertTrue(hasattr(act_1, 'mdd_impact'))
        self.assertTrue(hasattr(act_1, 'paa_impact'))
        self.assertTrue(hasattr(act_1, 'risk_transf_attach'))
        self.assertTrue(hasattr(act_1, 'risk_transf_cover'))

    def test_get_def_vars(self):
        """ Test def_source_vars function."""
        self.assertTrue(MeasureSet.get_def_file_var_names('xls') ==
                        READ_SET['XLS'][0])
        self.assertTrue(MeasureSet.get_def_file_var_names('.mat') ==
                        READ_SET['MAT'][0])

class TestContainer(unittest.TestCase):
    """Test MeasureSet as container."""
    def test_add_wrong_error(self):
        """Test error is raised when wrong ImpactFunc provided."""
        meas = MeasureSet()
        act_1 = Measure()
        with self.assertLogs('climada.entity.measures.measure_set', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.add_measure(act_1)
        self.assertIn("Input Measure's name not set.", cm.output[0])

        with self.assertLogs('climada.entity.measures.measure_set', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.add_measure(45)
        self.assertIn("Input value is not of type Measure.", cm.output[0])

    def test_remove_measure_pass(self):
        """Test remove_measure removes Measure of MeasureSet correcty."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        meas.add_measure(act_1)
        meas.remove_measure('Mangrove')
        self.assertEqual(0, len(meas._data))

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        meas.add_measure(act_1)
        with self.assertLogs('climada.entity.measures.measure_set', level='WARNING') as cm:
            meas.remove_measure('Seawall')
        self.assertIn('No Measure with name Seawall.', cm.output[0])

    def test_get_names_pass(self):
        """Test get_names function."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        meas.add_measure(act_1)
        self.assertEqual(1, len(meas.get_names()))
        self.assertEqual(['Mangrove'], meas.get_names())

        act_2 = Measure()
        act_2.name = 'Seawall'
        meas.add_measure(act_2)
        self.assertEqual(2, len(meas.get_names()))
        self.assertIn('Mangrove', meas.get_names())
        self.assertIn('Seawall', meas.get_names())

    def test_get_measure_pass(self):
        """Test normal functionality of get_measure method."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        meas.add_measure(act_1)
        self.assertIs(act_1, meas.get_measure('Mangrove'))

        act_2 = Measure()
        act_2.name = 'Seawall'
        meas.add_measure(act_2)
        self.assertIs(act_1, meas.get_measure('Mangrove'))
        self.assertIs(act_2, meas.get_measure('Seawall'))
        self.assertEqual(2, len(meas.get_measure()))

    def test_get_measure_wrong_error(self):
        """Test get_measure method with wrong inputs."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Seawall'
        meas.add_measure(act_1)
        self.assertEqual([], meas.get_measure('Mangrove'))

    def test_num_measures_pass(self):
        """Test num_measures function."""
        meas = MeasureSet()
        self.assertEqual(0, meas.num_measures())
        act_1 = Measure()
        act_1.name = 'Mangrove'
        meas.add_measure(act_1)
        self.assertEqual(1, meas.num_measures())
        meas.add_measure(act_1)
        self.assertEqual(1, meas.num_measures())

        act_2 = Measure()
        act_2.name = 'Seawall'
        meas.add_measure(act_2)
        self.assertEqual(2, meas.num_measures())

class TestChecker(unittest.TestCase):
    """Test check functionality of the MeasureSet class"""

    def test_check_wronginten_fail(self):
        """Wrong intensity definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.hazard_inten_imp = (1, 2, 3)
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        meas.add_measure(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.hazard_inten_imp size: 2 != 3.', \
                         cm.output[0])

    def test_check_wrongColor_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = (1, 2)
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.add_measure(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.color_rgb size: 3 != 2.', cm.output[0])

    def test_check_wrongMDD_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.add_measure(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Measure.mdd_impact has wrong dimensions.', cm.output[0])

    def test_check_wrongPAA_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2, 3, 4)
        act_1.hazard_inten_imp = (1, 2)
        meas.add_measure(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.paa_impact size: 2 != 4.', cm.output[0])

    def test_check_name_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'LaLa'
        meas._data['LoLo'] = act_1
        with self.assertRaises(ValueError) as error:
            meas.check()
        self.assertEqual('Wrong Measure.name: LoLo != LaLa', \
                         str(error.exception))

class TestAppend(unittest.TestCase):
    """Check append function"""
    def test_append_to_empty_same(self):
        """Append MeasureSet to empty one."""
        meas = MeasureSet()
        meas_add = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas_add.add_measure(act_1)

        meas.append(meas_add)
        meas.check()

        self.assertEqual(meas.num_measures(), 1)
        self.assertEqual(meas.get_names(), [act_1.name])

    def test_append_equal_same(self):
        """Append the same MeasureSet. The inital MeasureSet is obtained."""
        meas = MeasureSet()
        meas_add = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.add_measure(act_1)
        meas_add.add_measure(act_1)

        meas.append(meas_add)
        meas.check()

        self.assertEqual(meas.num_measures(), 1)
        self.assertEqual(meas.get_names(), [act_1.name])

    def test_append_different_append(self):
        """Append MeasureSet with same and new values. The actions
        with repeated name are overwritten."""
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)

        act_11 = Measure()
        act_11.name = 'Mangrove'
        act_11.color_rgb = np.array([1, 1, 1])
        act_11.mdd_impact = (1, 2)
        act_11.paa_impact = (1, 3)
        act_11.hazard_inten_imp = (1, 2)

        act_2 = Measure()
        act_2.name = 'Anything'
        act_2.color_rgb = np.array([1, 1, 1])
        act_2.mdd_impact = (1, 2)
        act_2.paa_impact = (1, 2)
        act_2.hazard_inten_imp = (1, 2)

        meas = MeasureSet()
        meas.add_measure(act_1)
        meas_add = MeasureSet()
        meas_add.add_measure(act_11)
        meas_add.add_measure(act_2)

        meas.append(meas_add)
        meas.check()

        self.assertEqual(meas.num_measures(), 2)
        self.assertEqual(meas.get_names(), [act_1.name, act_2.name])
        self.assertEqual(meas.get_measure(act_1.name).paa_impact, act_11.paa_impact)

class TestReadParallel(unittest.TestCase):
    """Check read function with several files"""

    def test_read_two_pass(self):
        """Both files are readed and appended."""
        descriptions = ['desc1','desc2']
        meas = MeasureSet([ENT_TEMPLATE_XLS, ENT_TEMPLATE_XLS], descriptions)
        self.assertEqual(meas.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(meas.tag.description, 'desc1 + desc2')
        self.assertEqual(meas.num_measures(), 7)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReadParallel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpact))
unittest.TextTestRunner(verbosity=2).run(TESTS)
