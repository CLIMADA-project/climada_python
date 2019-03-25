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

Test MeasureSet and Measure classes.
"""
import os
import unittest
import numpy as np

from climada.entity.measures.base import Measure
from climada.entity.measures.measure_set import MeasureSet
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_TODAY

ENT_TEST_MAT = os.path.join(os.path.dirname(__file__),
                            '../../exposures/test/data/demo_today.mat')

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

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the MeasuresExcel class"""

    def test_demo_file(self):
        """ Read demo excel file"""
        meas = MeasureSet()
        description = 'One single file.'
        meas.read_excel(ENT_DEMO_TODAY, description)

        # Check results
        n_meas = 4

        self.assertEqual(len(meas.get_measure()), n_meas)

        act_man = meas.get_measure('Mangroves')
        self.assertEqual(act_man.name, 'Mangroves')
        self.assertEqual(type(act_man.color_rgb), np.ndarray)
        self.assertEqual(len(act_man.color_rgb), 3)
        self.assertEqual(act_man.color_rgb[0], 0.1529)
        self.assertEqual(act_man.color_rgb[1], 0.251)
        self.assertEqual(act_man.color_rgb[2], 0.5451)
        self.assertEqual(act_man.cost, 1311768360.8515418)
        self.assertEqual(act_man.hazard_freq_cutoff, 0)
        self.assertEqual(act_man.hazard_inten_imp, (1, -4))
        self.assertEqual(act_man.mdd_impact, (1, 0))
        self.assertEqual(act_man.paa_impact, (1, 0))
        self.assertEqual(act_man.risk_transf_attach, 0)
        self.assertEqual(act_man.risk_transf_cover, 0)

        act_buil = meas.get_measure('Building code')
        self.assertEqual(act_buil.name, 'Building code')
        self.assertEqual(type(act_buil.color_rgb), np.ndarray)
        self.assertEqual(len(act_buil.color_rgb), 3)
        self.assertEqual(act_buil.color_rgb[0], 0.6980)
        self.assertEqual(act_buil.color_rgb[1], 0.8745)
        self.assertEqual(act_buil.color_rgb[2], 0.9333)
        self.assertEqual(act_buil.cost, 9200000000.0000000)
        self.assertEqual(act_buil.hazard_freq_cutoff, 0)
        self.assertEqual(act_buil.hazard_inten_imp, (1, 0))
        self.assertEqual(act_buil.mdd_impact, (0.75, 0))
        self.assertEqual(act_buil.paa_impact, (1, 0))
        self.assertEqual(act_buil.risk_transf_attach, 0)
        self.assertEqual(act_buil.risk_transf_cover, 0)

        self.assertEqual(meas.tag.file_name, ENT_DEMO_TODAY)
        self.assertEqual(meas.tag.description, description)

    def test_template_file_pass(self):
        """ Read template excel file"""
        meas = MeasureSet()
        meas.read_excel(ENT_TEMPLATE_XLS)

        self.assertEqual(len(meas.get_measure()), 7)

        name = 'elevate existing buildings'
        act_buil = meas.get_measure(name)
        self.assertEqual(act_buil.name, name)
        self.assertEqual(act_buil.haz_type, 'TS')
        self.assertTrue(np.array_equal(act_buil.color_rgb, np.array([0.84, 0.89, 0.70])))
        self.assertEqual(act_buil.cost, 3911963265.476649)

        self.assertEqual(act_buil.hazard_set, 'nil')
        self.assertEqual(act_buil.hazard_freq_cutoff, 0)
        self.assertEqual(act_buil.hazard_inten_imp, (1, -2))

        self.assertEqual(act_buil.exposures_set, 'nil')
        self.assertEqual(act_buil.exp_region_id, 0)

        self.assertEqual(act_buil.paa_impact, (0.9, 0))
        self.assertEqual(act_buil.mdd_impact, (0.9, -0.1))
        self.assertEqual(act_buil.imp_fun_map, 'nil')

        self.assertEqual(act_buil.risk_transf_attach, 0)
        self.assertEqual(act_buil.risk_transf_cover, 0)

        name = 'vegetation management'
        act_buil = meas.get_measure(name)
        self.assertEqual(act_buil.name, name)
        self.assertEqual(act_buil.haz_type, 'TC')
        self.assertTrue(np.array_equal(act_buil.color_rgb, np.array([0.76, 0.84, 0.60])))
        self.assertEqual(act_buil.cost,  63968125.00687534)

        self.assertEqual(act_buil.hazard_set, 'nil')
        self.assertEqual(act_buil.hazard_freq_cutoff, 0)
        self.assertEqual(act_buil.hazard_inten_imp, (1, -1))

        self.assertEqual(act_buil.exposures_set, 'nil')
        self.assertEqual(act_buil.exp_region_id, 0)

        self.assertEqual(act_buil.paa_impact, (0.8, 0))
        self.assertEqual(act_buil.mdd_impact, (1, 0))
        self.assertEqual(act_buil.imp_fun_map, 'nil')

        self.assertEqual(act_buil.risk_transf_attach, 0)
        self.assertEqual(act_buil.risk_transf_cover, 0)

        self.assertEqual(meas.get_measure('enforce building code').imp_fun_map, '1to3')

        self.assertEqual(meas.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(meas.tag.description, '')

class TestReaderMat(unittest.TestCase):
    """Test reader functionality of the MeasuresMat class"""

    def test_demo_file(self):
        # Read demo excel file
        meas = MeasureSet()
        description = 'One single file.'
        meas.read_mat(ENT_TEST_MAT, description)

        # Check results
        n_meas = 4

        self.assertEqual(len(meas.get_measure()), n_meas)

        act_man = meas.get_measure('Mangroves')
        self.assertEqual(act_man.name, 'Mangroves')
        self.assertEqual(act_man.haz_type, 'XX')
        self.assertEqual(type(act_man.color_rgb), np.ndarray)
        self.assertEqual(len(act_man.color_rgb), 3)
        self.assertEqual(act_man.color_rgb[0], 0.1529)
        self.assertEqual(act_man.color_rgb[1], 0.251)
        self.assertEqual(act_man.color_rgb[2], 0.5451)
        self.assertEqual(act_man.cost, 1311768360.8515418)

        self.assertEqual(act_man.hazard_freq_cutoff, 0)
        self.assertEqual(act_man.hazard_set, 'nil')
        self.assertEqual(act_man.hazard_inten_imp, (1, -4))

        self.assertEqual(act_man.exposures_set, 'nil')
        self.assertEqual(act_man.exp_region_id, 0)

        self.assertEqual(act_man.mdd_impact, (1, 0))
        self.assertEqual(act_man.paa_impact, (1, 0))
        self.assertEqual(act_man.imp_fun_map, 'nil')

        self.assertEqual(act_man.risk_transf_attach, 0)
        self.assertEqual(act_man.risk_transf_cover, 0)


        act_buil = meas.get_measure('Building code')
        self.assertEqual(act_buil.name, 'Building code')
        self.assertEqual(act_buil.haz_type, 'XX')
        self.assertEqual(type(act_buil.color_rgb), np.ndarray)
        self.assertEqual(len(act_buil.color_rgb), 3)
        self.assertEqual(act_buil.color_rgb[0], 0.6980)
        self.assertEqual(act_buil.color_rgb[1], 0.8745)
        self.assertEqual(act_buil.color_rgb[2], 0.9333)
        self.assertEqual(act_buil.cost, 9200000000.0000000)

        self.assertEqual(act_buil.hazard_freq_cutoff, 0)
        self.assertEqual(act_buil.hazard_set, 'nil')
        self.assertEqual(act_buil.hazard_inten_imp, (1, 0))

        self.assertEqual(act_buil.exposures_set, 'nil')
        self.assertEqual(act_buil.exp_region_id, 0)

        self.assertEqual(act_buil.mdd_impact, (0.75, 0))
        self.assertEqual(act_buil.paa_impact, (1, 0))
        self.assertEqual(act_man.imp_fun_map, 'nil')

        self.assertEqual(act_buil.risk_transf_attach, 0)
        self.assertEqual(act_buil.risk_transf_cover, 0)

        self.assertEqual(meas.tag.file_name, ENT_TEST_MAT)
        self.assertEqual(meas.tag.description, description)

class TestWriter(unittest.TestCase):
    """Test reader functionality of the MeasuresExcel class"""

    def test_write_read_file(self):
        """ Write and read excel file"""

        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.cost = 10
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        act_1.risk_transf_cover = 500

        act_11 = Measure()
        act_11.name = 'Something'
        act_11.color_rgb = np.array([1, 1, 1])
        act_11.mdd_impact = (1, 2)
        act_11.paa_impact = (1, 3)
        act_11.hazard_inten_imp = (1, 2)
        act_11.exp_region_id = 2

        act_2 = Measure()
        act_2.name = 'Anything'
        act_2.haz_type = 'Fl'
        act_2.color_rgb = np.array([1, 1, 1])
        act_2.mdd_impact = (1, 2)
        act_2.paa_impact = (1, 2)
        act_2.hazard_inten_imp = (1, 2)
        act_2.hazard_freq_cutoff = 30
        act_2.imp_fun_map = 'map'

        meas_set = MeasureSet()
        meas_set.add_measure(act_1)
        meas_set.add_measure(act_11)
        meas_set.add_measure(act_2)

        file_name = 'test_meas.xlsx'
        meas_set.write_excel(file_name)

        meas_read = MeasureSet()
        meas_read.read_excel(file_name, 'test')

        self.assertEqual(meas_read.tag.file_name, file_name)
        self.assertEqual(meas_read.tag.description, 'test')

        for meas in meas_read.get_measure():
            if meas.name == 'Mangrove':
                meas_ref = act_1
            elif meas.name == 'Something':
                meas_ref = act_11
            elif meas.name == 'Anything':
                meas_ref = act_2

            self.assertEqual(meas_ref.name, meas.name)
            self.assertEqual(meas_ref.haz_type, meas.haz_type)
            self.assertEqual(meas_ref.cost, meas.cost)
            self.assertEqual(meas_ref.hazard_set, meas.hazard_set)
            self.assertEqual(meas_ref.hazard_freq_cutoff, meas.hazard_freq_cutoff)
            self.assertEqual(meas_ref.exposures_set, meas.exposures_set)
            self.assertEqual(meas_ref.exp_region_id, meas.exp_region_id)
            self.assertTrue(np.array_equal(meas_ref.color_rgb, meas.color_rgb))
            self.assertEqual(meas_ref.mdd_impact, meas.mdd_impact)
            self.assertEqual(meas_ref.paa_impact, meas.paa_impact)
            self.assertEqual(meas_ref.hazard_inten_imp, meas.hazard_inten_imp)
            self.assertEqual(meas_ref.imp_fun_map, meas.imp_fun_map)
            self.assertEqual(meas_ref.risk_transf_attach, meas.risk_transf_attach)
            self.assertEqual(meas_ref.risk_transf_cover, meas.risk_transf_cover)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWriter))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
unittest.TextTestRunner(verbosity=2).run(TESTS)
