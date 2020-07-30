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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
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
        with self.assertLogs('climada.entity.measures.measure_set', level='WARNING') as cm:
            meas.append(act_1)
        self.assertIn("Input Measure's hazard type not set.", cm.output[0])

        with self.assertLogs('climada.entity.measures.measure_set', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.append(45)
        self.assertIn("Input value is not of type Measure.", cm.output[0])

    def test_remove_measure_pass(self):
        """Test remove_measure removes Measure of MeasureSet correcty."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        meas.remove_measure(name='Mangrove')
        self.assertEqual(0, meas.size())

    def test_remove_wrong_error(self):
        """Test error is raised when invalid inputs."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        with self.assertLogs('climada.entity.measures.measure_set', level='INFO') as cm:
            meas.remove_measure(name='Seawall')
        self.assertIn('No Measure with name Seawall.', cm.output[0])

    def test_get_names_pass(self):
        """Test get_names function."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        self.assertEqual(1, len(meas.get_names()))
        self.assertEqual({'FL': ['Mangrove']}, meas.get_names())

        act_2 = Measure()
        act_2.name = 'Seawall'
        act_2.haz_type = 'FL'
        meas.append(act_2)
        self.assertEqual(2, len(meas.get_names('FL')))
        self.assertIn('Mangrove', meas.get_names('FL'))
        self.assertIn('Seawall', meas.get_names('FL'))

    def test_get_measure_pass(self):
        """Test normal functionality of get_measure method."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        self.assertIs(act_1, meas.get_measure(name='Mangrove')[0])

        act_2 = Measure()
        act_2.name = 'Seawall'
        act_2.haz_type = 'FL'
        meas.append(act_2)
        self.assertIs(act_1, meas.get_measure(name='Mangrove')[0])
        self.assertIs(act_2, meas.get_measure(name='Seawall')[0])
        self.assertEqual(2, len(meas.get_measure('FL')))

    def test_get_measure_wrong_error(self):
        """Test get_measure method with wrong inputs."""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Seawall'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        self.assertEqual([], meas.get_measure('Mangrove'))

    def test_num_measures_pass(self):
        """Test num_measures function."""
        meas = MeasureSet()
        self.assertEqual(0, meas.size())
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'FL'
        meas.append(act_1)
        self.assertEqual(1, meas.size())
        meas.append(act_1)
        self.assertEqual(1, meas.size())

        act_2 = Measure()
        act_2.name = 'Seawall'
        act_2.haz_type = 'FL'
        meas.append(act_2)
        self.assertEqual(2, meas.size())

class TestChecker(unittest.TestCase):
    """Test check functionality of the MeasureSet class"""

    def test_check_wronginten_fail(self):
        """Wrong intensity definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.haz_type = 'TC'
        act_1.name = 'Mangrove'
        act_1.hazard_inten_imp = (1, 2, 3)
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        meas.append(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.hazard_inten_imp size: 2 != 3.',
                         cm.output[0])

    def test_check_wrongColor_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'DR'
        act_1.color_rgb = (1, 2)
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.append(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.color_rgb size: 3 != 2.', cm.output[0])

    def test_check_wrongMDD_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'DR'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.append(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Measure.mdd_impact has wrong dimensions.', cm.output[0])

    def test_check_wrongPAA_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'TC'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2, 3, 4)
        act_1.hazard_inten_imp = (1, 2)
        meas.append(act_1)

        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Invalid Measure.paa_impact size: 2 != 4.', cm.output[0])

    def test_check_name_fail(self):
        """Wrong measures definition"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'LaLa'
        act_1.haz_type = 'FL'
        meas._data['FL'] = dict()
        meas._data['FL']['LoLo'] = act_1

        with self.assertLogs('climada.entity.measures.measure_set', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                meas.check()
        self.assertIn('Wrong Measure.name: LoLo != LaLa', cm.output[0])

    def test_def_color(self):
        """Test default grey scale used when no color set"""
        meas = MeasureSet()
        act_1 = Measure()
        act_1.name = 'LaLa'
        act_1.haz_type = 'FL'
        meas.append(act_1)

        act_2 = Measure()
        act_2.name = 'LoLo'
        act_2.haz_type = 'FL'
        meas.append(act_2)

        meas.check()
        self.assertTrue(np.array_equal(meas.get_measure('FL', 'LaLa').color_rgb, np.ones(4)))
        self.assertTrue(np.allclose(meas.get_measure('FL', 'LoLo').color_rgb,
            np.array([0., 0., 0., 1.0])))

class TestExtend(unittest.TestCase):
    """Check extend function"""
    def test_extend_to_empty_same(self):
        """Extend MeasureSet to empty one."""
        meas = MeasureSet()
        meas_add = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'TC'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas_add.append(act_1)

        meas.extend(meas_add)
        meas.check()

        self.assertEqual(meas.size(), 1)
        self.assertEqual(meas.get_names(), {'TC': ['Mangrove']})

    def test_extend_equal_same(self):
        """Extend the same MeasureSet. The inital MeasureSet is obtained."""
        meas = MeasureSet()
        meas_add = MeasureSet()
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'TC'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        meas.append(act_1)
        meas_add.append(act_1)

        meas.extend(meas_add)
        meas.check()

        self.assertEqual(meas.size(), 1)
        self.assertEqual(meas.get_names(), {'TC': ['Mangrove']})

    def test_extend_different_extend(self):
        """Extend MeasureSet with same and new values. The actions
        with repeated name are overwritten."""
        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'TC'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)

        act_11 = Measure()
        act_11.name = 'Mangrove'
        act_11.haz_type = 'TC'
        act_11.color_rgb = np.array([1, 1, 1])
        act_11.mdd_impact = (1, 2)
        act_11.paa_impact = (1, 3)
        act_11.hazard_inten_imp = (1, 2)

        act_2 = Measure()
        act_2.name = 'Anything'
        act_2.haz_type = 'TC'
        act_2.color_rgb = np.array([1, 1, 1])
        act_2.mdd_impact = (1, 2)
        act_2.paa_impact = (1, 2)
        act_2.hazard_inten_imp = (1, 2)

        meas = MeasureSet()
        meas.append(act_1)
        meas_add = MeasureSet()
        meas_add.append(act_11)
        meas_add.append(act_2)

        meas.extend(meas_add)
        meas.check()

        self.assertEqual(meas.size(), 2)
        self.assertEqual(meas.get_names(), {'TC': ['Mangrove', 'Anything']})
        self.assertEqual(meas.get_measure(name=act_1.name)[0].paa_impact, act_11.paa_impact)

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the MeasuresExcel class"""

    def test_demo_file(self):
        """Read demo excel file"""
        meas = MeasureSet()
        description = 'One single file.'
        meas.read_excel(ENT_DEMO_TODAY, description)

        # Check results
        n_meas = 4

        self.assertEqual(meas.size(), n_meas)

        act_man = meas.get_measure(name='Mangroves')[0]
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

        act_buil = meas.get_measure(name='Building code')[0]
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
        """Read template excel file"""
        meas = MeasureSet()
        meas.read_excel(ENT_TEMPLATE_XLS)

        self.assertEqual(meas.size(), 7)

        name = 'elevate existing buildings'
        act_buil = meas.get_measure(name=name)[0]
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
        self.assertEqual(act_buil.risk_transf_cost_factor, 1)

        name = 'vegetation management'
        act_buil = meas.get_measure(name=name)[0]
        self.assertEqual(act_buil.name, name)
        self.assertEqual(act_buil.haz_type, 'TC')
        self.assertTrue(np.array_equal(act_buil.color_rgb, np.array([0.76, 0.84, 0.60])))
        self.assertEqual(act_buil.cost, 63968125.00687534)

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
        self.assertEqual(act_buil.risk_transf_cost_factor, 1)

        self.assertEqual(meas.get_measure(name='enforce building code')[0].imp_fun_map, '1to3')

        name = 'risk transfer'
        act_buil = meas.get_measure(name=name)[0]
        self.assertEqual(act_buil.name, name)
        self.assertEqual(act_buil.haz_type, 'TC')
        self.assertTrue(np.array_equal(act_buil.color_rgb, np.array([0.90, 0.72, 0.72])))
        self.assertEqual(act_buil.cost, 21000000)

        self.assertEqual(act_buil.hazard_set, 'nil')
        self.assertEqual(act_buil.hazard_freq_cutoff, 0)
        self.assertEqual(act_buil.hazard_inten_imp, (1, 0))

        self.assertEqual(act_buil.exposures_set, 'nil')
        self.assertEqual(act_buil.exp_region_id, 0)

        self.assertEqual(act_buil.paa_impact, (1, 0))
        self.assertEqual(act_buil.mdd_impact, (1, 0))
        self.assertEqual(act_buil.imp_fun_map, 'nil')

        self.assertEqual(act_buil.risk_transf_attach, 500000000)
        self.assertEqual(act_buil.risk_transf_cover, 1000000000)
        self.assertEqual(act_buil.risk_transf_cost_factor, 2)

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

        self.assertEqual(meas.size(), n_meas)

        act_man = meas.get_measure(name='Mangroves')[0]
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
        self.assertEqual(act_man.exp_region_id, [])

        self.assertEqual(act_man.mdd_impact, (1, 0))
        self.assertEqual(act_man.paa_impact, (1, 0))
        self.assertEqual(act_man.imp_fun_map, 'nil')

        self.assertEqual(act_man.risk_transf_attach, 0)
        self.assertEqual(act_man.risk_transf_cover, 0)


        act_buil = meas.get_measure(name='Building code')[0]
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
        self.assertEqual(act_buil.exp_region_id, [])

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
        """Write and read excel file"""

        act_1 = Measure()
        act_1.name = 'Mangrove'
        act_1.haz_type = 'TC'
        act_1.color_rgb = np.array([1, 1, 1])
        act_1.cost = 10
        act_1.mdd_impact = (1, 2)
        act_1.paa_impact = (1, 2)
        act_1.hazard_inten_imp = (1, 2)
        act_1.risk_transf_cover = 500

        act_11 = Measure()
        act_11.name = 'Something'
        act_11.haz_type = 'TC'
        act_11.color_rgb = np.array([1, 1, 1])
        act_11.mdd_impact = (1, 2)
        act_11.paa_impact = (1, 3)
        act_11.hazard_inten_imp = (1, 2)
        act_11.exp_region_id = [2]

        act_2 = Measure()
        act_2.name = 'Anything'
        act_2.haz_type = 'FL'
        act_2.color_rgb = np.array([1, 1, 1])
        act_2.mdd_impact = (1, 2)
        act_2.paa_impact = (1, 2)
        act_2.hazard_inten_imp = (1, 2)
        act_2.hazard_freq_cutoff = 30
        act_2.imp_fun_map = 'map'

        meas_set = MeasureSet()
        meas_set.append(act_1)
        meas_set.append(act_11)
        meas_set.append(act_2)

        file_name = os.path.join(DATA_DIR, 'test_meas.xlsx')
        meas_set.write_excel(file_name)

        meas_read = MeasureSet()
        meas_read.read_excel(file_name, 'test')

        self.assertEqual(meas_read.tag.file_name, file_name)
        self.assertEqual(meas_read.tag.description, 'test')

        meas_list = meas_read.get_measure('TC')
        meas_list.extend(meas_read.get_measure('FL'))

        for meas in meas_list:
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
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtend))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWriter))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConstructor))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
