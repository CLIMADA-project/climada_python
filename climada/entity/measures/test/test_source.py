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

Test MeasureSet from Excel file.
"""
import os
import unittest
import numpy as np

from climada.entity.measures.measure_set import MeasureSet
from climada.util.constants import ENT_TEMPLATE_XLS, ENT_DEMO_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the MeasuresExcel class"""

    def test_demo_file(self):
        """ Read demo excel file"""
        meas = MeasureSet()
        description = 'One single file.'
        meas.read(ENT_TEST_XLS, description)

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

        self.assertEqual(meas.tag.file_name, ENT_TEST_XLS)
        self.assertEqual(meas.tag.description, description)

    def test_template_file_pass(self):
        """ Read template excel file"""
        meas = MeasureSet()
        meas.read(ENT_TEMPLATE_XLS)

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
        meas.read(ENT_DEMO_MAT, description)

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

        self.assertEqual(meas.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(meas.tag.description, description)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderMat)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
unittest.TextTestRunner(verbosity=2).run(TESTS)
