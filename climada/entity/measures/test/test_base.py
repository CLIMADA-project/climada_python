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
import os
import unittest
import numpy as np

from climada.hazard.base import Hazard
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.measures.measure_set import MeasureSet
from climada.util.constants import ENT_DEMO_MAT

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, \
                        os.pardir, 'hazard', 'test', 'data')
HAZ_TEST_MAT = os.path.join(DATA_DIR, 'atl_prob_no_name.mat')

class TestApply(unittest.TestCase):
    """Test implement measures functions."""
    def test_change_imp_func_pass(self):
        """Test change_imp_func"""
        meas = MeasureSet()
        meas.read_mat(ENT_DEMO_MAT)
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
        new_imp = act_1._change_imp_func(imp_set).get_func('XX')[0]

        self.assertTrue(np.array_equal(new_imp.intensity, np.array([4., 24., 34., 44.,
            54., 64., 74., 84., 104.])))
        self.assertTrue(np.array_equal(new_imp.mdd, np.array([0, 0, 0.021857142857143, 0.035887500000000,
            0.053977415307403, 0.103534246575342, 0.180414000000000, 0.410796000000000, 0.410796000000000])))
        self.assertTrue(np.array_equal(new_imp.paa, np.array([0, 0.005000000000000, 0.042000000000000,
            0.160000000000000, 0.398500000000000, 0.657000000000000, 1.000000000000000,
            1.000000000000000, 1.000000000000000])))
        self.assertFalse(id(new_imp) == id(imp_tc))

    def test_change_hazard_pass(self):
        """Test change_hazard"""
        meas = MeasureSet()
        meas.read_mat(ENT_DEMO_MAT)
        act_1 = meas.get_measure('Seawall')
        
        haz = Hazard('TC', HAZ_TEST_MAT)
        exp = Exposures()
        exp.read_mat(ENT_DEMO_MAT)
        exp.rename(columns={'if_': 'if_TC'}, inplace=True)

        imp_set = ImpactFuncSet()
        imp_set.read_mat(ENT_DEMO_MAT)
        
        new_haz = act_1._change_hazard(exp, imp_set, haz)
        
        self.assertFalse(id(new_haz) == id(haz))
        
        pos_ref = np.array([6222, 13166, 7314, 11697, 7318, 7319, 7478, 5326, 5481, 7471,
                   7480, 6224, 4812, 5759, 777, 5530, 7476, 5489, 5528, 5529,
                   4813, 5329, 7192, 4284, 7195, 5527, 5490, 7479, 7311, 5352,
                   7194, 11698, 4283, 5979, 3330, 5977, 9052, 3895, 780,
                   7102, 5971, 8678, 4820, 5328, 6246, 11699, 12499, 7200, 3327,
                   779, 12148, 6247, 5485, 11695, 5950, 7433, 5948, 1077, 5949,
                   1071, 4097, 7103, 9054, 5140, 2430, 9051, 9053, 5945, 13200,
                   13501, 9135, 7698, 6250]) - 1 
        all_haz = np.arange(haz.intensity.shape[0])
        all_haz[pos_ref] = -1
        pos_null = np.argwhere(all_haz > 0).reshape(-1)
        for i_ev in pos_null:
            self.assertEqual(new_haz.intensity[i_ev, :].max(), 0)

    def test_apply_pass(self):
        """Test implement"""
        meas = MeasureSet()
        meas.read_mat(ENT_DEMO_MAT)
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
        new_exp, new_ifs, new_haz = act_1.apply(exposures, imp_set, hazard)

        self.assertFalse(id(new_ifs) == id(imp_tc))
        self.assertTrue(id(new_exp) == id(exposures))
        self.assertTrue(id(new_haz) == id(hazard))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestApply)
unittest.TextTestRunner(verbosity=2).run(TESTS)
