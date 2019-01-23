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

Test Impact class.
"""
import os
import unittest
import numpy as np

from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine.impact import Impact

HAZ_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'hazard/test/data/')
HAZ_TEST_MAT = os.path.join(HAZ_DIR, 'atl_prob_no_name.mat')
ENT_TEST_XLS = os.path.join(os.path.dirname(__file__), 'data/demo_today.xlsx')

class TestFreqCurve(unittest.TestCase):
    '''Test exceedence frequency curve computation'''
    def test_ref_value_pass(self):
        """Test result against reference value"""
        imp = Impact()
        imp.frequency = np.ones(10) * 6.211180124223603e-04
        imp.at_event = np.zeros(10)
        imp.at_event[0] = 0
        imp.at_event[1] = 0.400665463736549e9
        imp.at_event[2] = 3.150330960044466e9
        imp.at_event[3] = 3.715826406781887e9
        imp.at_event[4] = 2.900244271902339e9
        imp.at_event[5] = 0.778570745161971e9
        imp.at_event[6] = 0.698736262566472e9
        imp.at_event[7] = 0.381063674256423e9
        imp.at_event[8] = 0.569142464157450e9
        imp.at_event[9] = 0.467572545849132e9
        imp.unit = 'USD'

        ifc = imp.calc_freq_curve()
        self.assertEqual(10, len(ifc.return_per))
        self.assertEqual(1610.0000000000000, ifc.return_per[0])
        self.assertEqual(805.00000000000000, ifc.return_per[1])
        self.assertEqual(536.66666666666663, ifc.return_per[2])
        self.assertEqual(402.500000000000, ifc.return_per[3])
        self.assertEqual(322.000000000000, ifc.return_per[4])
        self.assertEqual(268.33333333333331, ifc.return_per[5])
        self.assertEqual(230.000000000000, ifc.return_per[6])
        self.assertEqual(201.250000000000, ifc.return_per[7])
        self.assertEqual(178.88888888888889, ifc.return_per[8])
        self.assertEqual(161.000000000000, ifc.return_per[9])
        self.assertEqual(10, len(ifc.impact))
        self.assertEqual(3.715826406781887e9, ifc.impact[0])
        self.assertEqual(3.150330960044466e9, ifc.impact[1])
        self.assertEqual(2.900244271902339e9, ifc.impact[2])
        self.assertEqual(0.778570745161971e9, ifc.impact[3])
        self.assertEqual(0.698736262566472e9, ifc.impact[4])
        self.assertEqual(0.569142464157450e9, ifc.impact[5])
        self.assertEqual(0.467572545849132e9, ifc.impact[6])
        self.assertEqual(0.400665463736549e9, ifc.impact[7])
        self.assertEqual(0.381063674256423e9, ifc.impact[8])
        self.assertEqual(0, ifc.impact[9])
        self.assertEqual('Exceedance frequency curve', ifc.label)
        self.assertEqual('USD', ifc.unit)

    def test_ref_value_rp_pass(self):
        """Test result against reference value with given return periods """
        imp = Impact()
        imp.frequency = np.ones(10) * 6.211180124223603e-04
        imp.at_event = np.zeros(10)
        imp.at_event[0] = 0
        imp.at_event[1] = 0.400665463736549e9
        imp.at_event[2] = 3.150330960044466e9
        imp.at_event[3] = 3.715826406781887e9
        imp.at_event[4] = 2.900244271902339e9
        imp.at_event[5] = 0.778570745161971e9
        imp.at_event[6] = 0.698736262566472e9
        imp.at_event[7] = 0.381063674256423e9
        imp.at_event[8] = 0.569142464157450e9
        imp.at_event[9] = 0.467572545849132e9
        imp.unit = 'USD'

        ifc = imp.calc_freq_curve(np.array([100, 500, 1000]))
        self.assertEqual(3, len(ifc.return_per))
        self.assertEqual(100, ifc.return_per[0])
        self.assertEqual(500, ifc.return_per[1])
        self.assertEqual(1000, ifc.return_per[2])
        self.assertEqual(3, len(ifc.impact))
        self.assertEqual(0, ifc.impact[0])
        self.assertEqual(2320408028.5695677, ifc.impact[1])
        self.assertEqual(3287314329.129928, ifc.impact[2])
        self.assertEqual('Exceedance frequency curve', ifc.label)
        self.assertEqual('USD', ifc.unit)

class TestOneExposure(unittest.TestCase):
    '''Test one_exposure function'''
    def test_ref_value_insure_pass(self):
        ''' Test result against reference value'''
        # Read demo entity values
        # Set the entity default file to the demo one
        ent = Entity()
        ent.read_excel(ENT_TEST_XLS)
        
        # Read default hazard file
        hazard = Hazard('TC', HAZ_TEST_MAT)
        # Create impact object
        impact = Impact()
        impact.at_event = np.zeros(hazard.intensity.shape[0])
        impact.eai_exp = np.zeros(len(ent.exposures.value))
        impact.tot_value = 0

        # Assign centroids to exposures
        ent.exposures.assign(hazard)

        # Compute impact for 6th exposure
        iexp = 5
        # Take its impact function
        imp_id = ent.exposures.if_TC[iexp]
        imp_fun = ent.impact_funcs.get_func(hazard.tag.haz_type, imp_id)[0]
        # Compute
        insure_flag = True
        impact._exp_impact(np.array([iexp]), ent.exposures, hazard, imp_fun, insure_flag)
        
        self.assertEqual(impact.eai_exp.size, ent.exposures.shape[0])
        self.assertEqual(impact.at_event.size, hazard.intensity.shape[0])
        
        events_pos = hazard.intensity[:, ent.exposures.centr_TC[iexp]].nonzero()[0]
        res_exp = np.zeros((ent.exposures.shape[0]))
        res_exp[iexp] = np.sum(impact.at_event[events_pos] * hazard.frequency[events_pos])
        self.assertTrue(np.array_equal(res_exp, impact.eai_exp))

        self.assertEqual(0, impact.at_event[12])
        # Check first 3 values
        self.assertEqual(0, impact.at_event[12])
        self.assertEqual(0, impact.at_event[41])
        self.assertEqual(1.0626600695059455e+06, impact.at_event[44])

        # Check intermediate values
        self.assertEqual(0, impact.at_event[6281])
        self.assertEqual(0, impact.at_event[4998])
        self.assertEqual(0, impact.at_event[9527])
        self.assertEqual(1.3318063850487845e+08, impact.at_event[7192])
        self.assertEqual(4.667108555054083e+06, impact.at_event[8624])

        # Check last 3 values
        self.assertEqual(0, impact.at_event[14349])
        self.assertEqual(0, impact.at_event[14347])
        self.assertEqual(0, impact.at_event[14309])

class TestCalc(unittest.TestCase):
    ''' Test impact calc method.'''

    def test_ref_value_pass(self):
        ''' Test result against reference value'''
        # Read default entity values
        ent = Entity()
        ent.read_excel(ENT_TEST_XLS)
        ent.check()

        # Read default hazard file
        hazard = Hazard('TC', HAZ_TEST_MAT)
        # Create impact object
        impact = Impact()

        # Assign centroids to exposures
        ent.exposures.assign(hazard)

        # Compute the impact over the whole exposures
        impact.calc(ent.exposures, ent.impact_funcs, hazard)
        
        # Check result
        num_events = len(hazard.event_id)
        num_exp = ent.exposures.shape[0]
        # Check relative errors as well when absolute value gt 1.0e-7
        # impact.at_event == EDS.damage in MATLAB
        self.assertEqual(num_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[int(num_events/2)])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809])
        self.assertEqual(7.076504723057619e+10, impact.at_event[12147])
        self.assertEqual(0, impact.at_event[num_events-1])
        # impact.eai_exp == EDS.ED_at_centroid in MATLAB
        self.assertEqual(num_exp, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.eai_exp[0])
        self.assertAlmostEqual(1.373490457046383e+08, \
                               impact.eai_exp[int(num_exp/2)], 6)
        self.assertTrue(np.isclose(1.373490457046383e+08, \
                                          impact.eai_exp[int(num_exp/2)]))
        self.assertAlmostEqual(1.066837260150042e+08, \
                               impact.eai_exp[num_exp-1], 6)
        self.assertTrue(np.isclose(1.066837260150042e+08, \
                                          impact.eai_exp[int(num_exp-1)]))
        # impact.tot_value == EDS.Value in MATLAB
        # impact.aai_agg == EDS.ED in MATLAB
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)
        self.assertTrue(np.isclose(6.512201157564421e+09, impact.aai_agg))
     
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOneExposure)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalc))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFreqCurve))
unittest.TextTestRunner(verbosity=2).run(TESTS)

