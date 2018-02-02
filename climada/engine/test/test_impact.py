"""
Test Impact class.
"""

import unittest
import numpy as np

from climada.util.constants import HAZ_DEMO_MAT, ENT_DEMO_XLS
from climada.entity.entity import Entity
from climada.hazard.base import Hazard
from climada.engine.impact import Impact

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
        self.assertEqual(' x ', ifc.label)
        self.assertEqual('USD', ifc.unit)

class TestOneExposure(unittest.TestCase):
    '''Test one_exposure function'''
    def test_ref_value_pass(self):
        ''' Test result against reference value'''
        # Read demo entity values
        # Set the entity default file to the demo one
        Entity.def_file = ENT_DEMO_XLS
        ent = Entity()
        
        # Read default hazard file
        hazard = Hazard(HAZ_DEMO_MAT, 'TC')
        # Create impact object
        impact = Impact()

        # Assign centroids to exposures
        ent.exposures.assign(hazard)

        # Compute impact for 6th exposure
        iexp = 5
        # Take its impact function
        imp_id = ent.exposures.impact_id[iexp]
        imp_fun = ent.impact_funcs.get_vulner(hazard.tag.type, imp_id)
        # Compute
        event_row, result = impact._one_exposure(iexp, ent.exposures, \
                                                  hazard, imp_fun)

        # Check sizes
        num_res = 1280
        self.assertEqual(num_res, len(event_row))
        self.assertEqual(num_res, len(result))

        # Check first 3 values
        self.assertEqual(0, result[0])
        self.assertEqual(0, result[1])
        self.assertEqual(1.0626600695059455e+06, result[2])
        self.assertEqual(12, event_row[0])
        self.assertEqual(41, event_row[1])
        self.assertEqual(44, event_row[2])

        # Check intermediate values
        self.assertEqual(0, result[678])
        self.assertEqual(0, result[543])
        self.assertEqual(0, result[982])
        self.assertEqual(1.3318063850487845e+08, result[750])
        self.assertEqual(4.6671085550540835e+06, result[917])
        self.assertEqual(6281, event_row[678])
        self.assertEqual(4998, event_row[543])
        self.assertEqual(9527, event_row[982])
        self.assertEqual(7192, event_row[750])
        self.assertEqual(8624, event_row[917])

        # Check last 3 values
        self.assertEqual(0, result[num_res-1])
        self.assertEqual(0, result[num_res-2])
        self.assertEqual(0, result[num_res-3])
        self.assertEqual(14349, event_row[num_res-1])
        self.assertEqual(14347, event_row[num_res-2])
        self.assertEqual(14309, event_row[num_res-3])

class TestCalc(unittest.TestCase):
    ''' Test impact calc method.'''

    def test_ref_value_pass(self):
        ''' Test result against reference value'''
        # Read default entity values
        Entity.def_file = ENT_DEMO_XLS
        ent = Entity()
        # Read default hazard file
        hazard = Hazard(HAZ_DEMO_MAT, 'TC')
        # Create impact object
        impact = Impact()

        # Assign centroids to exposures
        ent.exposures.assign(hazard)

        # Compute the impact over the whole exposures
        impact.calc(ent.exposures, ent.impact_funcs, hazard)

        # Check result
        num_events = len(hazard.event_id)
        num_exp = len(ent.exposures.id)
        # Check relative errors as well when absolute value gt 1.0e-7
        self.assertEqual(num_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[int(num_events/2)])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809])
        self.assertEqual(7.076504723057620e+10, impact.at_event[12147])
        self.assertEqual(0, impact.at_event[num_events-1])

        self.assertEqual(num_exp, len(impact.at_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.at_exp[0])
        self.assertAlmostEqual(1.373490457046383e+08, \
                               impact.at_exp[int(num_exp/2)], 6)
        self.assertEqual(True, np.isclose(1.373490457046383e+08, \
                                          impact.at_exp[int(num_exp/2)]))
        self.assertAlmostEqual(1.066837260150042e+08, \
                               impact.at_exp[num_exp-1], 6)
        self.assertEqual(True, np.isclose(1.066837260150042e+08, \
                                          impact.at_exp[int(num_exp-1)]))

        self.assertAlmostEqual(6.570532945599104e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.tot, 5)
        self.assertEqual(True, np.isclose(6.512201157564421e+09, impact.tot))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestOneExposure)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalc))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFreqCurve))
unittest.TextTestRunner(verbosity=2).run(TESTS)
