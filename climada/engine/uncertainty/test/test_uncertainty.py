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

Test uncertainty module.
"""



import unittest

from climada.entity import ImpactFunc, ImpactFuncSet
import numpy as np
from climada.entity import Entity
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5
from climada.entity import Exposures
from climada.hazard import Hazard
from climada.engine.uncertainty import UncVar, UncImpact, UncCostBenefit, Uncertainty
import scipy as sp
from pathos.pools import ProcessPool as Pool


def impf_dem(x_impf=1):
    impf = ImpactFunc()
    impf.haz_type = 'TC'
    impf.id = 1
    impf.intensity_unit = 'm/s'
    impf.intensity = np.linspace(0, 150, num=100)
    impf.mdd = np.repeat(1, len(impf.intensity))
    impf.paa = np.arange(0, len(impf.intensity)) / len(impf.intensity) * x_impf
    impf.check()
    impf_set = ImpactFuncSet()
    impf_set.append(impf)
    return impf_set

def exp_dem(x_exp=1):
    exp = Exposures()
    exp.read_hdf5(EXP_DEMO_H5)
    exp.value *= x_exp
    exp.check()
    return exp

def haz_dem(x_haz=1):
    haz= Hazard()
    haz.read_hdf5(HAZ_DEMO_H5)
    haz.intensity = haz.intensity.multiply(x_haz)
    return haz

# HAZ_TEST_MAT = '/Users/ckropf/Documents/Climada/climada_python/climada/hazard/test/data/atl_prob_no_name.mat'
# ENT_TEST_MAT = '/Users/ckropf/Documents/Climada/climada_python/climada/entity/exposures/test/data/demo_today.mat'
# def dummy_ent():
#     entity = Entity()
#     entity.read_mat(ENT_TEST_MAT)
#     entity.check()
#     entity.measures._data['TC'] = entity.measures._data.pop('XX')
#     for meas in entity.measures.get_measure('TC'):
#         meas.haz_type = 'TC'
#     entity.check()
#     return entity


class TestUncVar(unittest.TestCase):
    """ Test UncVar calss """
    
    def test_init_pass(self):
        
        impf = impf_dem
        distr_dict = {"x_impf": sp.stats.uniform(0.8,1.2)
              }
        impf_unc = UncVar(impf, distr_dict)
        self.assertTrue(
            np.array_equal(impf_unc.labels, ['x_impf'])
            )
        self.assertTrue(isinstance(impf_unc.distr_dict, dict))
        
    def test_evaluate_pass(self):
        
        impf = impf_dem
        distr_dict = {"x_impf": sp.stats.uniform(0.8, 1.2),
              }
        impf_unc = UncVar(impf, distr_dict)
        impf_eval = impf_unc.evaluate({'x_impf': 0.8})
        impf_true = impf_dem(x_impf = 0.8)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()['TC'][1]
        impf_func2 = impf_true.get_func()['TC'][1]
        self.assertTrue(
            np.array_equal(
                impf_func1.intensity, 
                impf_func2.intensity
                )
            )
        self.assertTrue(
            np.array_equal(
                impf_func1.mdd, 
                impf_func2.mdd
                )
            )
        self.assertTrue(
            np.array_equal(
                impf_func1.paa, 
                impf_func2.paa
                )
            )
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

    def test_plot_pass(self):
        impf = impf_dem()
        distr_dict = {'x_impf': sp.stats.norm(1, 2)
              }
        impf_unc = UncVar(impf, distr_dict)
        self.assertIsNotNone(impf_unc.plot());


class TestUncertainty(unittest.TestCase):
    """Test the Uncertainty class""" 

    # exp = exp_dem()
    # haz = haz_dem()
    # impf = impf_dem()

    # distr_dict = {"G": sp.stats.uniform(0.8,1),
    #               "v_half": sp.stats.uniform(50, 100),
    #               "vmin": sp.stats.uniform(15,30),
    #               "k": sp.stats.uniform(1, 5)
    #               }
    
    # impf_unc = UncVar(impf, distr_dict)

    # impf_unc.plot_distr()
    # unc = UncImpact(exp, impf_unc, haz)
    # unc.make_sample(N=10, sampling_kwargs = {'calc_second_order': False})
    # unc.plot_sample()
    # unc.calc_distribution(calc_eai_exp=False)
    # unc.calc_sensitivity(method_kwargs = {'calc_second_order': False})

    # unc.plot_distribution(['aai_agg', 'freq_curve'])
    # unc.plot_rp_distribution()
    # unc.plot_sensitivity()
    
    
    # unc.make_sample(N=1000)
    # unc.plot_sample()


# class TestUncertainty(unittest.TestCase):

#     exp = exp()
#     haz = haz()
#     impf = imp_fun_tc


#     pool = Pool()
#     haz_unc = UncVar(dummy_haz, {'x': sp.stats.norm(1, 1)})
#     ent = dummy_ent()
#     unc = UncCostBenefit(haz_unc, ent)
#     unc.make_sample(N=1)
#     unc.calc_distribution(pool=pool)
#     unc.calc_sensitivity()
#     pool.close()
#     pool.join()
#     pool.clear()
    
#     unc.plot_sensitivity(metric_list=list(unc.metrics.keys())[0:6])
#     unc.plot_distribution(metric_list=list(unc.metrics.keys())[0:6])

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUncVar)
    #TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertainty))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
