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
import pandas as pd
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

def make_imp_uncs():
    
    exp = exp_dem
    exp_distr = {"x_exp": sp.stats.uniform(0.8,2),
                  }
    exp_unc = UncVar(exp, exp_distr)
    
    impf = impf_dem
    impf_distr = {"x_impf": sp.stats.beta(0.5, 1),
              }
    impf_unc = UncVar(impf, impf_distr)
    
    haz = haz_dem
    haz_distr = {"x_haz": sp.stats.poisson(1),
                  }
    haz_unc = UncVar(haz, haz_distr)

    return exp_unc, impf_unc, haz_unc

    

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
        self.assertListEqual(impf_unc.labels, ['x_impf'])
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
        
    def test_vac_to_uncvar(self):
        
        exp = exp_dem()
        distr_dict = {"x_exp": sp.stats.uniform(0.8,1.2)
              }
        
        var = UncVar.var_to_uncvar(exp)
        self.assertDictEqual(var.distr_dict, {})
        self.assertTrue(isinstance(var.evaluate({}), Exposures))
        
        unc_var = UncVar.var_to_uncvar(UncVar(exp, distr_dict))
        self.assertDictEqual(unc_var.distr_dict, distr_dict)
        self.assertTrue(isinstance(var.evaluate({}), Exposures))

        

class TestUncertainty(unittest.TestCase):
    """Test the Uncertainty class""" 

    def test_init_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'impf': impf_unc,
                           'haz': haz_unc})
        self.assertDictEqual(unc.metrics, {})
        self.assertDictEqual(unc.sensitivity, {})
        
        self.assertEqual(unc.n_samples, 0)
        self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_haz', 'x_impf'})
        self.assertSetEqual(set(unc.problem['names']),
                            {'x_exp', 'x_haz', 'x_impf'})
        self.assertSetEqual(set(unc.distr_dict.keys()),
                            {"x_exp", "x_impf", "x_haz"})
        
        unc = Uncertainty(
            {'exp': exp_unc, 'impf': impf_unc}, 
            sample = pd.DataFrame({'x_exp': [1, 2], 'x_impf': [3, 4]}),
            metrics = {'aai_agg': pd.DataFrame({'aai_agg': [100, 200]})}
            )
        self.assertEqual(unc.n_samples, 2)
        self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_impf'})
        self.assertListEqual(list(unc.metrics['aai_agg']['aai_agg']), [100, 200])
        self.assertDictEqual(unc.sensitivity, {})
        
    def test_make_sample_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'haz': haz_unc})
        
        #default sampling saltelli
        unc.make_sample(N=1, sampling_kwargs = {'calc_second_order': True})
        self.assertEqual(unc.n_samples, 1*(2*2+2)) # N * (2 * D + 2)
        self.assertTrue(isinstance(unc.sample, pd.DataFrame))
        self.assertTrue(np.allclose(
            unc.sample['x_exp'],
            np.array([1.239453, 1.837109, 1.239453, 
                      1.239453, 1.837109, 1.837109]),
            rtol=1e-05
            ))
        self.assertListEqual(list(unc.sample['x_haz']),
                             [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        
        #latin sampling
        unc.make_sample(N=1, sampling_method='latin',
                        sampling_kwargs = {'seed': 11245})
        self.assertEqual(unc.n_samples, 1)
        self.assertTrue(isinstance(unc.sample, pd.DataFrame))
        print(unc.sample)
        self.assertTrue(np.allclose(
            unc.sample['x_exp'],
            np.array([2.58309]),
            rtol=1e-05
            ))
        self.assertListEqual(list(unc.sample['x_haz']), [2.0])
        
    def test_plot_sample_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        
        unc = Uncertainty({'exp': exp_unc,
                           'haz': haz_unc})
        
        unc.make_sample(N=1)
        unc.plot_sample()
        
    def test_est_comp_time_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'haz': haz_unc})

        unc.make_sample(N=1, sampling_kwargs = {'calc_second_order': False})
        est = unc.est_comp_time(0.12345)
        self.assertEqual(est, 1*(2+2) *  0.123) # N * (D + 2)
        
        pool = Pool(nodes=4)
        est = unc.est_comp_time(0.12345, pool)
        self.assertEqual(est, 1*(2+2) *  0.123 / 4) # N * (D + 2)
        pool.close()
        pool.join()
        pool.clear()
        
    def test_calc_sensitivty_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        sample = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'rp': pd.DataFrame({'rp100': [9, 10, 11, 12], 
                                       'rp250': [100, 110, 120, 130]
                                       }
                                      )
                   }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          sample = sample,
                          metrics = metrics)

        sens = unc.calc_sensitivity(
            method_kwargs = {'calc_second_order': False}
            )
        self.assertSetEqual(set(sens.keys()), {'rp'})
        self.assertSetEqual(set(sens['rp'].keys()), {'rp100', 'rp250'})
        self.assertSetEqual(set(sens['rp']['rp100'].keys()), {'S1', 'S1_conf',
                                                              'ST', 'ST_conf'})
        self.assertTrue(np.allclose(
                sens['rp']['rp100']['S1'],
                np.array([0.66666667, 1.33333333])
                )
            )
    def test_plot_sensitivity(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        sample = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'freq_curve': pd.DataFrame(
                {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
            }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          sample = sample,
                          metrics = metrics)

        unc.calc_sensitivity(method_kwargs = {'calc_second_order': False})
        unc.plot_sensitivity()
    
    def test_plot_distribution(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        sample = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'freq_curve': pd.DataFrame(
                {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
            }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          sample = sample,
                          metrics = metrics)

        unc.plot_distribution()
    
        
        
class TestUncImpact(unittest.TestCase):
    """Test the UncImpact class"""  

    def test_init_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        haz = haz_dem()

        unc = UncImpact(exp_unc, impf_unc, haz)
        
        self.assertSetEqual(set(unc.metrics.keys()),
             {'aai_agg', 'freq_curve', 'eai_exp', 'at_event'}
             )
        self.assertSetEqual(set(unc.unc_vars.keys()), {'exp', 'impf', 'haz'})
        
    def test_calc_distribution_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        haz = haz_dem()
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)
        
        unc.calc_distribution()
        
        self.assertListEqual(unc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc.calc_eai_exp, False)
        self.assertEqual(unc.calc_at_event, False)
        
        self.assertTrue(
            np.allclose(
                unc.metrics['aai_agg'].aai_agg,
                np.array([6.750486e+07, 1.000553e+08, 3.307738e+09,
                          3.307738e+09, 1.000553e+08,4.902708e+09])
                )
            )
        self.assertTrue(
            np.allclose(
                unc.metrics['freq_curve'].rp5,
                np.zeros(6)
                )
            )
        self.assertTrue(
            np.allclose(
                unc.metrics['freq_curve'].rp250,
                np.array([2.025634e+09, 3.002382e+09, 9.925607e+10,
                          9.925607e+10, 3.002382e+09, 1.471167e+11])
                )
            )
        self.assertTrue(unc.metrics['eai_exp'].empty)
        self.assertTrue(unc.metrics['at_event'].empty)
        
    def test_plot_distribution_pass(self):
        
        exp_unc, impf_unc, haz_unc = make_imp_uncs()
        haz = haz_dem()
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)
        unc.calc_distribution()
        unc.plot_rp_distribution()

        
        # self.assertDictEqual(unc.sensitivity, {})
        
        # self.assertEqual(unc.n_samples, 0)
        # self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_haz', 'x_impf'})
        # self.assertSetEqual(set(unc.problem['names']),
        #                     {'x_exp', 'x_haz', 'x_impf'})
        # self.assertSetEqual(set(unc.distr_dict.keys()),
        #                     {"x_exp", "x_impf", "x_haz"})
        
        # unc = Uncertainty(
        #     {'exp': exp_unc, 'impf': impf_unc}, 
        #     sample = pd.DataFrame({'x_exp': [1, 2], 'x_impf': [3, 4]}),
        #     metrics = {'aai_agg': pd.DataFrame({'aai_agg': [100, 200]})}
        #     )
        # self.assertEqual(unc.n_samples, 2)
        # self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_impf'})
        # self.assertListEqual(list(unc.metrics['aai_agg']['aai_agg']), [100, 200])
        # self.assertDictEqual(unc.sensitivity, {})       
    
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
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertainty))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncImpact))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
