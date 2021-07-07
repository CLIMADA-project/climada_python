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

Test uncertainty module.
"""



import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from pathos.pools import ProcessPool as Pool

from climada.entity import ImpactFunc, ImpactFuncSet
from climada.entity.entity_def import Entity
from climada.entity import Exposures
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5, ENT_DEMO_TODAY, ENT_DEMO_FUTURE
from climada.hazard import Hazard
from climada.engine.uncertainty import UncVar, UncImpact, UncCostBenefit, Uncertainty


def impf_dem(x_paa=1, x_mdd=1):
    impf = ImpactFunc()
    impf.haz_type = 'TC'
    impf.id = 1
    impf.intensity_unit = 'm/s'
    impf.intensity = np.linspace(0, 150, num=100)
    impf.mdd = np.repeat(1, len(impf.intensity)) * x_mdd
    impf.paa = np.arange(0, len(impf.intensity)) / len(impf.intensity) * x_paa
    impf.check()
    impf_set = ImpactFuncSet()
    impf_set.append(impf)
    return impf_set

def exp_dem(x_exp=1):
    exp = Exposures()
    exp.read_hdf5(EXP_DEMO_H5)
    exp.gdf.value *= x_exp
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
    impf_distr = {"x_paa": sp.stats.beta(0.5, 1),
                  "x_mdd": sp.stats.uniform(0.8, 1.2)
              }
    impf_unc = UncVar(impf, impf_distr)

    haz = haz_dem
    haz_distr = {"x_haz": sp.stats.poisson(1),
                  }
    haz_unc = UncVar(haz, haz_distr)

    return exp_unc, impf_unc, haz_unc


def ent_dem():
    entity = Entity()
    entity.read_excel(ENT_DEMO_TODAY)
    entity.exposures.ref_year = 2018
    entity.check()
    return entity

def ent_fut_dem():
    entity = Entity()
    entity.read_excel(ENT_DEMO_FUTURE)
    entity.exposures.ref_year = 2040
    entity.check()
    return entity


class TestUncVar(unittest.TestCase):
    """ Test UncVar class """

    def test_init_pass(self):

        impf = impf_dem
        distr_dict = {"x_paa": sp.stats.beta(0.5, 1),
                      "x_mdd": sp.stats.uniform(0.8, 1.2)
                      }
        impf_unc = UncVar(impf, distr_dict)
        self.assertListEqual(impf_unc.labels, ['x_paa', 'x_mdd'])
        self.assertTrue(isinstance(impf_unc.distr_dict, dict))

    def test_evaluate_pass(self):

        impf = impf_dem
        distr_dict = {"x_impf": sp.stats.uniform(0.8, 1.2),
              }
        impf_unc = UncVar(impf, distr_dict)
        impf_eval = impf_unc.uncvar_func(**{'x_paa': 0.8, 'x_mdd': 1.1})
        impf_true = impf_dem(x_paa=0.8, x_mdd=1.1)
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
        distr_dict = {"x_paa": sp.stats.beta(0.5, 1),
                      "x_mdd": sp.stats.uniform(0.8, 1.2)
              }
        impf_unc = UncVar(impf, distr_dict)
        self.assertIsNotNone(impf_unc.plot())
        plt.close()

    def test_vac_to_uncvar(self):

        exp = exp_dem()
        distr_dict = {"x_exp": sp.stats.uniform(0.8,1.2)
              }

        var = UncVar.var_to_uncvar(exp)
        self.assertDictEqual(var.distr_dict, {})
        self.assertTrue(isinstance(var.uncvar_func(), Exposures))

        unc_var = UncVar.var_to_uncvar(UncVar(exp, distr_dict))
        self.assertDictEqual(unc_var.distr_dict, distr_dict)
        self.assertTrue(isinstance(var.uncvar_func(), Exposures))



class TestUncertainty(unittest.TestCase):
    """Test the Uncertainty class"""

    def test_init_pass(self):
        """Test initiliazation uncertainty"""

        exp_unc, impf_unc, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'impf': impf_unc,
                           'haz': haz_unc})
        self.assertDictEqual(unc.metrics, {})
        self.assertDictEqual(unc.sensitivity, {})

        self.assertEqual(unc.n_samples, 0)
        self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_haz',
                                                    'x_paa', 'x_mdd'})
        self.assertSetEqual(set(unc.problem_sa['names']),
                            {'x_exp', 'x_haz', 'x_paa', 'x_mdd'})
        self.assertSetEqual(set(unc.distr_dict.keys()),
                            {"x_exp", "x_paa", "x_mdd", "x_haz"})

        unc = Uncertainty(
            {'exp': exp_unc, 'impf': impf_unc},
            samples = pd.DataFrame({'x_exp': [1, 2], 'x_paa': [3, 4],
                                    'x_mdd': [1, 2]}),
            metrics = {'aai_agg': pd.DataFrame({'aai_agg': [100, 200]})}
            )
        self.assertEqual(unc.n_samples, 2)
        self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_paa', 'x_mdd'})
        self.assertListEqual(list(unc.metrics['aai_agg']['aai_agg']), [100, 200])
        self.assertDictEqual(unc.sensitivity, {})

    def test_save_pass(self):
        """Test save samples"""

        exp_unc, impf_unc, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'impf': impf_unc,
                           'haz': haz_unc})
        unc.make_sample(1)
        filename = unc.save_samples_df()

        unc_imp = UncImpact(exp_unc, impf_unc, haz_unc)
        unc_imp.load_samples_df(filename)

        unc_imp.calc_distribution()
        unc_imp.calc_sensitivity()


    def test_make_sample_pass(self):
        """Test generate sample"""

        exp_unc, _ , haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'haz': haz_unc})

        #default sampling saltelli
        unc.make_sample(N=1, sampling_kwargs = {'calc_second_order': True})
        self.assertEqual(unc.n_samples, 1*(2*2+2)) # N * (2 * D + 2)
        self.assertTrue(isinstance(unc.samples_df, pd.DataFrame))
        self.assertTrue(np.allclose(
            unc.samples_df['x_exp'],
            np.array([1.239453, 1.837109, 1.239453,
                      1.239453, 1.837109, 1.837109]),
            rtol=1e-05
            ))
        self.assertListEqual(list(unc.samples_df['x_haz']),
                             [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

        #latin sampling
        unc.make_sample(N=1, sampling_method='latin',
                        sampling_kwargs = {'seed': 11245})
        self.assertEqual(unc.n_samples, 1)
        self.assertTrue(isinstance(unc.samples_df, pd.DataFrame))
        self.assertTrue(np.allclose(
            unc.samples_df['x_exp'],
            np.array([2.58309]),
            rtol=1e-05
            ))
        self.assertListEqual(list(unc.samples_df['x_haz']), [2.0])

    def test_plot_sample_pass(self):
        """Test plot sample"""

        exp_unc, _, haz_unc = make_imp_uncs()

        unc = Uncertainty({'exp': exp_unc,
                           'haz': haz_unc})

        unc.make_sample(N=1)
        unc.plot_sample()
        plt.close()

    def test_est_comp_time_pass(self):
        """Test estimate computation time"""

        exp_unc, _, haz_unc = make_imp_uncs()

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
        """Test compute sensitivity default"""

        exp_unc, _, haz_unc = make_imp_uncs()
        samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'rp': pd.DataFrame({'rp100': [9, 10, 11, 12],
                                       'rp250': [100, 110, 120, 130]
                                       }
                                      )
                   }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          samples = samples,
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

    def test_calc_sensitivty_XY_pass(self):
        """Test compute sensitvity method rbd_fast (variables names different
        from default)"""

        exp_unc, _, haz_unc = make_imp_uncs()
        samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'rp': pd.DataFrame({'rp100': [9.0, 10.0, 11.0, 12.0],
                                       'rp250': [100.0, 110.0, 120.0, 130.0]
                                       }
                                      )
                   }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          samples = samples,
                          metrics = metrics)

        sens = unc.calc_sensitivity(
            salib_method = 'rbd_fast',
            method_kwargs = {'M': 8}
            )
        self.assertSetEqual(set(sens.keys()), {'rp'})
        self.assertSetEqual(set(sens['rp'].keys()), {'rp100', 'rp250'})
        self.assertSetEqual(set(sens['rp']['rp100'].keys()), {'S1', 'names'})
        self.assertTrue(np.allclose(
                sens['rp']['rp100']['S1'],
                np.array([1.0, 1.0])
                )
            )


    def test_plot_sensitivity(self):
        """Test plot sensitivity indices first oder"""

        exp_unc, _, haz_unc = make_imp_uncs()
        samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'freq_curve': pd.DataFrame(
                {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
            }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          samples = samples,
                          metrics = metrics)

        unc.calc_sensitivity(method_kwargs = {'calc_second_order': False})
        unc.plot_sensitivity()
        plt.close()

        unc.calc_sensitivity(
            salib_method = 'rbd_fast',
            method_kwargs = {'M': 8}
            )
        unc.plot_sensitivity()
        plt.close()

    def plot_sensitivity_second_order(self):
        """Test plot sensitivity indices 2nd order"""

        exp_unc, impf_unc, _ = make_imp_uncs()
        haz = haz_dem()
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)

        unc.calc_distribution()
        unc.calc_sensitivity()
        unc.plot_sensitivity2d()
        plt.close()


    def test_plot_distribution(self):
        """Test plot metrics distribution"""

        exp_unc, _, haz_unc = make_imp_uncs()
        samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
                               'x_haz': [0.1, 0.2, 0.3, 0.4]})
        metrics = {'freq_curve': pd.DataFrame(
                {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
            }

        unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
                          samples = samples,
                          metrics = metrics)

        unc.plot_distribution()
        plt.close()



class TestUncImpact(unittest.TestCase):
    """Test the UncImpact class"""

    def test_init_pass(self):
        """Test impact initialization"""

        exp_unc, impf_unc, _ = make_imp_uncs()
        haz = haz_dem()

        unc = UncImpact(exp_unc, impf_unc, haz)

        self.assertSetEqual(set(unc.metrics.keys()),
             {'aai_agg', 'freq_curve', 'eai_exp', 'at_event'}
             )
        self.assertSetEqual(set(unc.unc_vars.keys()), {'exp', 'impf', 'haz'})

    def test_calc_distribution_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_imp_uncs()
        haz = haz_dem()
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)

        unc.calc_distribution(calc_eai_exp=False, calc_at_event=False)

        self.assertListEqual(unc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc.calc_eai_exp, False)
        self.assertEqual(unc.calc_at_event, False)

        self.assertTrue(
            np.allclose(
                unc.metrics['aai_agg'].aai_agg,
                np.array([9.600984e+07, 1.668144e+08, 8.068803e+08,
                          1.274945e+08, 1.071482e+09, 2.215182e+08,
                          1.401932e+09, 1.861671e+09])
                )
            )
        self.assertTrue(
            np.allclose(
                unc.metrics['freq_curve'].rp5,
                np.zeros(8)
                )
            )
        self.assertTrue(
            np.allclose(
                unc.metrics['freq_curve'].rp250,
                np.array([2.880990e+09, 5.005640e+09, 2.421225e+10,
                          3.825758e+09, 3.215222e+10, 6.647149e+09,
                          4.206811e+10, 5.586359e+10])
                )
            )
        self.assertTrue(unc.metrics['eai_exp'].empty)
        self.assertTrue(unc.metrics['at_event'].empty)

    def test_plot_distribution_pass(self):
        """Test plot the distribution (uncertainty) of the impact metrics)"""

        exp_unc, impf_unc, _ = make_imp_uncs()
        haz = haz_dem()
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)
        unc.calc_distribution()
        unc.plot_rp_distribution()
        plt.close()

    def test_plot_sensitivity_map_pass(self):
        """Test plot the map of the largest sensitivity index for eai_exp"""

        exp_unc, impf_unc, _ = make_imp_uncs()
        haz = haz_dem()

        #Default parameters
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1)
        unc.calc_distribution(calc_eai_exp=True)
        unc.calc_sensitivity()
        unc.plot_sensitivity_map(exp_unc.uncvar_func(x_exp= 1))
        plt.close()

        #Non-default parameters
        unc = UncImpact(exp_unc, impf_unc, haz)
        unc.make_sample(N=1, sampling_method='morris')
        unc.calc_distribution(calc_eai_exp=True)
        unc.calc_sensitivity(salib_method='morris')
        unc.plot_sensitivity_map(exp_unc.uncvar_func(x_exp= 1), salib_si='mu')
        plt.close()


class TestUncCostBenefit(unittest.TestCase):
    """Test the UncCostBenefit class"""

    def test_init_pass(self):
        """Test cost benefit initialization"""

        haz = haz_dem()
        ent = ent_dem()
        ent_fut = ent_fut_dem()

        unc = UncCostBenefit(haz_unc=haz, ent_unc=ent,
                             haz_fut_unc=haz, ent_fut_unc=ent_fut)

        self.assertSetEqual(set(unc.metrics.keys()),
             {'tot_climate_risk', 'benefit', 'cost_ben_ratio',
              'imp_meas_present', 'imp_meas_future'}
             )
        self.assertSetEqual(set(unc.unc_vars.keys()), {'haz', 'ent',
                                                       'haz_fut', 'ent_fut'})

    def test_calc_distribution_pass(self):
        """Test plot the distribution (uncertainty) of the impact metrics)"""

        haz_fut = haz_dem
        haz_distr = {"x_haz": sp.stats.uniform(1, 3),
                      }
        haz_fut_unc = UncVar(haz_fut, haz_distr)
        haz = haz_dem(x_haz=10)

        ent = ent_dem()
        ent_fut = ent_fut_dem()

        unc = UncCostBenefit(haz_unc=haz, ent_unc=ent,
                             haz_fut_unc=haz_fut_unc, ent_fut_unc=ent_fut)

        unc.make_sample(N=1)
        unc.calc_distribution()

        print(unc.metrics)

        self.assertTrue(
            np.allclose(unc.metrics['tot_climate_risk']['tot_climate_risk'],
                np.array([1.494806e+11, 1.262690e+11,
                          1.494806e+11, 1.262690e+11])
                )
            )
        self.assertTrue(
            np.allclose(unc.metrics['benefit']['Mangroves'],
                np.array([1.004445e+10, 2.403430e+09,
                          1.004445e+10, 2.403430e+09])
                )
            )
        self.assertTrue(
            np.allclose(unc.metrics['cost_ben_ratio']['Beach nourishment'],
                np.array([0.210822, 0.889260,
                          0.210822, 0.889260])
                )
            )

        self.assertTrue(
            np.allclose(unc.metrics['imp_meas_present']['Seawall-risk'],
                np.array([1.164395e+10, 1.164395e+10,
                          1.164395e+10, 1.164395e+10])
                )
            )

        self.assertTrue(
            np.allclose(unc.metrics['imp_meas_future']['Building code-risk'],
                np.array([2.556347e+09, 5.303379e+08,
                          2.556347e+09, 5.303379e+08])
                )
            )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUncVar)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertainty))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncImpact))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncCostBenefit))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
