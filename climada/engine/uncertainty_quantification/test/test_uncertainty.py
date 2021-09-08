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
import copy

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
from climada.engine.uncertainty_quantification import InputVar, CalcImpact, UncOutput, CalcCostBenefit


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

exp = Exposures()
exp.read_hdf5(EXP_DEMO_H5)
def exp_dem(x_exp=1, exp=exp):
    exp_tmp = exp.copy(deep=True)
    exp_tmp.gdf.value *= x_exp
    return exp_tmp

haz= Hazard()
haz.read_hdf5(HAZ_DEMO_H5)
def haz_dem(x_haz=1, haz=haz):
    haz_tmp = copy.deepcopy(haz)
    haz_tmp.intensity = haz_tmp.intensity.multiply(x_haz)
    return haz_tmp

def make_input_vars():

    exp = exp_dem
    exp_distr = {"x_exp": sp.stats.uniform(0.8,2),
                  }
    exp_unc = InputVar(exp, exp_distr)

    impf = impf_dem
    impf_distr = {"x_paa": sp.stats.beta(0.5, 1),
                  "x_mdd": sp.stats.uniform(0.8, 1.2)
              }
    impf_unc = InputVar(impf, impf_distr)

    haz = haz_dem
    haz_distr = {"x_haz": sp.stats.poisson(1),
                  }
    haz_unc = InputVar(haz, haz_distr)

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


class TestInputVar(unittest.TestCase):
    """ Test UncVar class """

    def test_init_pass(self):

        impf = impf_dem
        distr_dict = {"x_paa": sp.stats.beta(0.5, 1),
                      "x_mdd": sp.stats.uniform(0.8, 1.2)
                      }
        impf_iv = InputVar(impf, distr_dict)
        self.assertListEqual(impf_iv.labels, ['x_paa', 'x_mdd'])
        self.assertTrue(isinstance(impf_iv.distr_dict, dict))

    def test_evaluate_pass(self):

        impf = impf_dem
        distr_dict = {"x_paa": sp.stats.beta(0.5, 1),
                      "x_mdd": sp.stats.uniform(0.8, 0.4)
                      }
        impf_iv = InputVar(impf, distr_dict)

        #Direct function evaluate
        impf_eval = impf_iv.func(**{'x_paa': 0.8, 'x_mdd': 1.1})
        impf_true = impf_dem(x_paa=0.8, x_mdd=1.1)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()['TC'][1]
        impf_func2 = impf_true.get_func()['TC'][1]
        np.testing.assert_array_equal(
            impf_func1.intensity,
            impf_func2.intensity
            )
        np.testing.assert_array_equal(
            impf_func1.mdd,
            impf_func2.mdd
            )
        np.testing.assert_array_equal(
            impf_func1.paa,
            impf_func2.paa
            )
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

        #Specific evaluate
        impf_eval = impf_iv.evaluate(x_paa=0.8, x_mdd=1.1)
        impf_true = impf_dem(x_paa=0.8, x_mdd=1.1)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()['TC'][1]
        impf_func2 = impf_true.get_func()['TC'][1]
        np.testing.assert_array_equal(
            impf_func1.intensity,
            impf_func2.intensity
            )
        np.testing.assert_array_equal(
            impf_func1.mdd,
            impf_func2.mdd
            )
        np.testing.assert_array_equal(
            impf_func1.paa,
            impf_func2.paa
            )
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

        #Average evaluate (default)
        impf_eval = impf_iv.evaluate()
        impf_true = impf_dem(x_paa=0.3333333333333333, x_mdd=1.0)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()['TC'][1]
        impf_func2 = impf_true.get_func()['TC'][1]
        np.testing.assert_array_almost_equal(
            impf_func1.intensity,
            impf_func2.intensity
            )
        np.testing.assert_array_almost_equal(
            impf_func1.mdd,
            impf_func2.mdd
            )
        np.testing.assert_array_almost_equal(
            impf_func1.paa,
            impf_func2.paa
            )
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

    def test_plot_pass(self):
        impf = impf_dem()
        distr_dict = {"x_paa": sp.stats.beta(0.5, 1),
                      "x_mdd": sp.stats.uniform(0.8, 1.2),
                      "x_lit": sp.stats.randint(0, 10)
              }
        impf_iv = InputVar(impf, distr_dict)
        self.assertIsNotNone(impf_iv.plot())
        plt.close()

    def test_var_to_inputvar(self):

        exp = exp_dem()
        distr_dict = {"x_exp": sp.stats.uniform(0.8,1.2)
              }

        var = InputVar.var_to_inputvar(exp)
        self.assertDictEqual(var.distr_dict, {})
        self.assertTrue(isinstance(var.func(), Exposures))

        iv_var = InputVar.var_to_inputvar(InputVar(exp, distr_dict))
        self.assertDictEqual(iv_var.distr_dict, distr_dict)
        self.assertTrue(isinstance(iv_var, InputVar))

class TestOutput(unittest.TestCase):
    """Test the output class"""

    def test_init_pass(self):
        unc_out = UncOutput(pd.DataFrame())
        self.assertTrue(unc_out.samples_df.empty)

    def test_save_load_pass(self):
        """Test save and load output data"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)

        unc_data_save = unc_calc.make_sample(N=2, sampling_kwargs={'calc_second_order': True})
        filename = unc_data_save.to_hdf5()
        unc_data_load = UncOutput.from_hdf5(filename)
        for attr_save, val_save in unc_data_save.__dict__.items():
            if isinstance(val_save, pd.DataFrame):
                df_load = getattr(unc_data_load, attr_save)
                self.assertTrue(df_load.equals(val_save))
        self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
        self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
        filename.unlink()

        unc_data_save = unc_calc.uncertainty(unc_data_save, calc_eai_exp=True,
                                  calc_at_event=False)
        filename = unc_data_save.to_hdf5()
        unc_data_load = UncOutput.from_hdf5(filename)
        for attr_save, val_save in unc_data_save.__dict__.items():
            if isinstance(val_save, pd.DataFrame):
                df_load = getattr(unc_data_load, attr_save)
                self.assertTrue(df_load.equals(val_save))
        self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
        self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
        filename.unlink()

        unc_data_save = unc_calc.sensitivity(
            unc_data_save,
            sensitivity_kwargs = {'calc_second_order': True}
            )
        filename = unc_data_save.to_hdf5()
        unc_data_load = UncOutput.from_hdf5(filename)
        for attr_save, val_save in unc_data_save.__dict__.items():
            if isinstance(val_save, pd.DataFrame):
                df_load = getattr(unc_data_load, attr_save)
                self.assertTrue(df_load.equals(val_save))
        self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
        self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
        self.assertEqual(unc_data_load.sensitivity_method, unc_data_save.sensitivity_method)
        self.assertEqual(unc_data_load.sensitivity_kwargs, unc_data_save.sensitivity_kwargs)
        filename.unlink()

class TestCalcImpact(unittest.TestCase):
    """Test the calcluate impact uncertainty class"""

    def test_init_pass(self):
        """Test initiliazation uncertainty"""

        exp_iv, impf_iv, haz_iv = make_input_vars()
        unc_calc = CalcImpact(exp_iv, impf_iv, haz_iv)

        self.assertTupleEqual(
            unc_calc.input_var_names,
            ('exp_input_var', 'impf_input_var', 'haz_input_var')
            )
        self.assertTupleEqual(
            unc_calc.metric_names,
            ('aai_agg', 'freq_curve', 'at_event', 'eai_exp', 'tot_value')
            )
        self.assertEqual(unc_calc.value_unit, exp_iv.evaluate().value_unit)
        self.assertTrue(
            unc_calc.exp_input_var.evaluate(x_exp=1).gdf.equals(
                exp_dem(1).gdf)
            )
        impf1 = unc_calc.impf_input_var.evaluate(x_paa=1, x_mdd=1).get_func()['TC'][1]
        impf2 = impf_dem(1, 1).get_func()['TC'][1]
        np.testing.assert_array_almost_equal(
            impf1.calc_mdr(impf1.intensity),
            impf2.calc_mdr(impf2.intensity)
            )
        haz1 = unc_calc.haz_input_var.evaluate(x_haz=1)
        haz2 = haz_dem(1)
        self.assertListEqual(
            haz1.event_name, haz2.event_name
            )

    def test_make_sample_pass(self):
        """Test generate sample"""

        exp_unc, _ , haz_unc = make_input_vars()
        impf = impf_dem()

        unc_calc = CalcImpact(exp_unc, impf, haz_unc)

        #default sampling saltelli
        unc_data = unc_calc.make_sample(N=2, sampling_kwargs = {'calc_second_order': True})
        self.assertEqual(unc_data.n_samples, 2*(2*2+2)) # N * (2 * D + 2)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values,
            np.array(['x_exp', 'x_haz'])
            )

        # #latin sampling
        unc_data = unc_calc.make_sample(N=1, sampling_method='latin',
                        sampling_kwargs = {'seed': 11245})
        self.assertEqual(unc_data.n_samples, 1)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values,
            np.array(['x_exp', 'x_haz'])
            )


    def test_calc_uncertainty_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample( N=2)
        unc_data = unc_calc.uncertainty(unc_data, calc_eai_exp=False, calc_at_event=False)

        self.assertEqual(unc_data.unit, exp_dem().value_unit)
        self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc_calc.calc_eai_exp, False)
        self.assertEqual(unc_calc.calc_at_event, False)

        self.assertEqual(
            unc_data.aai_agg_unc_df.size,
            unc_data.n_samples
            )
        self.assertEqual(
            unc_data.tot_value_unc_df.size,
            unc_data.n_samples
            )

        self.assertEqual(
            unc_data.freq_curve_unc_df.size,
            unc_data.n_samples * len(unc_calc.rp)
            )
        self.assertTrue(unc_data.eai_exp_unc_df.empty)
        self.assertTrue(unc_data.at_event_unc_df.empty)

    def test_calc_uncertainty_pool_pass(self):
        """Test parallel compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample(N=2)

        pool = Pool(nodes=2)
        unc_data = unc_calc.uncertainty(unc_data, calc_eai_exp=False,
                             calc_at_event=False, pool=pool)
        pool.close()
        pool.join()
        pool.clear()

        self.assertEqual(unc_data.unit, exp_dem().value_unit)
        self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc_calc.calc_eai_exp, False)
        self.assertEqual(unc_calc.calc_at_event, False)

        self.assertEqual(
            unc_data.aai_agg_unc_df.size,
            unc_data.n_samples
            )
        self.assertEqual(
            unc_data.tot_value_unc_df.size,
            unc_data.n_samples
            )

        self.assertEqual(
            unc_data.freq_curve_unc_df.size,
            unc_data.n_samples * len(unc_calc.rp)
            )
        self.assertTrue(unc_data.eai_exp_unc_df.empty)
        self.assertTrue(unc_data.at_event_unc_df.empty)

    def test_calc_sensitivity_pass(self):
        """Test compute sensitivity default"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample(N=4, sampling_kwargs={'calc_second_order': True})
        unc_data = unc_calc.uncertainty(unc_data, calc_eai_exp=False,
                                  calc_at_event=False)

        unc_data = unc_calc.sensitivity(
            unc_data,
            sensitivity_kwargs = {'calc_second_order': True}
            )

        self.assertEqual(unc_data.sensitivity_method, 'sobol')
        self.assertTupleEqual(unc_data.sensitivity_kwargs,
                             tuple({'calc_second_order': 'True'}.items())
                             )

        for name, attr in unc_data.__dict__.items():
            if 'sens_df' in name:
                if 'eai' in name:
                    self.assertTrue(attr.empty)
                elif 'at_event' in name:
                    self.assertTrue(attr.empty)
                else:
                    np.testing.assert_array_equal(
                        attr.param.unique(),
                        np.array(['x_exp', 'x_paa', 'x_mdd'])
                        )

                    np.testing.assert_array_equal(
                        attr.si.unique(),
                        np.array(['S1', 'S1_conf', 'ST', 'ST_conf', 'S2', 'S2_conf'])
                        )

                    self.assertEqual(len(attr),
                                     len(unc_data.param_labels) * (4 + 3 + 3)
                                     )

    def test_calc_sensitivity_morris_pass(self):
        """Test compute sensitivity default"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample(N=4,
                             sampling_method='latin')
        unc_data = unc_calc.uncertainty(unc_data, calc_eai_exp=True,
                                  calc_at_event=True)

        unc_data = unc_calc.sensitivity(
            unc_data,
            sensitivity_method = 'morris'
            )

        self.assertEqual(unc_data.sensitivity_method, 'morris')
        self.assertTupleEqual(unc_data.sensitivity_kwargs,
                             tuple({}.items())
                             )

        for name, attr in unc_data.__dict__.items():
            if 'sens_df' in name:
                np.testing.assert_array_equal(
                    attr.param.unique(),
                    np.array(['x_exp', 'x_paa', 'x_mdd'])
                    )
                np.testing.assert_array_equal(
                    attr.si.unique(),
                    np.array(['mu', 'mu_star', 'sigma', 'mu_star_conf'])
                    )
                if 'eai' in name:
                    self.assertEqual(
                        attr.size,
                        len(unc_data.param_labels)*4*(len(exp_unc.evaluate().gdf) + 3)
                        )
                elif 'at_event' in name:
                    self.assertEqual(
                        attr.size,
                        len(unc_data.param_labels) * 4 * (haz.size + 3)
                        )
                else:
                    self.assertEqual(len(attr),
                                     len(unc_data.param_labels) * 4
                                     )


    # def test_calc_cb(self):
    #     haz_fut = haz_dem
    #     haz_distr = {"x_haz": sp.stats.uniform(1, 3),
    #                   }
    #     haz_fut_unc = InputVar(haz_fut, haz_distr)
    #     haz = haz_dem(x_haz=10)

    #     ent = ent_dem()
    #     ent_fut = ent_fut_dem()

    #     unc_data = UncOutput()
    #     unc = CalcCostBenefit(haz_input_var=haz, ent_input_var=ent,
    #                           haz_fut_input_var=haz_fut_unc, ent_fut_input_var=ent_fut)
    #     unc.make_sample(unc_data, N=1)
    #     unc.uncertainty(unc_data)
    #     self.assertEqual(unc_data.unit, exp_dem().value_unit)
    #     unc_data.plot_uncertainty()
    #     plt.close()

# class TestCalcCostBenefit(unittest.TestCase):
#     """Test the calcluate cost benefit uncertainty class"""

#     def test_init_pass(self):
#         """Test initiliazation uncertainty"""

#         exp_iv, impf_iv, haz_iv = make_input_vars()
#         unc_calc = CalcImpact(exp_iv, impf_iv, haz_iv)

#         self.assertTupleEqual(
#             unc_calc.input_var_names,
#             ('exp_input_var', 'impf_input_var', 'haz_input_var')
#             )
#         self.assertTupleEqual(
#             unc_calc.metric_names,
#             ('aai_agg', 'freq_curve', 'at_event', 'eai_exp', 'tot_value')
#             )
#         self.assertEqual(unc_calc.value_unit, exp_iv.evaluate().value_unit)
#         self.assertTrue(
#             unc_calc.exp_input_var.evaluate(x_exp=1).gdf.equals(
#                 exp_dem(1).gdf)
#             )
#         impf1 = unc_calc.impf_input_var.evaluate(x_paa=1, x_mdd=1).get_func()['TC'][1]
#         impf2 = impf_dem(1, 1).get_func()['TC'][1]
#         np.testing.assert_array_almost_equal(
#             impf1.calc_mdr(impf1.intensity),
#             impf2.calc_mdr(impf2.intensity)
#             )
#         haz1 = unc_calc.haz_input_var.evaluate(x_haz=1)
#         haz2 = haz_dem(1)
#         self.assertListEqual(
#             haz1.event_name, haz2.event_name
#             )

# class TestCalcImpact(unittest.TestCase):
#     """Test the calcluate impact uncertainty class"""

#     def test_init_pass(self):
#         """Test initiliazation uncertainty"""

#         unc_data = UncOutput()

#         exp_iv, impf_iv, haz_iv = make_input_vars()
#         unc_calc = CalcImpact(unc_data, exp_iv, impf_iv, haz_iv)

#         self.assertDictEqual(unc_data.metrics, {})
#         self.assertDictEqual(unc.sensitivity, {})

#         self.assertEqual(unc_data.n_samples, 0)
#         self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_haz',
#                                                     'x_paa', 'x_mdd'})
#         self.assertSetEqual(set(unc.problem_sa['names']),
#                             {'x_exp', 'x_haz', 'x_paa', 'x_mdd'})
#         self.assertSetEqual(set(unc.distr_dict.keys()),
#                             {"x_exp", "x_paa", "x_mdd", "x_haz"})

#         unc = Uncertainty(
#             {'exp': exp_unc, 'impf': impf_unc},
#             samples = pd.DataFrame({'x_exp': [1, 2], 'x_paa': [3, 4],
#                                     'x_mdd': [1, 2]}),
#             metrics = {'aai_agg': pd.DataFrame({'aai_agg': [100, 200]})}
#             )
#         self.assertEqual(unc.n_samples, 2)
#         self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_paa', 'x_mdd'})
#         self.assertListEqual(list(unc.metrics['aai_agg']['aai_agg']), [100, 200])
#         self.assertDictEqual(unc.sensitivity, {})

#     def test_make_sample_pass(self):
#         """Test generate sample"""

#         exp_unc, _ , haz_unc = make_imp_uncs()
#         impf = impf_dem()

#         unc_data = UncOutput()
#         unc_calc = CalcImpact(exp_unc, impf, haz_unc)

#         #default sampling saltelli
#         unc_calc.make_sample(unc_data, N=2, sampling_kwargs = {'calc_second_order': True})
#         self.assertEqual(unc_data.n_samples, 2*(2*2+2)) # N * (2 * D + 2)
#         self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
#         # self.assertTrue(np.allclose(
#         #     unc_data.samples_df['x_exp'],
#         #     np.array([1.239453, 1.837109, 1.239453,
#         #               1.239453, 1.837109, 1.837109]),
#         #     rtol=1e-05
#         #     ))
#         # self.assertListEqual(list(unc_data.samples_df['x_haz']),
#         #                       [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

#         # #latin sampling
#         unc_data = UncOutput()
#         unc_calc.make_sample(unc_data, N=1, sampling_method='latin',
#                         sampling_kwargs = {'seed': 11245})
#         self.assertEqual(unc_data.n_samples, 1)
#         self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
#         self.assertTrue(np.allclose(
#             unc_data.samples_df['x_exp'],
#             np.array([2.58309]),
#             rtol=1e-05
#             ))
#         self.assertListEqual(list(unc_data.samples_df['x_haz']), [2.0])


#     def test_calc_uncertainty_pass(self):
#         """Test compute the uncertainty distribution for an impact"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc_data = UncOutput()
#         unc_calc = CalcImpact(exp_unc, impf_unc, haz)
#         unc_calc.make_sample(unc_data, N=2)
#         unc_calc.uncertainty(unc_data, calc_eai_exp=False, calc_at_event=False)

#         self.assertEqual(unc_data.unit, exp_dem().value_unit)
#         self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
#         self.assertEqual(unc_calc.calc_eai_exp, False)
#         self.assertEqual(unc_calc.calc_at_event, False)

#         # self.assertTrue(
#         #     np.allclose(
#         #         unc_data.aai_agg_unc_df.aai_agg,
#         #         np.array([9.600984e+07, 1.668144e+08, 8.068803e+08,
#         #                   1.274945e+08, 1.071482e+09, 2.215182e+08,
#         #                   1.401932e+09, 1.861671e+09])
#         #         )
#         #     )
#         # self.assertTrue(
#         #     np.allclose(
#         #         unc_data.freq_curve_unc_df.rp5,
#         #         np.zeros(8)
#         #         )
#         #     )
#         # self.assertTrue(
#         #     np.allclose(
#         #         unc_data.freq_curve_unc_df.rp250,
#         #         np.array([2.880990e+09, 5.005640e+09, 2.421225e+10,
#         #                   3.825758e+09, 3.215222e+10, 6.647149e+09,
#         #                   4.206811e+10, 5.586359e+10])
#         #         )
#         #     )
#         self.assertTrue(unc_data.eai_exp_unc_df.empty)
#         self.assertTrue(unc_data.at_event_unc_df.empty)

#     def test_calc_sensitivity_pass(self):
#         """Test compute sensitivity default"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc_data = UncOutput()
#         unc_calc = CalcImpact(exp_unc, impf_unc, haz)
#         unc_calc.make_sample(unc_data, N=4, sampling_kwargs={'calc_second_order': True})
#         unc_data.plot_sample()
#         unc_calc.uncertainty(unc_data, calc_eai_exp=True,
#                                   calc_at_event=False)

#         unc_data.plot_uncertainty()
#         unc_data.plot_rp_uncertainty()
#         plt.close()

#         unc_calc.sensitivity(
#             unc_data,
#             sensitivity_kwargs = {'calc_second_order': True}
#             )
#         unc_data.plot_sensitivity()
#         unc_data.plot_sensitivity_second_order()
#         unc_data.plot_sensitivity_map(exp=exp_unc.func(1))
#         plt.close()

#     def test_calc_sensitivity_morris_pass(self):
#         """Test compute sensitivity default"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc_data = UncOutput()
#         unc_calc = CalcImpact(exp_unc, impf_unc, haz)
#         unc_calc.make_sample(unc_data, N=4, sampling_method='morris')
#         unc_calc.uncertainty(unc_data)

#         unc_calc.sensitivity(
#             unc_data,
#             sensitivity_method='morris'
#             )

#         unc_data.plot_sensitivity(salib_si='mu')
#         # self.assertSetEqual(set(sens.keys()), {'rp'})
#         # self.assertSetEqual(set(sens['rp'].keys()), {'rp100', 'rp250'})
#         # self.assertSetEqual(set(sens['rp']['rp100'].keys()), {'S1', 'S1_conf',
#         #                                                       'ST', 'ST_conf'})
#         # self.assertTrue(np.allclose(
#         #         sens['rp']['rp100']['S1'],
#         #         np.array([0.66666667, 1.33333333])
#         #         )
#         #     )
#     def test_calc_cb(self):
#         haz_fut = haz_dem
#         haz_distr = {"x_haz": sp.stats.uniform(1, 3),
#                       }
#         haz_fut_unc = InputVar(haz_fut, haz_distr)
#         haz = haz_dem(x_haz=10)

#         ent = ent_dem()
#         ent_fut = ent_fut_dem()

#         unc_data = UncOutput()
#         unc = CalcCostBenefit(haz_input_var=haz, ent_input_var=ent,
#                               haz_fut_input_var=haz_fut_unc, ent_fut_input_var=ent_fut)
#         unc.make_sample(unc_data, N=1)
#         unc.uncertainty(unc_data)
#         self.assertEqual(unc_data.unit, exp_dem().value_unit)
#         unc_data.plot_uncertainty()
#         plt.close()

#     def test_save_pass(self):
#         """Test save samples"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc_data_save = UncOutput()
#         unc_calc = CalcImpact(exp_unc, impf_unc, haz)
#         unc_calc.make_sample(unc_data_save , N=2, sampling_kwargs={'calc_second_order': True})
#         unc_calc.uncertainty(unc_data_save, calc_eai_exp=True,
#                                   calc_at_event=False)
#         unc_calc.sensitivity(
#             unc_data_save,
#             sensitivity_kwargs = {'calc_second_order': True}
#             )

#         filename = unc_data_save.save_hdf5()

#         unc_data_load = UncOutput.from_hdf5(filename)

#         for attr_save, val_save in unc_data_save.__dict__.items():
#             if isinstance(val_save, pd.DataFrame):
#                 df_load = getattr(unc_data_load, attr_save)
#                 self.assertTrue(df_load.equals(val_save))
#         self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
#         self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
#         self.assertEqual(unc_data_load.sensitivity_method, unc_data_save.sensitivity_method)
#         self.assertEqual(unc_data_load.sensitivity_kwargs, unc_data_save.sensitivity_kwargs)

# class TestUncertainty(unittest.TestCase):
#     """Test the Uncertainty class"""

#     def test_init_pass(self):
#         """Test initiliazation uncertainty"""

#         exp_unc, impf_unc, haz_unc = make_imp_uncs()

#         unc = Uncertainty({'exp': exp_unc,
#                            'impf': impf_unc,
#                            'haz': haz_unc})
#         self.assertDictEqual(unc.metrics, {})
#         self.assertDictEqual(unc.sensitivity, {})

#         self.assertEqual(unc.n_samples, 0)
#         self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_haz',
#                                                     'x_paa', 'x_mdd'})
#         self.assertSetEqual(set(unc.problem_sa['names']),
#                             {'x_exp', 'x_haz', 'x_paa', 'x_mdd'})
#         self.assertSetEqual(set(unc.distr_dict.keys()),
#                             {"x_exp", "x_paa", "x_mdd", "x_haz"})

#         unc = Uncertainty(
#             {'exp': exp_unc, 'impf': impf_unc},
#             samples = pd.DataFrame({'x_exp': [1, 2], 'x_paa': [3, 4],
#                                     'x_mdd': [1, 2]}),
#             metrics = {'aai_agg': pd.DataFrame({'aai_agg': [100, 200]})}
#             )
#         self.assertEqual(unc.n_samples, 2)
#         self.assertSetEqual(set(unc.param_labels), {'x_exp', 'x_paa', 'x_mdd'})
#         self.assertListEqual(list(unc.metrics['aai_agg']['aai_agg']), [100, 200])
#         self.assertDictEqual(unc.sensitivity, {})

#     def test_save_pass(self):
#         """Test save samples"""

#         exp_unc, impf_unc, haz_unc = make_imp_uncs()

#         unc = Uncertainty({'exp': exp_unc,
#                            'impf': impf_unc,
#                            'haz': haz_unc})
#         unc.make_sample(1)
#         filename = unc.save_samples_df()

#         unc_imp = UncImpact(exp_unc, impf_unc, haz_unc)
#         unc_imp.load_samples_df(filename)

#         unc_imp.calc_distribution()
#         unc_imp.calc_sensitivity()


#     def test_make_sample_pass(self):
#         """Test generate sample"""

#         exp_unc, _ , haz_unc = make_imp_uncs()

#         unc = Uncertainty({'exp': exp_unc,
#                            'haz': haz_unc})

#         #default sampling saltelli
#         unc.make_sample(N=1, sampling_kwargs = {'calc_second_order': True})
#         self.assertEqual(unc.n_samples, 1*(2*2+2)) # N * (2 * D + 2)
#         self.assertTrue(isinstance(unc.samples_df, pd.DataFrame))
#         self.assertTrue(np.allclose(
#             unc.samples_df['x_exp'],
#             np.array([1.239453, 1.837109, 1.239453,
#                       1.239453, 1.837109, 1.837109]),
#             rtol=1e-05
#             ))
#         self.assertListEqual(list(unc.samples_df['x_haz']),
#                              [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

#         #latin sampling
#         unc.make_sample(N=1, sampling_method='latin',
#                         sampling_kwargs = {'seed': 11245})
#         self.assertEqual(unc.n_samples, 1)
#         self.assertTrue(isinstance(unc.samples_df, pd.DataFrame))
#         self.assertTrue(np.allclose(
#             unc.samples_df['x_exp'],
#             np.array([2.58309]),
#             rtol=1e-05
#             ))
#         self.assertListEqual(list(unc.samples_df['x_haz']), [2.0])

#     def test_plot_sample_pass(self):
#         """Test plot sample"""

#         exp_unc, _, haz_unc = make_imp_uncs()

#         unc = Uncertainty({'exp': exp_unc,
#                            'haz': haz_unc})

#         unc.make_sample(N=1)
#         unc.plot_sample()
#         plt.close()

#     def test_est_comp_time_pass(self):
#         """Test estimate computation time"""

#         exp_unc, _, haz_unc = make_imp_uncs()

#         unc = Uncertainty({'exp': exp_unc,
#                            'haz': haz_unc})

#         unc.make_sample(N=1, sampling_kwargs = {'calc_second_order': False})
#         est = unc.est_comp_time(0.12345)
#         self.assertEqual(est, 1*(2+2) *  0.123) # N * (D + 2)

#         pool = Pool(nodes=4)
#         est = unc.est_comp_time(0.12345, pool)
#         self.assertEqual(est, 1*(2+2) *  0.123 / 4) # N * (D + 2)
#         pool.close()
#         pool.join()
#         pool.clear()

#     def test_calc_sensitivty_pass(self):
#         """Test compute sensitivity default"""

#         exp_unc, _, haz_unc = make_imp_uncs()
#         samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
#                                'x_haz': [0.1, 0.2, 0.3, 0.4]})
#         metrics = {'rp': pd.DataFrame({'rp100': [9, 10, 11, 12],
#                                        'rp250': [100, 110, 120, 130]
#                                        }
#                                       )
#                    }

#         unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
#                           samples = samples,
#                           metrics = metrics)

#         sens = unc.calc_sensitivity(
#             method_kwargs = {'calc_second_order': False}
#             )
#         self.assertSetEqual(set(sens.keys()), {'rp'})
#         self.assertSetEqual(set(sens['rp'].keys()), {'rp100', 'rp250'})
#         self.assertSetEqual(set(sens['rp']['rp100'].keys()), {'S1', 'S1_conf',
#                                                               'ST', 'ST_conf'})
#         self.assertTrue(np.allclose(
#                 sens['rp']['rp100']['S1'],
#                 np.array([0.66666667, 1.33333333])
#                 )
#             )

#     def test_calc_sensitivty_XY_pass(self):
#         """Test compute sensitvity method rbd_fast (variables names different
#         from default)"""

#         exp_unc, _, haz_unc = make_imp_uncs()
#         samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
#                                'x_haz': [0.1, 0.2, 0.3, 0.4]})
#         metrics = {'rp': pd.DataFrame({'rp100': [9.0, 10.0, 11.0, 12.0],
#                                        'rp250': [100.0, 110.0, 120.0, 130.0]
#                                        }
#                                       )
#                    }

#         unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
#                           samples = samples,
#                           metrics = metrics)

#         sens = unc.calc_sensitivity(
#             salib_method = 'rbd_fast',
#             method_kwargs = {'M': 8}
#             )
#         self.assertSetEqual(set(sens.keys()), {'rp'})
#         self.assertSetEqual(set(sens['rp'].keys()), {'rp100', 'rp250'})
#         self.assertSetEqual(set(sens['rp']['rp100'].keys()), {'S1', 'names'})
#         self.assertTrue(np.allclose(
#                 sens['rp']['rp100']['S1'],
#                 np.array([1.0, 1.0])
#                 )
#             )


#     def test_plot_sensitivity(self):
#         """Test plot sensitivity indices first oder"""

#         exp_unc, _, haz_unc = make_imp_uncs()
#         samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
#                                'x_haz': [0.1, 0.2, 0.3, 0.4]})
#         metrics = {'freq_curve': pd.DataFrame(
#                 {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
#             }

#         unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
#                           samples = samples,
#                           metrics = metrics)

#         unc.calc_sensitivity(method_kwargs = {'calc_second_order': False})
#         unc.plot_sensitivity()
#         plt.close()

#         unc.calc_sensitivity(
#             salib_method = 'rbd_fast',
#             method_kwargs = {'M': 8}
#             )
#         unc.plot_sensitivity()
#         plt.close()

#     def plot_sensitivity_second_order(self):
#         """Test plot sensitivity indices 2nd order"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc = UncImpact(exp_unc, impf_unc, haz)
#         unc.make_sample(N=1)

#         unc.calc_distribution()
#         unc.calc_sensitivity()
#         unc.plot_sensitivity2d()
#         plt.close()


#     def test_plot_distribution(self):
#         """Test plot metrics distribution"""

#         exp_unc, _, haz_unc = make_imp_uncs()
#         samples = pd.DataFrame({'x_exp': [1, 2, 3, 4],
#                                'x_haz': [0.1, 0.2, 0.3, 0.4]})
#         metrics = {'freq_curve': pd.DataFrame(
#                 {'rp100': [9, 10, 11, 12], 'rp250': [100, 110, 120, 130]})
#             }

#         unc = Uncertainty(unc_vars = {'exp': exp_unc, 'haz': haz_unc},
#                           samples = samples,
#                           metrics = metrics)

#         unc.plot_distribution()
#         plt.close()



# class TestUncImpact(unittest.TestCase):
#     """Test the UncImpact class"""

#     def test_init_pass(self):
#         """Test impact initialization"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()

#         unc = UncImpact(exp_unc, impf_unc, haz)

#         self.assertSetEqual(set(unc.metrics.keys()),
#              {'aai_agg', 'freq_curve', 'eai_exp', 'at_event'}
#              )
#         self.assertSetEqual(set(unc.unc_vars.keys()), {'exp', 'impf', 'haz'})

#     def test_calc_distribution_pass(self):
#         """Test compute the uncertainty distribution for an impact"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc = UncImpact(exp_unc, impf_unc, haz)
#         unc.make_sample(N=1)

#         unc.calc_distribution(calc_eai_exp=False, calc_at_event=False)

#         self.assertListEqual(unc.rp, [5, 10, 20, 50, 100, 250])
#         self.assertEqual(unc.calc_eai_exp, False)
#         self.assertEqual(unc.calc_at_event, False)

#         self.assertTrue(
#             np.allclose(
#                 unc.metrics['aai_agg'].aai_agg,
#                 np.array([9.600984e+07, 1.668144e+08, 8.068803e+08,
#                           1.274945e+08, 1.071482e+09, 2.215182e+08,
#                           1.401932e+09, 1.861671e+09])
#                 )
#             )
#         self.assertTrue(
#             np.allclose(
#                 unc.metrics['freq_curve'].rp5,
#                 np.zeros(8)
#                 )
#             )
#         self.assertTrue(
#             np.allclose(
#                 unc.metrics['freq_curve'].rp250,
#                 np.array([2.880990e+09, 5.005640e+09, 2.421225e+10,
#                           3.825758e+09, 3.215222e+10, 6.647149e+09,
#                           4.206811e+10, 5.586359e+10])
#                 )
#             )
#         self.assertTrue(unc.metrics['eai_exp'].empty)
#         self.assertTrue(unc.metrics['at_event'].empty)

#     def test_plot_distribution_pass(self):
#         """Test plot the distribution (uncertainty) of the impact metrics)"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()
#         unc = UncImpact(exp_unc, impf_unc, haz)
#         unc.make_sample(N=1)
#         unc.calc_distribution()
#         unc.plot_rp_distribution()
#         plt.close()

#     def test_plot_sensitivity_map_pass(self):
#         """Test plot the map of the largest sensitivity index for eai_exp"""

#         exp_unc, impf_unc, _ = make_imp_uncs()
#         haz = haz_dem()

#         #Default parameters
#         unc = UncImpact(exp_unc, impf_unc, haz)
#         unc.make_sample(N=1)
#         unc.calc_distribution(calc_eai_exp=True)
#         unc.calc_sensitivity()
#         unc.plot_sensitivity_map(exp_unc.uncvar_func(x_exp= 1))
#         plt.close()

#         #Non-default parameters
#         unc = UncImpact(exp_unc, impf_unc, haz)
#         unc.make_sample(N=1, sampling_method='morris')
#         unc.calc_distribution(calc_eai_exp=True)
#         unc.calc_sensitivity(salib_method='morris')
#         unc.plot_sensitivity_map(exp_unc.uncvar_func(x_exp= 1), salib_si='mu')
#         plt.close()


# class TestUncCostBenefit(unittest.TestCase):
#     """Test the UncCostBenefit class"""

#     def test_init_pass(self):
#         """Test cost benefit initialization"""

#         haz = haz_dem()
#         ent = ent_dem()
#         ent_fut = ent_fut_dem()

#         unc = UncCostBenefit(haz_unc=haz, ent_unc=ent,
#                              haz_fut_unc=haz, ent_fut_unc=ent_fut)

#         self.assertSetEqual(set(unc.metrics.keys()),
#              {'tot_climate_risk', 'benefit', 'cost_ben_ratio',
#               'imp_meas_present', 'imp_meas_future'}
#              )
#         self.assertSetEqual(set(unc.unc_vars.keys()), {'haz', 'ent',
#                                                        'haz_fut', 'ent_fut'})

#     def test_calc_distribution_pass(self):
#         """Test plot the distribution (uncertainty) of the impact metrics)"""

#         haz_fut = haz_dem
#         haz_distr = {"x_haz": sp.stats.uniform(1, 3),
#                       }
#         haz_fut_unc = UncVar(haz_fut, haz_distr)
#         haz = haz_dem(x_haz=10)

#         ent = ent_dem()
#         ent_fut = ent_fut_dem()

#         unc = UncCostBenefit(haz_unc=haz, ent_unc=ent,
#                              haz_fut_unc=haz_fut_unc, ent_fut_unc=ent_fut)

#         unc.make_sample(N=1)
#         unc.calc_distribution()

#         print(unc.metrics)

#         self.assertTrue(
#             np.allclose(unc.metrics['tot_climate_risk']['tot_climate_risk'],
#                 np.array([1.494806e+11, 1.262690e+11,
#                           1.494806e+11, 1.262690e+11])
#                 )
#             )
#         self.assertTrue(
#             np.allclose(unc.metrics['benefit']['Mangroves'],
#                 np.array([1.004445e+10, 2.403430e+09,
#                           1.004445e+10, 2.403430e+09])
#                 )
#             )
#         self.assertTrue(
#             np.allclose(unc.metrics['cost_ben_ratio']['Beach nourishment'],
#                 np.array([0.210822, 0.889260,
#                           0.210822, 0.889260])
#                 )
#             )

#         self.assertTrue(
#             np.allclose(unc.metrics['imp_meas_present']['Seawall-risk'],
#                 np.array([1.164395e+10, 1.164395e+10,
#                           1.164395e+10, 1.164395e+10])
#                 )
#             )

#         self.assertTrue(
#             np.allclose(unc.metrics['imp_meas_future']['Building code-risk'],
#                 np.array([2.556347e+09, 5.303379e+08,
#                           2.556347e+09, 5.303379e+08])
#                 )
#             )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInputVar)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOutput))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalcImpact))
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncCostBenefit))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
