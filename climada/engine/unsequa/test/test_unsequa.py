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

import copy
import time
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from tables.exceptions import HDF5ExtError

from climada.engine import ImpactCalc
from climada.engine.unsequa import (
    CalcCostBenefit,
    CalcDeltaImpact,
    CalcImpact,
    InputVar,
    UncOutput,
)
from climada.engine.unsequa.calc_base import LOGGER
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.entity.entity_def import Entity
from climada.hazard import Hazard
from climada.util.api_client import Client
from climada.util.constants import (
    ENT_DEMO_FUTURE,
    ENT_DEMO_TODAY,
    EXP_DEMO_H5,
    HAZ_DEMO_H5,
    TEST_UNC_OUTPUT_COSTBEN,
    TEST_UNC_OUTPUT_IMPACT,
)

test_unc_output_impact = Client().get_dataset_file(
    name=TEST_UNC_OUTPUT_IMPACT, status="test_dataset"
)
test_unc_output_costben = Client().get_dataset_file(
    name=TEST_UNC_OUTPUT_COSTBEN, status="test_dataset"
)


def impf_dem(x_paa=1, x_mdd=1):
    haz_type = "TC"
    id = 1
    intensity_unit = "m/s"
    intensity = np.linspace(0, 150, num=100)
    mdd = np.repeat(1, len(intensity)) * x_mdd
    paa = np.arange(0, len(intensity)) / len(intensity) * x_paa
    impf = ImpactFunc(haz_type, id, intensity, mdd, paa, intensity_unit)
    impf.check()
    impf_set = ImpactFuncSet([impf])
    return impf_set


def exp_dem(x_exp=1, exp=None):
    while not exp:
        try:
            exp = Exposures.from_hdf5(EXP_DEMO_H5)
        except HDF5ExtError:
            # possibly raised by pd.HDFStore when the file is locked by another process due to multiprocessing
            time.sleep(0.1)
    exp_tmp = exp.copy(deep=True)
    exp_tmp.gdf["value"] *= x_exp
    return exp_tmp


def haz_dem(x_haz=1, haz=None):
    haz = haz or Hazard.from_hdf5(HAZ_DEMO_H5)
    haz_tmp = copy.deepcopy(haz)
    haz_tmp.intensity = haz_tmp.intensity.multiply(x_haz)
    return haz_tmp


def make_input_vars():

    exp = exp_dem
    exp_distr = {
        "x_exp": sp.stats.uniform(0.8, 2),
    }
    exp_unc = InputVar(exp, exp_distr)

    impf = impf_dem
    impf_distr = {"x_paa": sp.stats.beta(0.5, 1), "x_mdd": sp.stats.uniform(0.8, 1.2)}
    impf_unc = InputVar(impf, impf_distr)

    haz = haz_dem
    haz_distr = {
        "x_haz": sp.stats.alpha(a=2, loc=1, scale=1),
    }
    haz_unc = InputVar(haz, haz_distr)

    return exp_unc, impf_unc, haz_unc


def ent_dem():
    entity = Entity.from_excel(ENT_DEMO_TODAY)
    entity.exposures.ref_year = 2018
    entity.check()
    return entity


def ent_fut_dem():
    entity = Entity.from_excel(ENT_DEMO_FUTURE)
    entity.exposures.ref_year = 2040
    entity.check()
    return entity


def make_costben_iv():

    entdem = ent_dem()
    ent_iv = InputVar.ent(
        impf_set_list=[entdem.impact_funcs],
        disc_rate=entdem.disc_rates,
        exp_list=[entdem.exposures],
        meas_set=entdem.measures,
        bounds_noise=[0.3, 1.9],
        bounds_cost=[0.5, 1.5],
        bounds_impfi=[-2, 5],
        haz_id_dict={"TC": [1]},
    )

    entfutdem = ent_fut_dem()
    entfut_iv = InputVar.entfut(
        impf_set_list=[entfutdem.impact_funcs],
        exp_list=[entfutdem.exposures],
        meas_set=entfutdem.measures,
        bounds_eg=[0.8, 1.5],
        bounds_mdd=[0.7, 0.9],
        bounds_paa=[1.3, 2],
        haz_id_dict={"TC": [1]},
    )

    return ent_iv, entfut_iv


class TestInputVar(unittest.TestCase):
    """Test UncVar class"""

    def test_init_pass(self):

        impf = impf_dem
        distr_dict = {
            "x_paa": sp.stats.beta(0.5, 1),
            "x_mdd": sp.stats.uniform(0.8, 1.2),
        }
        impf_iv = InputVar(impf, distr_dict)
        self.assertListEqual(impf_iv.labels, ["x_paa", "x_mdd"])
        self.assertTrue(isinstance(impf_iv.distr_dict, dict))

    def test_evaluate_pass(self):

        impf = impf_dem
        distr_dict = {
            "x_paa": sp.stats.beta(0.5, 1),
            "x_mdd": sp.stats.uniform(0.8, 0.4),
        }
        impf_iv = InputVar(impf, distr_dict)

        # Direct function evaluate
        impf_eval = impf_iv.func(**{"x_paa": 0.8, "x_mdd": 1.1})
        impf_true = impf_dem(x_paa=0.8, x_mdd=1.1)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()["TC"][1]
        impf_func2 = impf_true.get_func()["TC"][1]
        np.testing.assert_array_equal(impf_func1.intensity, impf_func2.intensity)
        np.testing.assert_array_equal(impf_func1.mdd, impf_func2.mdd)
        np.testing.assert_array_equal(impf_func1.paa, impf_func2.paa)
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

        # Specific evaluate
        impf_eval = impf_iv.evaluate(x_paa=0.8, x_mdd=1.1)
        impf_true = impf_dem(x_paa=0.8, x_mdd=1.1)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()["TC"][1]
        impf_func2 = impf_true.get_func()["TC"][1]
        np.testing.assert_array_equal(impf_func1.intensity, impf_func2.intensity)
        np.testing.assert_array_equal(impf_func1.mdd, impf_func2.mdd)
        np.testing.assert_array_equal(impf_func1.paa, impf_func2.paa)
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

        # Average evaluate (default)
        impf_eval = impf_iv.evaluate()
        impf_true = impf_dem(x_paa=0.3333333333333333, x_mdd=1.0)
        self.assertEqual(impf_eval.size(), impf_true.size())
        impf_func1 = impf_eval.get_func()["TC"][1]
        impf_func2 = impf_true.get_func()["TC"][1]
        np.testing.assert_array_almost_equal(impf_func1.intensity, impf_func2.intensity)
        np.testing.assert_array_almost_equal(impf_func1.mdd, impf_func2.mdd)
        np.testing.assert_array_almost_equal(impf_func1.paa, impf_func2.paa)
        self.assertEqual(impf_func1.id, impf_func2.id)
        self.assertEqual(impf_func1.haz_type, impf_func2.haz_type)

    def test_plot_pass(self):
        impf = impf_dem()
        distr_dict = {
            "x_paa": sp.stats.beta(0.5, 1),
            "x_mdd": sp.stats.uniform(0.8, 1.2),
            "x_lit": sp.stats.randint(0, 10),
        }
        impf_iv = InputVar(impf, distr_dict)
        self.assertIsNotNone(impf_iv.plot())
        plt.close()

    def test_var_to_inputvar(self):

        exp = exp_dem()
        distr_dict = {"x_exp": sp.stats.uniform(0.8, 1.2)}

        var = InputVar.var_to_inputvar(exp)
        self.assertDictEqual(var.distr_dict, {})
        self.assertTrue(isinstance(var.func(), Exposures))

        iv_var = InputVar.var_to_inputvar(InputVar(exp, distr_dict))
        self.assertDictEqual(iv_var.distr_dict, distr_dict)
        self.assertTrue(isinstance(iv_var, InputVar))


class TestOutput(unittest.TestCase):
    """Test the output class"""

    def test_init_pass(self):
        """Test initialization"""
        unc_out = UncOutput(pd.DataFrame())
        self.assertTrue(unc_out.samples_df.empty)

    def test_plot_unc_imp(self):
        """Test all impact plots"""
        unc_output = UncOutput.from_hdf5(test_unc_output_impact)
        plt_s = unc_output.plot_sample()
        self.assertIsNotNone(plt_s)
        plt.close()
        plt_u = unc_output.plot_uncertainty()
        self.assertIsNotNone(plt_u)
        plt.close()
        plt_rp = unc_output.plot_rp_uncertainty()
        self.assertIsNotNone(plt_rp)
        plt.close()
        plt_sens = unc_output.plot_rp_uncertainty()
        self.assertIsNotNone(plt_sens)
        plt.close()
        plt_sens_2 = unc_output.plot_sensitivity_second_order(salib_si="S1")
        self.assertIsNotNone(plt_sens_2)
        plt.close()
        plt_map = unc_output.plot_sensitivity_map()
        self.assertIsNotNone(plt_map)
        plt.close()

    def test_save_load_pass(self):
        """Test save and load output data"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)

        unc_data_save = unc_calc.make_sample(
            N=2, sampling_kwargs={"calc_second_order": True}
        )
        filename = unc_data_save.to_hdf5()
        unc_data_load = UncOutput.from_hdf5(filename)
        for attr_save, val_save in unc_data_save.__dict__.items():
            if isinstance(val_save, pd.DataFrame):
                df_load = getattr(unc_data_load, attr_save)
                self.assertTrue(df_load.equals(val_save))
        self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
        self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
        filename.unlink()

        unc_data_save = unc_calc.uncertainty(
            unc_data_save, calc_eai_exp=True, calc_at_event=False
        )
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
            unc_data_save, sensitivity_kwargs={"calc_second_order": True}
        )
        filename = unc_data_save.to_hdf5()
        unc_data_load = UncOutput.from_hdf5(filename)
        for attr_save, val_save in unc_data_save.__dict__.items():
            if isinstance(val_save, pd.DataFrame):
                df_load = getattr(unc_data_load, attr_save)
                self.assertTrue(df_load.equals(val_save))
        self.assertEqual(unc_data_load.sampling_method, unc_data_save.sampling_method)
        self.assertEqual(unc_data_load.sampling_kwargs, unc_data_save.sampling_kwargs)
        self.assertEqual(
            unc_data_load.sensitivity_method, unc_data_save.sensitivity_method
        )
        self.assertEqual(
            unc_data_load.sensitivity_kwargs, unc_data_save.sensitivity_kwargs
        )
        filename.unlink()


class TestCalcDelta(unittest.TestCase):
    """Test the calcluate delta impact uncertainty class"""

    def test_calc_uncertainty_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        haz2 = haz_dem()
        haz2.intensity *= 2
        unc_calc = CalcDeltaImpact(exp_unc, impf_dem(), haz, exp_dem(), impf_unc, haz2)
        unc_data = unc_calc.make_sample(N=2)
        unc_data = unc_calc.uncertainty(
            unc_data, calc_eai_exp=False, calc_at_event=False
        )

        for [x_exp, x_paa, x_mdd], delta_aai_aag in zip(
            unc_data.samples_df.values, unc_data.aai_agg_unc_df.values
        ):
            exp1 = exp_unc.evaluate(x_exp=x_exp)
            exp2 = exp_dem()
            impf1 = impf_dem()
            impf2 = impf_unc.evaluate(x_paa=x_paa, x_mdd=x_mdd)
            haz1 = haz

            imp1 = ImpactCalc(exp1, impf1, haz1).impact()
            imp2 = ImpactCalc(exp2, impf2, haz2).impact()

            self.assertAlmostEqual(
                (imp2.aai_agg - imp1.aai_agg) / imp1.aai_agg, delta_aai_aag
            )

        # test when computing absolute delta
        unc_data = unc_calc.uncertainty(
            unc_data, calc_eai_exp=False, calc_at_event=False, relative_delta=False
        )

        for [x_exp, x_paa, x_mdd], delta_aai_aag in zip(
            unc_data.samples_df.values, unc_data.aai_agg_unc_df.values
        ):
            exp1 = exp_unc.evaluate(x_exp=x_exp)
            exp2 = exp_dem()
            impf1 = impf_dem()
            impf2 = impf_unc.evaluate(x_paa=x_paa, x_mdd=x_mdd)
            haz1 = haz

            imp1 = ImpactCalc(exp1, impf1, haz1).impact()
            imp2 = ImpactCalc(exp2, impf2, haz2).impact()

            self.assertAlmostEqual(imp2.aai_agg - imp1.aai_agg, delta_aai_aag)

        self.assertEqual(unc_data.unit, exp_dem().value_unit)
        self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc_calc.calc_eai_exp, False)
        self.assertEqual(unc_calc.calc_at_event, False)

        self.assertEqual(unc_data.aai_agg_unc_df.size, unc_data.n_samples)
        self.assertEqual(
            unc_data.freq_curve_unc_df.size, unc_data.n_samples * len(unc_calc.rp)
        )
        self.assertTrue(unc_data.eai_exp_unc_df.empty)
        self.assertTrue(unc_data.at_event_unc_df.empty)

    def test_calc_sensitivity_pass(self):
        """Test compute sensitivity default for CalcDeltaImpact input"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        haz2 = haz_dem()
        haz2.intensity *= 2
        unc_calc = CalcDeltaImpact(exp_unc, impf_dem(), haz, exp_dem(), impf_unc, haz2)
        unc_data = unc_calc.make_sample(N=4)
        unc_data = unc_calc.uncertainty(
            unc_data, calc_eai_exp=False, calc_at_event=False
        )

        unc_data = unc_calc.sensitivity(
            unc_data, sensitivity_kwargs={"calc_second_order": True}
        )

        self.assertEqual(unc_data.sensitivity_method, "sobol")
        self.assertTupleEqual(
            unc_data.sensitivity_kwargs, tuple({"calc_second_order": "True"}.items())
        )

        for name, attr in unc_data.__dict__.items():
            if "sens_df" in name:
                if "eai" in name:
                    self.assertTrue(attr.empty)
                elif "at_event" in name:
                    self.assertTrue(attr.empty)
                else:
                    np.testing.assert_array_equal(
                        attr.param.unique(), np.array(["x_exp", "x_paa", "x_mdd"])
                    )

                    np.testing.assert_array_equal(
                        attr.si.unique(),
                        np.array(["S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf"]),
                    )

                    self.assertEqual(
                        len(attr), len(unc_data.param_labels) * (4 + 3 + 3)
                    )


class TestCalcImpact(unittest.TestCase):
    """Test the calcluate impact uncertainty class"""

    def test_init_pass(self):
        """Test initiliazation uncertainty"""

        exp_iv, impf_iv, haz_iv = make_input_vars()
        unc_calc = CalcImpact(exp_iv, impf_iv, haz_iv)

        self.assertTupleEqual(
            unc_calc._input_var_names,
            ("exp_input_var", "impf_input_var", "haz_input_var"),
        )
        self.assertTupleEqual(
            unc_calc._metric_names, ("aai_agg", "freq_curve", "at_event", "eai_exp")
        )
        self.assertEqual(unc_calc.value_unit, exp_iv.evaluate().value_unit)
        self.assertTrue(
            unc_calc.exp_input_var.evaluate(x_exp=1).gdf.equals(exp_dem(1).gdf)
        )
        impf1 = unc_calc.impf_input_var.evaluate(x_paa=1, x_mdd=1).get_func()["TC"][1]
        impf2 = impf_dem(1, 1).get_func()["TC"][1]
        np.testing.assert_array_almost_equal(
            impf1.calc_mdr(impf1.intensity), impf2.calc_mdr(impf2.intensity)
        )
        haz1 = unc_calc.haz_input_var.evaluate(x_haz=1)
        haz2 = haz_dem(1)
        self.assertListEqual(haz1.event_name, haz2.event_name)

    def test_make_sample_pass(self):
        """Test generate sample"""

        exp_unc, _, haz_unc = make_input_vars()
        impf = impf_dem()

        unc_calc = CalcImpact(exp_unc, impf, haz_unc)

        # default sampling saltelli
        unc_data = unc_calc.make_sample(
            N=2, sampling_kwargs={"calc_second_order": True}
        )
        self.assertEqual(unc_data.n_samples, 2 * (2 * 2 + 2))  # N * (2 * D + 2)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values, np.array(["x_exp", "x_haz"])
        )

        # latin sampling
        unc_data = unc_calc.make_sample(
            N=1, sampling_method="latin", sampling_kwargs={"seed": 11245}
        )
        self.assertEqual(unc_data.n_samples, 1)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values, np.array(["x_exp", "x_haz"])
        )

    def test_make_sample_ff_fail(self):
        """Test for warning and error messages when sampling using the 'ff' method"""

        exp_unc, impf_unc, haz_unc = make_input_vars()
        haz = haz_dem()

        # Warning ff sampling
        unc_calc = CalcImpact(exp_unc, impf_unc, haz_unc)
        warning_msg = "You are using the 'ff' sampler which does not require "
        "a value for N. The entered N value will be ignored"
        "in the sampling process."

        with self.assertLogs(LOGGER, level="WARNING") as logs:
            unc_data = unc_calc.make_sample(N=4, sampling_method="ff")
            self.assertEqual(len(logs.output), 1)
            self.assertIn(warning_msg, logs.output[0])

        # Error ff sampling
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        with self.assertRaises(ValueError) as cm:
            unc_data = unc_calc.make_sample(N=4, sampling_method="ff")
        the_exception = cm.exception
        self.assertEqual(
            the_exception.args[0],
            "The number of parameters must be a power of 2. "
            "To use the ff sampling method, you can generate "
            "dummy parameters to overcome this limitation."
            " See https://salib.readthedocs.io/en/latest/api.html",
        )

    def test_calc_uncertainty_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample(N=2)
        unc_data = unc_calc.uncertainty(
            unc_data, calc_eai_exp=False, calc_at_event=False
        )

        self.assertEqual(unc_data.unit, exp_dem().value_unit)
        self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc_calc.calc_eai_exp, False)
        self.assertEqual(unc_calc.calc_at_event, False)

        self.assertEqual(unc_data.aai_agg_unc_df.size, unc_data.n_samples)

        self.assertEqual(
            unc_data.freq_curve_unc_df.size, unc_data.n_samples * len(unc_calc.rp)
        )
        self.assertTrue(unc_data.eai_exp_unc_df.empty)
        self.assertTrue(unc_data.at_event_unc_df.empty)

    def test_calc_uncertainty_pool_pass(self):
        """Test parallel compute the uncertainty distribution for an impact"""

        exp_unc, impf_unc, _ = make_input_vars()
        haz = haz_dem()
        unc_calc = CalcImpact(exp_unc, impf_unc, haz)
        unc_data = unc_calc.make_sample(N=2)

        unc_data = unc_calc.uncertainty(
            unc_data, calc_eai_exp=False, calc_at_event=False, processes=4
        )

        self.assertEqual(unc_data.unit, exp_dem().value_unit)
        self.assertListEqual(unc_calc.rp, [5, 10, 20, 50, 100, 250])
        self.assertEqual(unc_calc.calc_eai_exp, False)
        self.assertEqual(unc_calc.calc_at_event, False)

        self.assertEqual(unc_data.aai_agg_unc_df.size, unc_data.n_samples)

        self.assertEqual(
            unc_data.freq_curve_unc_df.size, unc_data.n_samples * len(unc_calc.rp)
        )
        self.assertTrue(unc_data.eai_exp_unc_df.empty)
        self.assertTrue(unc_data.at_event_unc_df.empty)

    def test_calc_sensitivity_all_pass(self):
        """Test compute sensitivity using all different sensitivity methods"""

        # define input_vars
        exp_unc, impf_unc, haz_unc = make_input_vars()

        # dict to store the parameters and expected results for the tests
        test_dict = {
            "pawn": {
                "sampling_method": "saltelli",
                "sampling_kwargs": {},
                "N": 4,
                "sensitivity_kwargs": {"S": 10, "seed": 12345},
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["CV", 16],
                "test_si_value": [0.25000, 2],
            },
            "hdmr": {
                "sampling_method": "saltelli",
                "sampling_kwargs": {},
                "N": 100,
                "sensitivity_kwargs": {},
                "test_param_name": ["x_exp", 2],
                "test_si_name": ["Sa", 4],
                "test_si_value": [0.004658, 3],
            },
            "ff": {
                "sampling_method": "ff",
                "sampling_kwargs": {"seed": 12345},
                "N": 4,
                "sensitivity_kwargs": {"second_order": True},
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["IE", 4],
                "test_si_value": [865181825.901295, 10],
            },
            "sobol": {
                "sampling_method": "saltelli",
                "sampling_kwargs": {},
                "N": 4,
                "sensitivity_kwargs": {},
                "test_param_name": ["x_paa", 5],
                "test_si_name": ["ST", 8],
                "test_si_value": [0.313025, 10],
            },
            "dgsm": {
                "sampling_method": "finite_diff",
                "N": 4,
                "sampling_kwargs": {"seed": 12345},
                "sensitivity_kwargs": {
                    "num_resamples": 100,
                    "conf_level": 0.95,
                    "seed": 12345,
                },
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["dgsm", 8],
                "test_si_value": [1.697516e-01, 9],
            },
            "fast": {
                "sampling_method": "fast_sampler",
                "sampling_kwargs": {"M": 4, "seed": 12345},
                "N": 256,
                "sensitivity_kwargs": {"M": 4, "seed": 12345},
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["S1_conf", 8],
                "test_si_value": [0.671396, 1],
            },
            "rbd_fast": {
                "sampling_method": "saltelli",
                "sampling_kwargs": {},
                "N": 24,
                "sensitivity_kwargs": {"M": 4, "seed": 12345},
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["S1_conf", 4],
                "test_si_value": [0.152609, 4],
            },
            "morris": {
                "sampling_method": "morris",
                "sampling_kwargs": {"seed": 12345},
                "N": 4,
                "sensitivity_kwargs": {},
                "test_param_name": ["x_exp", 0],
                "test_si_name": ["mu", 1],
                "test_si_value": [5066460029.63911, 8],
            },
        }

        def test_sensitivity_method(
            exp_unc, impf_unc, haz_unc, sensitivity_method, param_dict
        ):
            """Function to test each seaprate sensitivity method"""
            unc_calc = CalcImpact(exp_unc, impf_unc, haz_unc)
            unc_data = unc_calc.make_sample(
                N=param_dict["N"],
                sampling_method=param_dict["sampling_method"],
                sampling_kwargs=param_dict["sampling_kwargs"],
            )
            unc_data = unc_calc.uncertainty(
                unc_data, calc_eai_exp=False, calc_at_event=False
            )

            # Call the sensitivity method with each method's specific arguments
            unc_data = unc_calc.sensitivity(
                unc_data,
                sensitivity_method=sensitivity_method,
                sensitivity_kwargs=param_dict["sensitivity_kwargs"],
            )

            self.assertEqual(
                param_dict["test_param_name"][0],
                unc_data.aai_agg_sens_df["param"][param_dict["test_param_name"][1]],
            )
            self.assertEqual(
                param_dict["test_si_name"][0],
                unc_data.aai_agg_sens_df["si"][param_dict["test_si_name"][1]],
            )
            self.assertAlmostEqual(
                param_dict["test_si_value"][0],
                unc_data.aai_agg_sens_df["aai_agg"][param_dict["test_si_value"][1]],
                places=5,
            )

            self.assertEqual(unc_data.aai_agg_unc_df.size, unc_data.n_samples)

            self.assertEqual(
                unc_data.freq_curve_unc_df.size, unc_data.n_samples * len(unc_calc.rp)
            )
            self.assertTrue(unc_data.eai_exp_unc_df.empty)
            self.assertTrue(unc_data.at_event_unc_df.empty)

        # loop over each method and do test
        for sensitivity_method, method_params in test_dict.items():
            test_sensitivity_method(
                exp_unc, impf_unc, haz_unc, sensitivity_method, method_params
            )


class TestCalcCostBenefit(unittest.TestCase):
    """Test the calcluate impact uncertainty class"""

    def test_init_pass(self):
        """Test initiliazation uncertainty"""

        ent_iv, ent_fut_iv = make_costben_iv()
        _, _, haz_iv = make_input_vars()

        unc_calc = CalcCostBenefit(haz_iv, ent_iv)

        self.assertTupleEqual(
            unc_calc._input_var_names,
            (
                "haz_input_var",
                "ent_input_var",
                "haz_fut_input_var",
                "ent_fut_input_var",
            ),
        )
        self.assertTupleEqual(
            unc_calc._metric_names,
            (
                "tot_climate_risk",
                "benefit",
                "cost_ben_ratio",
                "imp_meas_present",
                "imp_meas_future",
            ),
        )
        self.assertEqual(unc_calc.value_unit, ent_dem().exposures.value_unit)
        self.assertTrue(
            unc_calc.ent_input_var.evaluate(
                CO=None, IFi=None, EN=None, EL=0
            ).exposures.gdf.equals(ent_dem().exposures.gdf)
        )

        haz1 = unc_calc.haz_input_var.evaluate(x_haz=1)
        haz2 = haz_dem(1)
        self.assertListEqual(haz1.event_name, haz2.event_name)

        unc_calc = CalcCostBenefit(haz_iv, ent_iv, haz_iv, ent_fut_iv)

        self.assertTupleEqual(
            unc_calc._input_var_names,
            (
                "haz_input_var",
                "ent_input_var",
                "haz_fut_input_var",
                "ent_fut_input_var",
            ),
        )
        self.assertTupleEqual(
            unc_calc._metric_names,
            (
                "tot_climate_risk",
                "benefit",
                "cost_ben_ratio",
                "imp_meas_present",
                "imp_meas_future",
            ),
        )
        self.assertEqual(unc_calc.value_unit, ent_dem().exposures.value_unit)
        self.assertTrue(
            unc_calc.ent_input_var.evaluate(
                CO=None, IFi=None, EN=None
            ).exposures.gdf.equals(ent_dem().exposures.gdf)
        )
        self.assertTrue(
            unc_calc.ent_fut_input_var.evaluate(
                EG=None, MDD=None, PAA=None
            ).exposures.gdf.equals(ent_fut_dem().exposures.gdf)
        )

        haz1 = unc_calc.haz_input_var.evaluate(x_haz=1)
        haz2 = haz_dem(1)
        self.assertListEqual(haz1.event_name, haz2.event_name)

        haz3 = unc_calc.haz_fut_input_var.evaluate(x_haz=1)
        self.assertListEqual(haz3.event_name, haz2.event_name)

    def test_make_sample_pass(self):
        """Test generate sample"""

        ent_iv, ent_fut_iv = make_costben_iv()
        _, _, haz_iv = make_input_vars()

        unc_calc = CalcCostBenefit(haz_iv, ent_iv)

        # default sampling saltelli
        unc_data = unc_calc.make_sample(
            N=2, sampling_kwargs={"calc_second_order": True}
        )
        self.assertEqual(unc_data.n_samples, 2 * (2 * 4 + 2))  # N * (2 * D + 2)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values, np.array(["x_haz", "EN", "IFi", "CO"])
        )

        # #latin sampling
        unc_data = unc_calc.make_sample(
            N=1, sampling_method="latin", sampling_kwargs={"seed": 11245}
        )
        self.assertEqual(unc_data.n_samples, 1)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values, np.array(["x_haz", "EN", "IFi", "CO"])
        )

        unc_calc = CalcCostBenefit(haz_iv, ent_iv, haz_iv, ent_fut_iv)

        # default sampling saltelli
        unc_data = unc_calc.make_sample(
            N=2, sampling_kwargs={"calc_second_order": True}
        )
        self.assertEqual(unc_data.n_samples, 2 * (2 * 7 + 2))  # N * (2 * D + 2)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values,
            np.array(["x_haz", "EN", "IFi", "CO", "EG", "PAA", "MDD"]),
        )

        # #latin sampling
        unc_data = unc_calc.make_sample(
            N=1, sampling_method="latin", sampling_kwargs={"seed": 11245}
        )
        self.assertEqual(unc_data.n_samples, 1)
        self.assertTrue(isinstance(unc_data.samples_df, pd.DataFrame))
        np.testing.assert_array_equal(
            unc_data.samples_df.columns.values,
            np.array(["x_haz", "EN", "IFi", "CO", "EG", "PAA", "MDD"]),
        )

    def test_calc_uncertainty_pool_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        ent_iv, _ = make_costben_iv()
        _, _, haz_iv = make_input_vars()
        unc_calc = CalcCostBenefit(haz_iv, ent_iv)
        unc_data = unc_calc.make_sample(N=2)

        unc_data = unc_calc.uncertainty(unc_data, processes=2)

        self.assertEqual(unc_data.unit, ent_dem().exposures.value_unit)

        self.assertEqual(unc_data.tot_climate_risk_unc_df.size, unc_data.n_samples)
        self.assertEqual(
            unc_data.cost_ben_ratio_unc_df.size,
            unc_data.n_samples * 4,  # number of measures
        )
        self.assertEqual(unc_data.imp_meas_present_unc_df.size, 0)
        self.assertEqual(
            unc_data.imp_meas_future_unc_df.size,
            unc_data.n_samples * 4 * 5,  # All measures 4 and risks/benefits 5
        )


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInputVar)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOutput))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalcImpact))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalcDelta))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalcCostBenefit))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
