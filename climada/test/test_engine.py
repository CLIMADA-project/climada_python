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

Test engine.

"""

import copy
import time
import unittest

import numpy as np
import scipy as sp
from tables.exceptions import HDF5ExtError

from climada import CONFIG
from climada.engine import impact_data as im_d
from climada.engine.test.test_impact import dummy_impact
from climada.engine.unsequa import CalcCostBenefit, InputVar
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.entity.entity_def import Entity
from climada.hazard import Hazard
from climada.util.constants import (
    ENT_DEMO_FUTURE,
    ENT_DEMO_TODAY,
    EXP_DEMO_H5,
    HAZ_DEMO_H5,
)

DATA_DIR = CONFIG.engine.test_data.dir()
EMDAT_TEST_CSV = DATA_DIR.joinpath("emdat_testdata_BGD_USA_1970-2017.csv")


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
        # Possibly raised by pd.HDFStore when the file is locked by another process
        # due to multiprocessing
        except HDF5ExtError:
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


class TestEmdatProcessing(unittest.TestCase):
    def test_emdat_damage_yearlysum(self):
        """test emdat_impact_yearlysum yearly impact data extraction with scaling"""
        df = im_d.emdat_impact_yearlysum(
            EMDAT_TEST_CSV,
            countries=["Bangladesh", "USA"],
            hazard="Flood",
            year_range=(2015, 2017),
            reference_year=2000,
        )

        self.assertEqual(36, df.size)
        self.assertAlmostEqual(df["impact"].max(), 15150000000.0)
        self.assertAlmostEqual(df["impact_scaled"].min(), 10939000.0)
        self.assertEqual(df["year"][5], 2017)
        self.assertEqual(df["reference_year"].max(), 2000)
        self.assertIn("USA", list(df["ISO"]))
        self.assertIn(50, list(df["region_id"]))


class TestGDPScaling(unittest.TestCase):
    """test scaling of impact values proportional to GDP"""

    def test_scale_impact2refyear(self):
        """scale of impact values proportional to GDP"""
        impact_scaled = im_d.scale_impact2refyear(
            [10, 100, 1000, 100, 100],
            [1999, 2005, 2015, 2000, 2000],
            ["CZE", "CZE", "MEX", "MEX", "CZE"],
            reference_year=2015,
        )
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test fails in the following line:
        self.assertListEqual(impact_scaled, [28, 137, 1000, 163, 304])


class TestEmdatToImpact(unittest.TestCase):
    """Test import of EM-DAT data (as CSV) to Impact-instance (CLIMADA)"""

    def test_emdat_to_impact_scale(self):
        """test import DR EM-DAT to Impact() for 1 country and ref.year (scaling)"""
        impact_emdat = im_d.emdat_to_impact(
            EMDAT_TEST_CSV,
            "DR",
            year_range=[2010, 2016],
            countries=["USA"],
            hazard_type_emdat="Drought",
            reference_year=2016,
        )[0]
        self.assertEqual(5, impact_emdat.event_id.size)
        self.assertEqual(4, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn("2012-9235", impact_emdat.event_name)
        self.assertEqual(1, len(impact_emdat.eai_exp))
        self.assertAlmostEqual(impact_emdat.aai_agg, impact_emdat.eai_exp[0])
        self.assertAlmostEqual(
            0.14285714, np.unique(impact_emdat.frequency)[0], places=3
        )
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test failes in the following 2 lines:
        self.assertAlmostEqual(3.69, np.sum(impact_emdat.at_event * 1e-10), places=0)
        self.assertAlmostEqual(5.28, impact_emdat.aai_agg * 1e-9, places=0)


class TestCalcCostBenefit(unittest.TestCase):
    """Test the calcluate impact uncertainty class"""

    def test_calc_uncertainty_pass(self):
        """Test compute the uncertainty distribution for an impact"""

        ent_iv, ent_fut_iv = make_costben_iv()
        _, _, haz_iv = make_input_vars()
        unc_calc = CalcCostBenefit(haz_iv, ent_iv)
        unc_data = unc_calc.make_sample(N=2)
        unc_data = unc_calc.uncertainty(unc_data)

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

        unc_calc = CalcCostBenefit(haz_iv, ent_iv, haz_iv, ent_fut_iv)
        unc_data = unc_calc.make_sample(N=2)
        unc_data = unc_calc.uncertainty(unc_data)

        self.assertEqual(unc_data.unit, ent_dem().exposures.value_unit)

        self.assertEqual(unc_data.tot_climate_risk_unc_df.size, unc_data.n_samples)
        self.assertEqual(
            unc_data.cost_ben_ratio_unc_df.size,
            unc_data.n_samples * 4,  # number of measures
        )
        self.assertEqual(
            unc_data.imp_meas_present_unc_df.size,
            unc_data.n_samples * 4 * 5,  # All measures 4 and risks/benefits 5
        )
        self.assertEqual(
            unc_data.imp_meas_future_unc_df.size,
            unc_data.n_samples * 4 * 5,  # All measures 4 and risks/benefits 5
        )

    def test_calc_sensitivity_pass(self):
        """Test compute sensitivity default"""

        ent_iv, _ = make_costben_iv()
        _, _, haz_iv = make_input_vars()
        unc_calc = CalcCostBenefit(haz_iv, ent_iv)
        unc_data = unc_calc.make_sample(
            N=4, sampling_kwargs={"calc_second_order": True}
        )
        unc_data = unc_calc.uncertainty(unc_data)

        unc_data = unc_calc.sensitivity(
            unc_data, sensitivity_kwargs={"calc_second_order": True}
        )

        self.assertEqual(unc_data.sensitivity_method, "sobol")
        self.assertTupleEqual(
            unc_data.sensitivity_kwargs, tuple({"calc_second_order": "True"}.items())
        )

        for name, attr in unc_data.__dict__.items():
            if "sens_df" in name:
                if "imp_meas_present" in name:
                    self.assertTrue(attr.empty)
                else:
                    np.testing.assert_array_equal(
                        attr.param.unique(), np.array(["x_haz", "EN", "IFi", "CO"])
                    )

                    np.testing.assert_array_equal(
                        attr.si.unique(),
                        np.array(["S1", "S1_conf", "ST", "ST_conf", "S2", "S2_conf"]),
                    )

                    self.assertEqual(
                        len(attr), len(unc_data.param_labels) * (4 + 4 + 4)
                    )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmdatProcessing)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGDPScaling))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmdatToImpact))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalcCostBenefit))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
