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

Test CostBenefit class.
"""
from pathlib import Path
import copy
import unittest
import numpy as np

from climada.entity.entity_def import Entity
from climada.entity.disc_rates import DiscRates
from climada.hazard.base import Hazard
from climada.engine.cost_benefit import CostBenefit, risk_aai_agg, \
        risk_rp_100, risk_rp_250, _norm_values
from climada.engine import ImpactCalc
from climada.util.constants import ENT_DEMO_FUTURE, ENT_DEMO_TODAY
from climada.util.api_client import Client

from climada.test import get_test_file


HAZ_TEST_MAT = get_test_file('atl_prob_no_name')
ENT_TEST_MAT = get_test_file('demo_today', file_format='MAT-file')


class TestSteps(unittest.TestCase):
    """Test intermediate steps"""
    def test_calc_impact_measures_pass(self):
        """Test _calc_impact_measures against reference value"""
        self.assertTrue(HAZ_TEST_MAT.is_file(), "{} is not a file".format(HAZ_TEST_MAT))
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        self.assertTrue(ENT_TEST_MAT.is_file(), "{} is not a file".format(ENT_TEST_MAT))
        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.check()
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        entity.check()
        entity.exposures.assign_centroids(hazard)

        cost_ben = CostBenefit()
        cost_ben._calc_impact_measures(hazard, entity.exposures, entity.measures,
                                       entity.impact_funcs, when='future',
                                       risk_func=risk_aai_agg, save_imp=True)

        self.assertEqual(cost_ben.imp_meas_present, dict())
        self.assertEqual(cost_ben.cost_ben_ratio, dict())
        self.assertEqual(cost_ben.benefit, dict())
        self.assertEqual(cost_ben.tot_climate_risk, 0.0)
        self.assertEqual(cost_ben.present_year, 2016)
        self.assertEqual(cost_ben.future_year, 2030)

        self.assertEqual(cost_ben.imp_meas_future['no measure']['cost'], (0, 0))
        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['risk'],
                               6.51220115756442e+09, places=3)
        new_efc = cost_ben.imp_meas_future['no measure']['impact'].calc_freq_curve()
        self.assertTrue(
            np.allclose(new_efc.return_per,
                        cost_ben.imp_meas_future['no measure']['efc'].return_per))
        self.assertTrue(
            np.allclose(new_efc.impact, cost_ben.imp_meas_future['no measure']['efc'].impact))
        self.assertEqual(
            cost_ben.imp_meas_future['no measure']['impact'].at_event.nonzero()[0].size,
            841)
        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['impact'].at_event[14082],
                               8.801682862431524e+06, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['impact'].tot_value,
                               6.570532945599105e+11, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['impact'].aai_agg,
                               6.51220115756442e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['cost'][0],
                               1.3117683608515418e+09, places=3)
        self.assertEqual(cost_ben.imp_meas_future['Mangroves']['cost'][1], 1)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['risk'],
                               4.850407096284983e+09, places=3)
        new_efc = cost_ben.imp_meas_future['Mangroves']['impact'].calc_freq_curve()
        self.assertTrue(
            np.allclose(new_efc.return_per,
                        cost_ben.imp_meas_future['Mangroves']['efc'].return_per))
        self.assertTrue(
            np.allclose(new_efc.impact, cost_ben.imp_meas_future['Mangroves']['efc'].impact))
        self.assertEqual(
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event.nonzero()[0].size,
            665)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['impact'].at_event[13901],
                               1.29576562770977e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['impact'].tot_value,
                               6.570532945599105e+11, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['impact'].aai_agg,
                               4.850407096284983e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['cost'][0],
                               1.728000000000000e+09, places=3)
        self.assertEqual(cost_ben.imp_meas_future['Beach nourishment']['cost'][1], 1)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['risk'],
                               5.188921355413834e+09, places=3)
        new_efc = cost_ben.imp_meas_future['Beach nourishment']['impact'].calc_freq_curve()
        self.assertTrue(
            np.allclose(new_efc.return_per,
                        cost_ben.imp_meas_future['Beach nourishment']['efc'].return_per))
        self.assertTrue(
            np.allclose(new_efc.impact,
                        cost_ben.imp_meas_future['Beach nourishment']['efc'].impact))
        self.assertEqual(
            cost_ben.imp_meas_future['Beach nourishment']['impact'].at_event.nonzero()[0].size,
            702)
        self.assertEqual(cost_ben.imp_meas_future['Beach nourishment']['impact'].at_event[1110],
                         0.0)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['impact'].eai_exp[5],
                               1.1133679079730146e+08, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['impact'].tot_value,
                               6.570532945599105e+11, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['impact'].aai_agg,
                               5.188921355413834e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['cost'][0],
                               8.878779433630093e+09, places=3)
        self.assertEqual(cost_ben.imp_meas_future['Seawall']['cost'][1], 1)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['risk'],
                               4.736400526119911e+09, places=3)
        new_efc = cost_ben.imp_meas_future['Seawall']['impact'].calc_freq_curve()
        self.assertTrue(np.allclose(new_efc.return_per,
                                    cost_ben.imp_meas_future['Seawall']['efc'].return_per))
        self.assertTrue(np.allclose(new_efc.impact,
                                    cost_ben.imp_meas_future['Seawall']['efc'].impact))
        self.assertEqual(cost_ben.imp_meas_future['Seawall']['impact'].at_event.nonzero()[0].size,
                         73)
        self.assertEqual(cost_ben.imp_meas_future['Seawall']['impact'].at_event[1229], 0.0)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['impact'].tot_value,
                               6.570532945599105e+11, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['impact'].aai_agg,
                               4.736400526119911e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['cost'][0],
                               9.200000000000000e+09, places=3)
        self.assertEqual(cost_ben.imp_meas_future['Building code']['cost'][1], 1)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['risk'],
                               4.884150868173321e+09, places=3)
        new_efc = cost_ben.imp_meas_future['Building code']['impact'].calc_freq_curve()
        self.assertTrue(np.allclose(new_efc.return_per,
                                    cost_ben.imp_meas_future['Building code']['efc'].return_per))
        self.assertTrue(np.allclose(new_efc.impact,
                                    cost_ben.imp_meas_future['Building code']['efc'].impact))
        self.assertEqual(
            cost_ben.imp_meas_future['Building code']['impact'].at_event.nonzero()[0].size,
            841)
        self.assertEqual(cost_ben.imp_meas_future['Building code']['impact'].at_event[122], 0.0)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['impact'].eai_exp[11],
                               7.757060129393841e+07, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['impact'].tot_value,
                               6.570532945599105e+11, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['impact'].aai_agg,
                               4.884150868173321e+09, places=3)

    def test_cb_one_meas_pres_pass(self):
        """Test _cost_ben_one with different future"""
        meas_name = 'Mangroves'
        meas_val = dict()
        meas_val['cost'] = (1.3117683608515418e+09, 1)
        meas_val['risk'] = 4.826231151473135e+10
        meas_val['efc'] = None
        meas_val['risk_transf'] = 0

        imp_meas_present = dict()
        imp_meas_present['no measure'] = dict()
        imp_meas_present['no measure']['risk'] = 6.51220115756442e+09
        imp_meas_present['Mangroves'] = dict()
        imp_meas_present['Mangroves']['risk'] = 4.850407096284983e+09
        imp_meas_present['Mangroves']['risk_transf'] = 0

        imp_meas_future = dict()
        imp_meas_future['no measure'] = dict()
        imp_meas_future['no measure']['risk'] = 5.9506659786664024e+10

        cb = CostBenefit(present_year=2018, future_year=2040, imp_meas_present=imp_meas_present,
                         imp_meas_future=imp_meas_future)

        disc_rates = DiscRates()
        disc_rates.years = np.arange(2016, 2051)
        disc_rates.rates = np.ones(disc_rates.years.size) * 0.02

        time_dep = cb._time_dependency_array(1)

        cb._cost_ben_one(meas_name, meas_val, disc_rates, time_dep)
        self.assertAlmostEqual(cb.benefit[meas_name], 113345027690.81276, places=3)
        self.assertAlmostEqual(cb.cost_ben_ratio[meas_name], 0.011573232523528404)

    def test_cb_one_meas_fut_pass(self):
        """Test _cost_ben_one with same future"""
        meas_name = 'Mangroves'
        meas_val = dict()
        meas_val['cost'] = (1.3117683608515418e+09, 1)
        meas_val['risk'] = 4.850407096284983e+09
        meas_val['efc'] = None
        meas_val['risk_transf'] = 0

        imp_meas_future = dict()
        imp_meas_future['no measure'] = dict()
        imp_meas_future['no measure']['risk'] = 6.51220115756442e+09

        cb = CostBenefit(present_year=2018, future_year=2040, imp_meas_future=imp_meas_future)

        years = np.arange(2000, 2051)
        rates = np.ones(years.size) * 0.02
        disc_rates = DiscRates(years=years, rates=rates)

        time_dep = cb._time_dependency_array()

        cb._cost_ben_one(meas_name, meas_val, disc_rates, time_dep)
        self.assertAlmostEqual(cb.benefit[meas_name], 3.100583368954022e+10, places=3)
        self.assertAlmostEqual(cb.cost_ben_ratio[meas_name], 0.04230714690616641)

    def test_calc_cb_no_change_pass(self):
        """Test _calc_cost_benefit without present value against reference value"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        entity.check()
        entity.exposures.assign_centroids(hazard)

        cost_ben = CostBenefit()
        cost_ben._calc_impact_measures(hazard, entity.exposures, entity.measures,
                                       entity.impact_funcs, when='future',
                                       risk_func=risk_aai_agg, save_imp=True)

        cost_ben.present_year = 2018
        cost_ben.future_year = 2040
        cost_ben._calc_cost_benefit(entity.disc_rates)

        self.assertEqual(cost_ben.imp_meas_present, dict())
        self.assertEqual(len(cost_ben.imp_meas_future), 5)
        self.assertEqual(cost_ben.present_year, 2018)
        self.assertEqual(cost_ben.future_year, 2040)

        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Mangroves'], 0.04230714690616641)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Beach nourishment'], 0.06998836431681373)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Seawall'], 0.2679741183248266)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Building code'], 0.30286828677985717)

        self.assertAlmostEqual(cost_ben.benefit['Mangroves'], 3.100583368954022e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Beach nourishment'],
                               2.468981832719974e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Seawall'], 3.3132973770502796e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Building code'], 3.0376240767284798e+10, places=3)

        self.assertAlmostEqual(cost_ben.tot_climate_risk, 1.2150496306913972e+11, places=3)

    def test_calc_cb_change_pass(self):
        """Test _calc_cost_benefit with present value against reference value"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        entity.check()
        entity.exposures.assign_centroids(hazard)

        cost_ben = CostBenefit()
        cost_ben._calc_impact_measures(hazard, entity.exposures, entity.measures,
                                       entity.impact_funcs, when='present',
                                       risk_func=risk_aai_agg, save_imp=False)

        ent_future = Entity.from_excel(ENT_DEMO_FUTURE)
        ent_future.check()

        haz_future = copy.deepcopy(hazard)
        haz_future.intensity.data += 25
        ent_future.exposures.assign_centroids(haz_future)

        cost_ben._calc_impact_measures(haz_future, ent_future.exposures, ent_future.measures,
                                       ent_future.impact_funcs, when='future',
                                       risk_func=risk_aai_agg, save_imp=False)

        cost_ben.present_year = 2018
        cost_ben.future_year = 2040
        cost_ben._calc_cost_benefit(entity.disc_rates, imp_time_depen=1)

        self.assertEqual(cost_ben.present_year, 2018)
        self.assertEqual(cost_ben.future_year, 2040)
        self.assertAlmostEqual(cost_ben.tot_climate_risk, 5.768659152882021e+11, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_present['no measure']['risk'],
                               6.51220115756442e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Mangroves']['risk'],
                               4.850407096284983e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Beach nourishment']['risk'],
                               5.188921355413834e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Seawall']['risk'],
                               4.736400526119911e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Building code']['risk'],
                               4.884150868173321e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['risk'],
                               5.9506659786664024e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['risk'],
                               4.826231151473135e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['risk'],
                               5.0647250923231674e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['risk'],
                               21089567135.7345, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['risk'],
                               4.462999483999791e+10, places=3)

        self.assertAlmostEqual(cost_ben.benefit['Mangroves'], 113345027690.81276, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Beach nourishment'], 89444869971.53653, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Seawall'], 347977469896.1333, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Building code'], 144216478822.05154, places=2)

        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Mangroves'], 0.011573232523528404)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Beach nourishment'], 0.01931916274851638)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Seawall'], 0.025515385913577368)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Building code'], 0.06379298728650741)

        self.assertAlmostEqual(cost_ben.tot_climate_risk, 576865915288.2021, places=3)

    def test_time_array_pres_pass(self):
        """Test _time_dependency_array"""
        cb = CostBenefit(present_year=2018, future_year=2030)
        imp_time_depen = 1.0
        time_arr = cb._time_dependency_array(imp_time_depen)

        n_years = cb.future_year - cb.present_year + 1
        self.assertEqual(time_arr.size, n_years)
        self.assertTrue(np.allclose(time_arr[:-1], np.arange(0, 1, 1 / (n_years - 1))))
        self.assertEqual(time_arr[-1], 1)

        imp_time_depen = 0.5
        time_arr = cb._time_dependency_array(imp_time_depen)

        n_years = cb.future_year - cb.present_year + 1
        self.assertEqual(time_arr.size, n_years)
        self.assertTrue(np.allclose(time_arr, np.arange(n_years)**imp_time_depen /
                                    (n_years - 1)**imp_time_depen))

    def test_time_array_no_pres_pass(self):
        """Test _time_dependency_array"""
        cb = CostBenefit(present_year=2018, future_year=2030)
        time_arr = cb._time_dependency_array()

        n_years = cb.future_year - cb.present_year + 1
        self.assertEqual(time_arr.size, n_years)
        self.assertTrue(np.array_equal(time_arr, np.ones(n_years)))

    def test_npv_unaverted_no_pres_pass(self):
        """Test _npv_unaverted_impact"""
        cb = CostBenefit(present_year=2018, future_year=2030)
        risk_future = 1000
        years = np.arange(cb.present_year, cb.future_year + 1)
        rates = np.ones(years.size) * 0.025
        disc_rates = DiscRates(years=years, rates=rates)
        time_dep = np.linspace(0, 1, disc_rates.years.size)
        res = cb._npv_unaverted_impact(risk_future, disc_rates, time_dep,
                                       risk_present=None)

        self.assertEqual(
            res,
            disc_rates.net_present_value(cb.present_year, cb.future_year, time_dep * risk_future))

    def test_npv_unaverted_pres_pass(self):
        """Test _npv_unaverted_impact"""
        cb = CostBenefit(present_year=2018, future_year=2030)
        risk_future = 1000
        risk_present = 500
        years = np.arange(cb.present_year, cb.future_year + 1)
        rates = np.ones(years.size) * 0.025
        disc_rates = DiscRates(years=years, rates=rates)

        time_dep = np.linspace(0, 1, disc_rates.years.size)
        res = cb._npv_unaverted_impact(risk_future, disc_rates, time_dep, risk_present)


        tot_climate_risk = risk_present + (risk_future - risk_present) * time_dep
        self.assertEqual(res, disc_rates.net_present_value(cb.present_year,
                                                           cb.future_year,
                                                           tot_climate_risk))

    def test_norm_value(self):
        """Test _norm_values"""
        norm_fact, norm_name = _norm_values(1)
        self.assertEqual(norm_fact, 1)
        self.assertEqual(norm_name, "")

        norm_fact, norm_name = _norm_values(10)
        self.assertEqual(norm_fact, 1)
        self.assertEqual(norm_name, "")

        norm_fact, norm_name = _norm_values(100)
        self.assertEqual(norm_fact, 1)
        self.assertEqual(norm_name, "")

        norm_fact, norm_name = _norm_values(1001)
        self.assertEqual(norm_fact, 1000)
        self.assertEqual(norm_name, "k")

        norm_fact, norm_name = _norm_values(10000)
        self.assertEqual(norm_fact, 1000)
        self.assertEqual(norm_name, "k")

        norm_fact, norm_name = _norm_values(1.01e6)
        self.assertEqual(norm_fact, 1.0e6)
        self.assertEqual(norm_name, "m")

        norm_fact, norm_name = _norm_values(1.0e8)
        self.assertEqual(norm_fact, 1.0e6)
        self.assertEqual(norm_name, "m")

        norm_fact, norm_name = _norm_values(1.01e9)
        self.assertEqual(norm_fact, 1.0e9)
        self.assertEqual(norm_name, "bn")

        norm_fact, norm_name = _norm_values(1.0e10)
        self.assertEqual(norm_fact, 1.0e9)
        self.assertEqual(norm_name, "bn")

        norm_fact, norm_name = _norm_values(1.0e12)
        self.assertEqual(norm_fact, 1.0e9)
        self.assertEqual(norm_name, "bn")

    def test_combine_fut_pass(self):
        """Test combine_measures with present and future"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018

        fut_ent = copy.deepcopy(entity)
        fut_ent.exposures.ref_year = 2040
        fut_haz = copy.deepcopy(hazard)

        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, fut_haz, fut_ent, future_year=None,
                      risk_func=risk_aai_agg, imp_time_depen=None, save_imp=True)

        new_name = 'combine'
        new_color = np.array([0.1, 0.1, 0.1])
        new_cb = cost_ben.combine_measures(['Mangroves', 'Seawall'], new_name, new_color,
                                           entity.disc_rates, imp_time_depen=None,
                                           risk_func=risk_aai_agg)

        self.assertTrue(np.allclose(new_cb.color_rgb[new_name], new_color))

        new_imp = cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event
        new_imp += cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Seawall']['impact'].at_event
        new_imp = np.maximum(cost_ben.imp_meas_future['no measure']['impact'].at_event - new_imp,
                             0)

        self.assertTrue(np.allclose(new_cb.imp_meas_present[new_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_present[new_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_present['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(new_cb.imp_meas_present[new_name]['cost'][0],
                               cost_ben.imp_meas_present['Mangroves']['cost'][0] +
                               cost_ben.imp_meas_present['Seawall']['cost'][0])
        self.assertAlmostEqual(new_cb.imp_meas_present[new_name]['cost'][1], 1)
        self.assertTrue(np.allclose(
            new_cb.imp_meas_present[new_name]['efc'].impact,
            new_cb.imp_meas_present[new_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_present[new_name]['risk_transf'], 0)

        self.assertTrue(np.allclose(new_cb.imp_meas_future[new_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_future[new_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(
            new_cb.imp_meas_future[new_name]['cost'][0],
            cost_ben.imp_meas_future['Mangroves']['cost'][0]
            + cost_ben.imp_meas_future['Seawall']['cost'][0])
        self.assertAlmostEqual(new_cb.imp_meas_future[new_name]['cost'][1], 1)
        self.assertTrue(np.allclose(
            new_cb.imp_meas_future[new_name]['efc'].impact,
            new_cb.imp_meas_future[new_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_future[new_name]['risk_transf'], 0)

        self.assertAlmostEqual(new_cb.benefit[new_name], 51781337529.07264, places=3)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[new_name], 0.19679962474434248)

    def test_combine_current_pass(self):
        """Test combine_measures with only future"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018
        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, future_year=2040, risk_func=risk_aai_agg,
                      imp_time_depen=None, save_imp=True)

        new_name = 'combine'
        new_color = np.array([0.1, 0.1, 0.1])
        new_cb = cost_ben.combine_measures(['Mangroves', 'Seawall'], new_name, new_color,
                                           entity.disc_rates, imp_time_depen=None,
                                           risk_func=risk_aai_agg)

        self.assertTrue(np.allclose(new_cb.color_rgb[new_name], new_color))
        self.assertEqual(len(new_cb.imp_meas_present), 0)
        new_imp = cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event
        new_imp += cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Seawall']['impact'].at_event
        new_imp = np.maximum(cost_ben.imp_meas_future['no measure']['impact'].at_event - new_imp,
                             0)
        self.assertTrue(np.allclose(new_cb.imp_meas_future[new_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_future[new_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(
            new_cb.imp_meas_future[new_name]['cost'][0],
            cost_ben.imp_meas_future['Mangroves']['cost'][0]
            + cost_ben.imp_meas_future['Seawall']['cost'][0])
        self.assertAlmostEqual(new_cb.imp_meas_future[new_name]['cost'][1], 1)
        self.assertTrue(np.allclose(
            new_cb.imp_meas_future[new_name]['efc'].impact,
            new_cb.imp_meas_future[new_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_future[new_name]['risk_transf'], 0)
        self.assertAlmostEqual(new_cb.benefit[new_name], 51781337529.07264, places=3)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[new_name], 0.19679962474434248)

    def test_apply_transf_current_pass(self):
        """Test apply_risk_transfer with only future"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018
        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, future_year=2040, risk_func=risk_aai_agg,
                      imp_time_depen=None, save_imp=True)

        new_name = 'combine'
        new_color = np.array([0.1, 0.1, 0.1])
        risk_transf = (1.0e7, 15.0e11, 1)
        new_cb = cost_ben.combine_measures(['Mangroves', 'Seawall'], new_name, new_color,
                                           entity.disc_rates, imp_time_depen=None,
                                           risk_func=risk_aai_agg)
        new_cb.apply_risk_transfer(new_name, risk_transf[0], risk_transf[1],
                                   entity.disc_rates, cost_fix=0, cost_factor=risk_transf[2],
                                   imp_time_depen=1,
                                   risk_func=risk_aai_agg)

        tr_name = 'risk transfer (' + new_name + ')'
        new_imp = cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event
        new_imp += cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Seawall']['impact'].at_event
        new_imp = np.maximum(cost_ben.imp_meas_future['no measure']['impact'].at_event - new_imp,
                             0)
        imp_layer = np.minimum(np.maximum(new_imp - risk_transf[0], 0), risk_transf[1])
        risk_transfer = np.sum(
            imp_layer * cost_ben.imp_meas_future['no measure']['impact'].frequency)
        new_imp = np.maximum(new_imp - imp_layer, 0)

        self.assertTrue(np.allclose(new_cb.color_rgb[new_name], new_color))
        self.assertEqual(len(new_cb.imp_meas_present), 0)
        self.assertTrue(np.allclose(new_cb.imp_meas_future[tr_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_future[tr_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(
            new_cb.cost_ben_ratio[tr_name] * new_cb.benefit[tr_name],
            32106013195.316242, places=3)
        self.assertTrue(np.allclose(
            new_cb.imp_meas_future[tr_name]['efc'].impact,
            new_cb.imp_meas_future[tr_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_future[tr_name]['risk_transf'], risk_transfer)
        # benefit = impact layer
        self.assertAlmostEqual(new_cb.benefit[tr_name], 32106013195.316242, 4)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[tr_name], 1)

    def test_apply_transf_cost_fact_pass(self):
        """Test apply_risk_transfer with only future annd cost factor"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018
        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, future_year=2040, risk_func=risk_aai_agg,
                      imp_time_depen=None, save_imp=True)

        new_name = 'combine'
        new_color = np.array([0.1, 0.1, 0.1])
        risk_transf = (1.0e7, 15.0e11, 2)
        new_cb = cost_ben.combine_measures(['Mangroves', 'Seawall'], new_name, new_color,
                                           entity.disc_rates, imp_time_depen=None,
                                           risk_func=risk_aai_agg)
        new_cb.apply_risk_transfer(new_name, risk_transf[0], risk_transf[1],
                                   entity.disc_rates, cost_fix=0, cost_factor=risk_transf[2],
                                   imp_time_depen=1, risk_func=risk_aai_agg)

        tr_name = 'risk transfer (' + new_name + ')'
        new_imp = cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event
        new_imp += cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Seawall']['impact'].at_event
        new_imp = np.maximum(cost_ben.imp_meas_future['no measure']['impact'].at_event - new_imp,
                             0)
        imp_layer = np.minimum(np.maximum(new_imp - risk_transf[0], 0), risk_transf[1])
        risk_transfer = np.sum(
            imp_layer * cost_ben.imp_meas_future['no measure']['impact'].frequency)
        new_imp = np.maximum(new_imp - imp_layer, 0)

        self.assertTrue(np.allclose(new_cb.color_rgb[new_name], new_color))
        self.assertEqual(len(new_cb.imp_meas_present), 0)
        self.assertTrue(np.allclose(new_cb.imp_meas_future[tr_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_future[tr_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[tr_name] * new_cb.benefit[tr_name],
                               risk_transf[2] * 32106013195.316242, 4)
        self.assertTrue(
            np.allclose(new_cb.imp_meas_future[tr_name]['efc'].impact,
                        new_cb.imp_meas_future[tr_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_future[tr_name]['risk_transf'], risk_transfer)
        # benefit = impact layer
        self.assertAlmostEqual(new_cb.benefit[tr_name], 32106013195.316242, 4)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[tr_name], risk_transf[2])

    def test_apply_transf_future_pass(self):
        """Test apply_risk_transfer with present and future"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018

        fut_ent = copy.deepcopy(entity)
        fut_ent.exposures.ref_year = 2040

        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, ent_future=fut_ent, risk_func=risk_aai_agg,
                      imp_time_depen=None, save_imp=True)

        new_name = 'combine'
        new_color = np.array([0.1, 0.1, 0.1])
        risk_transf = (1.0e7, 15.0e11, 1)
        new_cb = cost_ben.combine_measures(['Mangroves', 'Seawall'], new_name, new_color,
                                           entity.disc_rates, imp_time_depen=None,
                                           risk_func=risk_aai_agg)
        new_cb.apply_risk_transfer(new_name, risk_transf[0], risk_transf[1],
                                   entity.disc_rates, cost_fix=0, cost_factor=risk_transf[2],
                                   imp_time_depen=1, risk_func=risk_aai_agg)

        tr_name = 'risk transfer (' + new_name + ')'
        new_imp = cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Mangroves']['impact'].at_event
        new_imp += cost_ben.imp_meas_future['no measure']['impact'].at_event - \
            cost_ben.imp_meas_future['Seawall']['impact'].at_event
        new_imp = np.maximum(cost_ben.imp_meas_future['no measure']['impact'].at_event - new_imp,
                             0)
        imp_layer = np.minimum(np.maximum(new_imp - risk_transf[0], 0), risk_transf[1])
        risk_transfer = np.sum(
            imp_layer * cost_ben.imp_meas_future['no measure']['impact'].frequency)
        new_imp = np.maximum(new_imp - imp_layer, 0)

        self.assertTrue(np.allclose(new_cb.color_rgb[new_name], new_color))
        self.assertEqual(len(new_cb.imp_meas_present), 3)
        self.assertTrue(np.allclose(new_cb.imp_meas_future[tr_name]['impact'].at_event, new_imp))
        self.assertTrue(np.allclose(new_cb.imp_meas_present[tr_name]['impact'].at_event, new_imp))
        self.assertAlmostEqual(
            new_cb.imp_meas_future[tr_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(
            new_cb.imp_meas_present[tr_name]['risk'],
            np.sum(new_imp * cost_ben.imp_meas_future['no measure']['impact'].frequency), 5)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[tr_name] * new_cb.benefit[tr_name],
                               69715165679.7042, places=3)
        self.assertTrue(
            np.allclose(new_cb.imp_meas_future[tr_name]['efc'].impact,
                        new_cb.imp_meas_future[tr_name]['impact'].calc_freq_curve().impact))
        self.assertAlmostEqual(new_cb.imp_meas_future[tr_name]['risk_transf'], risk_transfer)
        # benefit = impact layer
        self.assertAlmostEqual(new_cb.benefit[tr_name], 69715165679.7042, 4)
        self.assertAlmostEqual(new_cb.cost_ben_ratio[tr_name], 1)

    def test_remove_measure(self):
        """Test remove_measure method"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018
        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, future_year=2040, risk_func=risk_aai_agg,
                      imp_time_depen=None, save_imp=True)

        to_remove = 'Mangroves'
        self.assertTrue(to_remove in cost_ben.benefit.keys())
        cost_ben.remove_measure(to_remove)
        self.assertTrue(to_remove not in cost_ben.color_rgb.keys())
        self.assertTrue(to_remove not in cost_ben.benefit.keys())
        self.assertTrue(to_remove not in cost_ben.cost_ben_ratio.keys())
        self.assertTrue(to_remove not in cost_ben.imp_meas_future.keys())
        self.assertTrue(to_remove not in cost_ben.imp_meas_present.keys())
        self.assertEqual(len(cost_ben.imp_meas_present), 0)
        self.assertEqual(len(cost_ben.imp_meas_future), 4)
        self.assertEqual(len(cost_ben.color_rgb), 4)
        self.assertEqual(len(cost_ben.cost_ben_ratio), 3)
        self.assertEqual(len(cost_ben.benefit), 3)

class TestCalc(unittest.TestCase):
    """Test calc"""

    def test_calc_change_pass(self):
        """Test calc with future change"""
        # present
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.exposures.gdf.rename(columns={'impf_': 'impf_TC'}, inplace=True)
        entity.check()
        entity.exposures.ref_year = 2018

        # future
        ent_future = Entity.from_excel(ENT_DEMO_FUTURE)
        ent_future.check()
        ent_future.exposures.ref_year = 2040

        haz_future = copy.deepcopy(hazard)
        haz_future.intensity.data += 25

        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, haz_future, ent_future)

        self.assertEqual(cost_ben.present_year, 2018)
        self.assertEqual(cost_ben.future_year, 2040)
        self.assertAlmostEqual(cost_ben.tot_climate_risk, 5.768659152882021e+11, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_present['no measure']['risk'],
                               6.51220115756442e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Mangroves']['risk'],
                               4.850407096284983e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Beach nourishment']['risk'],
                               5.188921355413834e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Seawall']['risk'],
                               4.736400526119911e+09, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_present['Building code']['risk'],
                               4.884150868173321e+09, places=3)

        self.assertAlmostEqual(cost_ben.imp_meas_future['no measure']['risk'],
                               5.9506659786664024e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Mangroves']['risk'],
                               4.826231151473135e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Beach nourishment']['risk'],
                               5.0647250923231674e+10, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Seawall']['risk'],
                               21089567135.7345, places=3)
        self.assertAlmostEqual(cost_ben.imp_meas_future['Building code']['risk'],
                               4.462999483999791e+10, places=3)

        self.assertAlmostEqual(cost_ben.benefit['Mangroves'], 113345027690.81276, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Beach nourishment'], 89444869971.53653, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Seawall'], 347977469896.1333, places=2)
        self.assertAlmostEqual(cost_ben.benefit['Building code'], 144216478822.05154, places=2)

        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Mangroves'], 0.011573232523528404)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Beach nourishment'], 0.01931916274851638)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Seawall'], 0.025515385913577368)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Building code'], 0.06379298728650741)

        self.assertAlmostEqual(cost_ben.tot_climate_risk, 576865915288.2021, places=3)

    def test_calc_no_change_pass(self):
        """Test calc without future change"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        entity = Entity.from_excel(ENT_DEMO_TODAY)
        entity.check()
        entity.exposures.ref_year = 2018
        cost_ben = CostBenefit()
        cost_ben.calc(hazard, entity, future_year=2040)

        self.assertEqual(cost_ben.imp_meas_present, dict())
        self.assertEqual(len(cost_ben.imp_meas_future), 5)
        self.assertEqual(cost_ben.present_year, 2018)
        self.assertEqual(cost_ben.future_year, 2040)

        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Mangroves'], 0.04230714690616641)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Beach nourishment'], 0.06998836431681373)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Seawall'], 0.2679741183248266)
        self.assertAlmostEqual(cost_ben.cost_ben_ratio['Building code'], 0.30286828677985717)

        self.assertAlmostEqual(cost_ben.benefit['Mangroves'], 3.100583368954022e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Beach nourishment'],
                               2.468981832719974e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Seawall'], 3.3132973770502796e+10, places=3)
        self.assertAlmostEqual(cost_ben.benefit['Building code'], 3.0376240767284798e+10, places=3)

        self.assertAlmostEqual(cost_ben.tot_climate_risk, 1.2150496306913972e+11, places=3)

class TestRiskFuncs(unittest.TestCase):
    """Test risk functions definitions"""

    def test_impact(self):
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        impact = ImpactCalc(ent.exposures, ent.impact_funcs, hazard).impact()
        return impact

    def test_risk_aai_agg_pass(self):
        """Test risk_aai_agg"""
        impact = self.test_impact()
        risk = risk_aai_agg(impact)
        self.assertAlmostEqual(6.512201157564421e+09, risk, 5)
        self.assertTrue(np.isclose(6.512201157564421e+09, risk))

    def test_risk_rp_100_pass(self):
        """Test risk_rp_100"""
        impact = self.test_impact()
        exc_freq = impact.calc_freq_curve([100])

        risk = risk_rp_100(impact)
        self.assertAlmostEqual(exc_freq.impact[0], risk)

    def test_risk_rp_200_pass(self):
        """Test risk_rp_200"""
        impact = self.test_impact()
        exc_freq = impact.calc_freq_curve([250])

        risk = risk_rp_250(impact)
        self.assertAlmostEqual(exc_freq.impact[0], risk)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRiskFuncs)
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSteps))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
