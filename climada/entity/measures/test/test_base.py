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

Test MeasureSet and Measure classes.
"""
import unittest
import copy
from pathlib import Path

import numpy as np

from climada import CONFIG
from climada.hazard.base import Hazard
from climada.entity.entity_def import Entity
from climada.entity.exposures.base import Exposures, INDICATOR_IMPF
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.measures.measure_set import MeasureSet
from climada.entity.measures.base import Measure, IMPF_ID_FACT
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5
import climada.util.coordinates as u_coord
import climada.hazard.test as hazard_test
import climada.entity.exposures.test as exposures_test

DATA_DIR = CONFIG.measures.test_data.dir()

HAZ_TEST_MAT = Path(hazard_test.__file__).parent / 'data' / 'atl_prob_no_name.mat'
ENT_TEST_MAT = Path(exposures_test.__file__).parent / 'data' / 'demo_today.mat'

class TestApply(unittest.TestCase):
    """Test implement measures functions."""
    def test_change_imp_func_pass(self):
        """Test _change_imp_func"""
        meas = MeasureSet.from_mat(ENT_TEST_MAT)
        act_1 = meas.get_measure(name='Mangroves')[0]

        haz_type = 'XX'
        idx = 1
        intensity = np.arange(10, 100, 10)
        intensity[0] = 0.
        intensity[-1] = 100.
        mdd = np.array([0.0, 0.0, 0.021857142857143, 0.035887500000000,
                               0.053977415307403, 0.103534246575342, 0.180414000000000,
                               0.410796000000000, 0.410796000000000])
        paa = np.array([0, 0.005000000000000, 0.042000000000000, 0.160000000000000,
                               0.398500000000000, 0.657000000000000, 1.000000000000000,
                               1.000000000000000, 1.000000000000000])
        imp_tc = ImpactFunc(haz_type, idx, intensity, mdd, paa)
        imp_set = ImpactFuncSet([imp_tc])
        new_imp = act_1._change_imp_func(imp_set).get_func('XX')[0]

        self.assertTrue(np.array_equal(new_imp.intensity, np.array([4., 24., 34., 44.,
            54., 64., 74., 84., 104.])))
        self.assertTrue(np.array_equal(new_imp.mdd, np.array([0, 0, 0.021857142857143, 0.035887500000000,
            0.053977415307403, 0.103534246575342, 0.180414000000000, 0.410796000000000, 0.410796000000000])))
        self.assertTrue(np.array_equal(new_imp.paa, np.array([0, 0.005000000000000, 0.042000000000000,
            0.160000000000000, 0.398500000000000, 0.657000000000000, 1.000000000000000,
            1.000000000000000, 1.000000000000000])))
        self.assertFalse(id(new_imp) == id(imp_tc))

    def test_cutoff_hazard_pass(self):
        """Test _cutoff_hazard_damage"""
        meas = MeasureSet.from_mat(ENT_TEST_MAT)
        act_1 = meas.get_measure(name='Seawall')[0]

        haz = Hazard.from_mat(HAZ_TEST_MAT)
        exp = Exposures.from_mat(ENT_TEST_MAT)
        exp.gdf.rename(columns={'impf': 'impf_TC'}, inplace=True)
        exp.check()
        exp.assign_centroids(haz)

        imp_set = ImpactFuncSet.from_mat(ENT_TEST_MAT)

        new_haz = act_1._cutoff_hazard_damage(exp, imp_set, haz)

        self.assertFalse(id(new_haz) == id(haz))

        pos_no_null = np.array([6249, 7697, 9134, 13500, 13199, 5944, 9052, 9050, 2429,
                                5139, 9053, 7102, 4096, 1070, 5948, 1076, 5947, 7432,
                                5949, 11694, 5484, 6246, 12147, 778, 3326, 7199, 12498,
                               11698, 6245, 5327, 4819, 8677, 5970, 7101, 779, 3894,
                                9051, 5976, 3329, 5978, 4282, 11697, 7193, 5351, 7310,
                                7478, 5489, 5526, 7194, 4283, 7191, 5328, 4812, 5528,
                                5527, 5488, 7475, 5529, 776, 5758, 4811, 6223, 7479,
                                7470, 5480, 5325, 7477, 7318, 7317, 11696, 7313, 13165,
                                6221])
        all_haz = np.arange(haz.intensity.shape[0])
        all_haz[pos_no_null] = -1
        pos_null = np.argwhere(all_haz > 0).reshape(-1)
        for i_ev in pos_null:
            self.assertEqual(new_haz.intensity[i_ev, :].max(), 0)


    def test_cutoff_hazard_region_pass(self):
        """Test _cutoff_hazard_damage in specific region"""
        meas = MeasureSet.from_mat(ENT_TEST_MAT)
        act_1 = meas.get_measure(name='Seawall')[0]
        act_1.exp_region_id = [1]

        haz = Hazard.from_mat(HAZ_TEST_MAT)
        exp = Exposures.from_mat(ENT_TEST_MAT)
        exp.gdf['region_id'] = np.zeros(exp.gdf.shape[0])
        exp.gdf.region_id.values[10:] = 1
        exp.check()
        exp.assign_centroids(haz)

        imp_set = ImpactFuncSet.from_mat(ENT_TEST_MAT)

        new_haz = act_1._cutoff_hazard_damage(exp, imp_set, haz)

        self.assertFalse(id(new_haz) == id(haz))

        pos_no_null = np.array([6249, 7697, 9134, 13500, 13199, 5944, 9052, 9050, 2429,
                                5139, 9053, 7102, 4096, 1070, 5948, 1076, 5947, 7432,
                                5949, 11694, 5484, 6246, 12147, 778, 3326, 7199, 12498,
                               11698, 6245, 5327, 4819, 8677, 5970, 7101, 779, 3894,
                                9051, 5976, 3329, 5978, 4282, 11697, 7193, 5351, 7310,
                                7478, 5489, 5526, 7194, 4283, 7191, 5328, 4812, 5528,
                                5527, 5488, 7475, 5529, 776, 5758, 4811, 6223, 7479,
                                7470, 5480, 5325, 7477, 7318, 7317, 11696, 7313, 13165,
                                6221])
        all_haz = np.arange(haz.intensity.shape[0])
        all_haz[pos_no_null] = -1
        pos_null = np.argwhere(all_haz > 0).reshape(-1)
        centr_null = np.unique(exp.gdf.centr_[exp.gdf.region_id == 0])
        for i_ev in pos_null:
            self.assertEqual(new_haz.intensity[i_ev, centr_null].max(), 0)

    def test_change_exposures_impf_pass(self):
        """Test _change_exposures_impf"""
        meas = Measure(
            imp_fun_map='1to3',
            haz_type='TC',
        )

        imp_set = ImpactFuncSet()

        intensity = np.arange(10, 100, 10)
        mdd = np.arange(10, 100, 10)
        paa = np.arange(10, 100, 10)
        imp_tc = ImpactFunc("TC", 1, intensity, mdd, paa)
        imp_set.append(imp_tc)

        mdd = np.arange(10, 100, 10) * 2
        paa = np.arange(10, 100, 10) * 2
        imp_tc = ImpactFunc("TC", 3, intensity, mdd, paa)

        exp = Exposures.from_hdf5(EXP_DEMO_H5)
        new_exp = meas._change_exposures_impf(exp)

        self.assertEqual(new_exp.ref_year, exp.ref_year)
        self.assertEqual(new_exp.value_unit, exp.value_unit)
        self.assertEqual(new_exp.tag.file_name, exp.tag.file_name)
        self.assertEqual(new_exp.tag.description, exp.tag.description)
        self.assertTrue(np.array_equal(new_exp.gdf.value.values, exp.gdf.value.values))
        self.assertTrue(np.array_equal(new_exp.gdf.latitude.values, exp.gdf.latitude.values))
        self.assertTrue(np.array_equal(new_exp.gdf.longitude.values, exp.gdf.longitude.values))
        self.assertTrue(np.array_equal(exp.gdf[INDICATOR_IMPF + 'TC'].values, np.ones(new_exp.gdf.shape[0])))
        self.assertTrue(np.array_equal(new_exp.gdf[INDICATOR_IMPF + 'TC'].values, np.ones(new_exp.gdf.shape[0]) * 3))

    def test_change_all_hazard_pass(self):
        """Test _change_all_hazard method"""
        meas = Measure(hazard_set=HAZ_DEMO_H5)

        ref_haz = Hazard.from_hdf5(HAZ_DEMO_H5)

        hazard = Hazard('TC')
        new_haz = meas._change_all_hazard(hazard)

        self.assertEqual(new_haz.tag.file_name, ref_haz.tag.file_name)
        self.assertEqual(new_haz.tag.haz_type, ref_haz.tag.haz_type)
        self.assertTrue(np.array_equal(new_haz.frequency, ref_haz.frequency))
        self.assertTrue(np.array_equal(new_haz.date, ref_haz.date))
        self.assertTrue(np.array_equal(new_haz.orig, ref_haz.orig))
        self.assertTrue(np.array_equal(new_haz.centroids.coord, ref_haz.centroids.coord))
        self.assertTrue(np.array_equal(new_haz.intensity.data, ref_haz.intensity.data))
        self.assertTrue(np.array_equal(new_haz.fraction.data, ref_haz.fraction.data))

    def test_change_all_exposures_pass(self):
        """Test _change_all_exposures method"""
        meas = Measure(exposures_set=EXP_DEMO_H5)

        ref_exp = Exposures.from_hdf5(EXP_DEMO_H5)

        exposures = Exposures()
        exposures.gdf['latitude'] = np.ones(10)
        exposures.gdf['longitude'] = np.ones(10)
        new_exp = meas._change_all_exposures(exposures)

        self.assertEqual(new_exp.ref_year, ref_exp.ref_year)
        self.assertEqual(new_exp.value_unit, ref_exp.value_unit)
        self.assertEqual(new_exp.tag.file_name, ref_exp.tag.file_name)
        self.assertEqual(new_exp.tag.description, ref_exp.tag.description)
        self.assertTrue(np.array_equal(new_exp.gdf.value.values, ref_exp.gdf.value.values))
        self.assertTrue(np.array_equal(new_exp.gdf.latitude.values, ref_exp.gdf.latitude.values))
        self.assertTrue(np.array_equal(new_exp.gdf.longitude.values, ref_exp.gdf.longitude.values))

    def test_not_filter_exposures_pass(self):
        """Test _filter_exposures method with []"""
        meas = Measure(exp_region_id=[])

        exp = Exposures()
        imp_set = ImpactFuncSet()
        haz = Hazard('TC')

        new_exp = Exposures()
        new_impfs = ImpactFuncSet()
        new_haz = Hazard('TC')

        res_exp, res_ifs, res_haz = meas._filter_exposures(exp, imp_set, haz,
                                                           new_exp, new_impfs, new_haz)

        self.assertTrue(res_exp is new_exp)
        self.assertTrue(res_ifs is new_impfs)
        self.assertTrue(res_haz is new_haz)

        self.assertTrue(res_exp is not exp)
        self.assertTrue(res_ifs is not imp_set)
        self.assertTrue(res_haz is not haz)

    def test_filter_exposures_pass(self):
        """Test _filter_exposures method with two values"""
        meas = Measure(
            exp_region_id=[3, 4],
            haz_type='TC',
        )

        exp = Exposures.from_mat(ENT_TEST_MAT)
        exp.gdf.rename(columns={'impf_': 'impf_TC', 'centr_': 'centr_TC'}, inplace=True)
        exp.gdf['region_id'] = np.ones(exp.gdf.shape[0])
        exp.gdf.region_id.values[:exp.gdf.shape[0] // 2] = 3
        exp.gdf.region_id[0] = 4
        exp.check()

        imp_set = ImpactFuncSet.from_mat(ENT_TEST_MAT)

        haz = Hazard.from_mat(HAZ_TEST_MAT)
        exp.assign_centroids(haz)

        new_exp = copy.deepcopy(exp)
        new_exp.gdf['value'] *= 3
        new_exp.gdf['impf_TC'].values[:20] = 2
        new_exp.gdf['impf_TC'].values[20:40] = 3
        new_exp.gdf['impf_TC'].values[40:] = 1

        new_ifs = copy.deepcopy(imp_set)
        new_ifs.get_func('TC')[1].intensity += 1
        ref_ifs = copy.deepcopy(new_ifs)

        new_haz = copy.deepcopy(haz)
        new_haz.intensity *= 4

        res_exp, res_ifs, res_haz = meas._filter_exposures(exp, imp_set, haz,
            new_exp.copy(deep=True), new_ifs, new_haz)

        # unchanged meta data
        self.assertEqual(res_exp.ref_year, exp.ref_year)
        self.assertEqual(res_exp.value_unit, exp.value_unit)
        self.assertEqual(res_exp.tag.file_name, exp.tag.file_name)
        self.assertEqual(res_exp.tag.description, exp.tag.description)
        self.assertTrue(u_coord.equal_crs(res_exp.crs, exp.crs))
        self.assertFalse(hasattr(exp.gdf, "crs"))
        self.assertFalse(hasattr(res_exp.gdf, "crs"))

        # regions (that is just input data, no need for testing, but it makes the changed and unchanged parts obious)
        self.assertTrue(np.array_equal(res_exp.gdf.region_id.values[0], 4))
        self.assertTrue(np.array_equal(res_exp.gdf.region_id.values[1:25], np.ones(24) * 3))
        self.assertTrue(np.array_equal(res_exp.gdf.region_id.values[25:], np.ones(25)))

        # changed exposures
        self.assertTrue(np.array_equal(res_exp.gdf.value.values[:25], new_exp.gdf.value.values[:25]))
        self.assertTrue(np.all(np.not_equal(res_exp.gdf.value.values[:25], exp.gdf.value.values[:25])))
        self.assertTrue(np.all(np.not_equal(res_exp.gdf.impf_TC.values[:25], new_exp.gdf.impf_TC.values[:25])))
        self.assertTrue(np.array_equal(res_exp.gdf.latitude.values[:25], new_exp.gdf.latitude.values[:25]))
        self.assertTrue(np.array_equal(res_exp.gdf.longitude.values[:25], new_exp.gdf.longitude.values[:25]))

        # unchanged exposures
        self.assertTrue(np.array_equal(res_exp.gdf.value.values[25:], exp.gdf.value.values[25:]))
        self.assertTrue(np.all(np.not_equal(res_exp.gdf.value.values[25:], new_exp.gdf.value.values[25:])))
        self.assertTrue(np.array_equal(res_exp.gdf.impf_TC.values[25:], exp.gdf.impf_TC.values[25:]))
        self.assertTrue(np.array_equal(res_exp.gdf.latitude.values[25:], exp.gdf.latitude.values[25:]))
        self.assertTrue(np.array_equal(res_exp.gdf.longitude.values[25:], exp.gdf.longitude.values[25:]))

        # unchanged impact functions
        self.assertEqual(list(res_ifs.get_func().keys()), [meas.haz_type])
        self.assertEqual(res_ifs.get_func()[meas.haz_type][1].id, imp_set.get_func()[meas.haz_type][1].id)
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][1].intensity,
                                       imp_set.get_func()[meas.haz_type][1].intensity))
        self.assertEqual(res_ifs.get_func()[meas.haz_type][3].id, imp_set.get_func()[meas.haz_type][3].id)
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][3].intensity,
                                       imp_set.get_func()[meas.haz_type][3].intensity))

        # changed impact functions
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][1 + IMPF_ID_FACT].intensity,
                        ref_ifs.get_func()[meas.haz_type][1].intensity))
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][1 + IMPF_ID_FACT].paa,
                        ref_ifs.get_func()[meas.haz_type][1].paa))
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][1 + IMPF_ID_FACT].mdd,
                        ref_ifs.get_func()[meas.haz_type][1].mdd))
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][3 + IMPF_ID_FACT].intensity,
                        ref_ifs.get_func()[meas.haz_type][3].intensity))
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][3 + IMPF_ID_FACT].paa,
                        ref_ifs.get_func()[meas.haz_type][3].paa))
        self.assertTrue(np.array_equal(res_ifs.get_func()[meas.haz_type][3 + IMPF_ID_FACT].mdd,
                        ref_ifs.get_func()[meas.haz_type][3].mdd))

        # unchanged hazard
        self.assertTrue(np.array_equal(res_haz.intensity[:, :36].toarray(),
                        haz.intensity[:, :36].toarray()))
        self.assertTrue(np.array_equal(res_haz.intensity[:, 37:46].toarray(),
                        haz.intensity[:, 37:46].toarray()))
        self.assertTrue(np.array_equal(res_haz.intensity[:, 47:].toarray(),
                        haz.intensity[:, 47:].toarray()))

        # changed hazard
        self.assertTrue(np.array_equal(res_haz.intensity[[36, 46]].toarray(),
                        new_haz.intensity[[36, 46]].toarray()))

    def test_apply_ref_pass(self):
        """Test apply method: apply all measures but insurance"""
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        entity.check()

        new_exp, new_ifs, new_haz = entity.measures.get_measure('TC', 'Mangroves').apply(entity.exposures,
            entity.impact_funcs, hazard)

        self.assertTrue(new_exp is entity.exposures)
        self.assertTrue(new_haz is hazard)
        self.assertFalse(new_ifs is entity.impact_funcs)

        new_imp = new_ifs.get_func('TC')[0]
        self.assertTrue(np.array_equal(new_imp.intensity, np.array([4., 24., 34., 44.,
            54., 64., 74., 84., 104.])))
        self.assertTrue(np.allclose(new_imp.mdd, np.array([0, 0, 0.021857142857143, 0.035887500000000,
            0.053977415307403, 0.103534246575342, 0.180414000000000, 0.410796000000000, 0.410796000000000])))
        self.assertTrue(np.allclose(new_imp.paa, np.array([0, 0.005000000000000, 0.042000000000000,
            0.160000000000000, 0.398500000000000, 0.657000000000000, 1.000000000000000,
            1.000000000000000, 1.000000000000000])))

        new_imp = new_ifs.get_func('TC')[1]
        self.assertTrue(np.array_equal(new_imp.intensity, np.array([4., 24., 34., 44.,
            54., 64., 74., 84., 104.])))
        self.assertTrue(np.allclose(new_imp.mdd, np.array([0, 0, 0, 0.025000000000000,
            0.054054054054054, 0.104615384615385, 0.211764705882353, 0.400000000000000, 0.400000000000000])))
        self.assertTrue(np.allclose(new_imp.paa, np.array([0, 0.004000000000000, 0, 0.160000000000000,
            0.370000000000000, 0.650000000000000, 0.850000000000000, 1.000000000000000,
            1.000000000000000])))

    def test_calc_impact_pass(self):
        """Test calc_impact method: apply all measures but insurance"""

        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.exposures.gdf.rename(columns={'impf': 'impf_TC'}, inplace=True)
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        entity.measures.get_measure(name='Mangroves', haz_type='TC').haz_type = 'TC'
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        entity.check()

        imp, risk_transf = entity.measures.get_measure('TC', 'Mangroves').calc_impact(
                entity.exposures, entity.impact_funcs, hazard)

        self.assertAlmostEqual(imp.aai_agg, 4.850407096284983e+09, delta=1)
        self.assertAlmostEqual(imp.at_event[0], 0)
        self.assertAlmostEqual(imp.at_event[12], 1.470194187501225e+07)
        self.assertAlmostEqual(imp.at_event[41], 4.7226357936631286e+08)
        self.assertAlmostEqual(imp.at_event[11890], 1.742110428135755e+07)
        self.assertTrue(np.array_equal(imp.coord_exp[:, 0], entity.exposures.gdf.latitude))
        self.assertTrue(np.array_equal(imp.coord_exp[:, 1], entity.exposures.gdf.longitude))
        self.assertAlmostEqual(imp.eai_exp[0], 1.15677655725858e+08)
        self.assertAlmostEqual(imp.eai_exp[-1], 7.528669956120645e+07)
        self.assertAlmostEqual(imp.tot_value, 6.570532945599105e+11)
        self.assertEqual(imp.unit, 'USD')
        self.assertEqual(imp.imp_mat.shape, (0, 0))
        self.assertTrue(np.array_equal(imp.event_id, hazard.event_id))
        self.assertTrue(np.array_equal(imp.date, hazard.date))
        self.assertEqual(imp.event_name, hazard.event_name)
        self.assertEqual(imp.tag['exp'].file_name, entity.exposures.tag.file_name)
        self.assertEqual(imp.tag['haz'].file_name, hazard.tag.file_name)
        self.assertEqual(imp.tag['impf_set'].file_name, entity.impact_funcs.tag.file_name)
        self.assertEqual(risk_transf.aai_agg, 0)


    def test_calc_impact_transf_pass(self):
        """Test calc_impact method: apply all measures and insurance"""

        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        entity = Entity.from_mat(ENT_TEST_MAT)
        entity.exposures.gdf.rename(columns={'impf': 'impf_TC'}, inplace=True)
        entity.measures._data['TC'] = entity.measures._data.pop('XX')
        for meas in entity.measures.get_measure('TC'):
            meas.haz_type = 'TC'
        meas = entity.measures.get_measure(name='Beach nourishment', haz_type='TC')
        meas.haz_type = 'TC'
        meas.hazard_inten_imp = (1, 0)
        meas.mdd_impact = (1, 0)
        meas.paa_impact = (1, 0)
        meas.risk_transf_attach = 5.0e8
        meas.risk_transf_cover = 1.0e9
        entity.check()

        imp, risk_transf = entity.measures.get_measure(name='Beach nourishment', haz_type='TC').calc_impact(
                entity.exposures, entity.impact_funcs, hazard)

        self.assertAlmostEqual(imp.aai_agg, 6.280804242609713e+09)
        self.assertAlmostEqual(imp.at_event[0], 0)
        self.assertAlmostEqual(imp.at_event[12], 8.648764833437817e+07)
        self.assertAlmostEqual(imp.at_event[41], 500000000)
        self.assertAlmostEqual(imp.at_event[11890], 6.498096646836635e+07)
        self.assertTrue(np.array_equal(imp.coord_exp, np.array([])))
        self.assertTrue(np.array_equal(imp.eai_exp, np.array([])))
        self.assertAlmostEqual(imp.tot_value, 6.570532945599105e+11)
        self.assertEqual(imp.unit, 'USD')
        self.assertEqual(imp.imp_mat.shape, (0, 0))
        self.assertTrue(np.array_equal(imp.event_id, hazard.event_id))
        self.assertTrue(np.array_equal(imp.date, hazard.date))
        self.assertEqual(imp.event_name, hazard.event_name)
        self.assertEqual(imp.tag['exp'].file_name, entity.exposures.tag.file_name)
        self.assertEqual(imp.tag['haz'].file_name, hazard.tag.file_name)
        self.assertEqual(imp.tag['impf_set'].file_name, entity.impact_funcs.tag.file_name)
        self.assertEqual(risk_transf.aai_agg, 2.3139691495470852e+08)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestApply)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
