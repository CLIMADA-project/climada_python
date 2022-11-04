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

Test Impact class.
"""
import unittest
from unittest.mock import create_autospec, MagicMock, call, patch
import numpy as np
from scipy import sparse
import geopandas as gpd
from copy import deepcopy
from pathlib import Path

from climada import CONFIG
from climada.entity.entity_def import Entity
from climada.entity import Exposures, ImpactFuncSet
from climada.hazard.base import Hazard
from climada.engine import ImpactCalc, Impact
from climada.engine.impact_calc import LOGGER as ILOG
from climada.util.constants import ENT_DEMO_TODAY, DEMO_DIR
from climada.util.api_client import Client
from climada.util.config import Config

from climada.test import get_test_file


ENT = Entity.from_excel(ENT_DEMO_TODAY)
HAZ = Hazard.from_hdf5(get_test_file('test_tc_florida'))

DATA_FOLDER = DEMO_DIR / 'test-results'
DATA_FOLDER.mkdir(exist_ok=True)


def check_impact(self, imp, haz, exp, aai_agg, eai_exp, at_event, imp_mat_array=None):
    """Test properties of imapcts"""
    self.assertEqual(len(haz.event_id), len(imp.at_event))
    self.assertIsInstance(imp, Impact)
    np.testing.assert_allclose(imp.coord_exp[:,0], exp.gdf.latitude)
    np.testing.assert_allclose(imp.coord_exp[:,1], exp.gdf.longitude)
    self.assertAlmostEqual(imp.aai_agg, aai_agg, 3)
    np.testing.assert_allclose(imp.eai_exp, eai_exp, rtol=1e-5)
    np.testing.assert_allclose(imp.at_event, at_event, rtol=1e-5)
    if imp_mat_array is not None:
        np.testing.assert_allclose(imp.imp_mat.toarray().ravel(),
                                   imp_mat_array.ravel())


class TestImpactCalc(unittest.TestCase):
    """Test Impact calc methods"""
    def test_init(self):
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        self.assertEqual(icalc.n_exp_pnt, ENT.exposures.gdf.shape[0])
        self.assertEqual(icalc.n_events, HAZ.size)
        self.assertTrue(ENT.exposures.gdf.equals(icalc.exposures.gdf))
        np.testing.assert_array_equal(HAZ.event_id, icalc.hazard.event_id)
        np.testing.assert_array_equal(HAZ.event_name, icalc.hazard.event_name)

    def test_metrics(self):
        """Test methods to get impact metrics"""
        mat = sparse.csr_matrix(np.array(
            [[1, 0, 1],
             [2, 2, 0]]
            ))
        freq = np.array([1, 1/10])
        at_event = ImpactCalc.at_event_from_mat(mat)
        eai_exp = ImpactCalc.eai_exp_from_mat(mat, freq)
        aai_agg = ImpactCalc.aai_agg_from_eai_exp(eai_exp)
        np.testing.assert_array_equal(at_event, [2, 4])
        np.testing.assert_array_equal(eai_exp, [1.2, 0.2, 1])
        self.assertEqual(aai_agg, 2.4)

        ae, eai, aai = ImpactCalc.risk_metrics(mat, freq)
        self.assertEqual(aai, aai_agg)
        np.testing.assert_array_equal(at_event, ae)
        np.testing.assert_array_equal(eai_exp, eai)

    def test_apply_cover_to_mat(self):
        """Test methods to get insured metrics"""
        mat = sparse.csr_matrix(np.array(
            [[1, 0, 1],
             [2, 2, 0]]
            ))
        cover = np.array([0, 1, 10])
        imp = ImpactCalc.apply_cover_to_mat(mat, cover)
        np.testing.assert_array_equal(
            imp.todense(), np.array([[0, 0, 1], [0, 1, 0]])
            )

    def test_calc_impact_TC_pass(self):
        """Test compute impact"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        impact = icalc.impact()
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(7.076504723057620e+10, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[25], 6)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[49], 6)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)

        x = 0.6
        HAZf = deepcopy(HAZ)
        HAZf.fraction = HAZ.intensity.copy()
        HAZf.fraction.data.fill(x)
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZf)
        impact = icalc.impact(assign_centroids=False)
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(1.472482938320243e+08 * x, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(7.076504723057620e+10 * x, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08 * x, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(1.373490457046383e+08 * x, impact.eai_exp[25], 6)
        self.assertAlmostEqual(1.066837260150042e+08 * x, impact.eai_exp[49], 6)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09 * x, impact.aai_agg, 5)

    def test_calc_impact_RF_pass(self):
        haz = Hazard.from_hdf5(get_test_file('test_hazard_US_flood_random_locations'))
        exp = Exposures.from_hdf5(get_test_file('test_exposure_US_flood_random_locations'))
        impf_set = ImpactFuncSet.from_excel(Path(__file__).parent / 'data' / 'flood_imp_func_set.xls')
        icalc = ImpactCalc(exp, impf_set, haz)
        impact = icalc.impact(assign_centroids=False)
        aai_agg = 161436.05112960344
        eai_exp = np.array([
            1.61159701e+05, 1.33742847e+02, 0.00000000e+00, 4.21352988e-01,
            1.42185609e+02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00
            ])
        at_event = np.array([
            0.00000000e+00, 0.00000000e+00, 9.85233619e+04, 3.41245461e+04,
            7.73566566e+07, 0.00000000e+00, 0.00000000e+00
            ])
        imp_mat_array = np.array([
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 6.41965663e+04, 0.00000000e+00, 2.02249434e+02,
             3.41245461e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             3.41245461e+04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [7.73566566e+07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]
            ])
        check_impact(self, impact, haz, exp, aai_agg, eai_exp, at_event, imp_mat_array)

    def test_empty_impact(self):
        """Check that empty impact is returned if no centroids match the exposures"""
        exp = ENT.exposures.copy()
        exp.gdf['centr_TC'] = -1
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        impact = icalc.impact(assign_centroids=False)
        aai_agg = 0.0
        eai_exp = np.zeros(len(exp.gdf))
        at_event = np.zeros(HAZ.size)
        check_impact(self, impact, HAZ, exp, aai_agg, eai_exp, at_event, None)

        impact = icalc.impact(save_mat=True, assign_centroids=False)
        imp_mat_array = sparse.csr_matrix((HAZ.size, len(exp.gdf))).toarray()
        check_impact(self, impact, HAZ, exp, aai_agg, eai_exp, at_event, imp_mat_array)

    def test_single_event_impact(self):
        """Check impact for single event"""
        haz = HAZ.select([1])
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, haz)
        impact = icalc.impact()
        aai_agg = 0.0
        eai_exp = np.zeros(len(ENT.exposures.gdf))
        at_event = np.array([0])
        check_impact(self, impact, haz, ENT.exposures, aai_agg, eai_exp, at_event, None)
        impact = icalc.impact(save_mat=True, assign_centroids=False)
        imp_mat_array = sparse.csr_matrix((haz.size, len(ENT.exposures.gdf))).toarray()
        check_impact(self, impact, haz, ENT.exposures, aai_agg, eai_exp, at_event, imp_mat_array)


    def test_calc_impact_save_mat_pass(self):
        """Test compute impact with impact matrix"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        impact = icalc.impact()

        self.assertIsInstance(impact.imp_mat, sparse.csr_matrix)
        self.assertEqual(impact.imp_mat.shape, (HAZ.event_id.size,
                                                ENT.exposures.gdf.value.size))
        np.testing.assert_array_almost_equal_nulp(
            np.array(impact.imp_mat.sum(axis=1)).ravel(), impact.at_event, nulp=5)
        np.testing.assert_array_almost_equal_nulp(
            np.sum(impact.imp_mat.toarray() * impact.frequency[:, None], axis=0).reshape(-1),
            impact.eai_exp)

        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(7.076504723057620e+10, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[25], 6)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[49], 6)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)

    def test_calc_insured_impact_pass(self):
        """Test compute insured impact"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        with self.assertLogs(ILOG, level='INFO') as logs:
            impact = icalc.impact()
        self.assertIn("cover and/or deductible columns detected", logs.output[1])
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(62989686, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(657053294, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(3072092, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(2778593, impact.eai_exp[25], delta=1)
        self.assertAlmostEqual(2716548, impact.eai_exp[49], delta=1)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(143180396, impact.aai_agg, delta=1)

    def test_calc_insured_impact_no_cover(self):
        """Test compute insured impact"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        with self.assertLogs(ILOG, level='INFO') as logs:
            impact = icalc.impact(ignore_cover=True)
        self.assertIn("cover and/or deductible columns detected", logs.output[1])
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(147188636, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(70761282665, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(151847975, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(137341654, impact.eai_exp[25], delta=1)
        self.assertAlmostEqual(106676521, impact.eai_exp[49], delta=1)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6511839456, impact.aai_agg, delta=1)

    def test_calc_insured_impact_no_deductible(self):
        """Test compute insured impact"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        with self.assertLogs(ILOG, level='INFO') as logs:
            impact = icalc.impact(ignore_deductible=True)
        self.assertIn("cover and/or deductible columns detected", logs.output[1])
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(62989686, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(657053294, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(3072413, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(2778914, impact.eai_exp[25], delta=1)
        self.assertAlmostEqual(2716831, impact.eai_exp[49], delta=1)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(143195738, impact.aai_agg, delta=1)

    def test_calc_insured_impact_no_insurance(self):
        """Test compute insured impact"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        with self.assertLogs(ILOG, level='INFO') as logs:
            impact = icalc.impact(ignore_cover=True, ignore_deductible=True)
        self.assertEqual(logs.output, [
            "INFO:climada.engine.impact_calc:Calculating impact for 150 assets (>0) and 14450 events."
        ])
        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(147248293, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(70765047230, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(151855367, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(137349045, impact.eai_exp[25], delta=1)
        self.assertAlmostEqual(106683726, impact.eai_exp[49], delta=1)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6512201157, impact.aai_agg, delta=1)

    def test_calc_insured_impact_save_mat_pass(self):
        """Test compute impact with impact matrix"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        impact = icalc.impact(save_mat=True)

        self.assertIsInstance(impact.imp_mat, sparse.csr_matrix)
        self.assertEqual(impact.imp_mat.shape, (HAZ.event_id.size,
                                                ENT.exposures.gdf.value.size))
        np.testing.assert_array_almost_equal_nulp(
            np.array(impact.imp_mat.sum(axis=1)).ravel(), impact.at_event, nulp=5)
        np.testing.assert_array_almost_equal_nulp(
            np.sum(impact.imp_mat.toarray() * impact.frequency[:, None], axis=0).reshape(-1),
            impact.eai_exp)

        self.assertEqual(icalc.n_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[7225])
        self.assertAlmostEqual(62989686, impact.at_event[13809], delta=1)
        self.assertAlmostEqual(657053294, impact.at_event[12147], delta=1)
        self.assertEqual(0, impact.at_event[14449])
        self.assertEqual(icalc.n_exp_pnt, len(impact.eai_exp))
        self.assertAlmostEqual(3072092, impact.eai_exp[0], delta=1)
        self.assertAlmostEqual(2778593, impact.eai_exp[25], delta=1)
        self.assertAlmostEqual(2716548, impact.eai_exp[49], delta=1)
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(143180396, impact.aai_agg, delta=1)

    def test_minimal_exp_gdf(self):
        """Test obtain minimal exposures gdf"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        exp_min_gdf = icalc.minimal_exp_gdf('impf_TC', assign_centroids=True,
                                            ignore_cover=True, ignore_deductible=True)
        self.assertSetEqual(set(exp_min_gdf.columns),
                            set(['value', 'impf_TC', 'centr_TC']))
        np.testing.assert_array_equal(exp_min_gdf.value, ENT.exposures.gdf.value)
        np.testing.assert_array_equal(exp_min_gdf.impf_TC, ENT.exposures.gdf.impf_TC)
        np.testing.assert_array_equal(exp_min_gdf.centr_TC, ENT.exposures.gdf.centr_TC)

    def test_stitch_impact_matrix(self):
        """Check how sparse matrices from a generator are stitched together"""
        icalc = ImpactCalc(Exposures({'blank': [1, 2, 3, 4]}), ImpactFuncSet(), Hazard())
        icalc.hazard.event_id = np.array([1, 2, 3])
        icalc._orig_exp_idx = np.array([0, 1, 2, 3])

        imp_mat_gen = [
            (sparse.csr_matrix([[1.0, 1.0], [0.0, 1.0]]), np.array([0, 1])),
            (sparse.csr_matrix([[0.0, 0.0], [2.0, 2.0], [2.0, 2.0]]), np.array([1, 2])),
            (sparse.csr_matrix([[0.0], [0.0], [4.0]]), np.array([3])),
        ]
        mat = icalc.stitch_impact_matrix(imp_mat_gen)
        np.testing.assert_array_equal(
            mat.toarray(),
            [[1.0, 1.0, 0.0, 0.0], [0.0, 3.0, 2.0, 0.0], [0.0, 2.0, 2.0, 4.0]],
        )

    def test_apply_deductible_to_mat(self):
        """Test applying a deductible to an impact matrix"""
        hazard = create_autospec(HAZ)
        hazard.get_paa.return_value = sparse.csr_matrix([[1.0, 0.0], [0.1, 1.0]])

        mat = sparse.csr_matrix([[10.0, 20.0], [30.0, 40.0]])
        deductible = np.array([1.0, 0.5])

        centr_idx = np.ones(2)
        impf = None
        mat = ImpactCalc.apply_deductible_to_mat(mat, deductible, hazard, centr_idx, impf)
        np.testing.assert_array_equal(mat.toarray(), [[9.0, 20.0], [29.9, 39.5]])
        hazard.get_paa.assert_called_once_with(centr_idx, impf)

    def test_stitch_risk_metrics(self):
        """Test computing risk metrics from an impact matrix generator"""
        icalc = ImpactCalc(Exposures({'blank': [1, 2, 3]}), ImpactFuncSet(), Hazard())
        icalc.hazard.event_id = np.array([1, 2])
        icalc.hazard.frequency = np.array([2, 0.5])
        icalc._orig_exp_idx = np.array([0, 1, 2])

        # Matrices overlap at central exposure point
        imp_mat_gen = (
            (sparse.csc_matrix([[1.0, 0.0], [0.5, 1.0]]), np.array([0, 1])),
            (sparse.csc_matrix([[0.0, 2.0], [1.5, 1.0]]), np.array([1, 2])),
        )
        at_event, eai_exp, aai_agg = icalc.stitch_risk_metrics(imp_mat_gen)

        np.testing.assert_array_equal(at_event, [3.0, 4.0])
        np.testing.assert_array_equal(eai_exp, [2.25, 1.25, 4.5])
        self.assertEqual(aai_agg, 8.0)  # Sum of eai_exp


class TestImpactMatrixCalc(unittest.TestCase):
    """Verify the computation of the impact matrix"""

    def setUp(self):
        """Mock the methods called by 'impact_matrix'"""
        self.hazard = create_autospec(HAZ)
        self.hazard.get_mdr.return_value = sparse.csr_matrix(
            [[0.0, 0.5, -1.0], [1.0, 2.0, 1.0]]
        )
        self.hazard._get_fraction.return_value = sparse.csr_matrix(
            [[1.0, 1.0, 1.0], [-0.5, 0.5, 2.0]]
        )
        self.exposure_values = np.array([10.0, 20.0, -30.0])
        self.centroids = np.array([1, 2, 4])
        self.icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, self.hazard)

    def test_correct_calculation(self):
        """Assert that the calculation of the impact matrix is correct"""
        impact_matrix = self.icalc.impact_matrix(
            self.exposure_values, self.centroids, ENT.impact_funcs
        )
        np.testing.assert_array_equal(
            impact_matrix.toarray(), [[0.0, 10.0, 30.0], [-5.0, 20.0, -60.0]]
        )

        # Check if hazard methods were called with expected arguments
        with self.subTest("Internal call to hazard instance"):
            self.hazard.get_mdr.assert_called_once_with(
                self.centroids, ENT.impact_funcs
            )
            self.hazard._get_fraction.assert_called_once_with(self.centroids)

    def test_wrong_sizes(self):
        """Calling 'impact_matrix' with wrongly sized argument results in errors"""
        centroids = np.array([1, 2, 4, 5])  # Too long
        with self.assertRaises(ValueError):
            self.icalc.impact_matrix(self.exposure_values, centroids, ENT.impact_funcs)
        exposure_values = np.array([1.0])  # Too short
        with self.assertRaises(ValueError):
            self.icalc.impact_matrix(exposure_values, self.centroids, ENT.impact_funcs)


class TestImpactMatrixGenerator(unittest.TestCase):
    """Check the impact matrix generator"""

    def setUp(self):
        """"Initialize mocks"""
        # Alter the default config to enable chunking
        self._max_matrix_size = CONFIG.max_matrix_size.int()
        CONFIG.max_matrix_size = Config(val=1, root=CONFIG)

        # Mock the hazard
        self.hazard = create_autospec(HAZ)
        self.hazard.haz_type = "haz_type"
        self.hazard.centr_exp_col = "centr_col"
        self.hazard.size = 1

        # Mock the Impact function (set)
        self.impf = MagicMock(name="impact_function")
        self.impfset = create_autospec(ENT.impact_funcs)
        self.impfset.get_func.return_value = self.impf

        # Mock the impact matrix call
        self.icalc = ImpactCalc(ENT.exposures, self.impfset, self.hazard)
        self.icalc.impact_matrix = MagicMock()

        # Set up a dummy exposure dataframe
        self.exp_gdf = gpd.GeoDataFrame(
            {
                "impact_functions": [0, 11, 11],
                "centr_col": [0, 10, 20],
                "value": [0.0, 1.0, 2.0],
            }
        )

    def tearDown(self):
        """Reset the original config"""
        CONFIG.max_matrix_size = Config(val=self._max_matrix_size, root=CONFIG)

    def test_selection(self):
        """Verify the impact matrix generator returns the right values"""
        gen = self.icalc.imp_mat_gen(exp_gdf=self.exp_gdf, impf_col="impact_functions")
        out_list = [exp_idx for _, exp_idx in gen]

        np.testing.assert_array_equal(out_list, [[0], [1], [2]])

        # Verify calls
        self.impfset.get_func.assert_has_calls(
            [call(haz_type="haz_type", fun_id=0), call(haz_type="haz_type", fun_id=11),]
        )
        self.icalc.impact_matrix.assert_has_calls(
            [
                call(np.array([0.0]), np.array([0]), self.impf),
                call(np.array([1.0]), np.array([10]), self.impf),
                call(np.array([2.0]), np.array([20]), self.impf),
            ]
        )

    def test_chunking(self):
        """Verify that chunking works as expected"""
        # n_chunks = hazard.size * len(centr_idx) / max_size = 2 * 5 / 4 = 2.5
        CONFIG.max_matrix_size = Config(val=4, root=CONFIG)
        self.hazard.size = 2

        arr_len = 5
        exp_gdf = gpd.GeoDataFrame(
            {
                "impact_functions": np.zeros(arr_len, dtype=np.int64),
                "centr_col": np.array(list(range(arr_len))),
                "value": np.ones(arr_len, dtype=np.float64),
            }
        )
        gen = self.icalc.imp_mat_gen(exp_gdf=exp_gdf, impf_col="impact_functions")
        out_list = [exp_idx for _, exp_idx in gen]

        # Expect three chunks
        self.assertEqual(len(out_list[0]), 2)
        self.assertEqual(len(out_list[1]), 2)
        self.assertEqual(len(out_list[2]), 1)

    def test_chunk_error(self):
        """Assert that too large hazard results in error"""
        self.hazard.size = 2
        gen = self.icalc.imp_mat_gen(exp_gdf=self.exp_gdf, impf_col="impact_functions")
        with self.assertRaises(ValueError):
            list(gen)

    def test_empty_exp(self):
        """imp_mat_gen should return an empty iterator for an empty dataframe"""
        exp_gdf = gpd.GeoDataFrame({"impact_functions": [], "centr_col": [], "value": []})
        self.assertEqual(
            [],
            list(self.icalc.imp_mat_gen(exp_gdf=exp_gdf, impf_col="impact_functions")),
        )


class TestInsuredImpactMatrixGenerator(unittest.TestCase):
    """Verify the computation of the insured impact matrix"""
    def setUp(self):
        """"Initialize mocks"""
        hazard = create_autospec(HAZ)
        self.icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, hazard)
        self.icalc._orig_exp_idx = np.array([0, 1])
        self.icalc.hazard.centr_exp_col = "centr_col"
        self.icalc.hazard.haz_type = "haz_type"
        self.icalc.apply_deductible_to_mat = MagicMock(
            side_effect=["mat_deduct_1", "mat_deduct_2"]
        )
        self.icalc.apply_cover_to_mat = MagicMock(
            side_effect=["mat_cov_1", "mat_cov_2"]
        )
        self.icalc.impfset.get_func = MagicMock(side_effect=["impf_0", "impf_2"])

    def test_insured_mat_gen(self):
        """Test insured impact matrix generator"""
        exp_gdf = gpd.GeoDataFrame(
            {"impact_functions": [0, 2], "centr_col": [0, 10], "value": [1.0, 2.0],
             "deductible": [10.0, 20.0], "cover": [1.0, 100.0]}
        )
        imp_mat_gen = ((i, np.array([i])) for i in range(2))
        gen = self.icalc.insured_mat_gen(imp_mat_gen, exp_gdf, "impact_functions")
        out_list = list(gen)

        # Assert expected output
        self.assertEqual(len(out_list), 2)
        np.testing.assert_array_equal(
            [item[0] for item in out_list], ["mat_cov_1", "mat_cov_2"]
        )
        np.testing.assert_array_equal([item[1] for item in out_list], [[0], [1]])

        # Check if correct impf_id was selected
        self.icalc.impfset.get_func.assert_has_calls(
            [call(haz_type="haz_type", fun_id=0), call(haz_type="haz_type", fun_id=2),]
        )
        # Check if correct deductible and cent_idx were selected
        self.icalc.apply_deductible_to_mat.assert_has_calls(
            [
                call(0, np.array([10.0]), self.icalc.hazard, np.array([0]), "impf_0"),
                call(1, np.array([20.0]), self.icalc.hazard, np.array([10]), "impf_2"),
            ]
        )
        # Check if correct cover was selected
        self.icalc.apply_cover_to_mat.assert_has_calls(
            [
                call("mat_deduct_1", np.array([1.0])),
                call("mat_deduct_2", np.array([100.0])),
            ]
        )


class TestImpactMatrix(unittest.TestCase):
    """Test Impact matrix computation"""
    def setUp(self):
        """Initialize mock"""
        hazard = create_autospec(HAZ)
        impact_funcs = create_autospec(ENT.impact_funcs)
        self.icalc = ImpactCalc(ENT.exposures, impact_funcs, hazard)

        mdr = sparse.csr_matrix([[1.0, 0.0, 2.0], [-1.0, 0.5, 1.0]])
        mdr.eliminate_zeros()
        self.icalc.hazard.get_mdr.return_value = mdr
        fraction = sparse.csr_matrix([[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]])
        fraction.eliminate_zeros()
        self.icalc.hazard._get_fraction.return_value = fraction

    def test_impact_matrix(self):
        """Check if impact matrix calculations and calls to hazard are correct"""
        exp_values = np.array([1.0, 2.0, 4.0])
        centroid_idx = np.array([0, 2, 3])
        impact_matrix = self.icalc.impact_matrix(exp_values, centroid_idx, "impf")

        np.testing.assert_array_equal(
            impact_matrix.toarray(), [[1.0, 0.0, 8.0], [-1.0, 0.0, -4.0]]
        )
        self.icalc.hazard.get_mdr.assert_called_once_with(centroid_idx, "impf")
        self.icalc.hazard._get_fraction.assert_called_once_with(centroid_idx)


@patch.object(Impact, "from_eih")
class TestReturnImpact(unittest.TestCase):
    """Test the functionality of _return_impact without digging into the called methods

    This test patches the classmethod `Impact.from_eih` with a mock, so that the input
    variables don't need to make sense. The mock is passed to the test methods via the
    `from_eih_mock` argument for convenience.
    """

    def setUp(self):
        """Mock the methods called by _return_impact"""
        self.icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        self.icalc.stitch_impact_matrix = MagicMock(return_value="stitched_matrix")
        self.icalc.risk_metrics = MagicMock(
            return_value=("at_event", "eai_exp", "aai_agg")
        )
        self.icalc.stitch_risk_metrics = MagicMock(
            return_value=("at_event", "eai_exp", "aai_agg")
        )
        self.imp_mat_gen = "imp_mat_gen"

    def test_save_mat(self, from_eih_mock):
        """Test _return_impact when impact matrix is saved"""
        self.icalc._return_impact(self.imp_mat_gen, save_mat=True)
        from_eih_mock.assert_called_once_with(
            ENT.exposures,
            ENT.impact_funcs,
            HAZ,
            "at_event",
            "eai_exp",
            "aai_agg",
            "stitched_matrix",
        )

        self.icalc.stitch_impact_matrix.assert_called_once_with(self.imp_mat_gen)
        self.icalc.risk_metrics.assert_called_once_with(
            "stitched_matrix", HAZ.frequency
        )
        self.icalc.stitch_risk_metrics.assert_not_called()

    def test_skip_mat(self, from_eih_mock):
        """Test _return_impact when impact matrix is NOT saved"""
        self.icalc._return_impact(self.imp_mat_gen, save_mat=False)

        # Need to check every argument individually due to the last one being a matrix
        call_args = from_eih_mock.call_args.args
        self.assertEqual(call_args[0], ENT.exposures)
        self.assertEqual(call_args[1], ENT.impact_funcs)
        self.assertEqual(call_args[2], HAZ)
        self.assertEqual(call_args[3], "at_event")
        self.assertEqual(call_args[4], "eai_exp")
        self.assertEqual(call_args[5], "aai_agg")
        np.testing.assert_array_equal(
            from_eih_mock.call_args.args[-1], sparse.csr_matrix((0, 0)).toarray()
        )

        self.icalc.stitch_impact_matrix.assert_not_called()
        self.icalc.risk_metrics.assert_not_called()
        self.icalc.stitch_risk_metrics.assert_called_once_with(self.imp_mat_gen)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpactCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReturnImpact))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrix))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrixCalc))
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrixGenerator)
    )
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestInsuredImpactMatrixGenerator)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
