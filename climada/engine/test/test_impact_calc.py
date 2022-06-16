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
from unittest.mock import create_autospec, MagicMock, call
import numpy as np
from scipy import sparse
import pandas as pd

from climada import CONFIG
from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine import ImpactCalc
from climada.util.constants import ENT_DEMO_TODAY, DEMO_DIR
from climada.util.api_client import Client
from climada.util.config import Config


def get_haz_test_file(ds_name):
    # As this module is part of the installation test suite, we want tom make sure it is running
    # also in offline mode even when installing from pypi, where there is no test configuration.
    # So we set cache_enabled explicitly to true
    client = Client(cache_enabled=True)
    test_ds = client.get_dataset_info(name=ds_name, status='test_dataset')
    _, [haz_test_file] = client.download_dataset(test_ds)
    return haz_test_file


HAZ_TEST_MAT = get_haz_test_file('atl_prob_no_name')

ENT = Entity.from_excel(ENT_DEMO_TODAY)
HAZ = Hazard.from_mat(HAZ_TEST_MAT)

DATA_FOLDER = DEMO_DIR / 'test-results'
DATA_FOLDER.mkdir(exist_ok=True)


class TestImpactCalc(unittest.TestCase):
    """Test Impact calc methods"""
    def test_init(self):
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        self.assertEqual(icalc.n_exp_pnt, ENT.exposures.gdf.shape[0])
        self.assertEqual(icalc.n_events, HAZ.size)
        self.assertEqual(icalc.imp_mat.size, 0)
        self.assertTrue(ENT.exposures.gdf.equals(icalc.exposures.gdf))
        np.testing.assert_array_equal(HAZ.event_id, icalc.hazard.event_id)
        np.testing.assert_array_equal(HAZ.event_name, icalc.hazard.event_name)
        np.testing.assert_array_equal(icalc.deductible, ENT.exposures.gdf.deductible)
        np.testing.assert_array_equal(icalc.cover, ENT.exposures.gdf.cover)


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

    def test_calc_impact_pass(self):
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
        HAZf.fraction *= 0.6
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZf)
        impact = icalc.impact()
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

    def test_calc_impact_save_mat_pass(self):
        """Test compute impact with impact matrix"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
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
        impact = icalc.insured_impact()
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

    def test_calc_insured_impact_save_mat_pass(self):
        """Test compute impact with impact matrix"""
        exp = ENT.exposures.copy()
        exp.gdf.cover /= 1e3
        exp.gdf.deductible += 1e5
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        impact = icalc.insured_impact(save_mat=True)

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

    def test_calc_insured_impact_fail(self):
        """Test raise error for insured impact calc if no cover and
        no deductibles defined
        """
        exp = ENT.exposures.copy()
        exp.gdf = exp.gdf.drop(columns = ['cover', 'deductible'])
        icalc = ImpactCalc(exp, ENT.impact_funcs, HAZ)
        with self.assertRaises(AttributeError):
            icalc.insured_impact()

    def test_minimal_exp_gdf(self):
        """Test obtain minimal exposures gdf"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        exp_min_gdf = icalc.minimal_exp_gdf('impf_TC')
        self.assertSetEqual(
            set(exp_min_gdf.columns), set(['value', 'impf_TC', 'centr_TC'])
            )
        np.testing.assert_array_equal(exp_min_gdf.value, ENT.exposures.gdf.value)
        np.testing.assert_array_equal(exp_min_gdf.impf_TC, ENT.exposures.gdf.impf_TC)
        np.testing.assert_array_equal(exp_min_gdf.centr_TC, ENT.exposures.gdf.centr_TC)

    def test_stitch_impact_matrix(self):
        """Check how sparse matrices from a generator are stitched together"""
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        icalc.n_events = 3
        icalc.n_exp_pnt = 4

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
        icalc = ImpactCalc(ENT.exposures, ENT.impact_funcs, HAZ)
        icalc.n_events = 2
        icalc.n_exp_pnt = 3
        icalc.hazard.frequency = np.array([2, 0.5])

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
        # Mock the methods called by 'impact_matrix'
        self.hazard = create_autospec(HAZ)
        self.hazard.get_mdr.return_value = sparse.csr_matrix(
            [[0.0, 0.5, -1.0], [1.0, 2.0, 1.0]]
        )
        self.hazard.get_fraction.return_value = sparse.csr_matrix(
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
            self.hazard.get_fraction.assert_called_once_with(self.centroids)

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
        self.exp_gdf = pd.DataFrame(
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
        exp_gdf = pd.DataFrame(
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
        exp_gdf = pd.DataFrame({"impact_functions": [], "centr_col": [], "value": []})
        self.assertEqual(
            [],
            list(self.icalc.imp_mat_gen(exp_gdf=exp_gdf, impf_col="impact_functions")),
        )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpactCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrixCalc))
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrixGenerator)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
