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
import unittest.mock
import numpy as np
from scipy import sparse

from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine import ImpactCalc
from climada.util.constants import ENT_DEMO_TODAY, DEMO_DIR
from climada.util.api_client import Client


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

    def test_insured_matrics(self):
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


class TestImpactMatrixCalc(unittest.TestCase):
    """Verify the computation of the impact matrix"""

    def setUp(self):
        # Mock the methods called by 'impact_matrix'
        self.hazard = unittest.mock.create_autospec(HAZ)
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


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpactCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactMatrixCalc))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
