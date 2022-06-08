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
import numpy as np
from scipy import sparse

from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine.impact import Impact, ImpactCalc
from climada.util.constants import ENT_DEMO_TODAY, DEF_CRS, DEMO_DIR
from climada.util.api_client import Client
import climada.engine.test as engine_test


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
        np.testing.assert_array_equal(icalc.deductible, ENT.exposures.gdf.deductible)
        np.testing.assert_array_equal(icalc.cover, ENT.exposures.gdf.cover)
        self.assertEqual(icalc.imp_mat.size, 0)
        self.assertTrue(ENT.exposures.gdf.equals(icalc.exposures.gdf))
        self.assertEqual(HAZ.event_id, icalc.hazard.event_id)
        self.assertEqual(HAZ.event_name, icalc.hazard.event_name)

class TestCalc(unittest.TestCase):
    """Test impact calc method."""

    def test_ref_value_pass(self):
        """Test result against reference value"""
        # Read default entity values
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        # Read default hazard file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        # Create impact object
        impact = Impact()

        # Assign centroids to exposures
        ent.exposures.assign_centroids(hazard)

        # Compute the impact over the whole exposures
        impact.calc(ent.exposures, ent.impact_funcs, hazard)

        # Check result
        num_events = len(hazard.event_id)
        num_exp = ent.exposures.gdf.shape[0]
        # Check relative errors as well when absolute value gt 1.0e-7
        # impact.at_event == EDS.damage in MATLAB
        self.assertEqual(num_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[int(num_events / 2)])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809])
        self.assertAlmostEqual(7.076504723057620e+10, impact.at_event[12147])
        self.assertEqual(0, impact.at_event[num_events - 1])
        # impact.eai_exp == EDS.ED_at_centroid in MATLAB
        self.assertEqual(num_exp, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.eai_exp[0])
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[int(num_exp / 2)], 6)
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[int(num_exp / 2)], 5)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[num_exp - 1], 6)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[int(num_exp - 1)], 5)
        # impact.tot_value == EDS.Value in MATLAB
        # impact.aai_agg == EDS.ED in MATLAB
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)

    def test_calc_imp_mat_pass(self):
        """Test save imp_mat"""
        # Read default entity values
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        # Read default hazard file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        # Create impact object
        impact = Impact()

        # Assign centroids to exposures
        ent.exposures.assign_centroids(hazard)

        # Compute the impact over the whole exposures
        impact.calc(ent.exposures, ent.impact_funcs, hazard, save_mat=True)
        self.assertIsInstance(impact.imp_mat, sparse.csr_matrix)
        self.assertEqual(impact.imp_mat.shape, (hazard.event_id.size,
                                                ent.exposures.gdf.value.size))
        np.testing.assert_array_almost_equal_nulp(
            np.array(impact.imp_mat.sum(axis=1)).ravel(), impact.at_event, nulp=5)
        np.testing.assert_array_almost_equal_nulp(
            np.sum(impact.imp_mat.toarray() * impact.frequency[:, None], axis=0).reshape(-1),
            impact.eai_exp)

    def test_calc_impf_pass(self):
        """Execute when no impf_HAZ present, but only impf_"""
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        self.assertTrue('impf_TC' in ent.exposures.gdf.columns)
        ent.exposures.gdf.rename(columns={'impf_TC': 'impf_'}, inplace=True)
        self.assertFalse('impf_TC' in ent.exposures.gdf.columns)
        ent.check()

        # Read default hazard file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        # Create impact object
        impact = Impact()
        impact.calc(ent.exposures, ent.impact_funcs, hazard)

        # Check result
        num_events = len(hazard.event_id)
        num_exp = ent.exposures.gdf.shape[0]
        # Check relative errors as well when absolute value gt 1.0e-7
        # impact.at_event == EDS.damage in MATLAB
        self.assertEqual(num_events, len(impact.at_event))
        self.assertEqual(0, impact.at_event[0])
        self.assertEqual(0, impact.at_event[int(num_events / 2)])
        self.assertAlmostEqual(1.472482938320243e+08, impact.at_event[13809])
        self.assertEqual(7.076504723057620e+10, impact.at_event[12147])
        self.assertEqual(0, impact.at_event[num_events - 1])
        # impact.eai_exp == EDS.ED_at_centroid in MATLAB
        self.assertEqual(num_exp, len(impact.eai_exp))
        self.assertAlmostEqual(1.518553670803242e+08, impact.eai_exp[0])
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[int(num_exp / 2)], 6)
        self.assertAlmostEqual(1.373490457046383e+08, impact.eai_exp[int(num_exp / 2)], 5)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[num_exp - 1], 6)
        self.assertAlmostEqual(1.066837260150042e+08, impact.eai_exp[int(num_exp - 1)], 5)
        # impact.tot_value == EDS.Value in MATLAB
        # impact.aai_agg == EDS.ED in MATLAB
        self.assertAlmostEqual(6.570532945599105e+11, impact.tot_value)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)
        self.assertAlmostEqual(6.512201157564421e+09, impact.aai_agg, 5)



# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpactCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalc))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
