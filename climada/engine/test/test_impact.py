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

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz
from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine.impact import Impact
from climada.util.constants import ENT_DEMO_TODAY, DEF_CRS, DEMO_DIR
from climada.util.api_client import Client
import climada.util.coordinates as u_coord
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

class TestImpact(unittest.TestCase):
    """"Test initialization and more"""
    def test_from_eih_pass(self):
        exp = ENT.exposures
        exp.assign_centroids(HAZ)
        tot_value = exp.affected_total_value(HAZ)
        fake_eai_exp = np.arange(len(exp.gdf))
        fake_at_event = np.arange(HAZ.size)
        fake_aai_agg = np.sum(fake_eai_exp)
        imp = Impact.from_eih(exp, ENT.impact_funcs, HAZ,
                              fake_at_event, fake_eai_exp, fake_aai_agg)
        self.assertEqual(imp.crs, exp.crs)
        self.assertEqual(imp.aai_agg, fake_aai_agg)
        self.assertEqual(imp.imp_mat.size, 0)
        self.assertEqual(imp.unit, exp.value_unit)
        self.assertEqual(imp.tot_value, tot_value)
        np.testing.assert_array_almost_equal(imp.event_id, HAZ.event_id)
        np.testing.assert_array_almost_equal(imp.event_name, HAZ.event_name)
        np.testing.assert_array_almost_equal(imp.date, HAZ.date)
        np.testing.assert_array_almost_equal(imp.frequency, HAZ.frequency)
        np.testing.assert_array_almost_equal(imp.eai_exp, fake_eai_exp)
        np.testing.assert_array_almost_equal(imp.at_event, fake_at_event)
        np.testing.assert_array_almost_equal(
            imp.coord_exp,
            np.stack([exp.gdf.latitude.values, exp.gdf.longitude.values], axis=1)
            )


class TestFreqCurve(unittest.TestCase):
    """Test exceedence frequency curve computation"""
    def test_ref_value_pass(self):
        """Test result against reference value"""
        imp = Impact()
        imp.frequency = np.ones(10) * 6.211180124223603e-04
        imp.at_event = np.zeros(10)
        imp.at_event[0] = 0
        imp.at_event[1] = 0.400665463736549e9
        imp.at_event[2] = 3.150330960044466e9
        imp.at_event[3] = 3.715826406781887e9
        imp.at_event[4] = 2.900244271902339e9
        imp.at_event[5] = 0.778570745161971e9
        imp.at_event[6] = 0.698736262566472e9
        imp.at_event[7] = 0.381063674256423e9
        imp.at_event[8] = 0.569142464157450e9
        imp.at_event[9] = 0.467572545849132e9
        imp.unit = 'USD'

        ifc = imp.calc_freq_curve()
        self.assertEqual(10, len(ifc.return_per))
        self.assertEqual(1610.0000000000000, ifc.return_per[9])
        self.assertEqual(805.00000000000000, ifc.return_per[8])
        self.assertEqual(536.66666666666663, ifc.return_per[7])
        self.assertEqual(402.500000000000, ifc.return_per[6])
        self.assertEqual(322.000000000000, ifc.return_per[5])
        self.assertEqual(268.33333333333331, ifc.return_per[4])
        self.assertEqual(230.000000000000, ifc.return_per[3])
        self.assertEqual(201.250000000000, ifc.return_per[2])
        self.assertEqual(178.88888888888889, ifc.return_per[1])
        self.assertEqual(161.000000000000, ifc.return_per[0])
        self.assertEqual(10, len(ifc.impact))
        self.assertEqual(3.715826406781887e9, ifc.impact[9])
        self.assertEqual(3.150330960044466e9, ifc.impact[8])
        self.assertEqual(2.900244271902339e9, ifc.impact[7])
        self.assertEqual(0.778570745161971e9, ifc.impact[6])
        self.assertEqual(0.698736262566472e9, ifc.impact[5])
        self.assertEqual(0.569142464157450e9, ifc.impact[4])
        self.assertEqual(0.467572545849132e9, ifc.impact[3])
        self.assertEqual(0.400665463736549e9, ifc.impact[2])
        self.assertEqual(0.381063674256423e9, ifc.impact[1])
        self.assertEqual(0, ifc.impact[0])
        self.assertEqual('Exceedance frequency curve', ifc.label)
        self.assertEqual('USD', ifc.unit)

    def test_ref_value_rp_pass(self):
        """Test result against reference value with given return periods"""
        imp = Impact()
        imp.frequency = np.ones(10) * 6.211180124223603e-04
        imp.at_event = np.zeros(10)
        imp.at_event[0] = 0
        imp.at_event[1] = 0.400665463736549e9
        imp.at_event[2] = 3.150330960044466e9
        imp.at_event[3] = 3.715826406781887e9
        imp.at_event[4] = 2.900244271902339e9
        imp.at_event[5] = 0.778570745161971e9
        imp.at_event[6] = 0.698736262566472e9
        imp.at_event[7] = 0.381063674256423e9
        imp.at_event[8] = 0.569142464157450e9
        imp.at_event[9] = 0.467572545849132e9
        imp.unit = 'USD'

        ifc = imp.calc_freq_curve(np.array([100, 500, 1000]))
        self.assertEqual(3, len(ifc.return_per))
        self.assertEqual(100, ifc.return_per[0])
        self.assertEqual(500, ifc.return_per[1])
        self.assertEqual(1000, ifc.return_per[2])
        self.assertEqual(3, len(ifc.impact))
        self.assertEqual(0, ifc.impact[0])
        self.assertEqual(2320408028.5695677, ifc.impact[1])
        self.assertEqual(3287314329.129928, ifc.impact[2])
        self.assertEqual('Exceedance frequency curve', ifc.label)
        self.assertEqual('USD', ifc.unit)

class TestImpactPerYear(unittest.TestCase):
    """Test calc_impact_year_set method"""

    def test_impact_per_year_sum(self):
        """Test result against reference value with given events"""
        imp = Impact()
        imp.frequency = np.ones(10) * 6.211180124223603e-04
        imp.at_event = np.zeros(10)
        imp.at_event[0] = 0
        imp.at_event[1] = 0.400665463736549e9
        imp.at_event[2] = 3.150330960044466e9
        imp.at_event[3] = 3.715826406781887e9
        imp.at_event[4] = 2.900244271902339e9
        imp.at_event[5] = 0.778570745161971e9
        imp.at_event[6] = 0.698736262566472e9
        imp.at_event[7] = 0.381063674256423e9
        imp.at_event[8] = 0.569142464157450e9
        imp.at_event[9] = 0.467572545849132e9
        imp.unit = 'USD'
        imp.date = np.array([732801, 716160, 718313, 712468, 732802,
                             729285, 732931, 715419, 722404, 718351])

        iys_all = imp.impact_per_year()
        iys = imp.impact_per_year(all_years=False)
        iys_all_yr = imp.impact_per_year(year_range=(1975, 2000))
        iys_yr = imp.impact_per_year(all_years=False, year_range=[1975, 2000])
        iys_all_yr_1940 = imp.impact_per_year(all_years=True, year_range=[1940, 2000])
        self.assertEqual(np.around(sum([iys[year] for year in iys])),
                         np.around(sum(imp.at_event)))
        self.assertEqual(sum([iys[year] for year in iys]),
                         sum([iys_all[year] for year in iys_all]))
        self.assertEqual(len(iys), 7)
        self.assertEqual(len(iys_all), 57)
        self.assertIn(1951 and 1959 and 2007, iys_all)
        self.assertTrue(iys_all[1959] > 0)
        self.assertAlmostEqual(3598980534.468811, iys_all[2007])
        self.assertEqual(iys[1978], iys_all[1978])
        self.assertAlmostEqual(iys[1951], imp.at_event[3])
        # year range (yr):
        self.assertEqual(len(iys_yr), 2)
        self.assertEqual(len(iys_all_yr), 26)
        self.assertEqual(sum([iys_yr[year] for year in iys_yr]),
                         sum([iys_all_yr[year] for year in iys_all_yr]))
        self.assertIn(1997 and 1978, iys_yr)
        self.assertFalse(2007 in iys_yr)
        self.assertFalse(1959 in iys_yr)
        self.assertEqual(len(iys_all_yr_1940), 61)

    def test_impact_per_year_empty(self):
        """Test result for empty impact"""
        imp = Impact()
        iys_all = imp.impact_per_year()
        iys = imp.impact_per_year(all_years=False)
        self.assertEqual(len(iys), 0)
        self.assertEqual(len(iys_all), 0)

class TestIO(unittest.TestCase):
    """Test impact input/output methods."""

    def test_write_read_ev_test(self):
        """Test result against reference value"""
        # Create impact object
        num_ev = 10
        num_exp = 5
        imp_write = Impact()
        imp_write.tag = {'exp': Tag('file_exp.p', 'descr exp'),
                         'haz': TagHaz('TC', 'file_haz.p', 'descr haz'),
                         'impf_set': Tag()}
        imp_write.event_id = np.arange(num_ev)
        imp_write.event_name = ['event_' + str(num) for num in imp_write.event_id]
        imp_write.date = np.ones(num_ev)
        imp_write.coord_exp = np.zeros((num_exp, 2))
        imp_write.coord_exp[:, 0] = 1.5
        imp_write.coord_exp[:, 1] = 2.5
        imp_write.eai_exp = np.arange(num_exp) * 100
        imp_write.at_event = np.arange(num_ev) * 50
        imp_write.frequency = np.ones(num_ev) * 0.1
        imp_write.tot_value = 1000
        imp_write.aai_agg = 1001
        imp_write.unit = 'USD'

        file_name = DATA_FOLDER.joinpath('test.csv')
        imp_write.write_csv(file_name)

        imp_read = Impact.from_csv(file_name)
        np.testing.assert_array_equal(imp_write.event_id, imp_read.event_id)
        np.testing.assert_array_equal(imp_write.date, imp_read.date)
        np.testing.assert_array_equal(imp_write.coord_exp, imp_read.coord_exp)
        np.testing.assert_array_equal(imp_write.eai_exp, imp_read.eai_exp)
        np.testing.assert_array_equal(imp_write.at_event, imp_read.at_event)
        np.testing.assert_array_equal(imp_write.frequency, imp_read.frequency)
        self.assertEqual(imp_write.tot_value, imp_read.tot_value)
        self.assertEqual(imp_write.aai_agg, imp_read.aai_agg)
        self.assertEqual(imp_write.unit, imp_read.unit)
        self.assertEqual(
            0, len([i for i, j in zip(imp_write.event_name, imp_read.event_name) if i != j]))

    def test_write_read_exp_test(self):
        """Test result against reference value"""
        # Create impact object
        num_ev = 5
        num_exp = 10
        imp_write = Impact()
        imp_write.tag = {'exp': Tag('file_exp.p', 'descr exp'),
                         'haz': TagHaz('TC', 'file_haz.p', 'descr haz'),
                         'impf_set': Tag()}
        imp_write.event_id = np.arange(num_ev)
        imp_write.event_name = ['event_' + str(num) for num in imp_write.event_id]
        imp_write.date = np.ones(num_ev)
        imp_write.coord_exp = np.zeros((num_exp, 2))
        imp_write.coord_exp[:, 0] = 1.5
        imp_write.coord_exp[:, 1] = 2.5
        imp_write.eai_exp = np.arange(num_exp) * 100
        imp_write.at_event = np.arange(num_ev) * 50
        imp_write.frequency = np.ones(num_ev) * 0.1
        imp_write.tot_value = 1000
        imp_write.aai_agg = 1001
        imp_write.unit = 'USD'

        file_name = DATA_FOLDER.joinpath('test.csv')
        imp_write.write_csv(file_name)

        imp_read = Impact.from_csv(file_name)
        np.testing.assert_array_equal(imp_write.event_id, imp_read.event_id)
        np.testing.assert_array_equal(imp_write.date, imp_read.date)
        np.testing.assert_array_equal(imp_write.coord_exp, imp_read.coord_exp)
        np.testing.assert_array_equal(imp_write.eai_exp, imp_read.eai_exp)
        np.testing.assert_array_equal(imp_write.at_event, imp_read.at_event)
        np.testing.assert_array_equal(imp_write.frequency, imp_read.frequency)
        self.assertEqual(imp_write.tot_value, imp_read.tot_value)
        self.assertEqual(imp_write.aai_agg, imp_read.aai_agg)
        self.assertEqual(imp_write.unit, imp_read.unit)
        self.assertEqual(
            0, len([i for i, j in zip(imp_write.event_name, imp_read.event_name) if i != j]))
        self.assertIsInstance(imp_read.crs, str)

    def test_excel_io(self):
        """Test write and read in excel"""
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        imp_write = Impact()
        ent.exposures.assign_centroids(hazard)
        imp_write.calc(ent.exposures, ent.impact_funcs, hazard)
        file_name = DATA_FOLDER.joinpath('test.xlsx')
        imp_write.write_excel(file_name)

        imp_read = Impact.from_excel(file_name)

        np.testing.assert_array_equal(imp_write.event_id, imp_read.event_id)
        np.testing.assert_array_equal(imp_write.date, imp_read.date)
        np.testing.assert_array_equal(imp_write.coord_exp, imp_read.coord_exp)
        np.testing.assert_array_almost_equal_nulp(imp_write.eai_exp, imp_read.eai_exp, nulp=5)
        np.testing.assert_array_almost_equal_nulp(imp_write.at_event, imp_read.at_event, nulp=5)
        np.testing.assert_array_equal(imp_write.frequency, imp_read.frequency)
        self.assertEqual(imp_write.tot_value, imp_read.tot_value)
        self.assertEqual(imp_write.aai_agg, imp_read.aai_agg)
        self.assertEqual(imp_write.unit, imp_read.unit)
        self.assertEqual(
            0, len([i for i, j in zip(imp_write.event_name, imp_read.event_name) if i != j]))
        self.assertIsInstance(imp_read.crs, str)

    def test_write_imp_mat(self):
        """Test write_excel_imp_mat function"""
        impact = Impact()
        impact.imp_mat = np.zeros((5, 4))
        impact.imp_mat[0, :] = np.arange(4)
        impact.imp_mat[1, :] = np.arange(4) * 2
        impact.imp_mat[2, :] = np.arange(4) * 3
        impact.imp_mat[3, :] = np.arange(4) * 4
        impact.imp_mat[4, :] = np.arange(4) * 5
        impact.imp_mat = sparse.csr_matrix(impact.imp_mat)

        file_name = DATA_FOLDER.joinpath('test_imp_mat')
        impact.write_sparse_csr(file_name)
        read_imp_mat = Impact().read_sparse_csr(f'{file_name}.npz')
        for irow in range(5):
            np.testing.assert_array_equal(
                read_imp_mat[irow, :].toarray(), impact.imp_mat[irow, :].toarray())

class TestRPmatrix(unittest.TestCase):
    """Test computation of impact per return period for whole exposure"""
    def test_local_exceedance_imp_pass(self):
        """Test calc local impacts per return period"""
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
        # Compute the impact per return period over the whole exposures
        impact_rp = impact.local_exceedance_imp(return_periods=(10, 40))

        self.assertIsInstance(impact_rp, np.ndarray)
        self.assertEqual(impact_rp.size, 2 * ent.exposures.gdf.value.size)
        self.assertAlmostEqual(np.max(impact_rp), 2916964966.388219, places=5)
        self.assertAlmostEqual(np.min(impact_rp), 444457580.131494, places=5)

class TestRiskTrans(unittest.TestCase):
    """Test risk transfer methods"""
    def test_risk_trans_pass(self):
        """Test calc_risk_transfer"""
        # Create impact object
        imp = Impact()
        imp.event_id = np.arange(10)
        imp.event_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15]
        imp.date = np.arange(10)
        imp.coord_exp = np.array([[1, 2], [2, 3]])
        imp.crs = DEF_CRS
        imp.eai_exp = np.array([1, 2])
        imp.at_event = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 15])
        imp.frequency = np.ones(10) / 5
        imp.tot_value = 10
        imp.aai_agg = 100
        imp.unit = 'USD'
        imp.imp_mat = sparse.csr_matrix(np.empty((0, 0)))

        new_imp, imp_rt = imp.calc_risk_transfer(2, 10)
        self.assertEqual(new_imp.unit, imp.unit)
        self.assertEqual(new_imp.tot_value, imp.tot_value)
        np.testing.assert_array_equal(new_imp.imp_mat.toarray(), imp.imp_mat.toarray())
        self.assertEqual(new_imp.event_name, imp.event_name)
        np.testing.assert_array_almost_equal_nulp(new_imp.event_id, imp.event_id)
        np.testing.assert_array_almost_equal_nulp(new_imp.date, imp.date)
        np.testing.assert_array_almost_equal_nulp(new_imp.frequency, imp.frequency)
        np.testing.assert_array_almost_equal_nulp(new_imp.coord_exp, [])
        np.testing.assert_array_almost_equal_nulp(new_imp.eai_exp, [])
        np.testing.assert_array_almost_equal_nulp(new_imp.at_event, [0, 1, 2, 2, 2, 2, 2, 2, 2, 5])
        self.assertAlmostEqual(new_imp.aai_agg, 4.0)

        self.assertEqual(imp_rt.unit, imp.unit)
        self.assertEqual(imp_rt.tot_value, imp.tot_value)
        np.testing.assert_array_equal(imp_rt.imp_mat.toarray(), imp.imp_mat.toarray())
        self.assertEqual(imp_rt.event_name, imp.event_name)
        np.testing.assert_array_almost_equal_nulp(imp_rt.event_id, imp.event_id)
        np.testing.assert_array_almost_equal_nulp(imp_rt.date, imp.date)
        np.testing.assert_array_almost_equal_nulp(imp_rt.frequency, imp.frequency)
        np.testing.assert_array_almost_equal_nulp(imp_rt.coord_exp, [])
        np.testing.assert_array_almost_equal_nulp(imp_rt.eai_exp, [])
        np.testing.assert_array_almost_equal_nulp(imp_rt.at_event, [0, 0, 0, 1, 2, 3, 4, 5, 6, 10])
        self.assertAlmostEqual(imp_rt.aai_agg, 6.2)

    def test_transfer_risk_pass(self):
        """Test transfer risk"""
        imp = Impact()
        imp.at_event = np.array([1.5, 2, 3])
        imp.frequency = np.array([0.1, 0, 2])
        transfer_at_event, transfer_aai_agg = imp.transfer_risk(attachment=1, cover=2)
        self.assertTrue(transfer_aai_agg, 4.05)
        np.testing.assert_array_almost_equal(transfer_at_event, np.array([0.5, 1, 2]))

    def test_residual_risk_pass(self):
        """Test residual risk"""
        imp = Impact()
        imp.at_event = np.array([1.5, 2, 3])
        imp.frequency = np.array([0.1, 0, 2])
        residual_at_event, residual_aai_agg = imp.residual_risk(attachment=1, cover=1.5)
        self.assertTrue(residual_aai_agg, 3.1)
        np.testing.assert_array_almost_equal(residual_at_event, np.array([1, 1, 1.5]))

def dummy_impact():

    imp = Impact()
    imp.event_id = np.arange(6) + 10
    imp.event_name = [0, 1, 'two', 'three', 30, 31]
    imp.date = np.arange(6)
    imp.coord_exp = np.array([[1, 2], [1.5, 2.5]])
    imp.crs = DEF_CRS
    imp.eai_exp = np.array([7.2, 7.2])
    imp.at_event = np.array([0, 2, 4, 6, 60, 62])
    imp.frequency = np.array([1/6, 1/6, 1, 1, 1/30, 1/30])
    imp.tot_value = 7
    imp.aai_agg = 14.4
    imp.unit = 'USD'
    imp.imp_mat = sparse.csr_matrix(np.array([
        [0,0], [1,1], [2,2], [3,3], [30,30], [31,31]
        ]))

    return imp

class TestSelect(unittest.TestCase):
    """Test select method"""
    def test_select_event_id_pass(self):
        """Test select by event id"""

        imp = dummy_impact()
        sel_imp = imp.select(event_ids=[10, 11, 12])

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, [10, 11, 12])
        self.assertEqual(sel_imp.event_name, [0, 1, 'two'])
        np.testing.assert_array_equal(sel_imp.date, [0, 1, 2])
        np.testing.assert_array_almost_equal_nulp(sel_imp.frequency, [1/6, 1/6, 1])

        np.testing.assert_array_equal(sel_imp.at_event, [0, 2, 4])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0,0], [1,1], [2,2]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2, 1/6+2])
        self.assertEqual(sel_imp.aai_agg, 4+2/6)

        self.assertEqual(sel_imp.tot_value, 7)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2], [1.5, 2.5]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_event_name_pass(self):
        """Test select by event name"""

        imp = dummy_impact()
        sel_imp = imp.select(event_names=[0, 1, 'two'])

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, [10, 11, 12])
        self.assertEqual(sel_imp.event_name, [0, 1, 'two'])
        np.testing.assert_array_equal(sel_imp.date, [0, 1, 2])
        np.testing.assert_array_almost_equal_nulp(sel_imp.frequency, [1/6, 1/6, 1])

        np.testing.assert_array_equal(sel_imp.at_event, [0, 2, 4])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0,0], [1,1], [2,2]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2, 1/6+2])
        self.assertEqual(sel_imp.aai_agg, 4+2/6)

        self.assertEqual(sel_imp.tot_value, 7)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2], [1.5, 2.5]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_dates_pass(self):
        """Test select by event dates"""

        imp = dummy_impact()
        sel_imp = imp.select(dates=(0, 2))

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, [10, 11, 12])
        self.assertEqual(sel_imp.event_name, [0, 1, 'two'])
        np.testing.assert_array_equal(sel_imp.date, [0, 1, 2])
        np.testing.assert_array_almost_equal_nulp(sel_imp.frequency, [1/6, 1/6, 1])

        np.testing.assert_array_equal(sel_imp.at_event, [0, 2, 4])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0,0], [1,1], [2,2]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2, 1/6+2])
        self.assertEqual(sel_imp.aai_agg, 4+2/6)

        self.assertEqual(sel_imp.tot_value, 7)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2], [1.5, 2.5]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_coord_exp_pass(self):
        """ test select by exp coordinates """

        imp = dummy_impact()
        sel_imp = imp.select(coord_exp=np.array([1,2]))

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, imp.event_id)
        self.assertEqual(sel_imp.event_name, imp.event_name)
        np.testing.assert_array_equal(sel_imp.date, imp.date)
        np.testing.assert_array_equal(sel_imp.frequency, imp.frequency)

        np.testing.assert_array_equal(sel_imp.at_event, [0, 1, 2, 3, 30, 31])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0], [1], [2], [3], [30], [31]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2+3+1+31/30])
        self.assertEqual(sel_imp.aai_agg, 1/6+2+3+1+31/30)

        self.assertEqual(sel_imp.tot_value, None)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_event_identity_pass(self):
        """ test select same impact with event name, id and date """

        # Read default entity values
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        # Read default hazard file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        # Create impact object
        imp = Impact()

        # Assign centroids to exposures
        ent.exposures.assign_centroids(hazard)

        # Compute the impact over the whole exposures
        imp.calc(ent.exposures, ent.impact_funcs, hazard, save_mat=True)

        sel_imp = imp.select(event_ids=imp.event_id,
                             event_names=imp.event_name,
                             dates=(min(imp.date), max(imp.date))
                             )

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, imp.event_id)
        self.assertEqual(sel_imp.event_name, imp.event_name)
        np.testing.assert_array_equal(sel_imp.date, imp.date)
        np.testing.assert_array_equal(sel_imp.frequency, imp.frequency)

        np.testing.assert_array_equal(sel_imp.at_event, imp.at_event)
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), imp.imp_mat.todense())
        np.testing.assert_array_equal(sel_imp.eai_exp, imp.eai_exp)
        self.assertAlmostEqual(round(sel_imp.aai_agg,5), round(imp.aai_agg,5))

        self.assertEqual(sel_imp.tot_value, imp.tot_value)
        np.testing.assert_array_equal(sel_imp.coord_exp, imp.coord_exp)

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)


    def test_select_new_attributes(self):
        """Test if impact has new attributes """

        imp = dummy_impact()
        imp.new_per_ev =  ['a', 'b', 'c', 'd', 'e', 'f']
        sel_imp = imp.select(event_names=[0, 1, 'two'])

        self.assertEqual(sel_imp.new_per_ev, ['a', 'b', 'c'])

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, [10, 11, 12])
        self.assertEqual(sel_imp.event_name, [0, 1, 'two'])
        np.testing.assert_array_equal(sel_imp.date, [0, 1, 2])
        np.testing.assert_array_almost_equal_nulp(sel_imp.frequency, [1/6, 1/6, 1])

        np.testing.assert_array_equal(sel_imp.at_event, [0, 2, 4])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0,0], [1,1], [2,2]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2, 1/6+2])
        self.assertEqual(sel_imp.aai_agg, 4+2/6)

        self.assertEqual(sel_imp.tot_value, 7)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2], [1.5, 2.5]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_nothing(self):
        """Test select with no matches"""
        imp = dummy_impact()
        sel_imp = imp.select(event_ids=[100])
        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)
        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)
        self.assertEqual(sel_imp.event_id.size, 0)
        self.assertEqual(len(sel_imp.event_name), 0)
        self.assertEqual(sel_imp.date.size, 0)
        self.assertEqual(sel_imp.frequency.size, 0)
        self.assertEqual(sel_imp.at_event.size, 0)
        self.assertEqual(sel_imp.imp_mat.shape[0], 0)
        self.assertEqual(sel_imp.aai_agg, 0)

    def test_select_id_name_dates_pass(self):
        """Test select by event ids, names, and dates"""

        imp = dummy_impact()
        sel_imp = imp.select(event_ids=[0], event_names=[1, 'two'], dates=(0, 2))

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)

        np.testing.assert_array_equal(sel_imp.event_id, [10, 11, 12])
        self.assertEqual(sel_imp.event_name, [0, 1, 'two'])
        np.testing.assert_array_equal(sel_imp.date, [0, 1, 2])
        np.testing.assert_array_almost_equal_nulp(sel_imp.frequency, [1/6, 1/6, 1])

        np.testing.assert_array_equal(sel_imp.at_event, [0, 2, 4])
        np.testing.assert_array_equal(sel_imp.imp_mat.todense(), [[0,0], [1,1], [2,2]])
        np.testing.assert_array_almost_equal_nulp(sel_imp.eai_exp, [1/6+2, 1/6+2])
        self.assertEqual(sel_imp.aai_agg, 4+2/6)

        self.assertEqual(sel_imp.tot_value, 7)
        np.testing.assert_array_equal(sel_imp.coord_exp, [[1, 2], [1.5, 2.5]])

        self.assertIsInstance(sel_imp, Impact)
        self.assertIsInstance(sel_imp.imp_mat, sparse.csr_matrix)

    def test_select_imp_map_fail(self):
        """Test that selection fails if imp_mat is empty"""

        imp = dummy_impact()
        imp.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
        with self.assertRaises(ValueError):
            imp.select(event_ids=[0], event_names=[1, 'two'], dates=(0, 2))

class TestConvertExp(unittest.TestCase):
    def test__build_exp(self):
        """Test that an impact set can be converted to an exposure"""

        imp = dummy_impact()
        exp = imp._build_exp()
        np.testing.assert_array_equal(imp.eai_exp, exp.gdf['value'])
        np.testing.assert_array_equal(imp.coord_exp[:, 0], exp.gdf['latitude'])
        np.testing.assert_array_equal(imp.coord_exp[:, 1], exp.gdf['longitude'])
        self.assertTrue(u_coord.equal_crs(exp.crs, imp.crs))
        self.assertEqual(exp.value_unit, imp.unit)
        self.assertEqual(exp.ref_year, 0)

    def test__exp_build_event(self):
        """Test that a single events impact can be converted to an exposure"""

        imp = dummy_impact()
        event_id = imp.event_id[1]
        exp = imp._build_exp_event(event_id=event_id)
        np.testing.assert_array_equal(imp.imp_mat[1].todense().A1, exp.gdf['value'])
        np.testing.assert_array_equal(imp.coord_exp[:, 0], exp.gdf['latitude'])
        np.testing.assert_array_equal(imp.coord_exp[:, 1], exp.gdf['longitude'])
        self.assertTrue(u_coord.equal_crs(exp.crs, imp.crs))
        self.assertEqual(exp.value_unit, imp.unit)
        self.assertEqual(exp.ref_year, 0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFreqCurve)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactPerYear))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRPmatrix))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskTrans))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelect))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConvertExp))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpact))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
