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
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import numpy.testing as npt
from scipy import sparse
import h5py

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz
from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine import Impact, ImpactCalc
from climada.util.constants import ENT_DEMO_TODAY, DEF_CRS, DEMO_DIR, DEF_FREQ_UNIT
import climada.util.coordinates as u_coord

from climada.hazard.test.test_base import HAZ_TEST_TC


ENT :Entity = Entity.from_excel(ENT_DEMO_TODAY)
HAZ :Hazard = Hazard.from_hdf5(HAZ_TEST_TC)

DATA_FOLDER :Path = DEMO_DIR / 'test-results'
DATA_FOLDER.mkdir(exist_ok=True)


def dummy_impact():
    """Return an impact object for testing"""
    return Impact(
        event_id=np.arange(6) + 10,
        event_name=[0, 1, "two", "three", 30, 31],
        date=np.arange(6),
        coord_exp=np.array([[1, 2], [1.5, 2.5]]),
        crs=DEF_CRS,
        eai_exp=np.array([7.2, 7.2]),
        at_event=np.array([0, 2, 4, 6, 60, 62]),
        frequency=np.array([1 / 6, 1 / 6, 1, 1, 1 / 30, 1 / 30]),
        tot_value=7,
        aai_agg=14.4,
        unit="USD",
        frequency_unit="1/month",
        imp_mat=sparse.csr_matrix(
            np.array([[0, 0], [1, 1], [2, 2], [3, 3], [30, 30], [31, 31]])
        ),
        tag={
            "exp": Tag("file_exp.p", "descr exp"),
            "haz": TagHaz("TC", "file_haz.p", "descr haz"),
            "impf_set": Tag(),
        },
    )


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
        self.assertEqual(imp.frequency_unit, HAZ.frequency_unit)
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


class TestImpactConcat(unittest.TestCase):
    """test Impact.concat"""

    def setUp(self) -> None:
        """Create two dummy impacts"""
        self.imp1 = Impact(
            event_name=["ev1"],
            date=np.array([735449]),
            event_id=np.array([2]),
            frequency=np.array([0.1]),
            coord_exp=np.array([[45, 8]]),
            unit="cakes",
            frequency_unit="bar",
            crs=DEF_CRS,
            at_event=np.array([1]),
            eai_exp=np.array([0.1]),
            aai_agg=0.1,
            tot_value=5,
            imp_mat=sparse.csr_matrix([[1]]),
        )

        self.imp2 = Impact(
            event_name=["ev2", "ev3"],
            date=np.array([735450, 735451]),
            event_id=np.array([3, 4]),
            frequency=np.array([0.2, 0.3]),
            coord_exp=np.array([[45, 8]]),
            unit="cakes",
            frequency_unit="bar",
            crs=DEF_CRS,
            at_event=np.array([2, 3]),
            eai_exp=np.array([0.2]),
            aai_agg=0.2,
            tot_value=5,
            imp_mat=sparse.csr_matrix([[2], [3]]),
        )

    def test_check_exposure_pass(self):
        """test exposure checks"""
        # test crs
        with self.assertRaises(ValueError) as cm:
            self.imp2.crs = "OTHER"
            Impact.concat([self.imp1, self.imp2])
        self.assertIn("Attribute 'crs' must be unique among impacts", str(cm.exception))
        # reset crs
        self.imp2.crs = self.imp1.crs

        # test total exposure value
        with self.assertRaises(ValueError) as cm:
            self.imp2.tot_value = 1
            Impact.concat([self.imp1, self.imp2])
        self.assertIn(
            "Attribute 'tot_value' must be unique among impacts", str(cm.exception)
        )
        # reset exposure value
        self.imp2.tot_value = self.imp1.tot_value

        # test exposure coordinates
        with self.assertRaises(ValueError) as cm:
            self.imp2.coord_exp[0][0] = 0
            Impact.concat([self.imp1, self.imp2])
        self.assertIn(
            "The impacts have different exposure coordinates", str(cm.exception)
        )

    def test_event_ids(self):
        """Test if event IDs are handled correctly"""
        # Resetting
        imp = Impact.concat([self.imp1, self.imp2], reset_event_ids=True)
        np.testing.assert_array_equal(imp.event_id, [1, 2, 3])

        # Error on non-unique IDs
        self.imp2.event_id[0] = self.imp1.event_id[0]
        with self.assertRaises(ValueError) as cm:
            Impact.concat([self.imp1, self.imp2])
        self.assertIn("Duplicate event IDs: [2]", str(cm.exception))

    def test_empty_impact_matrix(self):
        """Test if empty impact matrices are handled correctly"""
        # One empty
        self.imp1.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
        with self.assertRaises(ValueError) as cm:
            Impact.concat([self.imp1, self.imp2])
        self.assertIn(
            "Impact matrices do not have the same number of exposure points",
            str(cm.exception),
        )

        # Both empty
        self.imp2.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
        imp = Impact.concat([self.imp1, self.imp2])
        np.testing.assert_array_equal(imp.imp_mat.toarray(), np.empty((0, 0)))

    def test_results(self):
        """Test results of impact.concat"""
        impact = Impact.concat([self.imp1, self.imp2])

        np.testing.assert_array_equal(impact.event_id, [2, 3, 4])
        np.testing.assert_array_equal(impact.event_name, ["ev1", "ev2", "ev3"])
        np.testing.assert_array_equal(impact.date, np.array([735449, 735450, 735451]))
        np.testing.assert_array_equal(impact.frequency, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(impact.eai_exp, np.array([0.3]))
        np.testing.assert_array_equal(impact.at_event, np.array([1, 2, 3]))
        self.assertAlmostEqual(impact.aai_agg, 0.3)
        np.testing.assert_array_equal(
            impact.imp_mat.toarray(), sparse.csr_matrix([[1], [2], [3]]).toarray()
        )
        np.testing.assert_array_equal(impact.coord_exp, self.imp1.coord_exp)
        self.assertEqual(impact.tot_value, self.imp1.tot_value)
        self.assertEqual(impact.unit, self.imp1.unit)
        self.assertEqual(impact.frequency_unit, self.imp1.frequency_unit)
        self.assertEqual(impact.crs, self.imp1.crs)


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
        imp.frequency_unit = '1/day'

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
        self.assertEqual('1/day', ifc.frequency_unit)

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
        imp.frequency_unit = '1/week'

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
        self.assertEqual('1/week', ifc.frequency_unit)

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
        imp_write.frequency_unit = '1/month'

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
        self.assertEqual(imp_write.frequency_unit, imp_read.frequency_unit)
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
        imp_write.frequency_unit = '1/month'

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
        self.assertEqual(imp_write.frequency_unit, imp_read.frequency_unit)
        self.assertEqual(
            0, len([i for i, j in zip(imp_write.event_name, imp_read.event_name) if i != j]))
        self.assertIsInstance(imp_read.crs, str)

    def test_excel_io(self):
        """Test write and read in excel"""
        ent = Entity.from_excel(ENT_DEMO_TODAY)
        ent.check()

        hazard = Hazard.from_hdf5(HAZ_TEST_TC)

        imp_write = ImpactCalc(ent.exposures, ent.impact_funcs, hazard).impact()
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
        self.assertEqual(imp_write.frequency_unit, imp_read.frequency_unit)
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
        hazard = Hazard.from_hdf5(HAZ_TEST_TC)

        # Compute the impact over the whole exposures
        impact = ImpactCalc(ent.exposures, ent.impact_funcs, hazard).impact(save_mat=True)
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
        imp.frequency_unit = '1/month'
        imp.imp_mat = sparse.csr_matrix(np.empty((0, 0)))

        new_imp, imp_rt = imp.calc_risk_transfer(2, 10)
        self.assertEqual(new_imp.unit, imp.unit)
        self.assertEqual(new_imp.frequency_unit, imp.frequency_unit)
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
        self.assertEqual(imp_rt.frequency_unit, imp.frequency_unit)
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


class TestSelect(unittest.TestCase):
    """Test select method"""
    def test_select_event_id_pass(self):
        """Test select by event id"""

        imp = dummy_impact()
        sel_imp = imp.select(event_ids=[10, 11, 12])

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        hazard = Hazard.from_hdf5(HAZ_TEST_TC)

        # Assign centroids to exposures
        ent.exposures.assign_centroids(hazard)

        # Compute the impact over the whole exposures
        imp = ImpactCalc(ent.exposures, ent.impact_funcs, hazard).impact(save_mat=True, assign_centroids=False)

        sel_imp = imp.select(event_ids=imp.event_id,
                             event_names=imp.event_name,
                             dates=(min(imp.date), max(imp.date))
                             )

        self.assertTrue(u_coord.equal_crs(sel_imp.crs, imp.crs))
        self.assertEqual(sel_imp.unit, imp.unit)
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)
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
        self.assertEqual(sel_imp.frequency_unit, imp.frequency_unit)

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


class TestImpactH5IO(unittest.TestCase):
    """Tests for reading and writing Impact from/to H5 files"""

    def setUp(self) -> None:
        """Create temporary directory and impact object"""
        self.tempdir = TemporaryDirectory()
        self.impact = dummy_impact()
        self.impact.event_name = list(
            map(str, range(6))
        )  # Writer does not support changing types
        self.filepath = Path(self.tempdir.name) / "file.h5"

    def tearDown(self) -> None:
        """Remove the temporary directory"""
        self.tempdir.cleanup()

    def _compare_file_to_imp(self, filepath, impact, dense_imp_mat):
        """Compare a file to an impact object"""
        with h5py.File(filepath, "r") as file:
            npt.assert_array_equal(file["event_id"], impact.event_id)
            npt.assert_array_equal(file["event_name"].asstr(), impact.event_name)
            npt.assert_array_equal(file["date"], impact.date)
            npt.assert_array_equal(file["coord_exp"], impact.coord_exp)
            self.assertEqual(file.attrs["crs"], DEF_CRS)
            npt.assert_array_equal(file["eai_exp"], impact.eai_exp)
            npt.assert_array_equal(file["at_event"], impact.at_event)
            npt.assert_array_equal(file["frequency"], impact.frequency)
            self.assertEqual(file.attrs["tot_value"], impact.tot_value)
            self.assertEqual(file.attrs["unit"], impact.unit)
            self.assertEqual(file.attrs["aai_agg"], impact.aai_agg)
            self.assertEqual(file.attrs["frequency_unit"], impact.frequency_unit)
            self.assertDictEqual(
                dict(**file["tag"]["exp"].attrs), impact.tag["exp"].__dict__
            )
            self.assertDictEqual(
                dict(**file["tag"]["haz"].attrs), impact.tag["haz"].__dict__
            )
            self.assertDictEqual(
                dict(**file["tag"]["impf_set"].attrs), impact.tag["impf_set"].__dict__
            )

            if dense_imp_mat:
                npt.assert_array_equal(file["imp_mat"], impact.imp_mat.toarray())
            else:
                npt.assert_array_equal(file["imp_mat"]["data"], impact.imp_mat.data)
                npt.assert_array_equal(
                    file["imp_mat"]["indices"], impact.imp_mat.indices
                )
                npt.assert_array_equal(file["imp_mat"]["indptr"], impact.imp_mat.indptr)
                npt.assert_array_equal(
                    file["imp_mat"].attrs["shape"], impact.imp_mat.shape
                )

    def _compare_impacts(self, impact_1, impact_2):
        """Compare to impact instances"""
        for name, value in impact_1.__dict__.items():
            self.assertIn(name, impact_2.__dict__)
            value_comp = getattr(impact_2, name)
            # NOTE: Tags do not compare
            if name == "tag":
                for key in value:
                    self.assertDictEqual(value[key].__dict__, value_comp[key].__dict__)
            elif isinstance(value, sparse.csr_matrix):
                npt.assert_array_equal(value.toarray(), value_comp.toarray())
            elif np.ndim(value) > 0:
                npt.assert_array_equal(value, value_comp)
            else:
                self.assertEqual(value, value_comp)
        npt.assert_array_equal(
            [attr for attr in impact_2.__dict__ if attr not in impact_1.__dict__], []
        )

    def test_write_hdf5(self):
        """Test writing an impact into an H5 file"""
        for dense in (True, False):
            with self.subTest(dense_imp_mat=dense):
                self.impact.write_hdf5(self.filepath, dense_imp_mat=dense)
                self._compare_file_to_imp(
                    self.filepath, self.impact, dense_imp_mat=dense
                )

    def test_write_hdf5_without_imp_mat(self):
        """Test writing an impact into an H5 file with an empty impact matrix"""
        self.impact.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
        self.test_write_hdf5()

    def test_write_hdf5_type_fail(self):
        """Test that writing attributes with varying types results in an error"""
        self.impact.event_name = [1, "a", 1.0, "b", "c", "d"]
        with self.assertRaises(TypeError) as cm:
            self.impact.write_hdf5(self.filepath)
        self.assertIn("No conversion path for dtype", str(cm.exception))

    def test_cycle_hdf5(self):
        """Test writing and reading the same object"""
        for dense in (True, False):
            with self.subTest(dense_imp_mat=dense):
                self.impact.write_hdf5(self.filepath, dense_imp_mat=dense)
                impact_read = Impact.from_hdf5(self.filepath)
                self._compare_impacts(self.impact, impact_read)

    def test_read_hdf5_minimal(self):
        """Try reading a basically empty file"""
        with h5py.File(self.filepath, "w") as file:
            file.create_dataset("imp_mat", data=np.empty((0, 0)))

        impact = Impact.from_hdf5(self.filepath)
        npt.assert_array_equal(impact.imp_mat.toarray(), np.empty((0, 0)))
        npt.assert_array_equal(impact.event_id, np.array([]))
        npt.assert_array_equal(impact.event_name, np.array([]))
        self.assertIsInstance(impact.event_name, list)
        npt.assert_array_equal(impact.date, np.array([]))
        npt.assert_array_equal(impact.coord_exp, np.array([]))
        npt.assert_array_equal(impact.eai_exp, np.array([]))
        npt.assert_array_equal(impact.at_event, np.array([]))
        npt.assert_array_equal(impact.frequency, np.array([]))
        self.assertEqual(impact.crs, DEF_CRS)
        self.assertEqual(impact.frequency_unit, DEF_FREQ_UNIT)
        self.assertEqual(impact.tot_value, 0)
        self.assertEqual(impact.aai_agg, 0)
        self.assertEqual(impact.unit, "")
        self.assertEqual(impact.tag, {})

    def test_read_hdf5_full(self):
        """Try reading a file full of data"""
        # Define the data
        imp_mat = sparse.csr_matrix(np.array([[1, 1, 1], [2, 2, 2]]))
        event_id = np.array([1, 2])
        event_name = ["a", "b"]
        date = np.array([10, 11])
        coord_exp = np.array([[1, 2], [1, 3], [2, 1]])
        eai_exp = np.array([0.1, 0.2, 0.3])
        at_event = np.array([1, 2])
        frequency = np.array([0.5, 0.6])
        crs = "EPSG:1"
        frequency_unit = "f_unit"
        tot_value = 100
        aai_agg = 200
        unit = "unit"
        haz_tag = dict(
            haz_type="haz_type", file_name="file_name", description="description"
        )
        exp_tag = dict(file_name="exp", description="exp")
        impf_set_tag = dict(file_name="impf_set", description="impf_set")

        def write_tag(group, tag_kwds):
            for key, value in tag_kwds.items():
                group.attrs[key] = value

        # Write the data
        with h5py.File(self.filepath, "w") as file:
            file.create_dataset("imp_mat", data=imp_mat.toarray())
            file.create_dataset("event_id", data=event_id)
            file.create_dataset(
                "event_name", data=event_name, dtype=h5py.string_dtype()
            )
            file.create_dataset("date", data=date)
            file.create_dataset("coord_exp", data=coord_exp)
            file.create_dataset("eai_exp", data=eai_exp)
            file.create_dataset("at_event", data=at_event)
            file.create_dataset("frequency", data=frequency)
            file.attrs["crs"] = crs
            file.attrs["frequency_unit"] = frequency_unit
            file.attrs["tot_value"] = tot_value
            file.attrs["aai_agg"] = aai_agg
            file.attrs["unit"] = unit
            for group, kwds in zip(
                ("haz", "exp", "impf_set"), (haz_tag, exp_tag, impf_set_tag)
            ):
                file.create_group(f"tag/{group}")
                write_tag(file["tag"][group], kwds)

        # Load and check
        impact = Impact.from_hdf5(self.filepath)
        npt.assert_array_equal(impact.imp_mat.toarray(), imp_mat.toarray())
        npt.assert_array_equal(impact.event_id, event_id)
        npt.assert_array_equal(impact.event_name, event_name)
        self.assertIsInstance(impact.event_name, list)
        npt.assert_array_equal(impact.date, date)
        npt.assert_array_equal(impact.coord_exp, coord_exp)
        npt.assert_array_equal(impact.eai_exp, eai_exp)
        npt.assert_array_equal(impact.at_event, at_event)
        npt.assert_array_equal(impact.frequency, frequency)
        self.assertEqual(impact.crs, crs)
        self.assertEqual(impact.frequency_unit, frequency_unit)
        self.assertEqual(impact.tot_value, tot_value)
        self.assertEqual(impact.aai_agg, aai_agg)
        self.assertEqual(impact.unit, unit)
        self.assertEqual(impact.tag["haz"].__dict__, haz_tag)
        self.assertEqual(impact.tag["exp"].__dict__, exp_tag)
        self.assertEqual(impact.tag["impf_set"].__dict__, impf_set_tag)

        # Check with sparse
        with h5py.File(self.filepath, "r+") as file:
            del file["imp_mat"]
            file.create_group("imp_mat")
            file["imp_mat"].create_dataset("data", data=[1, 2, 3])
            file["imp_mat"].create_dataset("indices", data=[1, 2, 0])
            file["imp_mat"].create_dataset("indptr", data=[0, 2, 3])
            file["imp_mat"].attrs["shape"] = (2, 3)
        impact = Impact.from_hdf5(self.filepath)
        npt.assert_array_equal(impact.imp_mat.toarray(), [[0, 1, 2], [3, 0, 0]])


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
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactH5IO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactConcat))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
