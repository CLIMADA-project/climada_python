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

Test Exposures from MATLAB file.
"""
import unittest
import copy

from climada import CONFIG
from climada.entity.exposures.base import Exposures, DEF_VAR_MAT

ENT_TEST_MAT = CURR_DIR = CONFIG.exposures.test_data.dir().joinpath('demo_today.mat')

class TestReader(unittest.TestCase):
    """Test reader functionality of the ExposuresMat class"""

    def test_read_demo_pass(self):
        """Read one single excel file"""
        # Read demo excel file
        expo = Exposures.from_mat(ENT_TEST_MAT)

        # Check results
        n_expos = 50

        self.assertEqual(expo.gdf.index.shape, (n_expos,))
        self.assertEqual(expo.gdf.index[0], 0)
        self.assertEqual(expo.gdf.index[n_expos - 1], n_expos - 1)

        self.assertEqual(expo.gdf.value.shape, (n_expos,))
        self.assertEqual(expo.gdf.value[0], 13927504367.680632)
        self.assertEqual(expo.gdf.value[n_expos - 1], 12624818493.687229)

        self.assertEqual(expo.gdf.deductible.shape, (n_expos,))
        self.assertEqual(expo.gdf.deductible[0], 0)
        self.assertEqual(expo.gdf.deductible[n_expos - 1], 0)

        self.assertEqual(expo.gdf.cover.shape, (n_expos,))
        self.assertEqual(expo.gdf.cover[0], 13927504367.680632)
        self.assertEqual(expo.gdf.cover[n_expos - 1], 12624818493.687229)

        self.assertIn('int', str(expo.gdf.impf_.dtype))
        self.assertEqual(expo.gdf.impf_.shape, (n_expos,))
        self.assertEqual(expo.gdf.impf_[0], 1)
        self.assertEqual(expo.gdf.impf_[n_expos - 1], 1)

        self.assertIn('int', str(expo.gdf.category_id.dtype))
        self.assertEqual(expo.gdf.category_id.shape, (n_expos,))
        self.assertEqual(expo.gdf.category_id[0], 1)
        self.assertEqual(expo.gdf.category_id[n_expos - 1], 1)

        self.assertIn('int', str(expo.gdf.centr_.dtype))
        self.assertEqual(expo.gdf.centr_.shape, (n_expos,))
        self.assertEqual(expo.gdf.centr_[0], 47)
        self.assertEqual(expo.gdf.centr_[n_expos - 1], 46)

        self.assertTrue('region_id' not in expo.gdf)

        self.assertEqual(expo.gdf.latitude.shape, (n_expos,))
        self.assertEqual(expo.gdf.latitude[0], 26.93389900000)
        self.assertEqual(expo.gdf.latitude[n_expos - 1], 26.34795700000)
        self.assertEqual(expo.gdf.longitude[0], -80.12879900000)
        self.assertEqual(expo.gdf.longitude[n_expos - 1], -80.15885500000)

        self.assertEqual(expo.ref_year, 2016)
        self.assertEqual(expo.value_unit, 'USD')
        self.assertEqual(expo.tag.file_name, str(ENT_TEST_MAT))

class TestObligatories(unittest.TestCase):
    """Test reading exposures obligatory values."""

    def test_no_value_fail(self):
        """Error if no values."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['val'] = 'no valid value'
        with self.assertRaises(KeyError):
            Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

    def test_no_impact_fail(self):
        """Error if no impact ids."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['impf'] = 'no valid value'
        with self.assertRaises(KeyError):
            Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

    def test_no_coord_fail(self):
        """Error if no coordinates."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['lat'] = 'no valid Latitude'
        with self.assertRaises(KeyError):
            Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        new_var_names['var_name']['lat'] = 'nLatitude'
        new_var_names['var_name']['lon'] = 'no valid Longitude'
        with self.assertRaises(KeyError):
            Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

class TestOptionals(unittest.TestCase):
    """Test reading exposures optional values."""

    def test_no_category_pass(self):
        """Not error if no category id."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['cat'] = 'no valid category'
        exp = Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        # Check results
        self.assertTrue('category_id' not in exp.gdf)

    def test_no_region_pass(self):
        """Not error if no region id."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['reg'] = 'no valid region'
        exp = Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        # Check results
        self.assertTrue('region_id' not in exp.gdf)

    def test_no_unit_pass(self):
        """Not error if no value unit."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['uni'] = 'no valid unit'
        exp = Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual('USD', exp.value_unit)

    def test_no_assigned_pass(self):
        """Not error if no value unit."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['ass'] = 'no valid assign'
        exp = Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        # Check results
        self.assertTrue('centr_' not in exp.gdf)

    def test_no_refyear_pass(self):
        """Not error if no value unit."""
        new_var_names = copy.deepcopy(DEF_VAR_MAT)
        new_var_names['var_name']['ref'] = 'no valid ref'
        exp = Exposures.from_mat(ENT_TEST_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual(2018, exp.ref_year)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptionals))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestObligatories))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
