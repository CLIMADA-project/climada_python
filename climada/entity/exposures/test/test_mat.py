"""
Test ExposuresMat class.
"""

import warnings
import unittest
import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.exposures.source_mat import ExposuresMat
from climada.util.constants import ENT_DEMO_MAT
from climada.util.config import config

class TestReader(unittest.TestCase):
    """Test reader functionality of the ExposuresMat class"""

    def test_read_demo_pass(self):
        """ Read one single excel file"""
        # Read demo excel file
        expo = ExposuresMat()
        description = 'One single file.'
        expo.read(ENT_DEMO_MAT, description)

        # Check results
        n_expos = 50

        self.assertEqual(type(expo.id[0]), np.int64)
        self.assertEqual(expo.id.shape, (n_expos,))
        self.assertEqual(expo.id[0], 0)
        self.assertEqual(expo.id[n_expos-1], n_expos-1)

        self.assertEqual(expo.value.shape, (n_expos,))
        self.assertEqual(expo.value[0], 13927504367.680632)
        self.assertEqual(expo.value[n_expos-1], 12624818493.687229)

        self.assertEqual(expo.deductible.shape, (n_expos,))
        self.assertEqual(expo.deductible[0], 0)
        self.assertEqual(expo.deductible[n_expos-1], 0)

        self.assertEqual(expo.cover.shape, (n_expos,))
        self.assertEqual(expo.cover[0], 13927504367.680632)
        self.assertEqual(expo.cover[n_expos-1], 12624818493.687229)

        self.assertEqual(type(expo.impact_id[0]), np.int64)
        self.assertEqual(expo.impact_id.shape, (n_expos,))
        self.assertEqual(expo.impact_id[0], 1)
        self.assertEqual(expo.impact_id[n_expos-1], 1)

        self.assertEqual(type(expo.category_id[0]), np.int64)
        self.assertEqual(expo.category_id.shape, (n_expos,))
        self.assertEqual(expo.category_id[0], 1)
        self.assertEqual(expo.category_id[n_expos-1], 1)

        self.assertEqual(type(expo.assigned[0]), np.int64)
        self.assertEqual(expo.assigned.shape, (n_expos,))
        self.assertEqual(expo.assigned[0], 47)
        self.assertEqual(expo.assigned[n_expos-1], 46)

        self.assertEqual(expo.region_id.shape, (0,))

        self.assertEqual(expo.coord.shape, (n_expos, 2))
        self.assertEqual(expo.coord[0][0], 26.93389900000)
        self.assertEqual(expo.coord[n_expos-1][0], 26.34795700000)
        self.assertEqual(expo.coord[0][1], -80.12879900000)
        self.assertEqual(expo.coord[n_expos-1][1], -80.15885500000)

        self.assertEqual(expo.ref_year, config["present_ref_year"])
        self.assertEqual(expo.value_unit, 'USD')
        self.assertEqual(expo.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(expo.tag.description, description)

    def test_check_demo_warning(self):
        """Check warning centroids when demo read."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ExposuresMat(ENT_DEMO_MAT)
            # Verify warnings thrown
            self.assertIn("Exposures.region_id not set.", \
                          str(w[0].message))

class TestObligatories(unittest.TestCase):
    """Test reading exposures obligatory values."""

    def test_no_value_fail(self):
        """Error if no values."""
        expo = ExposuresMat()
        expo.var['val'] = 'no valid value'
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT)

    def test_no_impact_fail(self):
        """Error if no impact ids."""
        expo = ExposuresMat()
        expo.var['imp'] = 'no valid impact'
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT)

    def test_no_coord_fail(self):
        """Error if no coordinates."""
        expo = ExposuresMat()
        expo.var['lat'] = 'no valid Latitude'
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT)

        expo.var['lat'] = 'Latitude'
        expo.var['lon'] = 'no valid Longitude'
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT)

class TestOptionals(unittest.TestCase):
    """Test reading exposures optional values."""

    def test_no_category_pass(self):
        """Not error if no category id."""
        expo = ExposuresMat()
        expo.var['cat'] = 'no valid category'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(0, expo.category_id.size)

    def test_no_region_pass(self):
        """Not error if no region id."""
        expo = ExposuresMat()
        expo.var['reg'] = 'no valid region'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(0, expo.region_id.size)

    def test_no_unit_pass(self):
        """Not error if no value unit."""
        expo = ExposuresMat()
        expo.var['uni'] = 'no valid value unit'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual('NA', expo.value_unit)

    def test_no_assigned_pass(self):
        """Not error if no value unit."""
        expo = ExposuresMat()
        expo.var['ass'] = 'no valid assign'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(0, expo.assigned.size)

    def test_no_refyear_pass(self):
        """Not error if no value unit."""
        expo = ExposuresMat()
        expo.var['ref'] = 'no valid ref'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(config["present_ref_year"], expo.ref_year)

class TestDefaults(unittest.TestCase):
    """Test reading exposures default values."""

    def test_no_cover_pass(self):
        """Check default values for excel file with no cover."""
        # Read demo excel file
        expo = ExposuresMat()
        # Change cover column name to simulate no present column
        expo.var['cov'] = 'Dummy'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(True, np.array_equal(expo.value, expo.cover))

    def test_no_deductible_pass(self):
        """Check default values for excel file with no deductible."""
        # Read demo excel file
        expo = ExposuresMat()
        # Change deductible column name to simulate no present column
        expo.var['ded'] = 'Dummy'
        expo.read(ENT_DEMO_MAT)

        # Check results
        self.assertEqual(True, np.array_equal(np.zeros(len(expo.value)), \
                                              expo.deductible))

class TestParsers(unittest.TestCase):
    """Test parser auxiliary functions"""

    def setUp(self):
        self.expo = hdf5.read(ENT_DEMO_MAT)
        self.expo = self.expo[ExposuresMat().sup_field_name]
        self.expo = self.expo[ExposuresMat().field_name]

    def test_parse_optional_exist_pass(self):
        """Check variable read if present."""
        var_ini = 0
        var = ExposuresMat._parse_optional(self.expo, var_ini, 'lat')
        self.assertEqual(50, len(var))

    def test_parse_optional_not_exist_pass(self):
        """Check pass if variable not present and initial value kept."""
        var_ini = 0
        var = ExposuresMat._parse_optional(self.expo, var_ini, 'Not Present')
        self.assertEqual(var_ini, var)

    def test_parse_default_exist_pass(self):
        """Check variable read if present."""
        def_val = 5
        var = ExposuresMat._parse_default(self.expo, 'lat', def_val)
        self.assertEqual(50, len(var))

    def test_parse_default_not_exist_pass(self):
        """Check pass if variable not present and default value is set."""
        def_val = 5
        var = ExposuresMat._parse_default(self.expo, 'Not Present', def_val)
        self.assertEqual(def_val, var)

if __name__ == '__main__':
    unittest.main()
