"""
Test Exposures from Excel file.
"""
import os
import unittest
import numpy as np
import pandas

from climada.entity.exposures import source as source
from climada.entity.exposures.base import Exposures
from climada.util.constants import ENT_TEMPLATE_XLS
from climada.util.config import CONFIG

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

class TestReader(unittest.TestCase):
    """Test reader functionality of the ExposuresExcel class"""

    def test_read_demo_pass(self):
        """ Read one single excel file"""
        # Read demo excel file
        expo = Exposures()
        description = 'One single file.'
        expo.read(ENT_TEST_XLS, description)

        # Check results
        n_expos = 50

        self.assertIn('int', str(expo.id.dtype))
        self.assertEqual(len(expo.id), n_expos)
        self.assertEqual(expo.id[0], 0)
        self.assertEqual(expo.id[n_expos-1], n_expos-1)

        self.assertEqual(len(expo.value), n_expos)
        self.assertEqual(expo.value[0], 13927504367.680632)
        self.assertEqual(expo.value[n_expos-1], 12624818493.687229)

        self.assertEqual(len(expo.deductible), n_expos)
        self.assertEqual(expo.deductible[0], 0)
        self.assertEqual(expo.deductible[n_expos-1], 0)

        self.assertEqual(len(expo.cover), n_expos)
        self.assertEqual(expo.cover[0], 13927504367.680632)
        self.assertEqual(expo.cover[n_expos-1], 12624818493.687229)

        self.assertIn('int', str(expo.impact_id.dtype))
        self.assertEqual(len(expo.impact_id), n_expos)
        self.assertEqual(expo.impact_id[0], 1)
        self.assertEqual(expo.impact_id[n_expos-1], 1)

        self.assertEqual(len(expo.category_id), 0)
        self.assertEqual(len(expo.region_id), 0)
        self.assertEqual(len(expo.assigned), 0)

        self.assertEqual(expo.coord.shape[0], n_expos)
        self.assertEqual(expo.coord.shape[1], 2)
        self.assertEqual(expo.coord[0][0], 26.93389900000)
        self.assertEqual(expo.coord[n_expos-1][0], 26.34795700000)
        self.assertEqual(expo.coord[0][1], -80.12879900000)
        self.assertEqual(expo.coord[n_expos-1][1], -80.15885500000)

        self.assertEqual(expo.ref_year, CONFIG['entity']["present_ref_year"])
        self.assertEqual(expo.value_unit, 'NA')
        self.assertEqual(expo.tag.file_name, ENT_TEST_XLS)
        self.assertEqual(expo.tag.description, description)

    def test_read_template_pass(self):
        """Check template excel file. Catch warning centroids not set."""
        # Read template file
        expo = Exposures()
        expo.read(ENT_TEMPLATE_XLS)

        # Check results
        n_expos = 24

        self.assertIn('int', str(expo.id.dtype))
        self.assertEqual(expo.id.shape, (n_expos,))
        self.assertEqual(expo.id[0], 0)
        self.assertEqual(expo.id[n_expos-1], n_expos-1)

        self.assertEqual(expo.value.shape, (n_expos,))
        self.assertEqual(expo.value[0], 13927504367.680632)
        self.assertEqual(expo.value[n_expos-1], 12597535489.94726)

        self.assertEqual(expo.deductible.shape, (n_expos,))
        self.assertEqual(expo.deductible[0], 0)
        self.assertEqual(expo.deductible[n_expos-1], 0)

        self.assertEqual(expo.cover.shape, (n_expos,))
        self.assertEqual(expo.cover[0], 13927504367.680632)
        self.assertEqual(expo.cover[n_expos-1], 12597535489.94726)

        self.assertIn('int', str(expo.impact_id.dtype))
        self.assertEqual(expo.impact_id.shape, (n_expos,))
        self.assertEqual(expo.impact_id[0], 1)
        self.assertEqual(expo.impact_id[n_expos-1], 1)

        self.assertIn('int', str(expo.category_id.dtype))        
        self.assertEqual(expo.category_id.shape, (n_expos,))
        self.assertEqual(expo.category_id[0], 1)
        self.assertEqual(expo.category_id[n_expos-1], 1)

        self.assertIn('int', str(expo.region_id.dtype)) 
        self.assertEqual(expo.region_id.shape, (n_expos,))
        self.assertEqual(expo.region_id[0], 1)
        self.assertEqual(expo.region_id[n_expos-1], 1)

        self.assertTrue(isinstance(expo.assigned, dict))
        self.assertEqual(expo.value_unit, 'USD')

        self.assertEqual(expo.coord.shape, (n_expos, 2))
        self.assertEqual(expo.coord[0][0], 26.93389900000)
        self.assertEqual(expo.coord[n_expos-1][0], 26.663149)
        self.assertEqual(expo.coord[0][1], -80.12879900000)
        self.assertEqual(expo.coord[n_expos-1][1], -80.151401)

        self.assertEqual(expo.ref_year, 2017)
        self.assertEqual(expo.tag.file_name, ENT_TEMPLATE_XLS)
        self.assertEqual(expo.tag.description, '')

    def test_check_template_warning(self):
        """Check warning centroids when template read."""
        with self.assertLogs('climada.util.checker', level='WARNING') as cm:
            Exposures(ENT_TEMPLATE_XLS)
        self.assertIn("Exposures.assigned not set.", cm.output[0])

    def test_check_demo_warning(self):
        """Check warning centroids when demo read."""
        with self.assertLogs('climada.util.checker', level='WARNING') as cm:
            Exposures(ENT_TEST_XLS)
        self.assertIn("Exposures.category_id not set.", cm.output[0])
        self.assertIn("Exposures.region_id not set.", cm.output[1])
        self.assertIn("Exposures.assigned not set.", cm.output[2])

class TestObligatories(unittest.TestCase):
    """Test reading exposures obligatory values."""

    def tearDown(self):
        source.DEF_VAR_EXCEL = {'sheet_name': {'exp': 'assets',
                                             'name': 'names'
                                            },
                              'col_name': {'lat' : 'Latitude',
                                           'lon' : 'Longitude',
                                           'val' : 'Value',
                                           'ded' : 'Deductible',
                                           'cov' : 'Cover',
                                           'imp' : 'DamageFunID',
                                           'cat' : 'Category_ID',
                                           'reg' : 'Region_ID',
                                           'uni' : 'Value unit',
                                           'ass' : 'centroid_index',
                                           'ref': 'reference_year',
                                           'item' : 'Item'
                                          }
                              }

    def test_no_value_fail(self):
        """Error if no values."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['val'] = 'no valid value'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_TEST_XLS, var_names=new_var_names)

    def test_no_impact_fail(self):
        """Error if no impact ids."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['imp'] = 'no valid impact'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_TEST_XLS, var_names=new_var_names)

    def test_no_coord_fail(self):
        """Error if no coordinates."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['lat'] = 'no valid Latitude'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_TEST_XLS, var_names=new_var_names)

        new_var_names['col_name']['lat'] = 'Latitude'
        new_var_names['col_name']['lon'] = 'no valid Longitude'
        with self.assertRaises(KeyError):
            expo.read(ENT_TEST_XLS, var_names=new_var_names)

class TestOptionals(unittest.TestCase):
    """Test reading exposures optional values."""

    def tearDown(self):
         source.DEF_VAR_EXCEL = {'sheet_name': {'exp': 'assets',
                                             'name': 'names'
                                            },
                              'col_name': {'lat' : 'Latitude',
                                           'lon' : 'Longitude',
                                           'val' : 'Value',
                                           'ded' : 'Deductible',
                                           'cov' : 'Cover',
                                           'imp' : 'DamageFunID',
                                           'cat' : 'Category_ID',
                                           'reg' : 'Region_ID',
                                           'uni' : 'Value unit',
                                           'ass' : 'centroid_index',
                                           'ref': 'reference_year',
                                           'item' : 'Item'
                                          }
                              }

    def test_no_category_pass(self):
        """Not error if no category id."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['cat'] = 'no valid category'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertEqual(0, expo.category_id.size)

    def test_no_region_pass(self):
        """Not error if no region id."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['reg'] = 'no valid region'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertEqual(0, expo.region_id.size)

    def test_no_unit_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['uni'] = 'no valid value unit'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertEqual('NA', expo.value_unit)

    def test_no_assigned_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['ass'] = 'no valid assign'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertEqual(0, len(expo.assigned))

    def test_no_refyear_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['ref'] = 'no valid ref'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertEqual(CONFIG['entity']["present_ref_year"], expo.ref_year)

class TestDefaults(unittest.TestCase):
    """Test reading exposures default values."""

    def tearDown(self):
         source.DEF_VAR_EXCEL = {'sheet_name': {'exp': 'assets',
                                             'name': 'names'
                                            },
                               'col_name': {'lat' : 'Latitude',
                                            'lon' : 'Longitude',
                                            'val' : 'Value',
                                            'ded' : 'Deductible',
                                            'cov' : 'Cover',
                                            'imp' : 'DamageFunID',
                                            'cat' : 'Category_ID',
                                            'reg' : 'Region_ID',
                                            'uni' : 'Value unit',
                                            'ass' : 'centroid_index',
                                            'ref': 'reference_year',
                                            'item' : 'Item'
                                           }
                               }

    def test_no_cover_pass(self):
        """Check default values for excel file with no cover."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['cov'] = 'Dummy'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertTrue(np.array_equal(expo.value, expo.cover))

    def test_no_deductible_pass(self):
        """Check default values for excel file with no deductible."""
        new_var_names = source.DEF_VAR_EXCEL
        new_var_names['col_name']['ded'] = 'Dummy'
        expo = Exposures()
        expo.read(ENT_TEST_XLS, var_names=new_var_names)

        # Check results
        self.assertTrue(np.array_equal(np.zeros(len(expo.value)), \
                                              expo.deductible))

class TestParsers(unittest.TestCase):
    """Test parser auxiliary functions"""

    def setUp(self):
        self.dfr = pandas.read_excel(ENT_TEMPLATE_XLS, 'assets')

    def test_parse_optional_exist_pass(self):
        """Check variable read if present."""
        var_ini = 0
        var = source._parse_xls_optional(self.dfr, var_ini, 'Latitude')
        self.assertEqual(24, len(var))

    def test_parse_optional_not_exist_pass(self):
        """Check pass if variable not present and initial value kept."""
        var_ini = 0
        var = source._parse_xls_optional(self.dfr, var_ini, 'Not Present')
        self.assertEqual(var_ini, var)

    def test_parse_default_exist_pass(self):
        """Check variable read if present."""
        def_val = 5
        var = source._parse_xls_default(self.dfr, 'Latitude', def_val)
        self.assertEqual(24, len(var))

    def test_parse_default_not_exist_pass(self):
        """Check pass if variable not present and default value is set."""
        def_val = 5
        var = source._parse_xls_default(self.dfr, 'Not Present', def_val)
        self.assertEqual(def_val, var)

    def test_refyear_exist_pass(self):
        """Check that the reference year is well read from template."""
        ref_year = source._parse_xls_ref_year(ENT_TEMPLATE_XLS, source.DEF_VAR_EXCEL)
        self.assertEqual(2017, ref_year)

    def test_refyear_not_exist_pass(self):
        """Check that the reference year is default if not present."""
        ref_year = source._parse_xls_ref_year(ENT_TEST_XLS, source.DEF_VAR_EXCEL)
        self.assertEqual(CONFIG['entity']["present_ref_year"], ref_year)      
        
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDefaults)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptionals))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestObligatories))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParsers))
unittest.TextTestRunner(verbosity=2).run(TESTS)
