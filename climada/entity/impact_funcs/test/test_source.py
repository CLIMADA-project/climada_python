"""
Test imp_funcsFuncs from MATLAB class.
"""

import unittest
import pandas

import climada.util.hdf5_handler as hdf5
from climada.entity.impact_funcs.base import ImpactFuncSet
from climada.entity.impact_funcs.source import DEF_VAR_MAT, DEF_VAR_EXCEL, _get_xls_funcs, _get_hdf5_funcs, _get_hdf5_unit, _get_hdf5_name
from climada.util.constants import ENT_DEMO_MAT, ENT_TEST_XLS, ENT_TEMPLATE_XLS

class TestReaderMat(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_demo_file_pass(self):
        """ Read demo excel file"""
        # Read demo excel file
        imp_funcs = ImpactFuncSet()
        description = 'One single file.'
        imp_funcs.read(ENT_DEMO_MAT, description)

        # Check results
        n_funcs = 2
        hazard = 'TC'
        first_id = 1
        second_id = 3

        self.assertEqual(len(imp_funcs._data), 1)
        self.assertEqual(len(imp_funcs._data[hazard]), n_funcs)

        # first function
        self.assertEqual(imp_funcs._data[hazard][first_id].id, 1)
        self.assertEqual(imp_funcs._data[hazard][first_id].name,
                         'Tropical cyclone default')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit, \
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][first_id].intensity.shape, \
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][first_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[8], 0.41079600)

        self.assertEqual(imp_funcs._data[hazard][first_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[8], 1)

        # second function
        self.assertEqual(imp_funcs._data[hazard][second_id].id, 3)
        self.assertEqual(imp_funcs._data[hazard][second_id].name,
                         'TC Building code')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit, \
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][second_id].intensity.shape, \
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][second_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[8], 0.4)

        self.assertEqual(imp_funcs._data[hazard][second_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[8], 1)

        # general information
        self.assertEqual(imp_funcs.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(imp_funcs.tag.description, description)

class TestGetsMat(unittest.TestCase):
    """Test functions to retrieve specific variables"""

    def setUp(self):
        self.imp = hdf5.read(ENT_DEMO_MAT)
        self.imp = self.imp[DEF_VAR_MAT['sup_field_name']]
        self.imp = self.imp[DEF_VAR_MAT['field_name']]

    def test_rows_pass(self):
        """Check get_funcs_rows."""
        funcs = _get_hdf5_funcs(self.imp, ENT_DEMO_MAT, DEF_VAR_MAT)
        self.assertEqual(len(funcs), 2)
        
        self.assertEqual(len(funcs[('TC', 1)]), 9)
        self.assertEqual(len(funcs[('TC', 3)]), 9)
        for i in range(9):
            self.assertEqual(funcs[('TC', 1)][i], i)
            self.assertEqual(funcs[('TC', 3)][i], 9 + i)

    def test_unit_pass(self):
        """Check get_imp_fun_unit"""
        funcs = _get_hdf5_funcs(self.imp, ENT_DEMO_MAT, DEF_VAR_MAT)
        fun_unit = _get_hdf5_unit(self.imp, funcs[('TC', 3)], \
                                    ENT_DEMO_MAT, DEF_VAR_MAT)
        self.assertEqual(fun_unit, 'm/s')

        fun_unit = _get_hdf5_unit(self.imp, \
                                 funcs[('TC', 1)], \
                                 ENT_DEMO_MAT, DEF_VAR_MAT)
        self.assertEqual(fun_unit, 'm/s')
        
    def test_name_pass(self):
        """Check get_imp_fun_unit"""
        funcs = _get_hdf5_funcs(self.imp, ENT_DEMO_MAT, DEF_VAR_MAT)
        fun_name = _get_hdf5_name(self.imp, funcs[('TC', 1)], \
                                    ENT_DEMO_MAT, DEF_VAR_MAT)
        self.assertEqual(fun_name, 'Tropical cyclone default')

        fun_name = _get_hdf5_name(self.imp, \
                                 funcs[('TC', 3)], \
                                 ENT_DEMO_MAT, DEF_VAR_MAT)
        self.assertEqual(fun_name, 'TC Building code')

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_demo_file_pass(self):
        """ Read demo excel file"""
        # Read demo excel file
        imp_funcs = ImpactFuncSet()
        description = 'One single file.'
        imp_funcs.read(ENT_TEST_XLS, description)

        # Check results
        n_funcs = 2
        hazard = 'TC'
        first_id = 1
        second_id = 3

        self.assertEqual(len(imp_funcs._data), 1)
        self.assertEqual(len(imp_funcs._data[hazard]), n_funcs)

        # first function
        self.assertEqual(imp_funcs._data[hazard][first_id].id, 1)
        self.assertEqual(imp_funcs._data[hazard][first_id].name,
                         'Tropical cyclone default')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit, \
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][first_id].intensity.shape, \
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][first_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].mdd[8], 0.41079600)

        self.assertEqual(imp_funcs._data[hazard][first_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][first_id].paa[8], 1)

        # second function
        self.assertEqual(imp_funcs._data[hazard][second_id].id, 3)
        self.assertEqual(imp_funcs._data[hazard][second_id].name,
                         'TC Building code')
        self.assertEqual(imp_funcs._data[hazard][first_id].intensity_unit, \
                         'm/s')

        self.assertEqual(imp_funcs._data[hazard][second_id].intensity.shape, \
                         (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[1], 20)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[2], 30)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[3], 40)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[4], 50)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[5], 60)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[6], 70)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[7], 80)
        self.assertEqual(imp_funcs._data[hazard][second_id].intensity[8], 100)

        self.assertEqual(imp_funcs._data[hazard][second_id].mdd.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].mdd[8], 0.4)

        self.assertEqual(imp_funcs._data[hazard][second_id].paa.shape, (9,))
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[0], 0)
        self.assertEqual(imp_funcs._data[hazard][second_id].paa[8], 1)

        # general information
        self.assertEqual(imp_funcs.tag.file_name, ENT_TEST_XLS)
        self.assertEqual(imp_funcs.tag.description, description)

    def test_template_file_pass(self):
        """ Read template excel file"""
        imp_funcs = ImpactFuncSet()
        imp_funcs.read(ENT_TEMPLATE_XLS)
        # Check some results
        self.assertEqual(len(imp_funcs._data), 10)
        self.assertEqual(len(imp_funcs._data['TC'][3].paa), 9)
        self.assertEqual(len(imp_funcs._data['EQ'][1].intensity), 14)
        self.assertEqual(len(imp_funcs._data['HS'][1].mdd), 16)
            
class TestFuncsExcel(unittest.TestCase):
    """Test reader functionality of the imp_funcsFuncsExcel class"""

    def test_distinct_funcs(self):
        """ Read demo excel file"""
        dfr = pandas.read_excel(ENT_TEST_XLS, DEF_VAR_EXCEL['sheet_name'])
        imp_funcs = _get_xls_funcs(dfr, DEF_VAR_EXCEL)
        self.assertEqual(imp_funcs[0], ('TC', 1))
        self.assertEqual(imp_funcs[1], ('TC', 3))

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderMat)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGetsMat))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFuncsExcel))
unittest.TextTestRunner(verbosity=2).run(TESTS)
