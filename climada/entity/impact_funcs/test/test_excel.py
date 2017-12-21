"""
Test ImpactFuncsExcel class.
"""

import unittest

from climada.entity.impact_funcs.source_excel import ImpactFuncsExcel
from climada.util.constants import ENT_DEMO_XLS

class TestReader(unittest.TestCase):
    """Test reader functionality of the ImpactFuncsExcel class"""

    def test_one_file(self):
        """ Read one single excel file"""

        # Read demo excel file
        impact = ImpactFuncsExcel()
        description = 'One single file.'
        impact._read(ENT_DEMO_XLS, description)

        # Check results
        n_funcs = 2
        hazard = 'TC'
        first_id = 1
        second_id = 3

        self.assertEqual(len(impact.data), 1)
        self.assertEqual(len(impact.data[hazard]), n_funcs)

        # first function
        self.assertEqual(impact.data[hazard][first_id].id, 1)
        self.assertEqual(impact.data[hazard][first_id].name,
                         'Tropical cyclone default')
        self.assertEqual(impact.data[hazard][first_id].intensity_unit, 'm/s')

        self.assertEqual(len(impact.data[hazard][first_id].intensity), 9)
        self.assertEqual(impact.data[hazard][first_id].intensity[0], 0)
        self.assertEqual(impact.data[hazard][first_id].intensity[1], 20)
        self.assertEqual(impact.data[hazard][first_id].intensity[2], 30)
        self.assertEqual(impact.data[hazard][first_id].intensity[3], 40)
        self.assertEqual(impact.data[hazard][first_id].intensity[4], 50)
        self.assertEqual(impact.data[hazard][first_id].intensity[5], 60)
        self.assertEqual(impact.data[hazard][first_id].intensity[6], 70)
        self.assertEqual(impact.data[hazard][first_id].intensity[7], 80)
        self.assertEqual(impact.data[hazard][first_id].intensity[8], 100)

        self.assertEqual(len(impact.data[hazard][first_id].mdd), 9)
        self.assertEqual(impact.data[hazard][first_id].mdd[0], 0)
        self.assertEqual(impact.data[hazard][first_id].mdd[8], 0.41079600)

        self.assertEqual(len(impact.data[hazard][first_id].paa), 9)
        self.assertEqual(impact.data[hazard][first_id].paa[0], 0)
        self.assertEqual(impact.data[hazard][first_id].paa[8], 1)

        # second function
        self.assertEqual(impact.data[hazard][second_id].id, 3)
        self.assertEqual(impact.data[hazard][second_id].name,
                         'TC Building code')
        self.assertEqual(impact.data[hazard][first_id].intensity_unit, 'm/s')

        self.assertEqual(len(impact.data[hazard][second_id].intensity), 9)
        self.assertEqual(impact.data[hazard][second_id].intensity[0], 0)
        self.assertEqual(impact.data[hazard][second_id].intensity[1], 20)
        self.assertEqual(impact.data[hazard][second_id].intensity[2], 30)
        self.assertEqual(impact.data[hazard][second_id].intensity[3], 40)
        self.assertEqual(impact.data[hazard][second_id].intensity[4], 50)
        self.assertEqual(impact.data[hazard][second_id].intensity[5], 60)
        self.assertEqual(impact.data[hazard][second_id].intensity[6], 70)
        self.assertEqual(impact.data[hazard][second_id].intensity[7], 80)
        self.assertEqual(impact.data[hazard][second_id].intensity[8], 100)

        self.assertEqual(len(impact.data[hazard][second_id].mdd), 9)
        self.assertEqual(impact.data[hazard][second_id].mdd[0], 0)
        self.assertEqual(impact.data[hazard][second_id].mdd[8], 0.4)

        self.assertEqual(len(impact.data[hazard][second_id].paa), 9)
        self.assertEqual(impact.data[hazard][second_id].paa[0], 0)
        self.assertEqual(impact.data[hazard][second_id].paa[8], 1)

        # general information
        self.assertEqual(impact.tag.file_name, ENT_DEMO_XLS)
        self.assertEqual(impact.tag.description, description)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestReader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
