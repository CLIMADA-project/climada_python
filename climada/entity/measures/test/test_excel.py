"""
Test MeasuresExcel class.
"""

import unittest
import numpy as np

from climada.entity.measures.source_excel import MeasuresExcel
from climada.util.constants import ENT_DEMO_XLS, ENT_TEMPLATE_XLS
 
class TestReader(unittest.TestCase):
    """Test reader functionality of the MeasuresExcel class"""

    def test_demo_file(self):
        # Read demo excel file
        meas = MeasuresExcel()
        description = 'One single file.'
        meas.read(ENT_DEMO_XLS, description)

        # Check results        
        n_meas = 4
        
        self.assertEqual(len(meas.data), n_meas)
        
        first = 0
        self.assertEqual(meas.data[first].name, 'Mangroves')
        self.assertEqual(type(meas.data[first].color_rgb), np.ndarray)
        self.assertEqual(len(meas.data[first].color_rgb), 3)
        self.assertEqual(meas.data[first].color_rgb[0], 0.1529)
        self.assertEqual(meas.data[first].color_rgb[1], 0.251)
        self.assertEqual(meas.data[first].color_rgb[2], 0.5451)
        self.assertEqual(meas.data[first].cost, 1311768360.8515418)
        self.assertEqual(meas.data[first].hazard_freq_cutoff, 0)
        self.assertEqual(meas.data[first].hazard_event_set, 'nil')
        self.assertEqual(meas.data[first].hazard_intensity, (1, -4))
        self.assertEqual(meas.data[first].mdd_impact, (1, 0))
        self.assertEqual(meas.data[first].paa_impact, (1, 0))
        self.assertEqual(meas.data[first].risk_transf_attach, 0)
        self.assertEqual(meas.data[first].risk_transf_cover, 0)
        
        self.assertEqual(meas.data[n_meas-1].name, 'Building code')
        self.assertEqual(type(meas.data[n_meas-1].color_rgb), np.ndarray)
        self.assertEqual(len(meas.data[n_meas-1].color_rgb), 3)
        self.assertEqual(meas.data[n_meas-1].color_rgb[0], 0.6980)
        self.assertEqual(meas.data[n_meas-1].color_rgb[1], 0.8745)
        self.assertEqual(meas.data[n_meas-1].color_rgb[2], 0.9333)
        self.assertEqual(meas.data[n_meas-1].cost, 9200000000.0000000)
        self.assertEqual(meas.data[n_meas-1].hazard_freq_cutoff, 0)
        self.assertEqual(meas.data[n_meas-1].hazard_event_set, 'nil')
        self.assertEqual(meas.data[n_meas-1].hazard_intensity, (1, 0))
        self.assertEqual(meas.data[n_meas-1].mdd_impact, (0.75, 0))
        self.assertEqual(meas.data[n_meas-1].paa_impact, (1, 0))
        self.assertEqual(meas.data[n_meas-1].risk_transf_attach, 0)
        self.assertEqual(meas.data[n_meas-1].risk_transf_cover, 0)

        self.assertEqual(meas.tag.file_name, ENT_DEMO_XLS)
        self.assertEqual(meas.tag.description, description)

    def test_template_file_pass(self):
        """ Read template excel file"""
        meas = MeasuresExcel()
        meas.read(ENT_TEMPLATE_XLS)
        # Check some results
        self.assertEqual(len(meas.data), 7)
        self.assertEqual(meas.data[4].paa_impact, (0.9, 0))
        self.assertEqual(meas.data[4].mdd_impact, (0.9, -0.1))
        self.assertEqual(meas.data[4].hazard_intensity, (1, -2))

    def test_wrong_file_fail(self):
        """ Read file intensity, fail."""
        meas = MeasuresExcel()
        meas.col_names['mdd_a'] = 'wrong name'
        with self.assertRaises(KeyError):
            meas.read(ENT_DEMO_XLS)

if __name__ == '__main__':
    unittest.main()
