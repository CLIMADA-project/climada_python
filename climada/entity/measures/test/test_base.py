"""
Test Measures class.
"""

import unittest
import numpy

from climada.entity.measures.base import Measures, Measure
from climada.util.constants import ENT_DEMO_XLS

class TestLoader(unittest.TestCase):
    """Test reader functionality of the Measures class"""

    def test_check_wronginten_fail(self):
        """Wrong intensity definition"""
        meas = Measures()
        meas.data.append(Measure())
        meas.data[0].hazard_intensity = (1, 2, 3)
        meas.data[0].color_rgb = numpy.array([1, 1, 1])
        meas.data[0].mdd_impact = (1, 2)
        meas.data[0].paa_impact = (1, 2)
        
        with self.assertRaises(ValueError) as error:
            meas.check()
        self.assertEqual('Invalid measure hazard intensity size: 2 != 3', \
                         str(error.exception))

    def test_check_wrongColor_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        meas.data.append(Measure())
        meas.data[0].color_rgb = (1, 2)
        meas.data[0].mdd_impact = (1, 2)
        meas.data[0].paa_impact = (1, 2)
        meas.data[0].hazard_intensity = (1, 2)
        
        with self.assertRaises(ValueError) as error:
            meas.check()
        self.assertEqual('Invalid measure colour RGB size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongMDD_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        meas.data.append(Measure())
        meas.data[0].color_rgb = numpy.array([1, 1, 1])
        meas.data[0].mdd_impact = (1)
        meas.data[0].paa_impact = (1, 2)
        meas.data[0].hazard_intensity = (1, 2)
        
        with self.assertRaises(ValueError) as error:
            meas.check()
            self.assertEqual('measure MDD impact has wrong dimensions.', \
                 str(error.exception))

    def test_check_wrongPAA_fail(self):
        """Wrong discount rates definition"""
        meas = Measures()
        meas.data.append(Measure())
        meas.data[0].color_rgb = numpy.array([1, 1, 1])
        meas.data[0].mdd_impact = (1, 2)
        meas.data[0].paa_impact = (1, 2, 3, 4)
        meas.data[0].hazard_intensity = (1, 2)
        
        with self.assertRaises(ValueError) as error:
            meas.check()
        self.assertEqual('Invalid measure PAA impact size: 2 != 4', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        meas = Measures()
        with self.assertRaises(NotImplementedError):
            meas.load(ENT_DEMO_XLS)

    def test_read_notimplemented(self):
        """Read function not implemented"""
        meas = Measures()
        with self.assertRaises(NotImplementedError):
            meas.read(ENT_DEMO_XLS)

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Measures(ENT_DEMO_XLS)

# Execute TestReader
suite_reader = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
unittest.TextTestRunner(verbosity=2).run(suite_reader)
