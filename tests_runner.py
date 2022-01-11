import sys
from pathlib import Path
import unittest
import matplotlib

from climada.util.config import SOURCE_DIR

def find_unit_tests():
    """select unit tests."""
    suite = unittest.TestLoader().discover('climada.entity.exposures.test')
    suite.addTest(unittest.TestLoader().discover('climada.entity.disc_rates.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.impact_funcs.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.measures.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.centroids.test'))
    suite.addTest(unittest.TestLoader().discover('climada.engine.test'))
    suite.addTest(unittest.TestLoader().discover('climada.engine.unsequa.test'))
    suite.addTest(unittest.TestLoader().discover('climada.util.test'))
    return suite

def find_integ_tests():
    """select integration tests."""
    suite = unittest.TestLoader().discover('climada.test')
    return suite

def main():
    """parse input argument: None, 'unit' or 'integ'. Execute accordingly."""
    if sys.argv[1:]:
        import xmlrunner
        arg = sys.argv[1]
        output = str(Path(__file__).parent.joinpath('tests_xml'))
        if arg == 'unit':
            xmlrunner.XMLTestRunner(output=output).run(find_unit_tests())
        elif arg == 'integ':
            xmlrunner.XMLTestRunner(output=output).run(find_integ_tests())
        else:
            xmlrunner.XMLTestRunner(output=output).run(
                unittest.TestLoader().discover(arg)
            )
    else:
        # execute without xml reports
        unittest.TextTestRunner(verbosity=2).run(find_unit_tests())

if __name__ == '__main__':
    matplotlib.use("Agg")
    sys.path.append(str(SOURCE_DIR))
    main()
