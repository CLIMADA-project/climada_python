#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest

def find_unit_tests():
    suite = unittest.TestLoader().discover('climada.entity.exposures.test')
    suite.addTest(unittest.TestLoader().discover('climada.entity.disc_rates.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.impact_funcs.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.measures.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.centroids.test'))
    suite.addTest(unittest.TestLoader().discover('climada.engine.test'))
    suite.addTest(unittest.TestLoader().discover('climada.util.test'))
    return suite

def find_integ_tests():
    suite = unittest.TestLoader().discover('climada.test')
    return suite

def main():
    # print command line arguments
    if sys.argv[1:]:
        import xmlrunner
        arg = sys.argv[1]
        output = 'tests_xml'
        if arg == 'unit':
            xmlrunner.XMLTestRunner(output=output).run(find_unit_tests())
        elif arg == 'integ':
            xmlrunner.XMLTestRunner(output=output).run(find_integ_tests())
    else:
        unittest.TextTestRunner(verbosity=2).run(find_unit_tests())

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    print(sys.path)
    main()
