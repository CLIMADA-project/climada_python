#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest
import matplotlib

from climada.util.constants import SOURCE_DIR

def find_install_tests():
    """select unit tests."""
    suite = unittest.TestLoader().discover(start_dir='climada.engine.test',
                                           pattern='test_cost_benefit.py')
    suite.addTest(unittest.TestLoader().discover(start_dir='climada.engine.test',
                                                 pattern='test_impact.py'))
    return suite

def main():
    """parse input argument: None or 'report'. Execute accordingly."""
    if sys.argv[1:]:
        import xmlrunner
        arg = sys.argv[1]
        if arg == 'report':
            output = os.path.join(SOURCE_DIR, '../tests_xml')
            xmlrunner.XMLTestRunner(output=output).run(find_install_tests())
    else:
        # execute without xml reports
        unittest.TextTestRunner(verbosity=2).run(find_install_tests())

if __name__ == '__main__':
    matplotlib.use("Agg")
    sys.path.append(os.getcwd())
    main()
