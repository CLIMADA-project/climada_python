#!/usr/bin/env python
# coding: utf-8

import os
import sys
import unittest
import matplotlib

def find_install_tests():
    """ select unit tests."""
    suite = unittest.TestLoader().discover('climada.engine.test.test_cost_benefit')
    suite.addTest(unittest.TestLoader().discover('climada.engine.test.test_impact'))
    return suite

def main():
    """ parse input argument: None, 'unit' or 'integ'. Execute accordingly."""
    unittest.TextTestRunner(verbosity=2).run(find_install_tests())

if __name__ == '__main__':
    matplotlib.use("Agg")
    sys.path.append(os.getcwd())
    main()
