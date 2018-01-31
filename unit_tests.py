#!/usr/bin/env python
# coding: utf-8

import unittest
import xmlrunner

def runner(output='python_tests_xml'):
    return xmlrunner.XMLTestRunner(output=output)

def find_tests():
    suite = unittest.TestLoader().discover('climada.entity.exposures.test')
    suite.addTest(unittest.TestLoader().discover('climada.entity.discounts.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.impact_funcs.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.measures.test'))
    suite.addTest(unittest.TestLoader().discover('climada.entity.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.test'))
    suite.addTest(unittest.TestLoader().discover('climada.hazard.centroids.test'))
    suite.addTest(unittest.TestLoader().discover('climada.engine.test'))
    suite.addTest(unittest.TestLoader().discover('climada.util.test'))
    return suite

if __name__ == '__main__':
    #unittest.TextTestRunner(verbosity=2).run(find_tests())
    runner().run(find_tests())
