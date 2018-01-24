#!/usr/bin/env python
# coding: utf-8

import unittest
import xmlrunner

def runner(output='python_tests_xml'):
    return xmlrunner.XMLTestRunner(
        output=output
    )

def find_tests():
    suite.addTest(unittest.TestLoader().discover('climada.test'))
    return suite

if __name__ == '__main__':
    #unittest.TextTestRunner(verbosity=2).run(find_tests())
    runner().run(find_tests())
