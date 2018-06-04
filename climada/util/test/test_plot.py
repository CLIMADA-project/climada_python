"""
Test plot module.
"""

import unittest

import climada.util.plot as plot

class TestFunc(unittest.TestCase):
    '''Test the auxiliary used with plot functions'''

    def test_get_coastlines_all_pass(self):
        '''Check get_coastlines function over whole earth'''
        lat, lon = plot.get_coastlines()
        self.assertEqual(410956, lat.size)
        self.assertEqual(-67.40048593499995, lat[0])
        self.assertEqual(80.59857819200005, lat[-1])

        self.assertEqual(410956, lon.size)
        self.assertEqual(59.916026238000086, lon[0])
        self.assertEqual(-66.77562415299991, lon[-1])

    def test_get_coastlines_pass(self):
        '''Check get_coastlines function in defined border'''
        border = (-100, 95, -55, 35)
        lat, lon = plot.get_coastlines(border)

        for lat_val, lon_val in zip(lat, lon):
            if lon_val < border[0] or lon_val > border[1]:
                self.assertTrue(False)
            if lat_val < border[2] or lat_val > border[3]:
                self.assertTrue(False)

        self.assertEqual(85383, lat.size)
        self.assertTrue(85383, lon.size)

TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFunc)
unittest.TextTestRunner(verbosity=2).run(TESTS)
