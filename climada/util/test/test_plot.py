"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test plot module.
"""

import unittest
import cartopy
import numpy as np
import matplotlib.pyplot as plt

import climada.util.plot as u_plot

class TestFuncs(unittest.TestCase):

    def test_get_transform_4326_pass(self):
        """Check _get_transformation for 4326 epsg."""
        res, unit = u_plot.get_transformation({'init': 'epsg:4326'})
        self.assertIsInstance(res, cartopy.crs.PlateCarree)
        self.assertEqual(unit, '°')

    def test_get_transform_3395_pass(self):
        """Check that assigned attribute is correctly set."""
        res, unit = u_plot.get_transformation({'init': 'epsg:3395'})
        self.assertIsInstance(res, cartopy.crs.Mercator)
        self.assertEqual(unit, 'm')

    def test_get_transform_3035_pass(self):
        """Check that assigned attribute is correctly set."""
        res, unit = u_plot.get_transformation({'init': 'epsg:3035'})
        self.assertIsInstance(res, cartopy._epsg._EPSGProjection)
        self.assertEqual(unit, 'm')
        
class TestPlots(unittest.TestCase):
    
    def test_geo_scatter_categorical(self):
        """Plots ones with geo_scatteR_categorical"""
        values = np.array([1, 2.0, 1, 'a'])
        coord = np.array([[26, 0], [26, 1], [28, 0], [29, 1]])
        u_plot.geo_scatter_categorical(values, coord, 'value', 'test plot',
                        cat_name=['int', 'float', 'string'],
                        pop_name=False, cmap='Dark2')
        plt.close()

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlots))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
