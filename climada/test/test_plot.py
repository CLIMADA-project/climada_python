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

test plots
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.engine.impact import Impact, ImpactFreqCurve
from climada.util.constants import HAZ_DEMO_MAT, ENT_DEMO_TODAY

class TestPlotter(unittest.TestCase):
    """Test plot functions."""

    def setUp(self):
        plt.ioff()

    def test_hazard_intensity_pass(self):
        """Generate all possible plots of the hazard intensity."""
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_DEMO_MAT)
        _, myax = hazard.plot_intensity(event=36)
        self.assertIn('Event ID 36: NNN_1185106_gen5', \
                      myax[0][0].get_title())

        _, myax = hazard.plot_intensity(event=-1)
        self.assertIn('1-largest Event. ID 3899: NNN_1190604_gen8', \
                      myax[0][0].get_title())

        _, myax = hazard.plot_intensity(event=-4)
        self.assertIn('4-largest Event. ID 5489: NNN_1192804_gen8', \
                      myax[0][0].get_title())

        _, myax = hazard.plot_intensity(event=0)
        self.assertIn('TC max intensity at each point', \
                      myax[0][0].get_title())

        myfig, _ = hazard.plot_intensity(centr=59)
        self.assertIn('Centroid ID 59: (29.0, -79.0)', \
                      myfig._suptitle.get_text())

        myfig, _ = hazard.plot_intensity(centr=-1)
        self.assertIn('1-largest Centroid. ID 100: (30.0, -75.0)', \
                      myfig._suptitle.get_text())

        myfig, _ = hazard.plot_intensity(centr=-4)
        self.assertIn('4-largest Centroid. ID 70: (30.0, -78.0)', \
                      myfig._suptitle.get_text())

        myfig, _ = hazard.plot_intensity(centr=0)
        self.assertIn('TC max intensity at each event', \
                      myfig._suptitle.get_text())

        _, myax = hazard.plot_intensity(event='NNN_1192804_gen8')
        self.assertIn('NNN_1192804_gen8', myax[0][0].get_title())

    def test_hazard_fraction_pass(self):
        """Generate all possible plots of the hazard fraction."""
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_DEMO_MAT)
        _, myax = hazard.plot_fraction(event=36)
        self.assertIn('Event ID 36: NNN_1185106_gen5', \
                      myax[0][0].get_title())

        _, myax = hazard.plot_fraction(event=-1)
        self.assertIn('1-largest Event. ID 11898: GORDON_gen7', \
                      myax[0][0].get_title())

        myfig, _ = hazard.plot_fraction(centr=59)
        self.assertIn('Centroid ID 59: (29.0, -79.0)', \
                      myfig._suptitle.get_text())

        myfig, _ = hazard.plot_fraction(centr=-1)
        self.assertIn('1-largest Centroid. ID 80: (30.0, -77.0)', \
                      myfig._suptitle.get_text())

    def test_exposures_value_pass(self):
        """Plot exposures values."""
        myexp = pd.read_excel(ENT_DEMO_TODAY)
        myexp = Exposures(myexp)
        myexp.check()
        myexp.tag.description = 'demo_today'
        _, myax= myexp.plot_hexbin()
        self.assertIn('demo_today', myax[0][0].get_title())

        myexp.tag.description = ''
        _, myax= myexp.plot_hexbin()
        self.assertIn('', myax[0][0].get_title())

    def test_impact_funcs_pass(self):
        """Plot diferent impact functions."""
        myfuncs = ImpactFuncSet()
        myfuncs.read_excel(ENT_DEMO_TODAY)
        _, myax = myfuncs.plot()
        self.assertEqual(2, len(myax))
        self.assertIn('TC 1: Tropical cyclone default', \
                      myax[0].title.get_text())
        self.assertIn('TC 3: TC Building code', myax[1].title.get_text())

    def test_impact_pass(self):
        """Plot impact exceedence frequency curves."""
        myent = Entity()
        myent.read_excel(ENT_DEMO_TODAY)
        myent.exposures.check()
        myhaz = Hazard('TC')
        myhaz.read_mat(HAZ_DEMO_MAT)
        myimp = Impact()
        myimp.calc(myent.exposures, myent.impact_funcs, myhaz)
        ifc = myimp.calc_freq_curve()
        myfig, _ = ifc.plot()
        self.assertIn('Exceedance frequency curve',\
                      myfig._suptitle.get_text())

        ifc2 = ImpactFreqCurve()
        ifc2.return_per = ifc.return_per
        ifc2.impact = 1.5e11 * np.ones(ifc2.return_per.size)
        ifc2.unit = ''
        ifc2.label = 'prove'
        ifc.plot_compare(ifc2)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
