"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

test plots
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx
import urllib

from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.engine import ImpactCalc, ImpactFreqCurve
from climada.util.constants import HAZ_DEMO_MAT, ENT_DEMO_TODAY

class TestPlotter(unittest.TestCase):
    """Test plot functions."""

    def setUp(self):
        plt.ioff()

    def test_hazard_intensity_pass(self):
        """Generate all possible plots of the hazard intensity."""
        hazard = Hazard.from_mat(HAZ_DEMO_MAT)
        hazard.event_name = [""] * hazard.event_id.size
        hazard.event_name[35] = "NNN_1185106_gen5"
        hazard.event_name[3898] = "NNN_1190604_gen8"
        hazard.event_name[5488] = "NNN_1192804_gen8"
        myax = hazard.plot_intensity(event=36)
        self.assertIn('Event ID 36: NNN_1185106_gen5', myax.get_title())

        myax = hazard.plot_intensity(event=-1)
        self.assertIn('1-largest Event. ID 3899: NNN_1190604_gen8', myax.get_title())

        myax = hazard.plot_intensity(event=-4)
        self.assertIn('4-largest Event. ID 5489: NNN_1192804_gen8', myax.get_title())

        myax = hazard.plot_intensity(event=0)
        self.assertIn('TC max intensity at each point', myax.get_title())

        myax = hazard.plot_intensity(centr=59)
        self.assertIn('Centroid 59: (30.0, -79.0)', myax.get_title())

        myax = hazard.plot_intensity(centr=-1)
        self.assertIn('1-largest Centroid. 99: (30.0, -75.0)', myax.get_title())

        myax = hazard.plot_intensity(centr=-4)
        self.assertIn('4-largest Centroid. 69: (30.0, -78.0)', myax.get_title())

        myax = hazard.plot_intensity(centr=0)
        self.assertIn('TC max intensity at each event', myax.get_title())

        myax = hazard.plot_intensity(event='NNN_1192804_gen8')
        self.assertIn('NNN_1192804_gen8', myax.get_title())

    def test_hazard_fraction_pass(self):
        """Generate all possible plots of the hazard fraction."""
        hazard = Hazard.from_mat(HAZ_DEMO_MAT)
        hazard.event_name = [""] * hazard.event_id.size
        hazard.event_name[35] = "NNN_1185106_gen5"
        hazard.event_name[11897] = "GORDON_gen7"
        myax = hazard.plot_fraction(event=36)
        self.assertIn('Event ID 36: NNN_1185106_gen5', myax.get_title())

        myax = hazard.plot_fraction(event=-1)
        self.assertIn('1-largest Event. ID 11898: GORDON_gen7', myax.get_title())

        myax = hazard.plot_fraction(centr=59)
        self.assertIn('Centroid 59: (30.0, -79.0)', myax.get_title())

        myax = hazard.plot_fraction(centr=-1)
        self.assertIn('1-largest Centroid. 79: (30.0, -77.0)', myax.get_title())

    def test_exposures_value_pass(self):
        """Plot exposures values."""
        myexp = pd.read_excel(ENT_DEMO_TODAY)
        myexp = Exposures(myexp)
        myexp.check()
        myexp.tag.description = 'demo_today'
        myax = myexp.plot_hexbin()
        self.assertIn('demo_today', myax.get_title())

        myexp.tag.description = ''
        myax = myexp.plot_hexbin()
        self.assertIn('', myax.get_title())

    def test_impact_funcs_pass(self):
        """Plot diferent impact functions."""
        myfuncs = ImpactFuncSet.from_excel(ENT_DEMO_TODAY)
        myax = myfuncs.plot()
        self.assertEqual(2, len(myax))
        self.assertIn('TC 1: Tropical cyclone default',
                      myax[0].title.get_text())
        self.assertIn('TC 3: TC Building code', myax[1].title.get_text())

    def test_impact_pass(self):
        """Plot impact exceedence frequency curves."""
        myent = Entity.from_excel(ENT_DEMO_TODAY)
        myent.exposures.check()
        myhaz = Hazard.from_mat(HAZ_DEMO_MAT)
        myhaz.event_name = [""] * myhaz.event_id.size
        myimp = ImpactCalc(myent.exposures, myent.impact_funcs, myhaz).impact()
        ifc = myimp.calc_freq_curve()
        myax = ifc.plot()
        self.assertIn('Exceedance frequency curve', myax.get_title())

        ifc2 = ImpactFreqCurve(
            return_per=ifc.return_per,
            impact=1.5e11 * np.ones(ifc.return_per.size),
            label='prove'
        )
        ifc2.plot(axis=myax)

    def test_ctx_osm_pass(self):
        """Test basemap function using osm images"""
        myexp = Exposures()
        myexp.gdf['latitude'] = np.array([30, 40, 50])
        myexp.gdf['longitude'] = np.array([0, 0, 0])
        myexp.gdf['value'] = np.array([1, 1, 1])
        myexp.check()

        try:
            myexp.plot_basemap(url=ctx.providers.OpenStreetMap.Mapnik)
        except urllib.error.HTTPError:
            self.assertEqual(1, 0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
