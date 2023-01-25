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
import copy
import unittest
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx

from climada.engine import ImpactCalc, ImpactFreqCurve, CostBenefit
from climada.entity import (Entity, ImpactFuncSet, Exposures, DiscRates, ImpfTropCyclone, Measure,
                            MeasureSet)
from climada.hazard import Hazard, Centroids
from climada.util.constants import HAZ_DEMO_MAT, ENT_DEMO_TODAY
from climada.util.api_client import Client


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

    def test_hazard_rp_intensity(self):
        """"Plot exceedance intensity maps for different return periods"""
        hazard = Hazard.from_mat(HAZ_DEMO_MAT)
        (axis1, axis2), _ = hazard.plot_rp_intensity([25, 50])
        self.assertEqual('Return period: 25 years', axis1.get_title())
        self.assertEqual('Return period: 50 years', axis2.get_title())

    def test_hazard_centroids(self):
        """Plot centroids scatter points over earth."""
        centroids = Centroids.from_base_grid()
        centroids.plot()

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

        myexp.plot_scatter()
        myexp.plot_basemap()
        myexp.plot_raster()

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

    def test_disc_rates(self):
        """Test plot function of discount rates."""
        years = np.arange(1950, 2100)
        rates = np.ones(years.size) * 0.014
        rates[51:55] = 0.025
        rates[95:120] = 0.035
        disc = DiscRates(years=years, rates=rates)
        disc.plot()

    def test_cost_benefit(self):
        """ Test plot functions of cost benefit"""

        # Load hazard from the data API
        client = Client()

        future_year = 2080
        haz_present = client.get_hazard('tropical_cyclone',
                                properties={'country_name': 'Haiti',
                                            'climate_scenario': 'historical',
                                            'nb_synth_tracks':'10'})
        haz_future = client.get_hazard('tropical_cyclone',
                                properties={'country_name': 'Haiti',
                                            'climate_scenario': 'rcp60',
                                            'ref_year': str(future_year),
                                            'nb_synth_tracks':'10'})

        # Create an exposure
        exp_present = client.get_litpop(country='Haiti')
        exp_future = copy.deepcopy(exp_present)
        exp_future.ref_year = future_year
        n_years = exp_future.ref_year - exp_present.ref_year + 1
        growth = 1.02 ** n_years
        exp_future.gdf['value'] = exp_future.gdf['value'] * growth
        # Create an impact function
        impf_tc = ImpfTropCyclone.from_emanuel_usa()
        impf_set = ImpactFuncSet([impf_tc])
        # Create adaptation measures
        meas_1 = Measure(
            haz_type='TC',
            name='Measure A',
            color_rgb=np.array([0.8, 0.1, 0.1]),
            cost=5000000000,
            hazard_inten_imp=(1, -5),
            risk_transf_cover=0,
        )

        meas_2 = Measure(
            haz_type='TC',
            name='Measure B',
            color_rgb=np.array([0.1, 0.1, 0.8]),
            cost=220000000,
            paa_impact=(1, -0.10),
        )

        meas_set = MeasureSet(measure_list=[meas_1, meas_2])
        # Create discount rates
        year_range = np.arange(exp_present.ref_year, exp_future.ref_year + 1)
        annual_discount_zero = np.zeros(n_years)
        discount_zero = DiscRates(year_range, annual_discount_zero)
        # Wrap the entity together
        entity_present = Entity(exposures=exp_present, disc_rates=discount_zero,
                                impact_func_set=impf_set, measure_set=meas_set)
        entity_future = Entity(exposures=exp_future, disc_rates=discount_zero,
                                impact_func_set=impf_set, measure_set=meas_set)
        # Create a cost benefit object
        costben = CostBenefit()
        costben.calc(haz_present, entity_present, haz_future=haz_future,
                    ent_future=entity_future, future_year=future_year,
                    imp_time_depen=1, save_imp=True)

        # Call the plotting functions
        costben.plot_cost_benefit()
        costben.plot_event_view((25, 50, 100, 250))
        costben.plot_waterfall_accumulated(haz_present, entity_present, entity_future)
        ax = costben.plot_waterfall(haz_present, entity_present,
                                    haz_future, entity_future)
        costben.plot_arrow_averted(axis = ax, in_meas_names=['Measure A', 'Measure B'],
                                    accumulate=True)
        CostBenefit._plot_list_cost_ben(cb_list = [costben])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
