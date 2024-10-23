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

Test Forecast class
"""

import datetime as dt
import unittest
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io import shapereader

from climada import CONFIG
from climada.engine.forecast import FORECAST_PLOT_DIR, Forecast
from climada.entity import ImpactFuncSet
from climada.entity.exposures.base import INDICATOR_IMPF, Exposures
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
from climada.hazard.storm_europe import StormEurope
from climada.util.constants import WS_DEMO_NC

HAZ_DIR = CONFIG.hazard.test_data.dir()


class TestCalc(unittest.TestCase):
    """Test calc and propety functions from the Forecast class"""

    def test_Forecast_calc_properties(self):
        """Test calc and propety functions from the Forecast class"""
        # hazard
        haz = StormEurope.from_cosmoe_file(
            HAZ_DIR.joinpath("storm_europe_cosmoe_forecast_vmax_testfile.nc"),
            run_datetime=dt.datetime(2018, 1, 1),
            event_date=dt.datetime(2018, 1, 3),
        )
        # exposure
        data = {}
        data["latitude"] = haz.centroids.lat
        data["longitude"] = haz.centroids.lon
        data["value"] = np.ones_like(data["latitude"]) * 100000
        data["deductible"] = np.zeros_like(data["latitude"])
        data[INDICATOR_IMPF + "WS"] = np.ones_like(data["latitude"])
        data["region_id"] = np.ones_like(data["latitude"], dtype=int) * 756
        expo = Exposures(gpd.GeoDataFrame(data=data))
        # vulnerability
        # generate vulnerability
        impact_function = ImpfStormEurope.from_welker()
        impact_function_set = ImpactFuncSet([impact_function])
        # create and calculate Forecast
        forecast = Forecast({dt.datetime(2018, 1, 1): haz}, expo, impact_function_set)
        forecast.calc()
        # test
        self.assertEqual(len(forecast.run_datetime), 1)
        self.assertEqual(forecast.run_datetime[0], dt.datetime(2018, 1, 1))
        self.assertEqual(forecast.event_date, dt.datetime(2018, 1, 3))
        self.assertEqual(forecast.lead_time().days, 2)
        self.assertEqual(
            forecast.summary_str(), "WS_NWP_run2018010100_event20180103_Switzerland"
        )
        self.assertAlmostEqual(forecast.ai_agg(), 26.347, places=1)
        self.assertAlmostEqual(forecast.ei_exp()[1], 7.941, places=1)
        self.assertEqual(len(forecast.hazard), 1)
        self.assertIsInstance(forecast.hazard[0], StormEurope)
        self.assertIsInstance(forecast.exposure, Exposures)
        self.assertIsInstance(forecast.vulnerability, ImpactFuncSet)

    def test_Forecast_init_raise(self):
        """Test calc and propety functions from the Forecast class"""
        # hazard with several event dates
        storms = StormEurope.from_footprints(WS_DEMO_NC)
        # exposure
        data = {}
        data["latitude"] = np.array([1, 2, 3])
        data["longitude"] = np.array([1, 2, 3])
        data["value"] = np.ones_like(data["latitude"]) * 100000
        data["deductible"] = np.zeros_like(data["latitude"])
        data[INDICATOR_IMPF + "WS"] = np.ones_like(data["latitude"])
        data["region_id"] = np.ones_like(data["latitude"], dtype=int) * 756
        expo = Exposures(gpd.GeoDataFrame(data=data))
        # vulnerability
        # generate vulnerability
        impact_function_set = ImpactFuncSet()
        # create and calculate Forecast
        with self.assertRaises(ValueError):
            Forecast({dt.datetime(2018, 1, 1): storms}, expo, impact_function_set)


class TestPlot(unittest.TestCase):
    """Test plotting functions from the Forecast class"""

    def test_Forecast_plot(self):
        """Test cplotting functions from the Forecast class"""
        ## given a forecast based on hazard exposure and vulnerability
        # hazard
        haz1 = StormEurope.from_cosmoe_file(
            HAZ_DIR.joinpath("storm_europe_cosmoe_forecast_vmax_testfile.nc"),
            run_datetime=dt.datetime(2018, 1, 1),
            event_date=dt.datetime(2018, 1, 3),
        )
        haz1.centroids.gdf.geometry = haz1.centroids.gdf.geometry.translate(-1.2, 0.6)
        haz2 = StormEurope.from_cosmoe_file(
            HAZ_DIR.joinpath("storm_europe_cosmoe_forecast_vmax_testfile.nc"),
            run_datetime=dt.datetime(2018, 1, 1),
            event_date=dt.datetime(2018, 1, 3),
        )
        haz2.centroids.gdf.geometry = haz2.centroids.gdf.geometry.translate(-1.2, 0.6)
        # exposure
        data = {}
        data["latitude"] = haz1.centroids.lat
        data["longitude"] = haz1.centroids.lon
        data["value"] = np.ones_like(data["latitude"]) * 100000
        data["deductible"] = np.zeros_like(data["latitude"])
        data[INDICATOR_IMPF + "WS"] = np.ones_like(data["latitude"])
        data["region_id"] = np.ones_like(data["latitude"], dtype=int) * 756
        expo = Exposures(gpd.GeoDataFrame(data=data))
        # vulnerability
        # generate vulnerability
        impact_function = ImpfStormEurope.from_welker()
        impact_function_set = ImpactFuncSet([impact_function])
        # create and calculate Forecast
        forecast = Forecast(
            {dt.datetime(2018, 1, 2): haz1, dt.datetime(2017, 12, 31): haz2},
            expo,
            impact_function_set,
        )
        forecast.calc()
        # create a file containing the polygons of Swiss cantons using natural earth
        cantons_file = CONFIG.local_data.save_dir.dir() / "CHE_cantons.shp"
        adm1_shape_file = shapereader.natural_earth(
            resolution="10m", category="cultural", name="admin_1_states_provinces"
        )
        if not cantons_file.exists():
            with fiona.open(adm1_shape_file, "r") as source:
                with fiona.open(cantons_file, "w", **source.meta) as sink:
                    for f in source:
                        if f["properties"]["adm0_a3"] == "CHE":
                            sink.write(f)
        ## test plotting functions
        # should save plot without failing
        forecast.plot_imp_map(
            run_datetime=dt.datetime(2017, 12, 31),
            explain_str="test text",
            polygon_file=str(cantons_file),
            save_fig=True,
            close_fig=True,
        )
        map_file_name = (
            forecast.summary_str(dt.datetime(2017, 12, 31)) + "_impact_map" + ".jpeg"
        )
        map_file_name_full = Path(FORECAST_PLOT_DIR) / map_file_name
        map_file_name_full.absolute().unlink(missing_ok=False)
        # should contain title strings
        ax = forecast.plot_hist(
            run_datetime=dt.datetime(2017, 12, 31),
            explain_str="test text",
            save_fig=False,
            close_fig=False,
        )
        title_artists = ax.get_figure().get_children()
        title_texts = [x.get_text() for x in title_artists if isinstance(x, plt.Text)]
        self.assertIn("test text", title_texts)
        self.assertIn("Wed 03 Jan 2018 00-24UTC", title_texts)
        self.assertIn("31.12.2017 00UTC +3d", title_texts)
        # should contain average impact in axes
        artists = ax.get_children()
        texts = [x.get_text() for x in artists if type(x) == plt.Text]
        self.assertIn("mean impact:\n 26  USD", texts)
        ax.get_figure().clf()
        # should contain title strings
        ax = forecast.plot_exceedence_prob(
            run_datetime=dt.datetime(2017, 12, 31),
            threshold=5000,
            explain_str="test text exceedence",
            save_fig=False,
            close_fig=False,
        )[0][0]
        title_artists = ax.get_figure().get_children()
        title_texts = [x.get_text() for x in title_artists if isinstance(x, plt.Text)]
        self.assertIn("test text exceedence", title_texts)
        self.assertIn("Wed 03 Jan 2018 00-24UTC", title_texts)
        self.assertIn("31.12.2017 00UTC +3d", title_texts)
        ax.get_figure().clf()
        forecast.plot_warn_map(
            str(cantons_file),
            decision_level="polygon",
            thresholds=[100000, 500000, 1000000, 5000000],
            probability_aggregation="mean",
            area_aggregation="sum",
            title="Building damage warning",
            explain_text="warn level based on aggregated damages",
            save_fig=False,
            close_fig=True,
        )
        forecast.plot_warn_map(
            str(cantons_file),
            decision_level="exposure_point",
            thresholds=[1, 1000, 5000, 5000000],
            probability_aggregation=0.2,
            area_aggregation=0.2,
            title="Building damage warning",
            explain_text="warn level based on aggregated damages",
            run_datetime=dt.datetime(2017, 12, 31),
            save_fig=False,
            close_fig=True,
        )
        forecast.plot_hexbin_ei_exposure()
        plt.close()
        # should fail because of invalid decision_level
        with self.assertRaises(ValueError) as cm:
            forecast.plot_warn_map(
                str(cantons_file),
                decision_level="test_fail",
                probability_aggregation=0.2,
                area_aggregation=0.2,
                title="Building damage warning",
                explain_text="warn level based on aggregated damages",
                save_fig=False,
                close_fig=True,
            )
        plt.close()
        self.assertIn("Parameter decision_level", str(cm.exception))
        # should fail because of invalid probability_aggregation
        with self.assertRaises(ValueError) as cm:
            forecast.plot_warn_map(
                str(cantons_file),
                decision_level="exposure_point",
                probability_aggregation="test_fail",
                area_aggregation=0.2,
                title="Building damage warning",
                explain_text="warn level based on aggregated damages",
                save_fig=False,
                close_fig=True,
            )
        plt.close()
        self.assertIn("Parameter probability_aggregation", str(cm.exception))
        # should fail because of invalid area_aggregation
        with self.assertRaises(ValueError) as cm:
            forecast.plot_warn_map(
                str(cantons_file),
                decision_level="exposure_point",
                probability_aggregation=0.2,
                area_aggregation="test_fail",
                title="Building damage warning",
                explain_text="warn level based on aggregated damages",
                save_fig=False,
                close_fig=True,
            )
        plt.close()
        self.assertIn("Parameter area_aggregation", str(cm.exception))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCalc)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlot))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
