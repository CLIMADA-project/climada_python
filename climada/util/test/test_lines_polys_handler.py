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

Test of lines_polys_handler
"""

import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np

from climada import CONFIG
from climada.entity import Exposures
import climada.util.lines_polys_handler as u_lp
import climada.util.coordinates as u_coord
from climada.util.api_client import Client
from climada.util.constants import DEMO_DIR, WS_DEMO_NC
from climada.engine.impact import Impact
from climada.hazard.storm_europe import StormEurope
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope


# Load gdfs and hazard and impact functions for tests
exp_poly = Client().get_exposures(
    'base', name='test_polygon_exp', status='test_dataset')
gdf_poly = exp_poly.gdf

# also atm in /climada_python/data/demo. Put on API. And only a subset, as
# 500 aren't needed
gdf_line = gpd.read_file(Path(DEMO_DIR,'nl_rails.gpkg'), driver="GPKG", crs="EPSG:4326")
gdf_line = gdf_line.iloc[np.arange(0,500,18)].set_crs(epsg=4326, allow_override=True)
gdf_line['impf_WS'] = 1

lon = np.array([6.88122345, 6.63770392, 6.3834594 , 6.08705493, 5.86532833,
       3.76552166, 5.04178393, 4.42513969, 4.89959365, 5.83199221,
       5.48188808, 5.15016205])
lat = np.array([53.15019278, 52.90814037, 52.48232657, 52.13002249, 51.26574748,
       51.30438894, 51.52944652, 51.98426911, 52.61538869, 53.10328543,
       52.54974468, 52.11286591])
gdf_point = gpd.GeoDataFrame({
    'name': np.arange(len(lat))},
    geometry=gpd.points_from_xy(lon, lat, crs="EPSG:4326"))

haz = StormEurope.from_footprints(WS_DEMO_NC, description='test_description')

impf = ImpfStormEurope.from_welker()
impf_set = ImpactFuncSet()
impf_set.append(impf)

COL_CHANGING = ['value', 'latitude', 'longitude', 'geometry', 'geometry_orig']

def check_unchanged_geom_gdf(self, gdf_geom, gdf_pnt):
    """Test properties that should not change"""
    for n in gdf_pnt.index.levels[1]:
        sub_gdf_pnt = gdf_pnt.xs(n, level=1)
        rows_sel = sub_gdf_pnt.index.to_numpy()
        sub_gdf = gdf_geom.loc[rows_sel]
        self.assertTrue(np.alltrue(sub_gdf.geometry.geom_equals(sub_gdf_pnt.geometry_orig)))
    for col in gdf_pnt.columns:
        if col not in COL_CHANGING:
            np.testing.assert_allclose(gdf_pnt[col].unique(), gdf_geom[col].unique())

class TestExposureGeomToPnt(unittest.TestCase):
    """Test Exposures to points functions"""

    def check_unchanged_exp(self, exp_geom, exp_pnt):
        """Test properties that should not change"""
        self.assertEqual(exp_geom.ref_year, exp_pnt.ref_year)
        self.assertEqual(exp_geom.value_unit, exp_pnt.value_unit)
        check_unchanged_geom_gdf(self, exp_geom.gdf, exp_pnt.gdf)

    def test_point_exposure_from_polygons(self):
        """Test disaggregation of polygons to points"""
        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=1, to_meters=False, disagg_met=u_lp.DisaggMethod.FIX, disagg_val=None)
        np.testing.assert_array_equal(exp_pnt.gdf.value, exp_poly.gdf.value)
        self.check_unchanged_exp(exp_poly, exp_pnt)

        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=0.5, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None)
        self.check_unchanged_exp(exp_poly, exp_pnt)
        val_avg = np.array([
            4.93449000e+10, 4.22202000e+10, 6.49988000e+10, 1.04223900e+11,
            1.04223900e+11, 5.85881000e+10, 1.11822300e+11, 8.54188667e+10,
            8.54188667e+10, 8.54188667e+10, 1.43895450e+11, 1.43895450e+11,
            1.16221500e+11, 3.70562500e+11, 1.35359600e+11, 3.83689000e+10
            ])
        np.testing.assert_allclose(exp_pnt.gdf.value, val_avg)
        lat = np.array([
            53.15019278, 52.90814037, 52.48232657, 52.23482697, 52.23482697,
            51.26574748, 51.30438894, 51.71676713, 51.71676713, 51.71676713,
            52.13772724, 52.13772724, 52.61538869, 53.10328543, 52.54974468,
            52.11286591
            ])
        np.testing.assert_allclose(exp_pnt.gdf.latitude, lat)

        #test
        res = 20000
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=res, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX, disagg_val=res**2   )
        self.check_unchanged_exp(exp_poly, exp_pnt)
        val = res**2
        self.assertEqual(np.unique(exp_pnt.gdf.value)[0], val)
        lat = np.array([
            53.13923671, 53.13923671, 53.13923671, 53.13923671, 53.43921725,
            53.43921725, 52.90782155, 52.90782155, 52.90782155, 52.90782155,
            52.90782155, 52.40180033, 52.40180033, 52.40180033, 52.40180033,
            52.40180033, 52.69674738, 52.69674738, 52.02540815, 52.02540815,
            52.02540815, 52.02540815, 52.02540815, 52.02540815, 52.31787025,
            52.31787025, 51.31813586, 51.31813586, 51.31813586, 51.49256036,
            51.49256036, 51.49256036, 51.49256036, 51.50407349, 51.50407349,
            51.50407349, 51.50407349, 51.50407349, 51.50407349, 51.50407349,
            51.50407349, 51.50407349, 51.79318374, 51.79318374, 51.79318374,
            51.92768703, 51.92768703, 51.92768703, 51.92768703, 51.92768703,
            51.92768703, 51.92768703, 52.46150801, 52.46150801, 52.46150801,
            52.75685438, 52.75685438, 52.75685438, 52.75685438, 53.05419711,
            53.08688006, 53.08688006, 53.08688006, 53.08688006, 53.08688006,
            53.38649582, 53.38649582, 53.38649582, 52.55795685, 52.55795685,
            52.55795685, 52.55795685, 52.23308448, 52.23308448
            ])
        np.testing.assert_allclose(exp_pnt.gdf.latitude, lat)

    def test_point_exposure_from_polygons_on_grid(self):
        """Test disaggregation of polygons to points on grid"""
        res = 0.1
        exp_poly.gdf = exp_poly.gdf[exp_poly.gdf['population']<400000]
        height, width, trafo = u_coord.pts_to_raster_meta(exp_poly.gdf.geometry.bounds, (res, res))
        x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)
        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=0.1, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None)
        exp_pnt_grid = u_lp.exp_geom_to_grid(exp_poly, (x_grid, y_grid), disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None)
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)
        for col in ['value', 'latitude', 'longitude']:
            np.testing.assert_allclose(exp_pnt.gdf[col], exp_pnt_grid.gdf[col])

        x_grid = np.append(x_grid, x_grid+10)
        y_grid = np.append(y_grid, y_grid+10)
        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=0.1, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None)
        exp_pnt_grid = u_lp.exp_geom_to_grid(exp_poly, (x_grid, y_grid), disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None)
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)
        for col in ['value', 'latitude', 'longitude']:
            np.testing.assert_allclose(exp_pnt.gdf[col], exp_pnt_grid.gdf[col])

class TestGeomImpactCalcs(unittest.TestCase):
    """Test main functions on impact calculation and impact aggregation"""

    def test_calc_geom_impact_polys(self):
        """ test calc_geom_impact() with polygons"""

        # TODO: update all values here with the most up-to-date version of
        # exp_poly from the Data API
        exp_poly = Client().get_exposures(
            'base', name='test_polygon_exp', status='test_dataset')
        gdf_poly = exp_poly.gdf
        # polygon exposures only
        imp1 = u_lp.calc_geom_impact(
            exp_poly, impf_set, haz, res=0.1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        self.assertEqual(len(haz.event_id), len(imp1.at_event))
        self.assertIsInstance(imp1, Impact)
        self.assertTrue(hasattr(imp1, 'geom_exp'))
        self.assertTrue(hasattr(imp1, 'coord_exp'))
        self.assertTrue(np.all(imp1.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp1.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp1.aai_agg, 2182703.085366719, 3)
        self.assertTrue(np.all(np.isclose(
            imp1.eai_exp,
            np.array([17554.08233195,   9896.48265036,  16862.31818246,
                      72055.81490662, 21485.93199464, 253701.42418527,
                      135031.5217457 , 387550.35813156, 352213.16031506,
                      480603.19106997, 203634.46630402, 232114.3335491 ]))))

        imp2 = u_lp.calc_geom_impact(
            exp_poly, impf_set, haz, res=10, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp2, Impact)
        self.assertTrue(hasattr(imp2, 'geom_exp'))
        self.assertTrue(hasattr(imp2, 'coord_exp'))
        self.assertTrue(np.all(imp2.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp2.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp2.aai_agg, 1282899.053069401, 3)
        self.assertTrue(np.all(np.isclose(
            imp2.eai_exp,
            np.array([ 8361.78802035,   7307.04698346,  12062.89257699,
                      35406.14977618, 12352.43204322,  77807.46608747,
                      128292.99535735, 231231.95252362, 131911.22622791,
                      537897.30570932,  83701.69475186,  16566.10301167]))))

        imp3 = u_lp.calc_geom_impact(
            exp_poly, impf_set, haz, res=800, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp3, Impact)
        self.assertTrue(hasattr(imp3, 'geom_exp'))
        self.assertTrue(hasattr(imp3, 'coord_exp'))
        self.assertTrue(np.all(imp3.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp3.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp3.aai_agg, 2317081.0602, 3)

        imp4 = u_lp.calc_geom_impact(
            exp_poly, impf_set, haz, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp4, Impact)
        self.assertTrue(hasattr(imp4, 'geom_exp'))
        self.assertTrue(hasattr(imp4, 'coord_exp'))
        self.assertTrue(np.all(imp4.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp4.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp4.aai_agg, 2326978.3422788195, 3)
        self.assertTrue(np.all(np.isclose(
            imp4.eai_exp,
            np.array([ 17558.22201377,  10796.36836336,  16239.35385599,  73254.21872128,
                    25202.52110382, 216510.67702673, 135412.73610909, 410197.10023667,
                   433400.62668497, 521005.95549878, 254979.4396249 , 212421.12303947]))))
        np.testing.assert_allclose(imp4.eai_exp, imp3.eai_exp, rtol=0.1)


        imp5 = u_lp.calc_geom_impact(
            exp_poly, impf_set, haz, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp5, Impact)
        self.assertTrue(hasattr(imp5, 'geom_exp'))
        self.assertTrue(hasattr(imp5, 'coord_exp'))
        self.assertTrue(np.all(imp5.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp5.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp5.aai_agg, 132.8155950020916, 3)
        self.assertTrue(np.all(np.isclose(
            imp5.eai_exp,
            np.array([3.55826479,  2.55715709,  2.49840826,  3.51427162,  4.30164506,
                   19.36203038,  5.28426336, 14.25330336, 37.29091663, 14.05986724,
                    6.88087542, 19.2545918 ]))))

        gdf_noval = gdf_poly.copy()
        gdf_noval.pop('value')
        exp_noval = Exposures(gdf_noval)

        imp6 = u_lp.calc_geom_impact(
            exp_noval, impf_set, haz, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM
            )
        self.assertTrue(np.all(np.isclose(
            imp5.eai_exp, imp6.eai_exp)))

        imp7 = u_lp.calc_geom_impact(
            exp_noval, impf_set, haz, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp7, Impact)
        self.assertTrue(hasattr(imp7, 'geom_exp'))
        self.assertTrue(hasattr(imp7, 'coord_exp'))
        self.assertTrue(np.all(imp7.geom_exp==exp_poly.gdf.geometry))
        self.assertEqual(len(imp7.coord_exp), len(exp_poly.gdf))
        self.assertAlmostEqual(imp7.aai_agg, 412832.8602866128, 3)
        self.assertTrue(np.all(np.isclose(
            imp7.eai_exp,
            np.array([8561.18507994,   6753.45186608,   8362.17243334,
                      18014.15630989, 8986.13653385,  36826.58179136,
                      27446.46387061,  45468.03772305, 130145.29903078,
                      54861.60197959,  26849.17587226,  40558.59779586]))))

    def test_calc_geom_impact_lines(self):
        """ test calc_geom_impact() with lines"""
        # line exposures only
        gdf_line_withvals = gdf_line.copy()
        gdf_line_withvals['value'] = np.arange(len(gdf_line_withvals))*1000
        exp_line = Exposures(gdf_line_withvals)
        exp_line_novals  = Exposures(gdf_line)

        imp1 = u_lp.calc_geom_impact(
            exp_line, impf_set, haz,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )

        self.assertEqual(len(haz.event_id), len(imp1.at_event))
        self.assertIsInstance(imp1, Impact)
        self.assertTrue(hasattr(imp1, 'geom_exp'))
        self.assertTrue(hasattr(imp1, 'coord_exp'))
        self.assertTrue(np.all(imp1.geom_exp==exp_line.gdf.geometry))
        self.assertEqual(len(imp1.coord_exp), len(exp_line.gdf))
        self.assertAlmostEqual(imp1.aai_agg, 0.5165320782563735, 3)
        self.assertTrue(np.all(np.isclose(
            imp1.eai_exp,
            np.array([0.        , 0.00035429, 0.0003158 , 0.00260505, 0.00532207,
                   0.00487179, 0.00443956, 0.00083155, 0.0025156 , 0.00721847,
                   0.00670862, 0.00725904, 0.00312695, 0.02160797, 0.03094104,
                   0.00647638, 0.00731211, 0.07057858, 0.02156532, 0.01191608,
                   0.00619134, 0.0780445 , 0.02992291, 0.02016611, 0.08015055,
                   0.04738706, 0.01257872, 0.02612462]))))


        imp2 = u_lp.calc_geom_impact(exp_line, impf_set, haz,
        res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.DIV,
        disagg_val=None, agg_met=u_lp.AggMethod.SUM)

        self.assertEqual(len(haz.event_id), len(imp1.at_event))
        self.assertIsInstance(imp2, Impact)
        self.assertTrue(hasattr(imp2, 'geom_exp'))
        self.assertTrue(hasattr(imp2, 'coord_exp'))
        self.assertTrue(np.all(imp2.geom_exp==exp_line.gdf.geometry))
        self.assertEqual(len(imp2.coord_exp), len(exp_line.gdf))
        self.assertAlmostEqual(imp2.aai_agg, 0.5232328825117059, 3)
        self.assertTrue(np.all(np.isclose(
            imp2.eai_exp, imp1.eai_exp, rtol=0.1)))

        imp3 = u_lp.calc_geom_impact(
            exp_line_novals, impf_set, haz,
            res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000, agg_met=u_lp.AggMethod.SUM
            )

        imp4 = u_lp.calc_geom_impact(
            exp_line, impf_set, haz,
            res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000, agg_met=u_lp.AggMethod.SUM
            )

        self.assertTrue(np.all(imp3.eai_exp==imp4.eai_exp))
        self.assertAlmostEqual(imp4.aai_agg, 2.8301447149208117)


    def test_calc_geom_impact_points(self):
        """ test calc_geom_impact() with points"""
        gdf_pnt_vals = gdf_point.copy()
        gdf_pnt_vals['value'] = np.arange(len(gdf_pnt_vals))*1000
        gdf_pnt_vals['impf_WS'] = 1
        gdf_pnt_novals = gdf_point.copy()
        gdf_pnt_novals['impf_WS'] = 1
        exp_pnt_vals = Exposures(gdf_pnt_vals)
        exp_pnt_novals = Exposures(gdf_pnt_novals)

        imp1 = u_lp.calc_geom_impact(
            exp_pnt_vals, impf_set, haz,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )

    def test_calc_geom_impact_mixed(self):
        """ test calc_geom_impact() with a mixed exp (points, lines and polygons) """
        # mixed exposures
        gdf_pnt_vals = gdf_point.copy()
        gdf_pnt_vals['value'] = np.arange(len(gdf_pnt_vals))*1000
        gdf_mix = gdf_line.append(gdf_poly).append(
            gdf_pnt_vals).reset_index(drop=True)
        gdf_mix_lp = gdf_line.append(gdf_poly).reset_index(drop=True)

        exp_mix = Exposures(gdf_mix)
        exp_mix_lp = Exposures(gdf_mix_lp)

        imp1 = u_lp.calc_geom_impact(
            exp_mix, impf_set, haz,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )

        imp2 = u_lp.calc_geom_impact(
            exp_mix_lp, impf_set, haz,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )



    def test_impact_pnt_agg(self):
        pass

    def test_aggregate_impact_mat(self):
        pass

class TestGdfGeomToPnt(unittest.TestCase):
    """Test Geodataframes to points and vice-versa functions"""

    def test_gdf_line_to_pnt(self):
        """"""
        pass

    def test_gdf_poly_to_pnt(self):
        """"""
        pass

    def test_disagg_values_div(self):
        """Test disaggregation divide value"""
        div_vals = np.array(
            [4.93449000e+10, 4.22202000e+10, 6.49988000e+10, 1.04223900e+11,
            1.04223900e+11, 5.85881000e+10, 1.11822300e+11, 8.54188667e+10,
            8.54188667e+10, 8.54188667e+10, 1.43895450e+11, 1.43895450e+11,
            1.16221500e+11, 3.70562500e+11, 1.35359600e+11, 3.83689000e+10]
            )

        # gdf_div = u_lp._disagg_values_div(gdf_poly_pnts)
        # np.testing.assert_allclose(div_vals, gdf_div.value)
        # not_checked = ['geometry', 'geometry_orig', 'value']
        # for col in gdf_div.columns:
        #     if col not in not_checked:
        #         np.testing.assert_allclose(gdf_div[col], gdf_poly_pnts[col])

    def test_poly_to_pnts(self):
        """Test polygon to points disaggregation"""
        gdf_pnt = u_lp._poly_to_pnts(gdf_poly, 1, False)
        check_unchanged_geom_gdf(self, gdf_poly, gdf_pnt)

    def test_lines_to_pnts(self):
        """Test lines to points disaggregation"""
        pass

    def test_pnts_per_line(self):
        """Test number of points per line for give resolutio"""
        self.assertEqual(u_lp._pnts_per_line(10, 1), 11)
        self.assertEqual(u_lp._pnts_per_line(1, 1), 2)
        self.assertEqual(u_lp._pnts_per_line(10, 1.5), 8)
        self.assertEqual(u_lp._pnts_per_line(10.5, 1), 12)

    def test_interp_one_poly(self):
        """"""
        pass

    def test_interp_one_poly_m(self):
        """"""
        pass

    def test_gdf_to_grid(self):
        """"""
        pass

class TestLPUtils(unittest.TestCase):
    """ """

    def test_pnt_line_poly_mask(self):
        """"""
        pnt, lines, poly = u_lp._pnt_line_poly_mask(gdf_poly)
        self.assertTrue(np.all(poly))
        self.assertTrue(np.all(lines==False))
        self.assertTrue(np.all(pnt==False))

    def test_get_equalarea_proj(self):
        """Test pass get locally cylindrical equalarea projection"""
        poly = exp_poly.gdf.geometry[0]
        proj = u_lp._get_equalarea_proj(poly)
        self.assertEqual(proj, '+proj=cea +lat_0=52.112866 +lon_0=5.150162 +units=m')

    def test_get_pyproj_trafo(self):
        """"""
        dest_crs = '+proj=cea +lat_0=52.112866 +lon_0=5.150162 +units=m'
        orig_crs = exp_poly.gdf.crs
        trafo = u_lp._get_pyproj_trafo(orig_crs, dest_crs)
        self.assertEqual(
            trafo.definition,
            'proj=pipeline step proj=unitconvert xy_in=deg' +
            ' xy_out=rad step proj=cea lat_0=52.112866 lon_0=5.150162 units=m'
            )

    def test_reproject_grid(self):
        """"""
        pass

    def test_reproject_poly(self):
        """"""
        pass

    def test_swap_geom_cols(self):
        """Test swap of geometry columns """
        gdf_orig = gdf_poly.copy()
        gdf_orig['new_geom'] = gdf_orig.geometry
        swap_gdf = u_lp._swap_geom_cols(gdf_orig, 'old_geom', 'new_geom')
        self.assertTrue(np.alltrue(swap_gdf.geometry.geom_equals(gdf_orig.new_geom)))


# Not needed, will come with another pull request.
# class TestImpactSetters(unittest.TestCase):
#     """ """

#     def test_set_imp_mat(self):
#         """ test set_imp_mat"""
#         pass

#     def test_eai_exp_from_mat(self):
#         """ test eai_exp_from_mat"""

#         pass

#     def test_at_event_from_mat(self):
#         """Test at_event_from_mat"""

#     def test_aai_agg_from_at_event(self):
#         """Test aai_agg_from_at_event"""
#         pass


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestExposureGeomToPnt)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGeomImpactCalcs))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGdfGeomToPnt))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLPUtils))
    unittest.TextTestRunner(verbosity=2).run(TESTS)

