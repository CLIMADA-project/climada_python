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

import geopandas as gpd
import numpy as np

from climada import CONFIG
from climada.entity import Exposures
import climada.util.lines_polys_handler as u_lp
import climada.util.coordinates as u_coord
from climada.util.api_client import Client

exp_poly = Client().get_exposures('base', name='test_polygon_exp', status='test_dataset')
gdf_poly = exp_poly.gdf

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
    """Test main functions on impact calculation, aggregation"""
    def test_calc_geom_impact(self):
        """"""
        pass

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

        gdf_div = u_lp._disagg_values_div(gdf_poly_pnts)
        np.testing.assert_allclose(div_vals, gdf_div.value)
        not_checked = ['geometry', 'geometry_orig', 'value']
        for col in gdf_div.columns:
            if col not in not_checked:
                np.testing.assert_allclose(gdf_div[col], gdf_poly_pnts[col])

    def test_poly_to_pnts(self):
        """Test polygon to points disaggregation"""
        gdf_pnt = u_lp._poly_to_pnts(gdf_poly, 1)
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
