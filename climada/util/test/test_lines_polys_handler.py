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

from climada.entity import Exposures
import climada.util.lines_polys_handler as u_lp

exp_poly = Exposures.from_hdf5('/Users/ckropf/climada/data/test_polygon_exp.hdf5')
gdf_poly = exp_poly.gdf

def check_unchanged_geom_gdf(self, gdf_geom, gdf_pnt):
    """Test properties that should not change"""
    for n in gdf_pnt.index.levels[1]:
        sub_gdf_pnt = gdf_pnt.xs(n, level=1)
        rows_sel = sub_gdf_pnt.index.to_numpy()
        sub_gdf = gdf_geom.loc[rows_sel]
        sub_gdf_pnt = sub_gdf_pnt
        self.assertTrue(np.alltrue(sub_gdf.geometry.geom_equals(sub_gdf_pnt.geometry_orig)))


class TestExposureGeomToPnt(unittest.TestCase):
    """Test Exposures to points functions"""

    def check_unchanged_exp(self, exp_geom, exp_pnt):
        """Test properties that should not change"""
        self.assertEqual(exp_geom.ref_year, exp_pnt.ref_year)
        self.assertEqual(exp_geom.value_unit, exp_pnt.value_unit)
        check_unchanged_geom_gdf(self, exp_geom.gdf, exp_pnt.gdf)
        # for n in exp_pnt.gdf.index.levels[1]:
        #     sub_gdf_pnt = exp_pnt.gdf.xs(n, level=1)
        #     rows_sel = sub_gdf_pnt.index.to_numpy()
        #     sub_gdf = exp_geom.gdf.iloc[rows_sel].reset_index()
        #     sub_gdf_pnt = sub_gdf_pnt.reset_index()
        #     self.assertTrue(np.alltrue(sub_gdf.geometry.geom_equals(sub_gdf_pnt.geometry_orig)))

    def test_point_exposure_from_polygons(self):
        """Test disaggregation of polygons to points"""
        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=1, to_meters=False, disagg='None')
        np.testing.assert_array_equal(exp_pnt.gdf.value, exp_poly.gdf.value)
        self.check_unchanged_exp(exp_poly, exp_pnt)

        #test
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=0.5, to_meters=False, disagg='avg')
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
        exp_pnt = u_lp.exp_geom_to_pnt(exp_poly, res=res, to_meters=True, disagg='surf')
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

class TestGdfGeomToPnt(unittest.TestCase):
    """Test Geodataframes to points and vice-versa functions"""

    def test_disagg_gdf_avg(self):
        """Test disaggregation average"""
        pass

    def test_disagg_gdf_val(self):
        """Test disaggregation value"""
        pass

    def test_poly_to_pnts(self):
        """Test polygon to points disaggregation"""
        gdf_pnt = u_lp.poly_to_pnts(gdf_poly, 1)
        check_unchanged_geom_gdf(self, gdf_poly, gdf_pnt)

    def test_poly_to_pnts_m(self):
        """Test polygon to points disaggregation in meter"""
        gdf_pnt = u_lp.poly_to_pnts_m(gdf_poly, 20000)
        check_unchanged_geom_gdf(self, gdf_poly, gdf_pnt)

    def test_lines_to_pnts(self):
        """Test polygon to points disaggregation"""
        pass

    def test_lines_to_pnts_m(self):
        """Test polygon to points disaggregation in meter"""
        pass


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestExposureGeomToPnt)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGdfGeomToPnt))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
