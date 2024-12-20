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

import copy
import unittest
from unittest.mock import DEFAULT, patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

import climada.util.coordinates as u_coord
import climada.util.lines_polys_handler as u_lp
from climada.engine import Impact, ImpactCalc
from climada.entity import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope
from climada.util.api_client import Client

# TODO: add tests for the private methods

# Load gdfs and hazard and impact functions for tests

HAZ = Client().get_hazard("storm_europe", name="test_haz_WS_nl", status="test_dataset")

EXP_POLY = Client().get_exposures(
    "base", name="test_polygon_exp", status="test_dataset"
)
EXP_POLY.gdf["impf_WS"] = 2
GDF_POLY = EXP_POLY.gdf

EXP_LINE = Client().get_exposures("base", name="test_line_exp", status="test_dataset")
GDF_LINE = EXP_LINE.gdf

EXP_POINT = Client().get_exposures("base", name="test_point_exp", status="test_dataset")
GDF_POINT = EXP_POINT.gdf

IMPF = ImpfStormEurope.from_welker()
IMPF2 = copy.deepcopy(IMPF)
IMPF2.id = 2
IMPF_SET = ImpactFuncSet([IMPF, IMPF2])

COL_CHANGING = ["value", "latitude", "longitude", "geometry", "geometry_orig"]


def check_unchanged_geom_gdf(self, gdf_geom, gdf_pnt):
    """Test properties that should not change"""
    for n in gdf_pnt.index.levels[1]:
        sub_gdf_pnt = gdf_pnt.xs(n, level=1)
        rows_sel = sub_gdf_pnt.index.to_numpy()
        sub_gdf = gdf_geom.loc[rows_sel]
        self.assertTrue(np.all(sub_gdf.geometry.geom_equals(sub_gdf_pnt.geometry_orig)))
    for col in gdf_pnt.columns:
        if col not in COL_CHANGING:
            np.testing.assert_allclose(gdf_pnt[col].unique(), gdf_geom[col].unique())


def check_impact(self, imp, haz, exp, aai_agg, eai_exp):
    """Test properties of imapcts"""
    self.assertEqual(len(haz.event_id), len(imp.at_event))
    self.assertIsInstance(imp, Impact)
    self.assertTrue(hasattr(imp, "geom_exp"))
    self.assertTrue(hasattr(imp, "coord_exp"))
    self.assertTrue(np.all(imp.geom_exp.sort_index() == exp.gdf.geometry.sort_index()))
    self.assertEqual(len(imp.coord_exp), len(exp.gdf))
    self.assertAlmostEqual(imp.aai_agg, aai_agg, 3)
    np.testing.assert_allclose(imp.eai_exp, eai_exp, rtol=1e-5)


class TestExposureGeomToPnt(unittest.TestCase):
    """Test Exposures to points functions"""

    def check_unchanged_exp(self, exp_geom, exp_pnt):
        """Test properties that should not change"""
        self.assertEqual(exp_geom.ref_year, exp_pnt.ref_year)
        self.assertEqual(exp_geom.value_unit, exp_pnt.value_unit)
        check_unchanged_geom_gdf(self, exp_geom.gdf, exp_pnt.gdf)

    def test_point_exposure_from_polygons(self):
        """Test disaggregation of polygons to points"""
        # test low res - one point per poly
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY,
            res=1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=None,
        )
        np.testing.assert_array_equal(exp_pnt.gdf["value"], EXP_POLY.gdf["value"])
        self.check_unchanged_exp(EXP_POLY, exp_pnt)

        # to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY,
            res=0.5,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        self.check_unchanged_exp(EXP_POLY, exp_pnt)
        val_avg = np.array(
            [
                4.93449000e10,
                4.22202000e10,
                6.49988000e10,
                1.04223900e11,
                1.04223900e11,
                5.85881000e10,
                1.11822300e11,
                8.54188667e10,
                8.54188667e10,
                8.54188667e10,
                1.43895450e11,
                1.43895450e11,
                1.16221500e11,
                3.70562500e11,
                1.35359600e11,
                3.83689000e10,
            ]
        )
        np.testing.assert_allclose(exp_pnt.value, val_avg)
        lat = np.array(
            [
                53.15019278,
                52.90814037,
                52.48232657,
                52.23482697,
                52.23482697,
                51.26574748,
                51.30438894,
                51.71676713,
                51.71676713,
                51.71676713,
                52.13772724,
                52.13772724,
                52.61538869,
                53.10328543,
                52.54974468,
                52.11286591,
            ]
        )
        np.testing.assert_allclose(exp_pnt.latitude, lat)

        # to_meters=TRUE, FIX, dissag_val
        res = 20000
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY,
            res=res,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=res**2,
        )
        self.check_unchanged_exp(EXP_POLY, exp_pnt)
        val = res**2
        self.assertEqual(np.unique(exp_pnt.value)[0], val)
        lat = np.array(
            [
                53.13923671,
                53.13923671,
                53.13923671,
                53.13923671,
                53.43921725,
                53.43921725,
                52.90782155,
                52.90782155,
                52.90782155,
                52.90782155,
                52.90782155,
                52.40180033,
                52.40180033,
                52.40180033,
                52.40180033,
                52.40180033,
                52.69674738,
                52.69674738,
                52.02540815,
                52.02540815,
                52.02540815,
                52.02540815,
                52.02540815,
                52.02540815,
                52.31787025,
                52.31787025,
                51.31813586,
                51.31813586,
                51.31813586,
                51.49256036,
                51.49256036,
                51.49256036,
                51.49256036,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.50407349,
                51.79318374,
                51.79318374,
                51.79318374,
                51.92768703,
                51.92768703,
                51.92768703,
                51.92768703,
                51.92768703,
                51.92768703,
                51.92768703,
                52.46150801,
                52.46150801,
                52.46150801,
                52.75685438,
                52.75685438,
                52.75685438,
                52.75685438,
                53.05419711,
                53.08688006,
                53.08688006,
                53.08688006,
                53.08688006,
                53.08688006,
                53.38649582,
                53.38649582,
                53.38649582,
                52.55795685,
                52.55795685,
                52.55795685,
                52.55795685,
                52.23308448,
                52.23308448,
            ]
        )
        np.testing.assert_allclose(exp_pnt.latitude, lat)

        # projected crs, to_meters=TRUE, FIX, dissag_val
        res = 20000
        EXP_POLY_PROJ = Exposures(GDF_POLY.to_crs(epsg=28992))
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY_PROJ,
            res=res,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=res**2,
        )
        self.check_unchanged_exp(EXP_POLY_PROJ, exp_pnt)
        val = res**2
        self.assertEqual(np.unique(exp_pnt.gdf["value"])[0], val)
        self.assertEqual(exp_pnt.gdf.crs, EXP_POLY_PROJ.gdf.crs)

    @patch.multiple(
        "climada.util.lines_polys_handler",
        _interp_one_poly=DEFAULT,
        _interp_one_poly_m=DEFAULT,
    )
    def test_point_exposure_from_polygons_reproject_call(
        self, _interp_one_poly, _interp_one_poly_m
    ):
        """Verify that the correct subroutine is called for a reprojected CRS"""
        # Just have the mocks return an empty geometry
        _interp_one_poly.return_value = Point(1.0, 1.0)
        _interp_one_poly_m.return_value = Point(1.0, 1.0)

        # Use geographical CRS
        EXP_POLY_PROJ = Exposures(GDF_POLY.to_crs(epsg=4326))
        res = 1
        u_lp.exp_geom_to_pnt(
            EXP_POLY_PROJ,
            res=res,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=res,
        )
        _interp_one_poly_m.assert_called()
        _interp_one_poly.assert_not_called()

        # Reset mocks
        _interp_one_poly.reset_mock()
        _interp_one_poly_m.reset_mock()

        # Use projected CRS
        EXP_POLY_PROJ = Exposures(GDF_POLY.to_crs(epsg=28992))
        u_lp.exp_geom_to_pnt(
            EXP_POLY_PROJ,
            res=res,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=res,
        )
        _interp_one_poly_m.assert_not_called()
        _interp_one_poly.assert_called()

    def test_point_exposure_from_polygons_on_grid(self):
        """Test disaggregation of polygons to points on grid"""
        exp_poly = EXP_POLY.copy()
        res = 0.1
        exp_poly.set_gdf(exp_poly.gdf[exp_poly.gdf["population"] < 400000])
        height, width, trafo = u_coord.pts_to_raster_meta(
            exp_poly.gdf.geometry.bounds, (res, res)
        )
        x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

        # to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_poly,
            res=0.1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        exp_pnt_grid = u_lp.exp_geom_to_grid(
            exp_poly,
            (x_grid, y_grid),
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)

        np.testing.assert_allclose(exp_pnt.value, exp_pnt_grid.value)
        np.testing.assert_allclose(exp_pnt.latitude, exp_pnt_grid.latitude)
        np.testing.assert_allclose(exp_pnt.longitude, exp_pnt_grid.longitude)

        x_grid = np.append(x_grid, x_grid + 10)
        y_grid = np.append(y_grid, y_grid + 10)
        # to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_poly,
            res=0.1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        exp_pnt_grid = u_lp.exp_geom_to_grid(
            exp_poly,
            (x_grid, y_grid),
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)

        np.testing.assert_allclose(exp_pnt.value, exp_pnt_grid.value)
        np.testing.assert_allclose(exp_pnt.latitude, exp_pnt_grid.latitude)
        np.testing.assert_allclose(exp_pnt.longitude, exp_pnt_grid.longitude)

    def test_point_exposure_from_lines(self):
        """Test disaggregation of lines to points"""
        # to_meters=False, FIX
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE,
            res=1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=None,
        )
        np.testing.assert_array_equal(exp_pnt.gdf["value"][:, 0], EXP_LINE.gdf["value"])
        self.check_unchanged_exp(EXP_LINE, exp_pnt)

        # to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE,
            res=1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        np.testing.assert_array_equal(exp_pnt.gdf["value"][:, 0], EXP_LINE.gdf["value"])
        self.check_unchanged_exp(EXP_LINE, exp_pnt)

        # to_meters=TRUE, FIX, dissag_val
        res = 20000
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE,
            res=res,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=res**2,
        )
        self.check_unchanged_exp(EXP_LINE, exp_pnt)
        val = res**2
        self.assertEqual(np.unique(exp_pnt.gdf["value"])[0], val)
        lat = np.array(
            [
                50.83944191,
                50.94706532,
                51.85008694,
                51.7524172,
                52.07732906,
                50.889641,
                51.90287148,
                51.53858598,
                52.30223675,
                53.15931081,
                51.61111058,
                52.05191342,
                52.3893,
                52.14520761,
                52.47715845,
                52.68641293,
                52.11355,
                51.90503849,
                52.49610201,
                51.8418,
                51.93188219,
                51.10694216,
                52.48596301,
                50.87543042,
                51.0801347,
                50.82145186,
                50.81341953,
                51.07235498,
                50.9105503,
            ]
        )
        np.testing.assert_allclose(exp_pnt.latitude, lat)


class TestGeomImpactCalcs(unittest.TestCase):
    """Test main functions on impact calculation and impact aggregation"""

    def test_calc_geom_impact_lines(self):
        """test calc_geom_impact() with lines"""
        # line exposures only
        exp_line_novals = Exposures(GDF_LINE.drop(columns="value"))

        imp1 = u_lp.calc_geom_impact(
            EXP_LINE,
            IMPF_SET,
            HAZ,
            res=0.05,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg1 = 2.114365936716238
        eai_exp1 = np.array(
            [
                8.58546479e-02,
                4.57753040e-02,
                1.07081794e-01,
                1.27160538e-02,
                8.60984331e-02,
                1.57751547e-01,
                2.32808488e-02,
                2.95520878e-02,
                4.06902083e-03,
                2.27553509e-01,
                5.29133033e-03,
                2.72705887e-03,
                8.48207692e-03,
                2.95633263e-02,
                4.88225543e-01,
                1.33011693e-03,
                1.03018186e-01,
                7.72573773e-02,
                5.48322256e-03,
                1.61239410e-02,
                1.13181160e-01,
                8.32840521e-02,
                2.99243546e-01,
                4.88901364e-02,
                1.71930351e-02,
                2.49435540e-02,
                2.96121155e-05,
                1.03654148e-02,
            ]
        )
        check_impact(self, imp1, HAZ, EXP_LINE, aai_agg1, eai_exp1)

        imp2 = u_lp.calc_geom_impact(
            EXP_LINE,
            IMPF_SET,
            HAZ,
            res=300,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        np.testing.assert_allclose(imp2.eai_exp, imp1.eai_exp, rtol=0.2)

        imp3 = u_lp.calc_geom_impact(
            exp_line_novals,
            IMPF_SET,
            HAZ,
            res=300,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg3 = 2.626753478142696
        eai_exp3 = np.array(
            [
                0.10307851,
                0.05544964,
                0.12810739,
                0.01736701,
                0.1092617,
                0.19785227,
                0.02959709,
                0.03617366,
                0.00464554,
                0.27378204,
                0.00670862,
                0.00329956,
                0.01030654,
                0.03324303,
                0.61571791,
                0.00215879,
                0.12245651,
                0.10379203,
                0.00536503,
                0.01881487,
                0.14592603,
                0.12312706,
                0.35965216,
                0.05581585,
                0.01968975,
                0.02843223,
                0.00241899,
                0.01451368,
            ]
        )
        check_impact(self, imp3, HAZ, exp_line_novals, aai_agg3, eai_exp3)

        imp4 = u_lp.calc_geom_impact(
            EXP_LINE,
            IMPF_SET,
            HAZ,
            res=300,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000,
            agg_met=u_lp.AggMethod.SUM,
        )
        np.testing.assert_array_equal(imp3.eai_exp, imp4.eai_exp)

    def test_calc_geom_impact_points(self):
        """test calc_geom_impact() with points"""
        imp1 = u_lp.calc_geom_impact(
            EXP_POINT,
            IMPF_SET,
            HAZ,
            res=0.05,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg1 = 0.0470814

        exp = EXP_POINT.copy()
        # exp.set_lat_lon()
        imp11 = ImpactCalc(exp, IMPF_SET, HAZ).impact()
        check_impact(self, imp1, HAZ, EXP_POINT, aai_agg1, imp11.eai_exp)

        imp2 = u_lp.calc_geom_impact(
            EXP_POINT,
            IMPF_SET,
            HAZ,
            res=500,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=1.0,
            agg_met=u_lp.AggMethod.SUM,
        )

        exp.gdf["value"] = 1.0
        imp22 = ImpactCalc(exp, IMPF_SET, HAZ).impact()
        aai_agg2 = 6.5454249333e-06
        check_impact(self, imp2, HAZ, EXP_POINT, aai_agg2, imp22.eai_exp)

    def test_calc_geom_impact_mixed(self):
        """test calc_geom_impact() with a mixed exp (points, lines and polygons)"""
        # mixed exposures
        gdf_mix = pd.concat([GDF_LINE, GDF_POLY, GDF_POINT]).reset_index(drop=True)
        exp_mix = Exposures(gdf_mix)

        imp1 = u_lp.calc_geom_impact(
            exp_mix,
            IMPF_SET,
            HAZ,
            res=0.05,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg1 = 2354303.3196003754
        eai_exp1 = np.array(
            [
                1.73069928e-04,
                8.80741357e-04,
                1.77657635e-01,
                1.06413744e-02,
                1.15405492e-02,
                3.40097761e-02,
                8.91658032e-03,
                4.19735141e-02,
                1.27160538e-02,
                2.43849980e-01,
                2.32808488e-02,
                5.47043065e-03,
                5.44984095e-03,
                5.80779958e-03,
                1.06361040e-01,
                4.67335812e-02,
                9.93703142e-02,
                8.48207692e-03,
                2.95633263e-02,
                1.30223646e-01,
                3.84600393e-01,
                2.05709279e-02,
                1.39919480e-01,
                1.61239410e-02,
                4.46991386e-02,
                1.30045513e-02,
                1.30045513e-02,
                6.91177788e-04,
                1.61063727e04,
                1.07420484e04,
                1.44746070e04,
                7.18796281e04,
                2.58806206e04,
                2.01316315e05,
                1.76071458e05,
                3.92482129e05,
                2.90364327e05,
                9.05399356e05,
                1.94728210e05,
                5.11729689e04,
                2.84224294e02,
                2.45938137e02,
                1.90644327e02,
                1.73925079e02,
                1.76091839e02,
                4.43054173e02,
                4.41378151e02,
                4.74316805e02,
                4.83873464e02,
                2.59001795e02,
                2.48200400e02,
                2.62995792e02,
            ]
        )
        check_impact(self, imp1, HAZ, exp_mix, aai_agg1, eai_exp1)

        imp2 = u_lp.calc_geom_impact(
            exp_mix,
            IMPF_SET,
            HAZ,
            res=5000,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg2 = 321653479.4607434
        eai_exp2 = np.array(
            [
                1.73069928e-04,
                8.80741357e-04,
                2.17736979e-01,
                6.48243461e-02,
                2.67262620e-02,
                3.55078893e-01,
                8.14081011e-02,
                4.36578022e-01,
                1.02605091e-01,
                3.45121722e-01,
                1.62144669e-01,
                1.45008544e-01,
                2.32808488e-02,
                2.73521532e-02,
                9.51399554e-02,
                2.25921717e-01,
                6.90427531e-01,
                5.29133033e-03,
                2.72705887e-03,
                8.48207692e-03,
                2.10403881e00,
                1.33011693e-03,
                3.14644100e-01,
                7.72573773e-02,
                5.48322256e-03,
                1.61239410e-02,
                2.68194832e-01,
                7.80273077e-02,
                1.48411299e06,
                1.09137411e06,
                1.62477251e06,
                1.43455724e07,
                2.94783633e06,
                1.06950486e07,
                3.17592949e07,
                4.58152749e07,
                3.94173129e07,
                1.48016265e08,
                1.87811203e07,
                5.41509882e06,
                1.24792652e04,
                1.20008305e04,
                1.43296472e04,
                3.15280802e04,
                3.32644558e04,
                3.19325625e04,
                3.11256252e04,
                3.20372742e04,
                1.67623417e04,
                1.64528393e04,
                1.47050883e04,
                1.37721978e04,
            ]
        )
        check_impact(self, imp2, HAZ, exp_mix, aai_agg2, eai_exp2)

        # Check non-default impact function id
        impfdouble = copy.deepcopy(IMPF2)
        impfdouble.mdd *= 2
        impf_set = ImpactFuncSet([IMPF, impfdouble])
        imp3 = u_lp.calc_geom_impact(
            exp_mix,
            impf_set,
            HAZ,
            res=0.05,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg3 = 4708604.47775332
        eai_exp3 = np.array(
            [
                1.73069928e-04,
                8.80741357e-04,
                1.77657635e-01,
                1.06413744e-02,
                1.15405492e-02,
                3.40097761e-02,
                8.91658032e-03,
                4.19735141e-02,
                1.27160538e-02,
                2.43849980e-01,
                2.32808488e-02,
                5.47043065e-03,
                5.44984095e-03,
                5.80779958e-03,
                1.06361040e-01,
                4.67335812e-02,
                9.93703142e-02,
                8.48207692e-03,
                2.95633263e-02,
                1.30223646e-01,
                3.84600393e-01,
                2.05709279e-02,
                1.39919480e-01,
                1.61239410e-02,
                4.46991386e-02,
                1.30045513e-02,
                1.30045513e-02,
                6.91177788e-04,
                3.22122197e04,
                2.14840968e04,
                2.89492139e04,
                1.43759256e05,
                5.17612411e04,
                4.02632630e05,
                3.52142916e05,
                7.84964258e05,
                5.80728653e05,
                1.81079871e06,
                3.89456421e05,
                1.02345938e05,
                5.68448588e02,
                4.91876274e02,
                3.81288655e02,
                3.47850159e02,
                3.52183678e02,
                8.86108346e02,
                8.82756302e02,
                9.48633609e02,
                9.67746928e02,
                5.18003590e02,
                4.96400801e02,
                5.25991584e02,
            ]
        )
        check_impact(self, imp3, HAZ, exp_mix, aai_agg3, eai_exp3)

    def test_impact_pnt_agg(self):
        """Test impact agreggation method"""
        gdf_mix = pd.concat([GDF_LINE, GDF_POLY, GDF_POINT]).reset_index(drop=True)
        exp_mix = Exposures(gdf_mix)

        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_mix,
            res=1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
        )
        imp_pnt = ImpactCalc(exp_pnt, IMPF_SET, HAZ).impact(save_mat=True)
        imp_agg = u_lp.impact_pnt_agg(imp_pnt, exp_pnt.gdf, u_lp.AggMethod.SUM)
        aai_agg = 1282901.0114188215
        eai_exp = np.array(
            [
                0.00000000e00,
                1.73069928e-04,
                3.71172778e-04,
                5.09568579e-04,
                8.43340681e-04,
                3.47906751e-03,
                3.00385618e-03,
                5.62430455e-03,
                9.07998787e-03,
                1.30641275e-02,
                6.18365411e-03,
                4.74934473e-03,
                8.34810476e-02,
                5.07280880e-02,
                1.02690634e-01,
                1.27160538e-02,
                8.60984331e-02,
                1.62144669e-01,
                2.32808488e-02,
                2.90389979e-02,
                4.06902083e-03,
                2.33667906e-01,
                5.29133033e-03,
                2.72705887e-03,
                8.48207692e-03,
                2.95633263e-02,
                4.01271600e-01,
                1.33011693e-03,
                9.94596852e-02,
                7.72573773e-02,
                5.48322256e-03,
                1.61239410e-02,
                4.14706673e-03,
                8.32840521e-02,
                2.87509619e-01,
                4.88901364e-02,
                1.71930351e-02,
                2.49435540e-02,
                2.96121155e-05,
                1.03654148e-02,
                8.36178802e03,
                7.30704698e03,
                1.20628926e04,
                3.54061498e04,
                1.23524320e04,
                7.78074661e04,
                1.28292995e05,
                2.31231953e05,
                1.31911226e05,
                5.37897306e05,
                8.37016948e04,
                1.65661030e04,
            ]
        )
        check_impact(self, imp_agg, HAZ, exp_mix, aai_agg, eai_exp)

    def test_calc_grid_impact_polys(self):
        """Test impact on grid for polygons"""
        import climada.util.coordinates as u_coord

        res = 0.1
        (_, _, xmax, ymax) = EXP_POLY.gdf.geometry.bounds.max()
        (xmin, ymin, _, _) = EXP_POLY.gdf.geometry.bounds.min()
        bounds = (xmin, ymin, xmax, ymax)
        height, width, trafo = u_coord.pts_to_raster_meta(bounds, (res, res))
        x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

        imp_g = u_lp.calc_grid_impact(
            exp=EXP_POLY,
            impf_set=IMPF_SET,
            haz=HAZ,
            grid=(x_grid, y_grid),
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg = 2319608.54202
        eai_exp = np.array(
            [
                17230.22051525,
                10974.85453081,
                14423.77523209,
                77906.29609785,
                22490.08925927,
                147937.83580832,
                132329.78961234,
                375082.82348148,
                514527.07490518,
                460185.19291995,
                265875.77587879,
                280644.81378238,
            ]
        )
        check_impact(self, imp_g, HAZ, EXP_POLY, aai_agg, eai_exp)

    def test_aggregate_impact_mat(self):
        """Private method"""
        pass


class TestGdfGeomToPnt(unittest.TestCase):
    """Test Geodataframes to points and vice-versa functions"""

    def test_gdf_line_to_pnt(self):
        """Test Lines to point dissagregation"""
        gdf_pnt = u_lp._line_to_pnts(GDF_LINE, 1, False)
        check_unchanged_geom_gdf(self, GDF_LINE, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_LINE["value"]), np.unique(gdf_pnt["value"])
        )

        gdf_pnt = u_lp._line_to_pnts(GDF_LINE, 1000, True)
        check_unchanged_geom_gdf(self, GDF_LINE, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_LINE["value"]), np.unique(gdf_pnt["value"])
        )

        gdf_pnt_d = u_lp._line_to_pnts(GDF_LINE.iloc[0:1], 0.01, False)
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.x.values,
            np.array(
                [
                    6.092507,
                    6.092895,
                    6.088363,
                    6.083726,
                    6.079199,
                    6.074582,
                    6.068896,
                    6.061939,
                    6.061839,
                ]
            ),
        )
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.y.values,
            np.array(
                [
                    50.876242,
                    50.866888,
                    50.857725,
                    50.84861,
                    50.839442,
                    50.830321,
                    50.82186,
                    50.814366,
                    50.80475,
                ]
            ),
        )

        # disaggregation in degrees and approximately same value in meters
        gdf_pnt_m = u_lp._line_to_pnts(GDF_LINE.iloc[0:1], 1000, True)
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.x, gdf_pnt_d.geometry.x, rtol=1e-2
        )
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.y, gdf_pnt_d.geometry.y, rtol=1e-2
        )

    def test_gdf_poly_to_pnts(self):
        """Test polygon to points disaggregation"""
        gdf_pnt = u_lp._poly_to_pnts(GDF_POLY, 1, False)
        check_unchanged_geom_gdf(self, GDF_POLY, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_POLY["value"]), np.unique(gdf_pnt["value"])
        )

        gdf_pnt = u_lp._poly_to_pnts(GDF_POLY, 5000, True)
        check_unchanged_geom_gdf(self, GDF_POLY, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_POLY["value"]), np.unique(gdf_pnt["value"])
        )

        gdf_pnt_d = u_lp._poly_to_pnts(GDF_POLY.iloc[0:1], 0.2, False)
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.x.values,
            np.array(
                [
                    6.9690605,
                    7.1690605,
                    6.3690605,
                    6.5690605,
                    6.7690605,
                    6.9690605,
                    7.1690605,
                    6.5690605,
                    6.7690605,
                ]
            ),
        )
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.y.values,
            np.array(
                [
                    53.04131655,
                    53.04131655,
                    53.24131655,
                    53.24131655,
                    53.24131655,
                    53.24131655,
                    53.24131655,
                    53.44131655,
                    53.44131655,
                ]
            ),
        )
        gdf_pnt_m = u_lp._poly_to_pnts(GDF_POLY.iloc[0:1], 15000, True)
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.x.values,
            np.array(
                [
                    6.84279696,
                    6.97754426,
                    7.11229155,
                    6.30380779,
                    6.43855509,
                    6.57330238,
                    6.70804967,
                    6.84279696,
                    6.97754426,
                ]
            ),
        )
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.y.values,
            np.array(
                [
                    53.0645655,
                    53.0645655,
                    53.0645655,
                    53.28896623,
                    53.28896623,
                    53.28896623,
                    53.28896623,
                    53.28896623,
                    53.28896623,
                ]
            ),
        )

    def test_pnts_per_line(self):
        """Test number of points per line for give resolution"""
        self.assertEqual(u_lp._pnts_per_line(10, 1), 10)
        self.assertEqual(u_lp._pnts_per_line(1, 1), 1)
        self.assertEqual(u_lp._pnts_per_line(10, 1.5), 7)
        self.assertEqual(u_lp._pnts_per_line(10.5, 1), 10)

    def test_line_fractions(self):
        """Test the division of lines into fractions"""
        length = 1
        res_fractions = {
            2: np.array([0.5]),
            0.8: np.array([0.5]),
            0.6: np.array([0.25, 0.75]),
            0.4: np.array([0.25, 0.75]),
        }
        for res, fraction in res_fractions.items():
            np.testing.assert_allclose(u_lp._line_fraction(length, res), fraction)

        length = 2
        res_fractions = {
            2: np.array([0.5]),
            0.8: np.array([0.25, 0.75]),
            0.6: np.array([0.166667, 0.5, 0.833333]),
            0.4: np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        }
        for res, fraction in res_fractions.items():
            np.testing.assert_allclose(
                u_lp._line_fraction(length, res), fraction, rtol=1e-04
            )

    def test_resolution_warning(self):
        lines = [
            LineString([[0, 0], [0, 2]]),
            LineString([[0, 0], [0, 12]]),
            LineString([[0, 0], [0, 20]]),
        ]
        gdf_lines = gpd.GeoDataFrame(geometry=lines)
        with self.assertLogs(
            "climada.util.lines_polys_handler", level="WARNING"
        ) as ctx:
            u_lp._line_to_pnts(gdf_lines, 1, False)
        self.assertEqual(
            ctx.records[0].message,
            f"{2} lines with a length < 10*resolution were found. "
            "Each of these lines is disaggregate to one point. "
            "Reaggregatint values will thus likely lead to overestimattion. "
            "Consider chosing a smaller resolution or filter out the short lines. ",
        )

    def test_gdf_to_grid(self):
        """"""
        pass

    def test_interp_one_poly(self):
        """Private method"""
        pass

    def test_interp_one_poly_m(self):
        """Private method"""
        pass

    def test_disagg_values_div(self):
        """Private method"""
        pass


class TestLPUtils(unittest.TestCase):
    """ """

    def test_pnt_line_poly_mask(self):
        """"""
        pnt, lines, poly = u_lp._pnt_line_poly_mask(GDF_POLY)
        self.assertTrue(np.all(poly))
        self.assertTrue(np.all(lines == False))
        self.assertTrue(np.all(pnt == False))

        pnt, lines, poly = u_lp._pnt_line_poly_mask(GDF_LINE)
        self.assertTrue(np.all(poly == False))
        self.assertTrue(np.all(lines))
        self.assertTrue(np.all(pnt == False))

        pnt, lines, poly = u_lp._pnt_line_poly_mask(GDF_POINT)
        self.assertTrue(np.all(poly == False))
        self.assertTrue(np.all(lines == False))
        self.assertTrue(np.all(pnt))

    def test_get_equalarea_proj(self):
        """Test pass get locally cylindrical equalarea projection"""
        poly = EXP_POLY.gdf.geometry[0]
        proj = u_lp._get_equalarea_proj(poly)
        self.assertEqual(proj, "+proj=cea +lat_0=53.150193 +lon_0=6.881223 +units=m")

    def test_get_pyproj_trafo(self):
        """"""
        dest_crs = "+proj=cea +lat_0=52.112866 +lon_0=5.150162 +units=m"
        orig_crs = EXP_POLY.gdf.crs
        trafo = u_lp._get_pyproj_trafo(orig_crs, dest_crs)
        self.assertEqual(
            trafo.definition,
            "proj=pipeline step proj=unitconvert xy_in=deg"
            + " xy_out=rad step proj=cea lat_0=52.112866 lon_0=5.150162 units=m",
        )

    def test_reproject_grid(self):
        """"""
        pass

    def test_reproject_poly(self):
        """"""
        pass

    def test_swap_geom_cols(self):
        """Test swap of geometry columns"""
        gdf_orig = GDF_POLY.copy()
        gdf_orig["new_geom"] = gdf_orig.geometry
        swap_gdf = u_lp._swap_geom_cols(gdf_orig, "old_geom", "new_geom")
        self.assertTrue(np.all(swap_gdf.geometry.geom_equals(gdf_orig.new_geom)))


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestExposureGeomToPnt)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGeomImpactCalcs))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGdfGeomToPnt))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLPUtils))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
