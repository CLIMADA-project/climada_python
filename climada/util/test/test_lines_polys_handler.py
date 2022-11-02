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

import numpy as np

from climada.entity import Exposures
import climada.util.lines_polys_handler as u_lp
import climada.util.coordinates as u_coord
from climada.util.api_client import Client
from climada.engine import Impact, ImpactCalc
from climada.hazard import Hazard
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.storm_europe import ImpfStormEurope

#TODO: add tests for the private methods

# Load gdfs and hazard and impact functions for tests

HAZ = Client().get_hazard('storm_europe', name='test_haz_WS_nl', status='test_dataset')

EXP_POLY = Client().get_exposures('base', name='test_polygon_exp', status='test_dataset')
GDF_POLY = EXP_POLY.gdf

EXP_LINE = Client().get_exposures('base', name='test_line_exp', status='test_dataset')
GDF_LINE = EXP_LINE.gdf

EXP_POINT = Client().get_exposures('base', name='test_point_exp', status='test_dataset')
GDF_POINT = EXP_POINT.gdf

IMPF = ImpfStormEurope.from_welker()
IMPF_SET = ImpactFuncSet([IMPF])

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

def check_impact(self, imp, haz, exp, aai_agg, eai_exp):
    """Test properties of imapcts"""
    self.assertEqual(len(haz.event_id), len(imp.at_event))
    self.assertIsInstance(imp, Impact)
    self.assertTrue(hasattr(imp, 'geom_exp'))
    self.assertTrue(hasattr(imp, 'coord_exp'))
    self.assertTrue(np.all(imp.geom_exp.sort_index()==exp.gdf.geometry.sort_index()))
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
        #test low res - one point per poly
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY, res=1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.FIX, disagg_val=None
            )
        np.testing.assert_array_equal(exp_pnt.gdf.value, EXP_POLY.gdf.value)
        self.check_unchanged_exp(EXP_POLY, exp_pnt)

        #to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY, res=0.5, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        self.check_unchanged_exp(EXP_POLY, exp_pnt)
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

        #to_meters=TRUE, FIX, dissag_val
        res = 20000
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_POLY, res=res, to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX, disagg_val=res**2
            )
        self.check_unchanged_exp(EXP_POLY, exp_pnt)
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
        exp_poly = EXP_POLY.copy()
        res = 0.1
        exp_poly.set_gdf(exp_poly.gdf[exp_poly.gdf['population']<400000])
        height, width, trafo = u_coord.pts_to_raster_meta(
            exp_poly.gdf.geometry.bounds, (res, res)
            )
        x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

        #to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_poly, res=0.1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        exp_pnt_grid = u_lp.exp_geom_to_grid(
            exp_poly, (x_grid, y_grid),
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)
        for col in ['value', 'latitude', 'longitude']:
            np.testing.assert_allclose(exp_pnt.gdf[col], exp_pnt_grid.gdf[col])

        x_grid = np.append(x_grid, x_grid+10)
        y_grid = np.append(y_grid, y_grid+10)
        #to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_poly, res=0.1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        exp_pnt_grid = u_lp.exp_geom_to_grid(
            exp_poly, (x_grid, y_grid),
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        self.check_unchanged_exp(exp_poly, exp_pnt_grid)
        for col in ['value', 'latitude', 'longitude']:
            np.testing.assert_allclose(exp_pnt.gdf[col], exp_pnt_grid.gdf[col])


    def test_point_exposure_from_lines(self):
        """Test disaggregation of lines to points"""
        #test start and end point per line
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE, res=1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.FIX, disagg_val=None
            )
        np.testing.assert_array_equal(exp_pnt.gdf.value[:,0], EXP_LINE.gdf.value)
        np.testing.assert_array_equal(exp_pnt.gdf.value[:,0], exp_pnt.gdf.value[:,1])
        self.check_unchanged_exp(EXP_LINE, exp_pnt)

        #to_meters=False, DIV
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE, res=1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV, disagg_val=None
            )
        np.testing.assert_array_equal(exp_pnt.gdf.value[:,0], EXP_LINE.gdf.value/2)
        np.testing.assert_array_equal(exp_pnt.gdf.value[:,0], exp_pnt.gdf.value[:,1])
        self.check_unchanged_exp(EXP_LINE, exp_pnt)

        #to_meters=TRUE, FIX, dissag_val
        res = 20000
        exp_pnt = u_lp.exp_geom_to_pnt(
            EXP_LINE, res=res, to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX, disagg_val=res**2
            )
        self.check_unchanged_exp(EXP_LINE, exp_pnt)
        val = res**2
        self.assertEqual(np.unique(exp_pnt.gdf.value)[0], val)
        lat = np.array([
            50.8794    , 50.8003    , 50.955     , 50.9198    , 51.921     ,
            51.83477563, 51.77826097, 51.6732    , 52.078     , 52.0788    ,
            50.8963    , 50.8967    , 51.9259    , 51.925     , 51.5457    ,
            51.5285    , 52.2614    , 52.3091    , 53.1551    , 53.1635    ,
            51.6814    , 51.61111058, 51.5457    , 52.0518    , 52.052     ,
            52.3893    , 52.3893    , 52.1543    , 52.1413    , 52.4735    ,
            52.4784    , 52.6997    , 52.6448    , 52.1139    , 52.1132    ,
            51.9222    , 51.8701    , 52.4943    , 52.4929    , 51.8402    ,
            51.8434    , 51.9255    , 51.9403    , 51.2019    , 51.10694216,
            50.9911    , 52.4919    , 52.4797    , 50.8557    , 50.8627    ,
            51.0757    , 51.0821    , 50.8207    , 50.8223    , 50.817     ,
            50.8093    , 51.0723    , 51.0724    , 50.9075    , 50.9141
            ])
        np.testing.assert_allclose(exp_pnt.gdf.latitude, lat)

class TestGeomImpactCalcs(unittest.TestCase):
    """Test main functions on impact calculation and impact aggregation"""

    def test_calc_geom_impact_polys(self):
        """ test calc_geom_impact() with polygons"""
        #to_meters=False, DIV, res=0.1, SUM
        imp1 = u_lp.calc_geom_impact(
            EXP_POLY, IMPF_SET, HAZ, res=0.1, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        aai_agg = 2182703.085366719
        eai_exp = np.array([
            17554.08233195, 9896.48265036,16862.31818246,
            72055.81490662, 21485.93199464, 253701.42418527,
            135031.5217457, 387550.35813156, 352213.16031506,
            480603.19106997, 203634.46630402, 232114.3335491
            ])
        check_impact(self, imp1, HAZ, EXP_POLY, aai_agg, eai_exp)


        #to_meters=False, DIV, res=10, SUM
        imp2 = u_lp.calc_geom_impact(
            EXP_POLY, IMPF_SET, HAZ, res=10, to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        aai_agg2 = 1282899.0530
        eai_exp2 = np.array([
            8361.78802035, 7307.04698346, 12062.89257699,
            35406.14977618, 12352.43204322, 77807.46608747,
            128292.99535735, 231231.95252362, 131911.22622791,
            537897.30570932, 83701.69475186, 16566.10301167
            ])
        check_impact(self, imp2, HAZ, EXP_POLY, aai_agg2, eai_exp2)


        #to_meters=True, DIV, res=800, SUM
        imp3 = u_lp.calc_geom_impact(
            EXP_POLY, IMPF_SET, HAZ, res=800, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        self.assertIsInstance(imp3, Impact)
        self.assertTrue(hasattr(imp3, 'geom_exp'))
        self.assertTrue(hasattr(imp3, 'coord_exp'))
        self.assertTrue(np.all(imp3.geom_exp==EXP_POLY.gdf.geometry))
        self.assertEqual(len(imp3.coord_exp), len(EXP_POLY.gdf))
        self.assertAlmostEqual(imp3.aai_agg, 2317081.0602, 3)

        #to_meters=True, DIV, res=1000, SUM
        imp4 = u_lp.calc_geom_impact(
            EXP_POLY, IMPF_SET, HAZ, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM)
        aai_agg4 = 2326978.3422
        eai_exp4 = np.array([
            17558.22201377, 10796.36836336, 16239.35385599,
            73254.21872128, 25202.52110382, 216510.67702673,
            135412.73610909, 410197.10023667, 433400.62668497,
            521005.95549878, 254979.4396249, 212421.12303947
            ])
        check_impact(self, imp4, HAZ, EXP_POLY, aai_agg4, eai_exp4)


        #to_meters=True, DIV, res=1000, SUM, dissag_va=10e6
        imp5 = u_lp.calc_geom_impact(
            EXP_POLY, IMPF_SET, HAZ, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg5 = 132.81559
        eai_exp5 = np.array([
            3.55826479, 2.55715709, 2.49840826, 3.51427162, 4.30164506,
            19.36203038, 5.28426336, 14.25330336, 37.29091663,
            14.05986724, 6.88087542, 19.2545918
            ])
        check_impact(self, imp5, HAZ, EXP_POLY, aai_agg5, eai_exp5)

        gdf_noval = GDF_POLY.copy()
        gdf_noval.pop('value')
        exp_noval = Exposures(gdf_noval)
        #to_meters=True, DIV, res=950, SUM, dissag_va=10e6
        imp6 = u_lp.calc_geom_impact(
            exp_noval, IMPF_SET, HAZ, res=950, to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM
            )
        np.testing.assert_allclose(imp5.eai_exp, imp6.eai_exp, rtol=0.1)

        #to_meters=True, FIX, res=1000, SUM, dissag_va=10e6
        imp7 = u_lp.calc_geom_impact(
            exp_noval, IMPF_SET, HAZ, res=1000, to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=10e6, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg7 = 412832.86028
        eai_exp7 = np.array([
            8561.18507994, 6753.45186608, 8362.17243334,
            18014.15630989, 8986.13653385, 36826.58179136,
            27446.46387061, 45468.03772305, 130145.29903078,
            54861.60197959, 26849.17587226, 40558.59779586
            ])
        check_impact(self, imp7, HAZ, EXP_POLY, aai_agg7, eai_exp7)


    def test_calc_geom_impact_lines(self):
        """ test calc_geom_impact() with lines"""
        # line exposures only
        exp_line_novals  = Exposures(GDF_LINE.drop(columns='value'))

        imp1 = u_lp.calc_geom_impact(
            EXP_LINE, IMPF_SET, HAZ,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg1 = 2.18359
        eai_exp1 =  np.array([
            8.50634478e-02, 4.24820916e-02, 1.04429093e-01, 1.27160538e-02,
            8.60539827e-02, 1.75262423e-01, 2.32808488e-02, 2.92552267e-02,
            4.26205598e-03, 2.31991466e-01, 5.29133033e-03, 2.72705887e-03,
            8.87954091e-03, 2.95633263e-02, 5.61356696e-01, 1.33011693e-03,
            9.95247490e-02, 7.72573773e-02, 6.12233710e-03, 1.61239410e-02,
            1.14566573e-01, 7.45522678e-02, 2.95181528e-01, 4.64021003e-02,
            1.45806743e-02, 2.49435540e-02, 2.96121155e-05, 1.03654148e-02
            ])
        check_impact(self, imp1, HAZ, EXP_LINE, aai_agg1, eai_exp1)


        imp2 = u_lp.calc_geom_impact(
            EXP_LINE, IMPF_SET, HAZ,
            res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )
        np.testing.assert_allclose(imp2.eai_exp, imp1.eai_exp, rtol=0.1)

        imp3 = u_lp.calc_geom_impact(
            exp_line_novals, IMPF_SET, HAZ,
            res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg3 = 2.830144
        eai_exp3 = np.array([
            0.10973467, 0.05930568, 0.1291031 , 0.02170876, 0.11591773,
            0.20360855, 0.03329673, 0.03672271, 0.00779005, 0.28260995,
            0.01006294, 0.00989869, 0.01279569, 0.04986454, 0.62946471,
            0.00431759, 0.12464957, 0.12455043, 0.01734576, 0.02508649,
            0.15109773, 0.12019767, 0.36631115, 0.06004143, 0.05308581,
            0.04738706, 0.00483797, 0.01935157
            ])
        check_impact(self, imp3, HAZ, exp_line_novals, aai_agg3, eai_exp3)

        imp4 = u_lp.calc_geom_impact(
            EXP_LINE, IMPF_SET, HAZ,
            res=300, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=5000, agg_met=u_lp.AggMethod.SUM
            )
        np.testing.assert_array_equal(imp3.eai_exp, imp4.eai_exp)


    def test_calc_geom_impact_points(self):
        """ test calc_geom_impact() with points"""
        imp1 = u_lp.calc_geom_impact(
            EXP_POINT, IMPF_SET, HAZ,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg1 =  0.0470814

        exp = EXP_POINT.copy()
        exp.set_lat_lon()
        imp11 = ImpactCalc(exp, IMPF_SET, HAZ).impact()
        check_impact(self, imp1, HAZ, EXP_POINT, aai_agg1, imp11.eai_exp)

        imp2 = u_lp.calc_geom_impact(
            EXP_POINT, IMPF_SET, HAZ,
            res=500, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=1.0, agg_met=u_lp.AggMethod.SUM
            )

        exp.gdf['value'] = 1.0
        imp22 = ImpactCalc(exp, IMPF_SET, HAZ).impact()
        aai_agg2 = 6.5454249333e-06
        check_impact(self, imp2, HAZ, EXP_POINT, aai_agg2, imp22.eai_exp)

    def test_calc_geom_impact_mixed(self):
        """ test calc_geom_impact() with a mixed exp (points, lines and polygons) """
        # mixed exposures
        gdf_mix = GDF_LINE.append(GDF_POLY).append(GDF_POINT).reset_index(drop=True)
        exp_mix = Exposures(gdf_mix)

        imp1 = u_lp.calc_geom_impact(
            exp_mix, IMPF_SET, HAZ,
            res=0.05, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg1 = 2354303.388829326
        eai_exp1 = np.array(
            [5.44242706e-04, 7.83583295e-03, 1.83750670e-01, 1.73511269e-02,
            1.94180761e-02, 3.90576163e-02, 1.10985612e-02, 1.86135108e-01,
            6.14306427e-02, 6.16206874e-02, 8.56458490e-03, 8.81751253e-03,
            4.26205598e-03, 8.12498654e-02, 1.57396460e-01, 6.00203189e-03,
            3.19600253e-01, 1.46198876e-01, 1.29361932e-01, 1.33011693e-03,
            1.38153438e-01, 4.20094145e-02, 9.14516636e-02, 3.61084945e-02,
            4.75139931e-02, 7.99620467e-02, 9.23306174e-02, 1.04525623e-01,
            1.61059946e+04, 1.07420484e+04, 1.44746070e+04, 7.18796281e+04,
            2.58806206e+04, 2.01316315e+05, 1.76071458e+05, 3.92482129e+05,
            2.90364327e+05, 9.05399356e+05, 1.94728210e+05, 5.11729689e+04,
            2.84224294e+02, 2.45938137e+02, 1.90644327e+02, 1.73925079e+02,
            1.76091839e+02, 4.43054173e+02, 4.41378151e+02, 4.74316805e+02,
            4.83873464e+02, 2.59001795e+02, 2.48200400e+02, 2.62995792e+02
            ])
        check_impact(self, imp1, HAZ, exp_mix, aai_agg1, eai_exp1)

        imp2 = u_lp.calc_geom_impact(
            exp_mix, IMPF_SET, HAZ,
            res=5000, to_meters=True, disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=None, agg_met=u_lp.AggMethod.SUM
            )
        aai_agg2 = 321653482.41806
        eai_exp2 = np.array([
            5.44242706e-04, 4.83197677e-03, 4.12448052e-01, 1.34215052e-01,
            2.55089453e-01, 3.82348309e-01, 2.24599809e-01, 2.57801309e-01,
            3.67620642e-01, 5.24002585e-01, 5.62882027e-02, 6.17225877e-02,
            8.52411196e-03, 4.87499192e-01, 9.09740934e-01, 8.01838920e-03,
            7.96127932e-02, 1.34945299e+00, 9.06839997e-01, 4.01295245e-01,
            5.93452277e-01, 8.40188290e-02, 4.67806576e-01, 8.21743744e-02,
            2.48612395e-01, 1.24387821e-01, 3.48131313e-01, 5.53983704e-01,
            1.48411250e+06, 1.09137411e+06, 1.62477251e+06, 1.43455724e+07,
            2.94783633e+06, 1.06950486e+07, 3.17592949e+07, 4.58152749e+07,
            3.94173129e+07, 1.48016265e+08, 1.87811203e+07, 5.41509882e+06,
            1.24792652e+04, 1.20008305e+04, 1.43296472e+04, 3.15280802e+04,
            3.32644558e+04, 3.19325625e+04, 3.11256252e+04, 3.20372742e+04,
            1.67623417e+04, 1.64528393e+04, 1.47050883e+04, 1.37721978e+04
            ])
        check_impact(self, imp2, HAZ, exp_mix, aai_agg2, eai_exp2)

    def test_impact_pnt_agg(self):
        """Test impact agreggation method"""
        gdf_mix = GDF_LINE.append(GDF_POLY).append(GDF_POINT).reset_index(drop=True)
        exp_mix = Exposures(gdf_mix)

        exp_pnt = u_lp.exp_geom_to_pnt(
            exp_mix, res=1, to_meters=False, disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None
            )
        imp_pnt = ImpactCalc(exp_pnt, IMPF_SET, HAZ).impact(save_mat=True)
        imp_agg = u_lp.impact_pnt_agg(imp_pnt, exp_pnt.gdf, u_lp.AggMethod.SUM)
        aai_agg = 1282901.377219451
        eai_exp = np.array([
            1.73069928e-04, 8.80741357e-04, 4.32240819e-03, 8.62816073e-03,
            2.21441154e-02, 1.09329988e-02, 8.58546479e-02, 4.62370081e-02,
            8.99584440e-02, 1.27160538e-02, 8.60317575e-02, 2.02440009e-01,
            2.32808488e-02, 2.86159458e-02, 4.26205598e-03, 2.40051484e-01,
            5.29133033e-03, 2.72705887e-03, 8.87954091e-03, 2.95633263e-02,
            6.33106879e-01, 1.33011693e-03, 1.11120718e-01, 7.72573773e-02,
            6.12233710e-03, 1.61239410e-02, 1.01492204e-01, 7.45522678e-02,
            1.41155415e-01, 1.53820450e-01, 2.27951125e-02, 2.23629697e-02,
            8.59651753e-03, 5.98415680e-03, 1.24717770e-02, 1.24717770e-02,
            1.48060577e-05, 1.48060577e-05, 5.18270742e-03, 5.18270742e-03,
            8.36178802e+03, 7.30704698e+03, 1.20628926e+04, 3.54061498e+04,
            1.23524320e+04, 7.78074661e+04, 1.28292995e+05, 2.31231953e+05,
            1.31911226e+05, 5.37897306e+05, 8.37016948e+04, 1.65661030e+04
            ])
        check_impact(self, imp_agg, HAZ, exp_mix, aai_agg, eai_exp)

    def test_calc_grid_impact_polys(self):
        """Test impact on grid for polygons"""
        import climada.util.coordinates as u_coord
        res = 0.1
        (_, _, xmax, ymax) = EXP_POLY.gdf.geometry.bounds.max()
        (xmin, ymin, _, _) = EXP_POLY.gdf.geometry.bounds.min()
        bounds = (xmin, ymin, xmax, ymax)
        height, width, trafo = u_coord.pts_to_raster_meta(
            bounds, (res, res)
            )
        x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

        imp_g = u_lp.calc_grid_impact(
                    exp=EXP_POLY, impf_set=IMPF_SET, haz=HAZ,
                    grid=(x_grid, y_grid), disagg_met=u_lp.DisaggMethod.DIV,
                    disagg_val=None, agg_met=u_lp.AggMethod.SUM
                    )
        aai_agg = 2319608.54202
        eai_exp = np.array([
            17230.22051525,  10974.85453081,  14423.77523209,  77906.29609785,
            22490.08925927, 147937.83580832, 132329.78961234, 375082.82348148,
            514527.07490518, 460185.19291995, 265875.77587879, 280644.81378238
            ])
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
            np.unique(GDF_LINE.value), np.unique(gdf_pnt.value)
            )

        gdf_pnt = u_lp._line_to_pnts(GDF_LINE, 1000, True)
        check_unchanged_geom_gdf(self, GDF_LINE, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_LINE.value), np.unique(gdf_pnt.value)
            )

        gdf_pnt_d = u_lp._line_to_pnts(GDF_LINE.iloc[0:1], 0.01, False)
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.x.values,
            np.array([
                6.0885    , 6.09416494, 6.09160809, 6.08743533, 6.08326257,
                6.0791987 , 6.07509502, 6.07016232, 6.0640264 , 6.06085342,
                6.06079
                ])
            )
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.y.values,
            np.array([
                50.8794    , 50.87275494, 50.86410478, 50.85590192, 50.84769906,
                50.83944191, 50.83120479, 50.82346045, 50.81661416, 50.80861974,
                50.8003
                ])
            )
        gdf_pnt_m = u_lp._line_to_pnts(GDF_LINE.iloc[0:1], 1000, True)
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.x,
            gdf_pnt_d.geometry.x)
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.y,
            gdf_pnt_d.geometry.y)

    def test_gdf_poly_to_pnts(self):
        """Test polygon to points disaggregation"""
        gdf_pnt = u_lp._poly_to_pnts(GDF_POLY, 1, False)
        check_unchanged_geom_gdf(self, GDF_POLY, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_POLY.value), np.unique(gdf_pnt.value)
            )

        gdf_pnt = u_lp._poly_to_pnts(GDF_POLY, 5000, True)
        check_unchanged_geom_gdf(self, GDF_POLY, gdf_pnt)
        np.testing.assert_array_equal(
            np.unique(GDF_POLY.value), np.unique(gdf_pnt.value)
            )

        gdf_pnt_d = u_lp._poly_to_pnts(GDF_POLY.iloc[0:1], 0.2, False)
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.x.values,
            np.array([
                6.9690605, 7.1690605, 6.3690605, 6.5690605, 6.7690605, 6.9690605,
                7.1690605, 6.5690605, 6.7690605
                ])
            )
        np.testing.assert_allclose(
            gdf_pnt_d.geometry.y.values,
            np.array([
                53.04131655, 53.04131655, 53.24131655, 53.24131655, 53.24131655,
                53.24131655, 53.24131655, 53.44131655, 53.44131655
                ])
            )
        gdf_pnt_m = u_lp._poly_to_pnts(GDF_POLY.iloc[0:1], 15000, True)
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.x.values,
            np.array([
                6.84279696, 6.97754426, 7.11229155, 6.30380779, 6.43855509,
                6.57330238, 6.70804967, 6.84279696, 6.97754426
                ])
            )
        np.testing.assert_allclose(
            gdf_pnt_m.geometry.y.values,
            np.array([
                53.0645655 , 53.0645655 , 53.0645655 , 53.28896623, 53.28896623,
                53.28896623, 53.28896623, 53.28896623, 53.28896623
                ])
            )


    def test_pnts_per_line(self):
        """Test number of points per line for give resolution"""
        self.assertEqual(u_lp._pnts_per_line(10, 1), 11)
        self.assertEqual(u_lp._pnts_per_line(1, 1), 2)
        self.assertEqual(u_lp._pnts_per_line(10, 1.5), 8)
        self.assertEqual(u_lp._pnts_per_line(10.5, 1), 12)

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
        self.assertTrue(np.all(lines==False))
        self.assertTrue(np.all(pnt==False))

        pnt, lines, poly = u_lp._pnt_line_poly_mask(GDF_LINE)
        self.assertTrue(np.all(poly==False))
        self.assertTrue(np.all(lines))
        self.assertTrue(np.all(pnt==False))

        pnt, lines, poly = u_lp._pnt_line_poly_mask(GDF_POINT)
        self.assertTrue(np.all(poly==False))
        self.assertTrue(np.all(lines==False))
        self.assertTrue(np.all(pnt))


    def test_get_equalarea_proj(self):
        """Test pass get locally cylindrical equalarea projection"""
        poly = EXP_POLY.gdf.geometry[0]
        proj = u_lp._get_equalarea_proj(poly)
        self.assertEqual(proj, '+proj=cea +lat_0=53.150193 +lon_0=6.881223 +units=m')

    def test_get_pyproj_trafo(self):
        """"""
        dest_crs = '+proj=cea +lat_0=52.112866 +lon_0=5.150162 +units=m'
        orig_crs = EXP_POLY.gdf.crs
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
        gdf_orig = GDF_POLY.copy()
        gdf_orig['new_geom'] = gdf_orig.geometry
        swap_gdf = u_lp._swap_geom_cols(gdf_orig, 'old_geom', 'new_geom')
        self.assertTrue(np.alltrue(swap_gdf.geometry.geom_equals(gdf_orig.new_geom)))


# Not needed, metehods will be incorporated in to ImpactCalc in another
# pull request
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
