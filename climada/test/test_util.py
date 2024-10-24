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

Tests on lines_polys_handlers.

"""

import unittest

import numpy as np

import climada.util.lines_polys_handler as u_lp
from climada.engine import Impact
from climada.entity import Exposures
from climada.util.test.test_lines_polys_handler import (
    EXP_POLY,
    GDF_POLY,
    HAZ,
    IMPF_SET,
    check_impact,
)


class TestGeomImpactCalcs(unittest.TestCase):
    """Test main functions on impact calculation and impact aggregation"""

    def test_calc_geom_impact_polys(self):
        """test calc_geom_impact() with polygons"""
        # to_meters=False, DIV, res=0.1, SUM
        imp1 = u_lp.calc_geom_impact(
            EXP_POLY,
            IMPF_SET,
            HAZ,
            res=0.1,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg = 2182703.085366719
        eai_exp = np.array(
            [
                17554.08233195,
                9896.48265036,
                16862.31818246,
                72055.81490662,
                21485.93199464,
                253701.42418527,
                135031.5217457,
                387550.35813156,
                352213.16031506,
                480603.19106997,
                203634.46630402,
                232114.3335491,
            ]
        )
        check_impact(self, imp1, HAZ, EXP_POLY, aai_agg, eai_exp)

        # to_meters=False, DIV, res=10, SUM
        imp2 = u_lp.calc_geom_impact(
            EXP_POLY,
            IMPF_SET,
            HAZ,
            res=10,
            to_meters=False,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg2 = 1282899.0530
        eai_exp2 = np.array(
            [
                8361.78802035,
                7307.04698346,
                12062.89257699,
                35406.14977618,
                12352.43204322,
                77807.46608747,
                128292.99535735,
                231231.95252362,
                131911.22622791,
                537897.30570932,
                83701.69475186,
                16566.10301167,
            ]
        )
        check_impact(self, imp2, HAZ, EXP_POLY, aai_agg2, eai_exp2)

        # to_meters=True, DIV, res=800, SUM
        imp3 = u_lp.calc_geom_impact(
            EXP_POLY,
            IMPF_SET,
            HAZ,
            res=800,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        self.assertIsInstance(imp3, Impact)
        self.assertTrue(hasattr(imp3, "geom_exp"))
        self.assertTrue(hasattr(imp3, "coord_exp"))
        self.assertTrue(np.all(imp3.geom_exp == EXP_POLY.gdf.geometry))
        self.assertEqual(len(imp3.coord_exp), len(EXP_POLY.gdf))
        self.assertAlmostEqual(imp3.aai_agg, 2317081.0602, 3)

        # to_meters=True, DIV, res=1000, SUM
        imp4 = u_lp.calc_geom_impact(
            EXP_POLY,
            IMPF_SET,
            HAZ,
            res=1000,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=None,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg4 = 2326978.3422
        eai_exp4 = np.array(
            [
                17558.22201377,
                10796.36836336,
                16239.35385599,
                73254.21872128,
                25202.52110382,
                216510.67702673,
                135412.73610909,
                410197.10023667,
                433400.62668497,
                521005.95549878,
                254979.4396249,
                212421.12303947,
            ]
        )
        check_impact(self, imp4, HAZ, EXP_POLY, aai_agg4, eai_exp4)

        # to_meters=True, DIV, res=1000, SUM, dissag_va=10e6
        imp5 = u_lp.calc_geom_impact(
            EXP_POLY,
            IMPF_SET,
            HAZ,
            res=1000,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg5 = 132.81559
        eai_exp5 = np.array(
            [
                3.55826479,
                2.55715709,
                2.49840826,
                3.51427162,
                4.30164506,
                19.36203038,
                5.28426336,
                14.25330336,
                37.29091663,
                14.05986724,
                6.88087542,
                19.2545918,
            ]
        )
        check_impact(self, imp5, HAZ, EXP_POLY, aai_agg5, eai_exp5)

        gdf_noval = GDF_POLY.copy()
        gdf_noval.pop("value")
        exp_noval = Exposures(gdf_noval)
        # to_meters=True, DIV, res=950, SUM, dissag_va=10e6
        imp6 = u_lp.calc_geom_impact(
            exp_noval,
            IMPF_SET,
            HAZ,
            res=950,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.DIV,
            disagg_val=10e6,
            agg_met=u_lp.AggMethod.SUM,
        )
        np.testing.assert_allclose(imp5.eai_exp, imp6.eai_exp, rtol=0.1)

        # to_meters=True, FIX, res=1000, SUM, dissag_va=10e6
        imp7 = u_lp.calc_geom_impact(
            exp_noval,
            IMPF_SET,
            HAZ,
            res=1000,
            to_meters=True,
            disagg_met=u_lp.DisaggMethod.FIX,
            disagg_val=10e6,
            agg_met=u_lp.AggMethod.SUM,
        )
        aai_agg7 = 412832.86028
        eai_exp7 = np.array(
            [
                8561.18507994,
                6753.45186608,
                8362.17243334,
                18014.15630989,
                8986.13653385,
                36826.58179136,
                27446.46387061,
                45468.03772305,
                130145.29903078,
                54861.60197959,
                26849.17587226,
                40558.59779586,
            ]
        )
        check_impact(self, imp7, HAZ, EXP_POLY, aai_agg7, eai_exp7)


# Execute tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestGeomImpactCalcs)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
