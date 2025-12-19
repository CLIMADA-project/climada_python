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

Test impf_TropCycl class.
"""

import unittest

import numpy as np
import pandas as pd

from climada.entity.impact_funcs.trop_cyclone import ImpfSetTropCyclone, ImpfTropCyclone


class TestEmanuelFormula(unittest.TestCase):
    """Impact function interpolation test"""

    def test_default_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpfTropCyclone.from_emanuel_usa()
        self.assertEqual(imp_fun.name, "Emanuel 2011")
        self.assertEqual(imp_fun.haz_type, "TC")
        self.assertEqual(imp_fun.id, 1)
        self.assertEqual(imp_fun.intensity_unit, "m/s")
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 121, 5)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((25,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:6], np.zeros((6,))))
        np.testing.assert_allclose(
            imp_fun.mdd[6:25],
            [
                0.0006753419543492556,
                0.006790495604105169,
                0.02425254393374475,
                0.05758706257339458,
                0.10870556455111065,
                0.1761433569521351,
                0.2553983618763961,
                0.34033822528795565,
                0.4249447743109498,
                0.5045777092933046,
                0.576424302849412,
                0.6393091739184916,
                0.6932203123193963,
                0.7388256596555696,
                0.777104531116526,
                0.8091124649261859,
                0.8358522190681132,
                0.8582150905529946,
                0.8769633232141456,
            ],
        )

    def test_values_pass(self):
        """Compute mdr interpolating values."""
        imp_fun = ImpfTropCyclone.from_emanuel_usa(
            impf_id=5, intensity=np.arange(0, 6, 1), v_thresh=2, v_half=5, scale=0.5
        )
        self.assertEqual(imp_fun.name, "Emanuel 2011")
        self.assertEqual(imp_fun.haz_type, "TC")
        self.assertEqual(imp_fun.id, 5)
        self.assertEqual(imp_fun.intensity_unit, "m/s")
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 6, 1)))
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones((6,))))
        self.assertTrue(np.array_equal(imp_fun.mdd[0:3], np.zeros((3,))))
        self.assertTrue(
            np.array_equal(
                imp_fun.mdd[3:],
                np.array(
                    [0.017857142857142853, 0.11428571428571425, 0.250000000000000]
                ),
            )
        )

    def test_wrong_shape(self):
        """Set shape parameters."""
        with self.assertRaises(ValueError):
            imp_fun = ImpfTropCyclone.from_emanuel_usa(
                impf_id=5, v_thresh=2, v_half=1, intensity=np.arange(0, 6, 1)
            )

    def test_wrong_scale(self):
        """Set shape parameters."""
        with self.assertRaises(ValueError):
            imp_fun = ImpfTropCyclone.from_emanuel_usa(
                impf_id=5, scale=2, intensity=np.arange(0, 6, 1)
            )


class TestCalibratedImpfSet(unittest.TestCase):
    """Test inititation of IFS with regional calibrated TC IFs
    based on Eberenz et al. (2020)"""

    def test_default_values_pass(self):
        """Test return TDR optimized IFs (TDR=1)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet()
        v_halfs = ImpfSetTropCyclone.calibrated_regional_vhalf()
        # extract IF for region WP4
        impf_wp4 = impfs.get_func(fun_id=9)[0]
        self.assertIn("TC", impfs.get_ids().keys())
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs.get_ids()["TC"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(impf_wp4.intensity_unit, "m/s")
        self.assertEqual(impf_wp4.name, "North West Pacific")
        self.assertAlmostEqual(v_halfs["WP2"], 188.4, places=7)
        self.assertAlmostEqual(v_halfs["ROW"], 110.1, places=7)
        self.assertListEqual(list(impf_wp4.intensity), list(np.arange(0, 121, 5)))
        self.assertEqual(impf_wp4.paa.min(), 1.0)
        self.assertEqual(impf_wp4.mdd.min(), 0.0)
        self.assertAlmostEqual(impf_wp4.mdd.max(), 0.15779133833203, places=5)
        self.assertAlmostEqual(impf_wp4.calc_mdr(75), 0.02607326527808, places=5)

    def test_RMSF_pass(self):
        """Test return RMSF optimized impact function set (RMSF=minimum)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet("RMSF")
        v_halfs = ImpfSetTropCyclone.calibrated_regional_vhalf(
            calibration_approach="RMSF"
        )
        # extract IF for region NA1
        impf_na1 = impfs.get_func(fun_id=1)[0]
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs.get_ids()["TC"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(impf_na1.intensity_unit, "m/s")
        self.assertEqual(impf_na1.name, "Caribbean and Mexico")
        self.assertAlmostEqual(v_halfs["NA1"], 59.6, places=7)
        self.assertAlmostEqual(v_halfs["ROW"], 73.4, places=7)
        self.assertListEqual(list(impf_na1.intensity), list(np.arange(0, 121, 5)))
        self.assertEqual(impf_na1.mdd.min(), 0.0)
        self.assertAlmostEqual(impf_na1.mdd.max(), 0.95560418241669, places=5)
        self.assertAlmostEqual(impf_na1.calc_mdr(75), 0.7546423895457, places=5)

    def test_quantile_pass(self):
        """Test return impact function set from quantile of inidividual event fitting (EDR=1)"""
        impfs = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet("EDR")
        impfs_p10 = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet("EDR", q=0.1)
        # extract IF for region SI
        impf_si = impfs.get_func(fun_id=5)[0]
        impf_si_p10 = impfs_p10.get_func(fun_id=5)[0]
        self.assertEqual(impfs.size(), 10)
        self.assertEqual(impfs_p10.size(), 10)
        self.assertEqual(impf_si.intensity_unit, "m/s")
        self.assertEqual(impf_si_p10.name, "South Indian")
        self.assertAlmostEqual(impf_si_p10.mdd.max(), 0.99999999880, places=5)
        self.assertAlmostEqual(impf_si.calc_mdr(30), 0.01620503041, places=5)
        intensity = np.random.randint(26, impf_si.intensity.max())
        self.assertTrue(impf_si.calc_mdr(intensity) < impf_si_p10.calc_mdr(intensity))

    def test_get_countries_per_region(self):
        """Test static get_countries_per_region()"""
        ifs = ImpfSetTropCyclone()
        out = ifs.get_countries_per_region("NA2")
        self.assertEqual(out[0], "USA and Canada")
        self.assertEqual(out[1], 2)
        self.assertListEqual(out[2], [124, 840])
        self.assertListEqual(out[3], ["CAN", "USA"])

    def test_get_countries_per_region_all_or_none(self):
        ifs = ImpfSetTropCyclone()
        out = ifs.get_countries_per_region()
        out2 = ifs.get_countries_per_region("all")
        self.assertEqual(out, out2)
        self.assertDictEqual(
            out[0],
            {
                "NA1": "Caribbean and Mexico",
                "NA2": "USA and Canada",
                "NI": "North Indian",
                "OC": "Oceania",
                "SI": "South Indian",
                "WP1": "South East Asia",
                "WP2": "Philippines",
                "WP3": "China Mainland",
                "WP4": "North West Pacific",
                "ROW": "Rest of The World",
            },
        )
        self.assertDictEqual(
            out[1],
            {
                "NA1": 1,
                "NA2": 2,
                "NI": 3,
                "OC": 4,
                "SI": 5,
                "WP1": 6,
                "WP2": 7,
                "WP3": 8,
                "WP4": 9,
                "ROW": 10,
            },
        )
        # fmt: off
        self.assertDictEqual(out[2],
                             {'NA1': [
                                 533, 660, 32, 28, 44, 84, 60, 68, 52, 152, 170, 132, 188, 192,
                                 136, 212, 214, 218, 238, 312, 308, 320, 254, 328, 340, 332,
                                 388, 659, 662, 484, 500, 474, 558, 591, 604, 630, 600, 654,
                                 222, 740, 534, 796, 780, 858, 670, 862, 92, 850
                             ],
                              'NA2': [124, 840],
                              'NI': [
                                  4, 784, 51, 31, 50, 48, 64, 262, 232, 231, 268, 356, 364,
                                  368, 376, 400, 398, 417, 414, 422, 144, 462, 104, 496,
                                  524, 512, 586, 634, 682, 706, 760, 762, 795, 800, 860, 887
                              ],
                              'OC': [
                                  16, 36, 184, 242, 583, 316, 296, 584, 580, 540, 574,
                                  570, 520, 554, 612, 585, 598, 258, 90, 772, 626, 776,
                                  798, 548, 876, 882
                              ],
                              'SI': [
                                  180, 174, 450, 466, 508, 480, 454, 748, 834, 710, 716
                              ],
                              'WP1': [116, 360, 418, 458, 764, 704],
                              'WP2': [608],
                              'WP3': [156],
                              'WP4': [344, 392, 410, 446, 158],
                              'ROW': [
                                  24, 248, 8, 20, 10, 260, 40, 108, 56, 204, 535, 854,
                                  100, 70, 652, 112, 76, 96, 74, 72, 140, 166, 756, 384,
                                  120, 178, 531, 162, 196, 203, 276, 208, 12, 818, 732,
                                  724, 233, 246, 250, 234, 266, 826, 831, 288, 292, 324,
                                  270, 624, 226, 300, 304, 334, 191, 348, 833, 86, 372,
                                  352, 380, 832, 404, 430, 434, 438, 426, 440, 442, 428,
                                  663, 504, 492, 498, 807, 470, 499, 478, 175, 516, 562,
                                  566, 528, 578, 616, 408, 620, 275, 638, 642, 643, 646,
                                  729, 686, 702, 239, 744, 694, 674, 666, 688, 728, 678,
                                  703, 705, 752, 690, 148, 768, 788, 792, 804, 581, 336,
                                  983, 894
                              ]})

        self.assertDictEqual(out[3],
                             {
                                 "NA1": [
                                     "ABW", "AIA", "ARG", "ATG", "BHS", "BLZ", "BMU", "BOL", "BRB", "CHL", "COL",
                                     "CPV", "CRI", "CUB", "CYM", "DMA", "DOM", "ECU", "FLK", "GLP", "GRD", "GTM",
                                     "GUF", "GUY", "HND", "HTI", "JAM", "KNA", "LCA", "MEX", "MSR", "MTQ", "NIC",
                                     "PAN", "PER", "PRI", "PRY", "SHN", "SLV", "SUR", "SXM", "TCA", "TTO", "URY",
                                     "VCT", "VEN", "VGB", "VIR",
                                 ],
                                 "NA2": ["CAN", "USA"],
                                 "NI": [
                                     "AFG", "ARE", "ARM", "AZE", "BGD", "BHR", "BTN", "DJI", "ERI", "ETH", "GEO",
                                     "IND", "IRN", "IRQ", "ISR", "JOR", "KAZ", "KGZ", "KWT", "LBN", "LKA", "MDV",
                                     "MMR", "MNG", "NPL", "OMN", "PAK", "QAT", "SAU", "SOM", "SYR", "TJK", "TKM",
                                     "UGA", "UZB", "YEM",
                                 ],
                                 "OC": [
                                     "ASM", "AUS", "COK", "FJI", "FSM", "GUM", "KIR", "MHL", "MNP", "NCL", "NFK",
                                     "NIU", "NRU", "NZL", "PCN", "PLW", "PNG", "PYF", "SLB", "TKL", "TLS", "TON",
                                     "TUV", "VUT", "WLF", "WSM",
                                 ],
                                 "SI": [
                                     "COD", "COM", "MDG", "MLI", "MOZ", "MUS", "MWI", "SWZ", "TZA", "ZAF", "ZWE",
                                 ],
                                 "WP1": ["KHM", "IDN", "LAO", "MYS", "THA", "VNM"],
                                 "WP2": ["PHL"],
                                 "WP3": ["CHN"],
                                 "WP4": ["HKG", "JPN", "KOR", "MAC", "TWN"],
                                 "ROW": [
                                     "AGO", "ALA", "ALB", "AND", "ATA", "ATF", "AUT", "BDI", "BEL", "BEN", "BES",
                                     "BFA", "BGR", "BIH", "BLM", "BLR", "BRA", "BRN", "BVT", "BWA", "CAF", "CCK",
                                     "CHE", "CIV", "CMR", "COG", "CUW", "CXR", "CYP", "CZE", "DEU", "DNK", "DZA",
                                     "EGY", "ESH", "ESP", "EST", "FIN", "FRA", "FRO", "GAB", "GBR", "GGY", "GHA",
                                     "GIB", "GIN", "GMB", "GNB", "GNQ", "GRC", "GRL", "HMD", "HRV", "HUN", "IMN",
                                     "IOT", "IRL", "ISL", "ITA", "JEY", "KEN", "LBR", "LBY", "LIE", "LSO", "LTU",
                                     "LUX", "LVA", "MAF", "MAR", "MCO", "MDA", "MKD", "MLT", "MNE", "MRT", "MYT",
                                     "NAM", "NER", "NGA", "NLD", "NOR", "POL", "PRK", "PRT", "PSE", "REU", "ROU",
                                     "RUS", "RWA", "SDN", "SEN", "SGP", "SGS", "SJM", "SLE", "SMR", "SPM", "SRB",
                                     "SSD", "STP", "SVK", "SVN", "SWE", "SYC", "TCD", "TGO", "TUN", "TUR", "UKR",
                                     "UMI", "VAT", "XKO", "ZMB",
                                     ],
                             }
                             )

    # fmt: on

    def test_get_imf_id_regions_per_countries(self):
        """Test get_impf_id_regions_per_countries()"""
        ifs = ImpfSetTropCyclone()
        impf_id_reg_id_reg_name = ifs.get_impf_id_regions_per_countries(
            countries=["CHE"]
        )

        # the first element of impf_id_reg_id_reg_name [0] is the impact function id,
        # the second [1] is the region id, the third [2] is the region name.
        self.assertEqual(impf_id_reg_id_reg_name[0][0], 10)
        self.assertEqual(impf_id_reg_id_reg_name[1][0], "ROW")
        self.assertEqual(impf_id_reg_id_reg_name[2][0], "Rest of The World")
        impf_id_reg_id_reg_name = ifs.get_impf_id_regions_per_countries(countries=[756])
        self.assertEqual(impf_id_reg_id_reg_name[0][0], 10)
        self.assertEqual(impf_id_reg_id_reg_name[1][0], "ROW")
        self.assertEqual(impf_id_reg_id_reg_name[2][0], "Rest of The World")

        impf_id_reg_id_reg_name = ifs.get_impf_id_regions_per_countries(
            countries=["CHE", 268]
        )
        # CHE
        self.assertEqual(impf_id_reg_id_reg_name[0][0], 10)
        self.assertEqual(impf_id_reg_id_reg_name[1][0], "ROW")
        self.assertEqual(impf_id_reg_id_reg_name[2][0], "Rest of The World")
        # GEO (georgia, 268)
        self.assertEqual(impf_id_reg_id_reg_name[0][1], 3)
        self.assertEqual(impf_id_reg_id_reg_name[1][1], "NI")
        self.assertEqual(impf_id_reg_id_reg_name[2][1], "North Indian")


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmanuelFormula)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalibratedImpfSet))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
