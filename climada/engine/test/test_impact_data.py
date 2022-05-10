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

Test Impact class.
"""
import unittest
import numpy as np
import warnings

from climada import CONFIG
from climada.util.constants import DEMO_DIR

import climada.engine.impact_data as im_d

DATA_DIR = CONFIG.engine.test_data.dir()
EMDAT_TEST_CSV = DATA_DIR.joinpath('emdat_testdata_BGD_USA_1970-2017.csv')
EMDAT_TEST_CSV_FAKE = DATA_DIR.joinpath('emdat_testdata_fake_2007-2011.csv')
EMDAT_2020_CSV_DEMO = DEMO_DIR.joinpath('demo_emdat_impact_data_2020.csv')

class TestEmdatImport(unittest.TestCase):
    """Test import of EM-DAT data (as CSV) for impact data analysis"""

    def test_clean_emdat_df_2018_load(self):
        """load selected sub sample from CSV, return DataFrame.
            here: from 2018 EM-DAT version to 2018 target_version"""

        df = im_d.clean_emdat_df(EMDAT_TEST_CSV, countries=['Bangladesh'], hazard='TC',
                                 year_range=[2000, 2017], target_version=2018)
        self.assertIn('ISO', df.columns)
        self.assertIn('Year', df.columns)
        iso3 = list(df.ISO.unique())
        years = np.arange(df.Year.min(), df.Year.max() + 1)

        self.assertListEqual(['BGD'], iso3)
        self.assertEqual(18, len(years))
        self.assertEqual(2017, years[-1])
        self.assertEqual(2010, years[10])
        self.assertEqual(450, df.size)
        self.assertEqual(8978541, df['Total affected'].max())
        self.assertIn('Tropical cyclone', list(df['Disaster subtype']))
        self.assertFalse(False in list(df['Disaster subtype'] == 'Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster subtype']))

    def test_emdat_df_2018_to_2020_load(self):
        """load selected sub sample from CSV, return DataFrame
                here: from 2018 EM-DAT version to 2020 target_version"""
        df = im_d.clean_emdat_df(EMDAT_TEST_CSV, countries=['USA'], hazard='TC',
                                 year_range=[2000, 2017], target_version=2020)
        self.assertIn('ISO', df.columns)
        self.assertIn('Year', df.columns)
        iso3 = list(df.ISO.unique())
        years = np.arange(df.Year.min(), df.Year.max() + 1)
        self.assertListEqual(['USA'], iso3)
        self.assertEqual(18, len(years))
        self.assertEqual(2017, years[-1])
        self.assertEqual(2010, years[10])
        self.assertEqual(1634, df.size)
        self.assertEqual(60000000, df["Insured Damages ('000 US$)"].max())
        self.assertIn('Tropical cyclone', list(df['Disaster Subtype']))
        self.assertFalse(False in list(df['Disaster Subtype'] == 'Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster Subtype']))

    def test_emdat_df_2020_load(self):
        """load selected sub sample from CSV, return DataFrame
                here: from 2020 EM-DAT version to 2020 target_version"""
        df = im_d.clean_emdat_df(EMDAT_2020_CSV_DEMO, countries=['THA', 'Viet Nam'], hazard='TC',
                                 year_range=[2005, 2008], target_version=2020)
        self.assertIn('ISO', df.columns)
        self.assertIn('Year', df.columns)
        iso3 = list(df.ISO.unique())
        years = np.arange(df.Year.min(), df.Year.max() + 1)
        self.assertIn('THA', iso3)
        self.assertIn('VNM', iso3)
        self.assertNotIn('USA', iso3)
        self.assertNotIn('TWN', iso3)
        self.assertEqual(4, len(years))
        self.assertEqual(2008, years[-1])
        self.assertEqual(2006, years[1])
        self.assertEqual(43, df.columns.size)
        self.assertEqual(688, df.size)
        self.assertEqual(624000, df["Total Damages ('000 US$)"].max())
        self.assertIn('Tropical cyclone', list(df['Disaster Subtype']))
        self.assertFalse(False in list(df['Disaster Subtype'] == 'Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster Subtype']))

class TestGDPScaling(unittest.TestCase):
    """test scaling of impact values proportional to GDP"""
    def test_scale_impact2refyear(self):
        """scale of impact values proportional to GDP"""
        impact_scaled = im_d.scale_impact2refyear([10, 100, 1000, 100, 100],
                                                  [1999, 2005, 2015, 2000, 2000],
                                                  ['CZE', 'CZE', 'MEX', 'MEX', 'CZE'],
                                                  reference_year=2015)
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test fails in the following line:
        self.assertListEqual(impact_scaled, [28, 137, 1000, 165, 304])

class TestEmdatProcessing(unittest.TestCase):
    def test_emdat_impact_event_2018(self):
        """test emdat_impact_event event impact data extraction, version 2018"""
        df = im_d.emdat_impact_event(EMDAT_TEST_CSV, countries=['Bangladesh', 'USA'],
                                     hazard='Drought', year_range=[2015, 2017],
                                     reference_year=2017, version=2018)

        self.assertEqual(46, df.size)
        self.assertEqual('2017-9550', df['Disaster No.'][1])
        self.assertEqual(df["Total damage ('000 US$)"][0],
                         df["impact"][0] * 1e-3)
        self.assertEqual(df["impact_scaled"][1],
                         df["impact"][1])
        self.assertEqual(df["Total damage ('000 US$)"][1], 2500000)
        self.assertEqual(df["Total damage ('000 US$)"][0], 1800000)
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test failes in the following 1 lines:
        self.assertAlmostEqual(df["impact_scaled"][0] * 1e-7,
                               192.7868, places=0)
        self.assertIn('USA', list(df['ISO']))
        self.assertIn('Drought', list(df['Disaster type']))
        self.assertEqual(2017, df['reference_year'].min())

    def test_emdat_impact_event_2020(self):
        """test emdat_impact_event event impact data extraction, version 2020"""
        df = im_d.emdat_impact_event(EMDAT_TEST_CSV, countries=['Bangladesh', 'USA'],
                                     hazard='Drought', year_range=[2015, 2017],
                                     reference_year=2000, version=2020)

        self.assertEqual(96, df.size)
        self.assertEqual('2017-9550', df['Dis No'][1])
        self.assertEqual(df["Total Damages ('000 US$)"][0],
                         df["impact"][0] * 1e-3)
        self.assertNotEqual(df["impact_scaled"][1],
                            df["impact"][1])
        self.assertEqual(df["Total Damages ('000 US$)"][1], 2500000)
        self.assertEqual(df["Total Damages ('000 US$)"][0], 1800000)
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test failes in the following line:
        self.assertAlmostEqual(df["impact_scaled"][0] * 1e-9,
                               1.012, places=0)
        self.assertIn('USA', list(df['ISO']))
        self.assertIn('Drought', list(df['Disaster Type']))
        self.assertEqual(2000, df['reference_year'].min())

    def test_emdat_impact_yearlysum_no_futurewarning(self):
        """Ensure that no FutureWarning is issued"""
        with warnings.catch_warnings():
            # Make sure that FutureWarning will cause an error
            warnings.simplefilter("error", category=FutureWarning)
            im_d.emdat_impact_yearlysum(
                EMDAT_TEST_CSV,
                countries=["Bangladesh", "USA"],
                hazard="Flood",
                year_range=(2015, 2017),
                reference_year=None,
                imp_str="Total Affected",
            )

    def test_emdat_affected_yearlysum(self):
        """test emdat_impact_yearlysum yearly impact data extraction"""
        df = im_d.emdat_impact_yearlysum(EMDAT_TEST_CSV, countries=['Bangladesh', 'USA'],
                                         hazard='Flood', year_range=(2015, 2017),
                                         reference_year=None, imp_str="Total Affected")

        self.assertEqual(36, df.size)
        self.assertEqual(df["impact"][1], 91000)
        self.assertEqual(df.impact.sum(), 11517946)
        self.assertEqual(df["year"][5], 2017)

        self.assertIn('USA', list(df['ISO']))
        self.assertIn('BGD', list(df['ISO']))

    def test_emdat_damage_yearlysum(self):
        """test emdat_impact_yearlysum yearly impact data extraction with scaling"""
        df = im_d.emdat_impact_yearlysum(EMDAT_TEST_CSV, countries=['Bangladesh', 'USA'],
                                         hazard='Flood', year_range=(2015, 2017),
                                         reference_year=2000)

        self.assertEqual(36, df.size)
        self.assertAlmostEqual(df.impact.max(), 15150000000.0)
        self.assertEqual(df.impact_scaled.min(), 10943000.0)
        self.assertEqual(df["year"][5], 2017)
        self.assertEqual(df["reference_year"].max(), 2000)
        self.assertIn('USA', list(df['ISO']))
        self.assertIn(50, list(df['region_id']))

    def test_emdat_countries_by_hazard_2020_pass(self):
        """test to get list of countries impacted by tropical cyclones from 2000 to 2019"""
        iso3_codes, country_names = im_d.emdat_countries_by_hazard(EMDAT_2020_CSV_DEMO,
                                                                   hazard='TC',
                                                                   year_range=(2000, 2019))

        self.assertIn('Réunion', country_names)
        self.assertEqual('Sri Lanka', country_names[4])
        self.assertEqual('BLZ', iso3_codes[3])
        self.assertEqual(len(country_names), len(iso3_codes))
        self.assertEqual(100, len(iso3_codes))

class TestEmdatToImpact(unittest.TestCase):
    """Test import of EM-DAT data (as CSV) to Impact-instance (CLIMADA)"""
    def test_emdat_to_impact_all_countries_pass(self):
        """test import EM-DAT to Impact() for all countries in CSV"""
        # =====================================================================
        # emdat_to_impact(emdat_file_csv, hazard_type_climada, \
        #                 year_range=None, countries=None, hazard_type_emdat=None, \
        #                 reference_year=None, imp_str="Total Damages ('000 US$)")
        # =====================================================================

        # file 1: version 2020
        _impact_emdat2020, countries2020 = im_d.emdat_to_impact(EMDAT_2020_CSV_DEMO, 'TC')
        # file 2: version 2018
        impact_emdat, countries = im_d.emdat_to_impact(EMDAT_TEST_CSV, 'TC')

        self.assertEqual(142, impact_emdat.event_id.size)
        self.assertEqual(141, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2013-0138', impact_emdat.event_name)
        self.assertEqual('USA', countries[0])
        self.assertEqual('BGD', countries[1])
        self.assertEqual(len(countries), len(impact_emdat.eai_exp))
        self.assertEqual(2, len(impact_emdat.eai_exp))
        self.assertEqual(impact_emdat.date.size, impact_emdat.frequency.size)
        self.assertAlmostEqual(555861710000 * 1e-5, np.sum(impact_emdat.at_event) * 1e-5, places=0)
        self.assertAlmostEqual(0.0208333333333, np.unique(impact_emdat.frequency)[0], places=7)
        self.assertAlmostEqual(11580452291.666666, impact_emdat.aai_agg, places=0)
        self.assertAlmostEqual(109456249.99999999, impact_emdat.eai_exp[1], places=0)
        self.assertAlmostEqual(11470996041.666666, impact_emdat.eai_exp[0], places=0)
        self.assertIn('SPI', countries2020)
        self.assertNotIn('SPI', countries)

    def test_emdat_to_impact_scale(self):
        """test import DR EM-DAT to Impact() for 1 country and ref.year (scaling)"""
        impact_emdat = im_d.emdat_to_impact(EMDAT_TEST_CSV, 'DR',
                                            year_range=[2010, 2016], countries=['USA'],
                                            hazard_type_emdat='Drought',
                                            reference_year=2016)[0]
        self.assertEqual(5, impact_emdat.event_id.size)
        self.assertEqual(4, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2012-9235', impact_emdat.event_name)
        self.assertEqual(1, len(impact_emdat.eai_exp))
        self.assertAlmostEqual(impact_emdat.aai_agg, impact_emdat.eai_exp[0])
        self.assertAlmostEqual(0.14285714, np.unique(impact_emdat.frequency)[0], places=3)
        # scaled impact value might change if worldbank input data changes,
        # check magnitude and adjust if test failes in the following 2 lines:
        self.assertAlmostEqual(3.69, np.sum(impact_emdat.at_event * 1e-10), places=0)
        self.assertAlmostEqual(5.28, impact_emdat.aai_agg * 1e-9, places=0)

    def test_emdat_to_impact_fakedata(self):
        """test import TC EM-DAT to Impact() for all countries in CSV"""
        impact_emdat, countries = im_d.emdat_to_impact(EMDAT_TEST_CSV_FAKE, 'FL',
                                                       hazard_type_emdat='Flood')
        self.assertEqual(6, impact_emdat.event_id.size)
        self.assertEqual(5, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2008-0001', impact_emdat.event_name)
        self.assertEqual('CHE', countries[0])
        self.assertEqual('DEU', countries[1])
        self.assertEqual(len(countries), len(impact_emdat.eai_exp))
        self.assertEqual(2, len(impact_emdat.eai_exp))
        self.assertAlmostEqual(11000000.0, np.sum(impact_emdat.at_event))
        self.assertAlmostEqual(0.2, np.unique(impact_emdat.frequency)[0])
        self.assertAlmostEqual(2200000.0, impact_emdat.aai_agg)
        self.assertAlmostEqual(200000.0, impact_emdat.eai_exp[1])  # DEU
        self.assertAlmostEqual(2000000.0, impact_emdat.eai_exp[0])  # CHE

    def test_emdat_to_impact_2020format(self):
        """test import TC EM-DAT to Impact() from new 2020 EMDAT format CSV"""
        df1 = im_d.clean_emdat_df(EMDAT_2020_CSV_DEMO, hazard='TC',
                                  countries='PHL', year_range=(2013, 2013))
        df2 = im_d.emdat_impact_event(EMDAT_2020_CSV_DEMO, countries='PHL', hazard='TC',
                                      year_range=(2013, 2013), reference_year=None,
                                      imp_str='Total Affected')
        impact_emdat, _countries = im_d.emdat_to_impact(EMDAT_2020_CSV_DEMO, 'TC',
                                                       countries='PHL',
                                                       year_range=(2013, 2013),
                                                       imp_str="Total Affected")
        # compare number of entries for all steps:
        self.assertEqual(len(df1.index), len(df2.index))
        self.assertEqual(impact_emdat.event_id.size, len(df1.index))
        # TC events in EM-DAT in the Philipppines, 2013:
        self.assertEqual(8, impact_emdat.event_id.size)
        # People affected by TC events in the Philippines in 2013 (AAI):
        self.assertAlmostEqual(17944571., impact_emdat.aai_agg, places=0)
        # People affected by Typhoon Hayian in the Philippines:
        self.assertAlmostEqual(1.610687e+07, impact_emdat.at_event[4], places=0)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmdatImport)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGDPScaling))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmdatProcessing))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmdatToImpact))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
