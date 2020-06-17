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

Test Impact class.
"""
import os
import unittest
import numpy as np

import climada.engine.impact_data as im_d

DATA_FOLDER = os.path.join(os.path.dirname(__file__) , 'data')
EMDAT_TEST_CSV = os.path.join(DATA_FOLDER, 'emdat_testdata_BGD_USA_1970-2017.csv')
EMDAT_TEST_CSV_FAKE = os.path.join(DATA_FOLDER, 'emdat_testdata_fake_2007-2011.csv')

class TestEmdatImport(unittest.TestCase):
    '''Test import of EM-DAT data (as CSV) for impact data analysis'''
    def test_emdat_df_2018_load(self):
        """load selected sub sample from CSV, return DataFrame"""
        df, years, iso3 = im_d.emdat_df_load('Bangladesh', 'TC', \
                                             EMDAT_TEST_CSV, [2000, 2017], \
                                             target_version=2018)

        self.assertEqual('BGD', iso3)
        self.assertEqual(18, len(years))
        self.assertEqual(2017, years[-1])
        self.assertEqual(2010, years[10])
        self.assertEqual(475, df.size)
        self.assertEqual(8978541, df['Total affected'].max())
        self.assertIn('Tropical cyclone', list(df['Disaster subtype']))
        self.assertFalse(False in list(df['Disaster subtype']=='Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster subtype']))

    def test_emdat_df_2020_load(self):
        """load selected sub sample from CSV, return DataFrame"""
        df, years, iso3 = im_d.emdat_df_load('Bangladesh', 'TC', \
                                             EMDAT_TEST_CSV, [2000, 2017], \
                                             target_version=2020)

        self.assertEqual('BGD', iso3)
        self.assertEqual(18, len(years))
        self.assertEqual(2017, years[-1])
        self.assertEqual(2010, years[10])
        self.assertEqual(1075, df.size)
        self.assertEqual(8978541, df['Total Affected'].max())
        self.assertIn('Tropical cyclone', list(df['Disaster Subtype']))
        self.assertFalse(False in list(df['Disaster Subtype']=='Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster Subtype']))

    def test_emdat_impact_event_2018(self):
        """test emdat_impact_event event impact data extraction"""
        df = im_d.emdat_impact_event(['Bangladesh', 'USA'], 'Drought', \
                                             EMDAT_TEST_CSV, [2015, 2017], \
                                             reference_year = 2017, target_version=2018)

        self.assertEqual(92, df.size)
        self.assertEqual('2017-9550', df['Disaster No.'][3])
        self.assertEqual(df["Total damage ('000 US$)"][1], \
                            df["Total damage ('000 US$) scaled"][1])
        self.assertEqual(df["Total damage ('000 US$)"][1], 2500000000.0)
        self.assertEqual(df["Total damage ('000 US$)"][0], 1800000000.0)
        self.assertAlmostEqual(df["Total damage ('000 US$) scaled"][0], \
                                  1925085683.1166406)
        self.assertIn('USA', list(df['ISO']))
        self.assertIn('Drought', list(df['Disaster type']))
        self.assertEqual(2017, df['reference_year'].min())

    def test_emdat_impact_event(self):
        """test emdat_impact_event event impact data extraction"""
        df = im_d.emdat_impact_event(['Bangladesh', 'USA'], 'Drought', \
                                             EMDAT_TEST_CSV, [2015, 2017], \
                                             reference_year = 2017)

        self.assertEqual(184, df.size)
        self.assertEqual('2017-9550', df['Dis No'][3])
        self.assertEqual(df["Total Damages ('000 US$)"][1], \
                            df["Total Damages ('000 US$) scaled"][1])
        self.assertEqual(df["Total Damages ('000 US$)"][1], 2500000000.0)
        self.assertEqual(df["Total Damages ('000 US$)"][0], 1800000000.0)
        self.assertAlmostEqual(df["Total Damages ('000 US$) scaled"][0], \
                                  1925085683.1166406)
        self.assertIn('USA', list(df['ISO']))
        self.assertIn('Drought', list(df['Disaster Type']))
        self.assertEqual(2017, df['reference_year'].min())

    def test_emdat_impact_yearlysum(self):
        """test emdat_impact_yearlysum yearly impact data extraction"""
        df = im_d.emdat_impact_yearlysum(['Bangladesh', 'USA'], 'Flood', \
                                             EMDAT_TEST_CSV, [2015, 2017], \
                                             imp_str = 'Total Affected')
        self.assertEqual(36, df.size)
        self.assertEqual(df["impact"][1], 1900000)
        self.assertEqual(df.impact.sum(), 11517946)
        self.assertEqual(df["year"][5], 2017)

        self.assertIn('USA', list(df['ISO3']))
        self.assertIn('BGD', list(df['ISO3']))
        self.assertEqual(0, df['reference_year'].max())

class TestEmdatToImpact(unittest.TestCase):
    """Test import of EM-DAT data (as CSV) to Impact-instance (CLIMADA)"""
    def test_emdat_to_impact_all_countries(self):
        """test import TC EM-DAT to Impact() for all countries in CSV"""
        impact_emdat, countries = im_d.emdat_to_impact(EMDAT_TEST_CSV, \
                                        hazard_type_climada='TC')
        self.assertEqual(142, impact_emdat.event_id.size)
        self.assertEqual(141, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2013-0138', impact_emdat.event_name)
        self.assertEqual('BGD', countries[0])
        self.assertEqual('USA', countries[1])
        self.assertEqual(len(countries), len(impact_emdat.eai_exp))
        self.assertEqual(2, len(impact_emdat.eai_exp))
        self.assertEqual(impact_emdat.date.size, impact_emdat.frequency.size)
        self.assertAlmostEqual(555861710000, np.sum(impact_emdat.at_event))
        self.assertAlmostEqual(0.0208333333333, np.unique(impact_emdat.frequency)[0])
        self.assertAlmostEqual(11580452291.666666, impact_emdat.aai_agg)
        self.assertAlmostEqual(109456249.99999999, impact_emdat.eai_exp[0])
        self.assertAlmostEqual(11470996041.666666, impact_emdat.eai_exp[1])

    def test_emdat_to_impact_scale(self):
        """test import DR EM-DAT to Impact() for 1 country and ref.year (scaling)"""    
        impact_emdat = im_d.emdat_to_impact(EMDAT_TEST_CSV,
                                        year_range=[2010, 2016], countries=['USA'],\
                                        hazard_type_emdat='Drought', \
                                        reference_year=2016)[0]
        self.assertEqual(10, impact_emdat.event_id.size)
        self.assertEqual(9, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2012-9235', impact_emdat.event_name)
        self.assertEqual(1, len(impact_emdat.eai_exp))
        self.assertAlmostEqual(impact_emdat.aai_agg, impact_emdat.eai_exp[0])
        self.assertAlmostEqual(0.14285714, np.unique(impact_emdat.frequency)[0])
        self.assertAlmostEqual(73850951957.43886, np.sum(impact_emdat.at_event))
        self.assertAlmostEqual(10550135993.919838, impact_emdat.aai_agg)

    def test_emdat_to_impact_fakedata(self):
        """test import TC EM-DAT to Impact() for all countries in CSV"""
        impact_emdat, countries = im_d.emdat_to_impact(EMDAT_TEST_CSV_FAKE, \
                                        hazard_type_emdat='Flood')
        self.assertEqual(6, impact_emdat.event_id.size)
        self.assertEqual(5, impact_emdat.event_id[-1])
        self.assertEqual(0, impact_emdat.event_id[0])
        self.assertIn('2008-0001', impact_emdat.event_name)
        self.assertEqual('DEU', countries[0])
        self.assertEqual('CHE', countries[1])
        self.assertEqual(len(countries), len(impact_emdat.eai_exp))
        self.assertEqual(2, len(impact_emdat.eai_exp))
        self.assertAlmostEqual(11000000.0, np.sum(impact_emdat.at_event))
        self.assertAlmostEqual(0.2, np.unique(impact_emdat.frequency)[0])
        self.assertAlmostEqual(2200000.0, impact_emdat.aai_agg)
        self.assertAlmostEqual(200000.0, impact_emdat.eai_exp[0]) # DEU
        self.assertAlmostEqual(2000000.0, impact_emdat.eai_exp[1]) # CHE

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmdatImport)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmdatToImpact))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
