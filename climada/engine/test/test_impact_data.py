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

import climada.engine.impact_data as im_d

DATA_FOLDER = os.path.join(os.path.dirname(__file__) , 'data')

class TestEmdatImport(unittest.TestCase):
    '''Test import of EM-DAT data (as CSV) for impact data analysis'''
    def test_emdat_df_load(self):
        """load selected sub sample from CSV, return DataFrame"""
        emdat_file_csv = os.path.join(DATA_FOLDER, \
                                    'emdat_testdata_BGD_USA_1970-2017.csv')
        df, years, iso3 = im_d.emdat_df_load('Bangladesh', 'TC', \
                                             emdat_file_csv, [2000, 2017])

        self.assertEqual('BGD', iso3)
        self.assertEqual(18, len(years))
        self.assertEqual(2017, years[-1])
        self.assertEqual(2010, years[10])
        self.assertEqual(475, df.size)
        self.assertEqual(8978541, df['Total affected'].max())
        self.assertIn('Tropical cyclone', list(df['Disaster subtype']))
        self.assertFalse(False in list(df['Disaster subtype']=='Tropical cyclone'))
        self.assertFalse('Flood' in list(df['Disaster subtype']))

    def test_emdat_impact_event(self):
        """test emdat_impact_event event impact data extraction"""
        emdat_file_csv = os.path.join(DATA_FOLDER, \
                                    'emdat_testdata_BGD_USA_1970-2017.csv')
        df = im_d.emdat_impact_event(['Bangladesh', 'USA'], 'Drought', \
                                             emdat_file_csv, [2015, 2017], \
                                             reference_year = 2017)

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

    def test_emdat_impact_yearlysum(self):
        """test emdat_impact_yearlysum yearly impact data extraction"""
        emdat_file_csv = os.path.join(DATA_FOLDER, \
                                    'emdat_testdata_BGD_USA_1970-2017.csv')
        df = im_d.emdat_impact_yearlysum(['Bangladesh', 'USA'], 'Flood', \
                                             emdat_file_csv, [2015, 2017], \
                                             imp_str = 'Total affected')
        self.assertEqual(36, df.size)
        self.assertEqual(df["impact"][1], 1900000)
        self.assertEqual(df.impact.sum(), 11517946)
        self.assertEqual(df["year"][5], 2017)

        self.assertIn('USA', list(df['ISO3']))
        self.assertIn('BGD', list(df['ISO3']))
        self.assertEqual(0, df['reference_year'].max())

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestEmdatImport)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCalc))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
