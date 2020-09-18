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

Test SpamAgrar base class: Unity tests
"""
import unittest
import pandas as pd

from climada.entity.exposures.spam_agrar import SpamAgrar

class TestSetCountry(unittest.TestCase):
    """Test country_iso function."""
    def test_iso3_country_name(self):
        """iso3 and country name produce same output:"""
        ent = SpamAgrar()
        testdata = self.init_testdata()

        testdata1, teststr1 = ent._spam_set_country(testdata, country='AAA')
        testdata2, teststr2 = ent._spam_set_country(testdata, country='countryA')
        self.assert_pd_frame_equal(testdata1, testdata2)
        self.assertEqual(teststr1, 'AAA')
        self.assertEqual(teststr2, 'countryA')

    def test_all_names_given(self):
        """all names given and returned as string:"""
        ent = SpamAgrar()
        testdata = self.init_testdata()

        testdata1, teststr1 = ent._spam_set_country(testdata, name_adm2='municB')
        testdata2, teststr2 = ent._spam_set_country(testdata, country='AAA',
                                                    name_adm1='kantonB',
                                                    name_adm2='municB')
        self.assert_pd_frame_equal(testdata1, testdata2)
        self.assertEqual(teststr1, ' municB')
        self.assertEqual(teststr2, 'AAA kantonB municB')

    def test_invalid_country(self):
        """invalid country name produces unchanged dataframe:"""
        ent = SpamAgrar()
        testdata = self.init_testdata()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            testdata1 = ent._spam_set_country(testdata, country='XXX')[0]
        self.assertIn('Country name not found in data: XXX', cm.output[0])
        self.assert_pd_frame_equal(testdata1, testdata)

    def test_invalid_adm2(self):
        """invalid admin 2 name produces unchanged dataframe:"""
        ent = SpamAgrar()
        testdata = self.init_testdata()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            testdata1, teststr1 = ent._spam_set_country(testdata, name_adm2='XXX')
        self.assertIn('Admin2 not found in data: XXX', cm.output[0])
        self.assert_pd_frame_equal(testdata1, testdata)
        self.assertEqual(teststr1, 'global')

    def test_invalid_adm1(self):
        """invalid admin 1 name produces unchanged dataframe:"""
        ent = SpamAgrar()
        testdata = self.init_testdata()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            testdata1, teststr1 = ent._spam_set_country(testdata, name_adm1='stateC',
                                                        name_adm2='XXX')
        testdata2 = ent._spam_set_country(testdata, name_adm1='stateC')[0]
        self.assertIn('Admin2 not found in data: XXX', cm.output[0])
        self.assert_pd_frame_equal(testdata1, testdata2)
        self.assertEqual(teststr1, ' stateC')

    @staticmethod
    def init_testdata():
        testdata = pd.DataFrame(columns=['iso3', 'dat1', 'dat2',
                                         'name_cntr', 'name_adm1',
                                         'name_adm2'])
        testdata.loc[0, 'iso3'] = 'AAA'
        testdata.loc[1, 'iso3'] = 'AAA'
        testdata.loc[2, 'iso3'] = 'CCC'
        testdata.loc[0, 'dat1'] = 1
        testdata.loc[1, 'dat1'] = 2
        testdata.loc[2, 'dat1'] = 3
        testdata.loc[0, 'dat2'] = 11
        testdata.loc[1, 'dat2'] = 12
        testdata.loc[2, 'dat2'] = 13
        testdata.loc[0, 'name_cntr'] = 'countryA'
        testdata.loc[1, 'name_cntr'] = 'countryA'
        testdata.loc[2, 'name_cntr'] = 'countryC'
        testdata.loc[0, 'name_adm1'] = 'kantonA'
        testdata.loc[1, 'name_adm1'] = 'kantonB'
        testdata.loc[2, 'name_adm1'] = 'stateC'
        testdata.loc[0, 'name_adm2'] = 'municA'
        testdata.loc[1, 'name_adm2'] = 'municB'
        testdata.loc[2, 'name_adm2'] = 'municC'
        return testdata

    @staticmethod
    def assert_pd_frame_equal(df1, df2, **kwds):
        """Assert that two dataframes are equal, ignoring ordering of columns"""
        from pandas.util.testing import assert_frame_equal
        return assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1),
                                  check_names=True, **kwds)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSetCountry)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestXYZ))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
