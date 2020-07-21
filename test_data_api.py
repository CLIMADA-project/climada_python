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

Test files_handler module.
"""

import os
import pandas as pd
import unittest
import urllib
import xmlrunner

# solve version problem in pandas-datareader-0.6.0. see:
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-
# importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

from climada.entity.exposures.nightlight import NOAA_SITE, NASA_SITE, BM_FILENAMES
from climada.hazard.tc_tracks import IBTRACS_URL, IBTRACS_FILE
from climada.hazard.tc_tracks_forecast import TCForecast
from climada.util.finance import WORLD_BANK_WEALTH_ACC, WORLD_BANK_INC_GRP
from climada.util.files_handler import download_file, download_ftp
from climada.util.constants import SOURCE_DIR

class TestDataAvail(unittest.TestCase):
    """Test availability of data used through APIs"""

    def test_noaa_nl_pass(self):
        """Test NOAA nightlights used in BlackMarble."""
        file_down = download_file(NOAA_SITE + 'F101992.v4.tar')
        os.remove(file_down)

    def test_nasa_nl_pass(self):
        """Test NASA nightlights used in BlackMarble and LitPop."""
        url = NASA_SITE + BM_FILENAMES[0]
        file_down = download_file(url.replace('*', str(2016)))
        os.remove(file_down)

    def test_wb_wealth_pass(self):
        """Test world bank's wealth data"""
        file_down = download_file(WORLD_BANK_WEALTH_ACC)
        os.remove(file_down)

    def test_wb_lev_hist_pass(self):
        """Test world bank's historical income group levels data"""
        file_down = download_file(WORLD_BANK_INC_GRP)
        os.remove(file_down)

    # TODO: FILE_GWP_WEALTH2GDP_FACTORS

    def test_wb_api_pass(self):
        """Test World Bank API"""
        wb.download(indicator='NY.GDP.MKTP.CD', country='CHE', start=1960, end=2030)

    def test_ne_api_pass(self):
        """Test Natural Earth API"""
        url = 'http://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        file_down = download_file(url)
        os.remove(file_down)

    def test_ibtracs_pass(self):
        download_ftp("/".join([IBTRACS_URL, IBTRACS_FILE]), IBTRACS_FILE)
        os.remove(IBTRACS_FILE)

    def test_ecmwf_tc_bufr(self):
        """Test availability ECMWF essentials TC forecast."""
        fcast = TCForecast.fetch_bufr_ftp()
        [f.close() for f in fcast]

# Execute Tests
if __name__ == '__main__':
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDataAvail)
    xmlrunner.XMLTestRunner(output=os.path.join(SOURCE_DIR, '../tests_xml')).run(TESTS)
