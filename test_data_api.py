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

from pathlib import Path
import pandas as pd
import unittest
import urllib
import xmlrunner
import datetime as dt

# solve version problem in pandas-datareader-0.6.0. see:
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-
# importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

from climada.entity.exposures.nightlight import NOAA_SITE, NASA_SITE, BM_FILENAMES
from climada.hazard.tc_tracks import IBTRACS_URL, IBTRACS_FILE
from climada.hazard.tc_tracks_forecast import TCForecast
from climada.util.finance import WORLD_BANK_WEALTH_ACC, WORLD_BANK_INC_GRP
from climada.util.dwd_icon_loader import (download_icon_grib,
                                          delete_icon_grib,
                                          download_icon_centroids_file)
from climada.util.files_handler import download_file, download_ftp

class TestDataAvail(unittest.TestCase):
    """Test availability of data used through APIs"""

    def test_noaa_nl_pass(self):
        """Test NOAA nightlights used in BlackMarble."""
        file_down = download_file(NOAA_SITE + 'F101992.v4.tar')
        Path(file_down).unlink()

    def test_nasa_nl_pass(self):
        """Test NASA nightlights used in BlackMarble and LitPop."""
        url = NASA_SITE + BM_FILENAMES[0]
        file_down = download_file(url.replace('*', str(2016)))
        Path(file_down).unlink()

    def test_wb_wealth_pass(self):
        """Test world bank's wealth data"""
        file_down = download_file(WORLD_BANK_WEALTH_ACC)
        Path(file_down).unlink()

    def test_wb_lev_hist_pass(self):
        """Test world bank's historical income group levels data"""
        file_down = download_file(WORLD_BANK_INC_GRP)
        Path(file_down).unlink()

    # TODO: FILE_GWP_WEALTH2GDP_FACTORS

    def test_wb_api_pass(self):
        """Test World Bank API"""
        wb.download(indicator='NY.GDP.MKTP.CD', country='CHE', start=1960, end=2030)

    def test_ne_api_pass(self):
        """Test Natural Earth API"""
        url = 'http://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
        file_down = download_file(url)
        Path(file_down).unlink()

    def test_ibtracs_pass(self):
        download_ftp("/".join([IBTRACS_URL, IBTRACS_FILE]), IBTRACS_FILE)
        Path(IBTRACS_FILE).unlink()

    def test_ecmwf_tc_bufr(self):
        """Test availability ECMWF essentials TC forecast."""
        fcast = TCForecast.fetch_bufr_ftp()
        [f.close() for f in fcast]

    def test_icon_forecast_download(self):
        """Test availability of DWD icon forecast."""
        run_date = dt.datetime.today().replace(hour=0,
                                               minute=0,
                                               second=0,
                                               microsecond=0)
        icon_file = download_icon_grib(run_date,max_lead_time=1)
        self.assertEqual(len(icon_file), 1)
        delete_icon_grib(run_date,max_lead_time=1) #deletes icon_file
        self.assertFalse(Path(icon_file[0]).exists())

    def test_icon_centroids_download(self):
        """Test availablility of DWD icon grid information."""
        grid_file = download_icon_centroids_file()
        Path(grid_file).unlink()

# Execute Tests
if __name__ == '__main__':
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDataAvail)
    xmlrunner.XMLTestRunner(output=str(Path(__file__).parent.joinpath('tests_xml'))).run(TESTS)
