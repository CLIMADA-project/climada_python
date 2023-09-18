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

Test files_handler module.
"""

from pathlib import Path
from sys import dont_write_bytecode
import pandas as pd
import unittest
import xmlrunner
import datetime as dt

import numpy as np
from pandas_datareader import wb

from climada import CONFIG
from climada.entity.exposures.litpop.nightlight import BM_FILENAMES, download_nl_files
from climada.hazard.tc_tracks import IBTRACS_URL, IBTRACS_FILE
from climada.util.finance import WORLD_BANK_WEALTH_ACC, WORLD_BANK_INC_GRP
from climada.util.dwd_icon_loader import (download_icon_grib,
                                          delete_icon_grib,
                                          download_icon_centroids_file)
from climada.util.files_handler import download_file, download_ftp

class TestDataAvail(unittest.TestCase):
    """Test availability of data used through APIs"""

    def test_noaa_nl_pass(self):
        """Test NOAA nightlights used in BlackMarble."""
        file_down = download_file(f'{CONFIG.exposures.litpop.nightlights.noaa_url.str()}/F101992.v4.tar')
        Path(file_down).unlink()

    def test_nasa_nl_pass(self):
        """Test NASA nightlights used in BlackMarble and LitPop."""
        req_files = np.zeros(len(BM_FILENAMES))
        req_files[0] = 1
        year = 2016
        dwnl_path = CONFIG.local_data.save_dir.dir()
        dwnl_file = dwnl_path.joinpath(BM_FILENAMES[0] % year)
        self.assertFalse(dwnl_file.is_file())
        download_nl_files(req_files=req_files, dwnl_path=dwnl_path, year=year)
        self.assertTrue(dwnl_file.is_file())
        dwnl_file.unlink()

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
        url = 'https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip'
        file_down = download_file(url)
        Path(file_down).unlink()

    def test_ibtracs_pass(self):
        download_ftp("/".join([IBTRACS_URL, IBTRACS_FILE]), IBTRACS_FILE)
        Path(IBTRACS_FILE).unlink()

    def test_icon_eu_forecast_download(self):
        """Test availability of DWD icon forecast."""
        run_datetime = dt.datetime.utcnow() - dt.timedelta(hours=5)
        run_datetime = run_datetime.replace(hour=run_datetime.hour//12*12,
                                            minute=0,
                                            second=0,
                                            microsecond=0)
        icon_file = download_icon_grib(run_datetime,max_lead_time=1)
        self.assertEqual(len(icon_file), 1)
        delete_icon_grib(run_datetime,max_lead_time=1) #deletes icon_file
        self.assertFalse(Path(icon_file[0]).exists())

    def test_icon_d2_forecast_download(self):
        """Test availability of DWD icon forecast."""
        run_datetime = dt.datetime.utcnow() - dt.timedelta(hours=5)
        run_datetime = run_datetime.replace(hour=run_datetime.hour//12*12,
                                            minute=0,
                                            second=0,
                                            microsecond=0)
        icon_file = download_icon_grib(run_datetime,
                                       model_name='icon-d2-eps',
                                       max_lead_time=1)
        self.assertEqual(len(icon_file), 1)
        delete_icon_grib(run_datetime,
                         model_name='icon-d2-eps',
                         max_lead_time=1) #deletes icon_file
        self.assertFalse(Path(icon_file[0]).exists())

    def test_icon_centroids_download(self):
        """Test availablility of DWD icon grid information."""
        grid_file = download_icon_centroids_file()
        Path(grid_file).unlink()
        grid_file = download_icon_centroids_file(model_name='icon-d2-eps')
        Path(grid_file).unlink()

# Execute Tests
if __name__ == '__main__':
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDataAvail)
    xmlrunner.XMLTestRunner(output=str(Path(__file__).parent.joinpath('tests_xml'))).run(TESTS)
