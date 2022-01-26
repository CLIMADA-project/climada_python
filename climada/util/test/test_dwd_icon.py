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

import unittest
from pathlib import Path
import datetime as dt
import numpy as np

from climada.util.dwd_icon_loader import (download_icon_grib,
                                          delete_icon_grib,
                                          _create_icon_grib_name,
                                          download_icon_centroids_file)

from climada.util.constants import SYSTEM_DIR


class TestCreateIconName(unittest.TestCase):
    """Test _create_icon_grib_name function"""
    def test_icon_name(self):
        """Correct strings created"""
        url, file_name, lead_times = _create_icon_grib_name(
            dt.datetime(2021, 2, 2),
            max_lead_time=56,
            )
        self.assertEqual(url, ('https://opendata.dwd.de/weather/nwp/'+
                               'icon-eu-eps/grib/00/vmax_10m/')
                         )
        self.assertEqual(file_name,
                         ('icon-eu-eps_europe_icosahedral_single-level_'+
                          '2021020200_{lead_i:03}_vmax_10m.grib2.bz2')
                         )
        np.testing.assert_array_equal(lead_times,
                                      np.concatenate([np.arange(1,49),
                                                      [51,54,]])
                                      )

        def test_leadtime_warning(self):
            """Adjustment for wrong leadtime"""
            url, file_name, lead_times = _create_icon_grib_name(
                dt.datetime(2021, 2, 2),
                max_lead_time=240,
                )
            self.assertEqual(lead_times.max(),120)


class TestDownloadIcon(unittest.TestCase):
    """Test download_icon_grib function"""
    def test_download_icon(self):
        """Value Error if date to old"""
        try:
            with self.assertRaises(ValueError):
                download_icon_grib(dt.datetime(2020,1,1))
        except IOError:
            pass


class TestDownloadIconCentroids(unittest.TestCase):
    """Test download_icon_centroids_file function"""
    def test_download_icon(self):
        """Value Error if model unknown"""
        with self.assertRaises(ValueError):
            download_icon_centroids_file(model_name='icon')


class TestDeleteIcon(unittest.TestCase):
    """Test delete_icon_grib function"""

    def test_file_not_exist_warning(self):
        """test warning if file does not exist"""

        with self.assertLogs('climada.util.dwd_icon_loader', 'WARNING') as cm:
            delete_icon_grib(dt.datetime(1908, 2, 2),
                                 max_lead_time=1,
                                 )
        self.assertEqual(len(cm.output), 1)
        self.assertIn('does not exist and could not be deleted', cm.output[0])

    def test_rm_file(self):
        """test if file is removed"""
        url, file_name, lead_times = _create_icon_grib_name(
                dt.datetime(1908, 2, 2),
                max_lead_time=1,
                )
        file_name_i = SYSTEM_DIR.absolute().joinpath(
            file_name.format(lead_i=lead_times[0])
            )
        Path(file_name_i).touch()
        delete_icon_grib(dt.datetime(1908, 2, 2),
                         max_lead_time=1,
                         download_dir=SYSTEM_DIR
                         )
        self.assertFalse(Path(file_name_i).exists())


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCreateIconName)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDownloadIcon))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDownloadIconCentroids))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeleteIcon))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
