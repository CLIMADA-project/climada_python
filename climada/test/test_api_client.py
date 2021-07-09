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

Test save module.
"""
from pathlib import Path
import unittest
from shutil import rmtree

from climada import CONFIG
from climada.util.api_client import Client, Download

DATA_DIR = CONFIG.test_data.dir()


class TestClient(unittest.TestCase):
    """Test data api client methods."""

    def test_data_type(self):
        """"""
        lpdt = Client().get_data_type("litpop")
        self.assertEqual(lpdt.data_type, 'litpop')
        self.assertEqual(lpdt.data_type_group, 'exposures')

    def test_data_types(self):
        """"""
        exdts = Client().get_data_types("exposures")
        self.assertTrue('litpop' in [exdt.data_type for exdt in exdts])

    def test_datasets(self):
        """"""
        datasets = Client().get_datasets(status=None, name='FAOSTAT_data_producer_prices')
        self.assertEqual(len(datasets), 1)

    def test_dataset(self):
        """"""
        client = Client()
        
        dataset = client.get_dataset(name='FAOSTAT_data_producer_prices')
        self.assertEqual(dataset.version, 'v1')
        self.assertEqual(len(dataset.files), 1)
        self.assertEqual(dataset.files[0].file_size, 26481)
        self.assertEqual(dataset.data_type.data_type, 'crop_production')

        dataset2 = client.get_dataset_by_uuid(dataset.uuid)
        self.assertEqual(dataset, dataset2)

    def test_download_file(self):
        """"""
        client = Client()
        client.MAX_WAITING_PERIOD = 0.1
        dataset = client.get_dataset(name='FAOSTAT_data_producer_prices')

        # test failure
        def fail(x, y):
            raise Download.Failed("on purpose")
        self.assertRaises(Download.Failed,
            client.download_file, DATA_DIR, dataset.files[0], check=fail)
        self.assertFalse(DATA_DIR.joinpath(dataset.files[0].file_name).is_file())

        # test success
        download = client.download_file(DATA_DIR, dataset.files[0])
        self.assertEqual(download, DATA_DIR / dataset.files[0].file_name)
        self.assertTrue(download.is_file())
        self.assertEqual(download.stat().st_size, dataset.files[0].file_size)
        download.unlink()
        self.assertFalse(download.is_file())

    def test_download_dataset(self):
        """"""
        client = Client()
        client.MAX_WAITING_PERIOD = 0.1

        dataset = client.get_dataset(name='test_write_raster')
        downloads = client.download_dataset(dataset, target_dir=DATA_DIR)
        self.assertEqual(len(downloads), 2)
        for download, dsfile in zip(downloads, dataset.files):
            self.assertTrue(download.is_file())
            self.assertEqual(download.stat().st_size, dsfile.file_size)
            self.assertEqual(download.name, dsfile.file_name)
            self.assertEqual(download.parent.name, dataset.version)
            self.assertEqual(download.parent.parent.name, dataset.name)
            self.assertEqual(download.parent.parent.parent.name, dataset.data_type.data_type)
            download.unlink()
        rm_empty_dir(download.parent.parent.parent)


def rm_empty_dir(folder):
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            rm_empty_dir(subfolder)
    try:
        folder.rmdir()
    except:
        pass


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestClient)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
