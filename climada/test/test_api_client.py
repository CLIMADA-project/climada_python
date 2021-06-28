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
import unittest

from climada import CONFIG
from climada.util.api_client import Client

DATA_DIR = CONFIG.util.test_data.str()
IN_CONFIG = CONFIG.local_data.system.str()


class TestClient(unittest.TestCase):
    """Test data api client methods."""

    def setUp(self):
        CONFIG.local_data.system._val = DATA_DIR

    def tearDown(self):
        CONFIG.local_data.system._val = IN_CONFIG

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
        datasets = Client().get_datasets(status=None, name='emdat_testdata_fake_2007-2011')
        self.assertEqual(len(datasets), 1)

    def test_dataset(self):
        """"""
        dataset = Client().get_dataset(name='emdat_testdata_fake_2007-2011')
        self.assertEqual(dataset.version, 'v1')
        self.assertEqual(len(dataset.files), 1)
        self.assertEqual(dataset.files[0].file_size, 969)
        self.assertEqual(dataset.data_type.data_type, 'impact')


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestClient)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
