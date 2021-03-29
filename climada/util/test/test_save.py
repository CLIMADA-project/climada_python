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
from pathlib import Path

from climada import CONFIG
from climada.util.save import save, load

DATA_DIR = CONFIG.util.test_data.str()
IN_CONFIG = CONFIG.local_data.save_dir.str()

class TestSave(unittest.TestCase):
    """Test save function"""

    def setUp(self):
        CONFIG.local_data.save_dir._val = DATA_DIR

    def tearDown(self):
        CONFIG.local_data.save_dir._val = IN_CONFIG

    def test_entity_in_save_dir(self):
        """Returns the same list if its length is correct."""
        file_name = 'save_test.pkl'
        ent = {'value': [1, 2, 3]}
        with self.assertLogs('climada.util.save', level='INFO') as cm:
            save(file_name, ent)
        self.assertTrue(CONFIG.local_data.save_dir.dir().joinpath(file_name).is_file())
        self.assertTrue((file_name in cm.output[0]) or
                        (file_name in cm.output[1]))

    def test_load_pass(self):
        """Load previously saved variable"""
        file_name = 'save_test.pkl'
        ent = {'value': [1, 2, 3]}
        save(file_name, ent)
        res = load(file_name)
        self.assertTrue('value' in res)
        self.assertTrue(res['value'] == ent['value'])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
