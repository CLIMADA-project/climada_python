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

Test save module.
"""
import os
import copy
import unittest

from climada.util.save import save, load

from climada.util.config import CONFIG

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

IN_CONFIG = copy.copy(CONFIG['local_data']['save_dir'])

class TestSave(unittest.TestCase):
    """Test save function"""

    def setUp(self):
        CONFIG['local_data']['save_dir'] = DATA_DIR

    def tearDown(self):
        CONFIG['local_data']['save_dir'] = IN_CONFIG

    def test_entity_in_save_dir(self):
        """Returns the same list if its length is correct."""
        file_name = 'save_test.pkl'
        ent = {'value': [1, 2, 3]}
        with self.assertLogs('climada.util.save', level='INFO') as cm:
            save(file_name, ent)
        self.assertTrue(os.path.isfile(os.path.join(DATA_DIR, file_name)))
        self.assertTrue((file_name in cm.output[0]) or
                        (file_name in cm.output[1]))

    def test_load_pass(self):
        """Load previously saved variable"""
        file_name = 'save_test.pkl'
        ent = {'value': [1, 2, 3]}
        save(file_name, ent)
        res = load(file_name)
        self.assertTrue(os.path.isfile(os.path.join(DATA_DIR, file_name)))
        self.assertTrue('value' in res)
        self.assertTrue(res['value'] == ent['value'])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
