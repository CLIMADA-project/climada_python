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

Test Tag class
"""

import unittest

from climada.entity.tag import Tag

class TestAppend(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def test_append_different_increase(self):
        """Appends an other tag correctly."""
        tag1 = Tag('file_name1.mat', 'dummy file 1')
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

        tag2 = Tag('file_name2.mat', 'dummy file 2')

        tag1.append(tag2)

        self.assertEqual('file_name1.mat + file_name2.mat', tag1.file_name)
        self.assertEqual('dummy file 1 + dummy file 2', tag1.description)

    def test_append_equal_same(self):
        """Appends an other tag correctly."""
        tag1 = Tag('file_name1.mat', 'dummy file 1')
        tag2 = Tag('file_name1.mat', 'dummy file 1')

        tag1.append(tag2)
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

    def test_append_empty(self):
        """Appends an other tag correctly."""
        tag1 = Tag('file_name1.mat', 'dummy file 1')
        tag2 = Tag()

        tag1.append(tag2)
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

        tag1 = Tag()
        tag2 = Tag('file_name1.mat', 'dummy file 1')

        tag1.append(tag2)
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestAppend)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
