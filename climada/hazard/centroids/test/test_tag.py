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

Test Tag class
"""

import unittest

from climada.hazard.centroids.tag import Tag

class TestAppend(unittest.TestCase):
    """Test append function of Tag class"""

    def test_append_different_increase(self):
        """Appends an other tag correctly."""
        tag1 = Tag('file_name1.mat', 'dummy file 1')
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

        tag2 = Tag('file_name2.mat', 'dummy file 2')
        
        tag1.append(tag2)
        
        self.assertEqual(['file_name1.mat', 'file_name2.mat'], tag1.file_name)
        self.assertEqual(['dummy file 1', 'dummy file 2'], tag1.description)

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

class TestJoin(unittest.TestCase):
    """Test joining functions and string formation Tag class."""

    def test_one_str_pass(self):
        """ Test __str__ method with one file"""
        tag = Tag('file_name1.mat', 'dummy file 1')
        self.assertEqual(str(tag), 
                         ' File: file_name1\n Description: dummy file 1')

    def test_teo_str_pass(self):
        """ Test __str__ method with one file"""
        tag1 = Tag('file1.mat', 'desc1')
        tag2 = Tag('file2.xls', 'desc2')
        tag1.append(tag2)
        self.assertEqual(str(tag1), 
                         ' File: file1 + file2\n Description: desc1 + desc2')

    def test_join_names_pass(self):
        """ Test join_file_names function."""
        tag1 = Tag('file1', 'desc1')
        tag2 = Tag('file2', 'desc2')
        tag1.append(tag2)
        join_name = tag1.join_file_names()
        self.assertEqual('file1 + file2', join_name)

    def test_join_descr_pass(self):
        """ Test join_descriptions function."""
        tag1 = Tag('file1', 'desc1')
        tag2 = Tag('file2', 'desc2')
        tag1.append(tag2)
        join_desc = tag1.join_descriptions()
        self.assertEqual('desc1 + desc2', join_desc)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestAppend)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestJoin))
unittest.TextTestRunner(verbosity=2).run(TESTS)
