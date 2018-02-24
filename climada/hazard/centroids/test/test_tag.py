"""
Test Tag class
"""

import unittest

from climada.hazard.centroids.tag import Tag

class TestTag(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

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

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTag)
unittest.TextTestRunner(verbosity=2).run(TESTS)
