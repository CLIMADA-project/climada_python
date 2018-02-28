"""
Test Tag class
"""

import unittest

from climada.hazard.tag import Tag as TagHazard

class TestTag(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def test_append_right_pass(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        self.assertEqual('TC', tag1.haz_type)

        tag2 = TagHazard('file_name2.mat', 'TC', 'dummy file 2')
        
        tag1.append(tag2)
        
        self.assertEqual(['file_name1.mat', 'file_name2.mat'], tag1.file_name)
        self.assertEqual(['dummy file 1', 'dummy file 2'], tag1.description)
        self.assertEqual('TC', tag1.haz_type)

    def test_append_wrong_pass(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        tag2 = TagHazard('file_name2.mat', 'EQ', 'dummy file 2')
        with self.assertLogs('climada.hazard.tag', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                tag1.append(tag2)
        self.assertIn("Hazards of different type can't be appended: " \
                         + "TC != EQ.", cm.output[0])

    def test_equal_same(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        tag2 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        tag1.append(tag2)        
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        self.assertEqual('TC', tag1.haz_type)
        
    def test_append_empty(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        tag2 = TagHazard()
        tag2.haz_type = 'TC'
        
        tag1.append(tag2)    
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        
        tag1 = TagHazard()
        tag1.haz_type = 'TC'
        tag2 = TagHazard('file_name1.mat', 'TC', 'dummy file 1')
        
        tag1.append(tag2)
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTag)
unittest.TextTestRunner(verbosity=2).run(TESTS)
