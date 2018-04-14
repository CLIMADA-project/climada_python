"""
Test Tag class
"""

import unittest

from climada.hazard.tag import Tag as TagHazard

class TestTag(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def test_append_right_pass(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        self.assertEqual('TC', tag1.haz_type)

        tag2 = TagHazard('TC', 'file_name2.mat', 'dummy file 2')
        
        tag1.append(tag2)
        
        self.assertEqual(['file_name1.mat', 'file_name2.mat'], tag1.file_name)
        self.assertEqual(['dummy file 1', 'dummy file 2'], tag1.description)
        self.assertEqual('TC', tag1.haz_type)

    def test_append_wrong_pass(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        tag2 = TagHazard('EQ', 'file_name2.mat', 'dummy file 2')
        with self.assertLogs('climada.hazard.tag', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                tag1.append(tag2)
        self.assertIn("Hazards of different type can't be appended: " \
                         + "TC != EQ.", cm.output[0])

    def test_equal_same(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        tag2 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        tag1.append(tag2)        
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        self.assertEqual('TC', tag1.haz_type)
        
    def test_append_empty(self):
        """Appends an other tag correctly."""
        tag1 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        tag2 = TagHazard()
        tag2.haz_type = 'TC'
        
        tag1.append(tag2)    
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)
        
        tag1 = TagHazard()
        tag1.haz_type = 'TC'
        tag2 = TagHazard('TC', 'file_name1.mat', 'dummy file 1')
        
        tag1.append(tag2)
        self.assertEqual('file_name1.mat', tag1.file_name)
        self.assertEqual('dummy file 1', tag1.description)

    def test_str_pass(self):
        """ Test __str__ method """
        tag = TagHazard('EQ', 'file_name1.mat', 'dummy file 1')
        self.assertEqual(str(tag), ' Type: EQ\n File: file_name1.mat\n Description: dummy file 1')

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestTag)
unittest.TextTestRunner(verbosity=2).run(TESTS)
