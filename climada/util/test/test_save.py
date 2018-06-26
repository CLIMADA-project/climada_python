"""
Test save module.
"""
import unittest

from climada.util.save import save

class TestSave(unittest.TestCase):
    """Test save function"""
    def test_entity_in_save_dir(self):
        """Returns the same list if its length is correct."""
        ent = {'value': [1, 2, 3]}
        with self.assertLogs('climada.util.save', level='INFO') as cm:
            save('save_test.pkl', ent)
        self.assertTrue(('save_test.pkl' in cm.output[0]) or \
                        ('save_test.pkl' in cm.output[1]))        
            
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
unittest.TextTestRunner(verbosity=2).run(TESTS)
