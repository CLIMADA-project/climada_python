"""
Test save module.
"""

import unittest

from climada.util.save import save
from climada.entity.entity import Entity
from climada.util.constants import ENT_DEMO_XLS

class TestSave(unittest.TestCase):
    """Test save function"""
    def test_entity_in_save_dir(self):
        """Returns the same list if its length is correct."""
        ent = Entity(ENT_DEMO_XLS)
        save('entity_demo.pkl', ent)
                
    def test_inexistent_folder_fail(self):
        """Raise error if folder does not exists."""
        ent = Entity(ENT_DEMO_XLS)
        with self.assertLogs('climada.util.save', level='ERROR') as cm:
            with self.assertRaises(FileNotFoundError):
                save('../wrong/entity_demo.pkl', ent)
        self.assertIn('Folder not found:', cm.output[0])
        self.assertIn('wrong', cm.output[0])
            
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
unittest.TextTestRunner(verbosity=2).run(TESTS)
