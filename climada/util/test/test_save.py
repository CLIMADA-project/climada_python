"""
Test save module.
"""

import unittest

import climada.util.save as save
from climada.entity.entity import Entity
from climada.util.constants import ENT_DEMO_XLS

class TestSave(unittest.TestCase):
    """Test save function"""
    def test_entity_in_save_dir(self):
        """Returns the same list if its length is correct."""
        ent = Entity(ENT_DEMO_XLS)
        save.save('entity_demo.pkl', ent)
                
    def test_inexistent_folder_fail(self):
        """Raise error if folder does not exists."""
        ent = Entity(ENT_DEMO_XLS)
        with self.assertRaises(ValueError) as error:
            save.save('../wrong/entity_demo.pkl', ent)
        self.assertIn('wrong', str(error.exception))
            
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
unittest.TextTestRunner(verbosity=2).run(TESTS)
