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
        with self.assertLogs('climada.util.save', level='INFO') as cm:
            save('entity_demo.pkl', ent)
        self.assertTrue(('entity_demo.pkl' in cm.output[0]) or \
                        ('entity_demo.pkl' in cm.output[1]))        
            
# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSave)
unittest.TextTestRunner(verbosity=2).run(TESTS)
