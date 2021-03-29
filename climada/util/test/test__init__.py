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

Test config module.
"""
import unittest
import logging

from climada.util import log_level

class TestUtilInit(unittest.TestCase):
    """Test util __init__ methods"""

    def test_log_level_pass(self):
        """Test log level context manager passes"""
        #Check loggers are set to level
        with self.assertLogs('climada', level='INFO') as cm:
             with log_level('WARNING'):
                logging.getLogger('climada').info('info')
                logging.getLogger('climada').error('error')
                self.assertEqual(cm.output, ['ERROR:climada:error'])
        #Check if only climada loggers level change
        with self.assertLogs('matplotlib', level='DEBUG') as cm:
            with log_level('ERROR', name_prefix='climada'):
                logging.getLogger('climada').info('info')
            logging.getLogger('matplotlib').debug('debug')
            self.assertEqual(cm.output, ['DEBUG:matplotlib:debug'])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUtilInit)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
