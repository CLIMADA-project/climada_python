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

"""

import unittest
import numpy as np
import scipy.sparse as sparse

from climada.util.checker import check_oligatories, check_optionals

class DummyClass(object):

    vars_oblig = {'id', 'array', 'sparse_arr'}
    vars_opt = {'list', 'array_opt'}

    def __init__(self):
        self.id = np.arange(25)
        self.array = np.arange(25)
        self.array_opt = np.arange(25)
        self.list = np.arange(25).tolist()
        self.sparse_arr = sparse.csr.csr_matrix(np.zeros((25, 2)))
        self.name = 'name class'

class TestChecks(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def test_check_oligatories_pass(self):
        """Correct DummyClass definition"""
        dummy = DummyClass()
        check_oligatories(dummy.__dict__, dummy.vars_oblig, "DummyClass.",
                          dummy.id.size, dummy.id.size, 2)

    def test_check_oligatories_fail(self):
        """Wrong DummyClass definition"""
        dummy = DummyClass()
        dummy.array = np.arange(3)
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                check_oligatories(dummy.__dict__, dummy.vars_oblig, "DummyClass.",
                                  dummy.id.size, dummy.id.size, 2)
        self.assertIn('Invalid DummyClass.array size: 25 != 3.', cm.output[0])

        dummy = DummyClass()
        dummy.sparse_arr = sparse.csr.csr_matrix(np.zeros((25, 1)))
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                check_oligatories(dummy.__dict__, dummy.vars_oblig, "DummyClass.",
                                  dummy.id.size, dummy.id.size, 2)
        self.assertIn('Invalid DummyClass.sparse_arr column size: 2 != 1.', cm.output[0])

    def test_check_optionals_pass(self):
        """Correct DummyClass definition"""
        dummy = DummyClass()
        check_optionals(dummy.__dict__, dummy.vars_opt, "DummyClass.",
                        dummy.id.size)

    def test_check_optionals_fail(self):
        """Correct DummyClass definition"""
        dummy = DummyClass()
        dummy.array_opt = np.arange(3)
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                check_optionals(dummy.__dict__, dummy.vars_opt, "DummyClass.",
                                dummy.id.size)
        self.assertIn('Invalid DummyClass.array_opt size: 25 != 3.', cm.output[0])

        dummy.array_opt = np.array([], int)
        with self.assertLogs('climada.util.checker', level='DEBUG') as cm:
            check_optionals(dummy.__dict__, dummy.vars_opt, "DummyClass.",
                            dummy.id.size)
        self.assertIn('DummyClass.array_opt not set.', cm.output[0])

        dummy = DummyClass()
        dummy.list = np.arange(3).tolist()
        with self.assertLogs('climada.util.checker', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                check_optionals(dummy.__dict__, dummy.vars_opt, "DummyClass.",
                                dummy.id.size)
        self.assertIn('Invalid DummyClass.list size: 25 != 3.', cm.output[0])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecks)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
