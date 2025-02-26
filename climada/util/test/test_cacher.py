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

Test cacher module.
"""

import shutil
import tempfile
import time
import unittest
from concurrent.futures import thread
from pathlib import Path

from climada.util.cacher import Cacher, cached


@cached(Cacher(sqlite=".c.db"))
def f(a, b, c):
    return str(time.time())


class TestCacher(unittest.TestCase):
    """Test Cacher methods"""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir)

    @cached(Cacher(sqlite=".c.db", cachedir=".cdir"))
    def g(self, a, b):
        return str(time.time())

    def test_dir_cacher(self):
        o = f(44, b=55, c=66)
        r = self.g(44, 55)
        time.sleep(0.001)
        self.assertEqual(o, f(44, b=55, c=66))
        self.assertEqual(r, self.g(44, 55))
        self.assertNotEqual(o, f(44, 55, c=66))
