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

from climada.util.config import Config, CONFIG

class TestConfig(unittest.TestCase):
    """Test Config methods"""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_from_dict(self):
        """Check the creation and use of a Config object."""
        dct = {'a': 4.,
               'b': [0, 1., '2', {'c': 'c'}, [[11, 12], [21, 22]]]}
        conf = Config.from_dict(dct)
        self.assertEqual(conf.a.float(), 4.0)
        self.assertEqual(str(conf), '{a: 4.0, b: [0, 1.0, 2, {c: c}, [[11, 12], [21, 22]]]}')
        try:
            conf.a.int()
            self.fail("this should have failed with `<class 'float'>, not int`")
        except:
            pass
        self.assertEqual(conf.b.get(0).int(), 0)
        self.assertEqual(conf.b.int(0), 0)
        self.assertEqual(conf.b.float(1), 1.)
        self.assertEqual(conf.b.str(2), '2')
        self.assertEqual(conf.b.get(3).c.str(), 'c')
        self.assertEqual(conf.b.get(4, 1, 0).int(), 21)
        self.assertEqual(conf.b.get(4, 1).int(1), 22)
        self.assertEqual(conf.b.get(4).list(0)[1].int(), 12)
        self.assertEqual(conf.b.list(4)[0].int(0), 11)

    def test_substitute(self):
        global CONFIG
        """Check the substitution of references."""
        dct = {'a': 'https://{b.c}/{b.d}.{b.e}', 'b': {'c': 'host', 'd': 'page', 'e': 'domain'}}
        conf = Config.from_dict(dct)
        self.assertEqual(conf.a._root, conf._root)
        self.assertEqual(conf.a.str(), 'https://host/page.domain')


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestConfig)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
