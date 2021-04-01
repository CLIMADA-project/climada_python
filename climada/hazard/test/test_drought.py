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

Tests on Drought Hazard"""


import unittest

from climada.hazard.drought import Drought


class TestReader(unittest.TestCase):
    """Test loading functions from the Drought class"""
    def test(self):

        drought = Drought()
        drought.set_area(44.5, 5, 50, 12)

        hazard_set = drought.setup()

        self.assertEqual(hazard_set.tag.haz_type, 'DR')
        self.assertEqual(hazard_set.size, 114)
        self.assertEqual(hazard_set.centroids.size, 130)
        self.assertEqual(hazard_set.intensity[112, 111], -1.6286273002624512)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
