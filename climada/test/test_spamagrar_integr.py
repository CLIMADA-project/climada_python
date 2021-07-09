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

Tests on SPAM agriculture exposures.
"""

import unittest
# import numpy as np

from climada.entity.exposures.spam_agrar import SpamAgrar


class TestDefault(unittest.TestCase):
    """Test SPAM aggregated value (V_agg) exposure. (default settings)"""

    def test_global_pass(self):
        """Test global entity for default parameters and ent.select:"""
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar()
        ent_select = ent.gdf[ent.gdf.region_id == 208]  # select Denmark only
        self.assertIn('Lat. range: -55.375 to +71.125.', cm.output[0])
        self.assertIn('Lon. range: -179.125 to +179.958.', cm.output[1])
        self.assertIn("Total V_agg TA global: 1301919384722.2 USD.", cm.output[2])
        self.assertTrue(ent.gdf.region_id.min() == 4)
        self.assertTrue(ent.gdf.region_id[10000] == 246)
        self.assertTrue(ent.gdf.region_id.max() == 894)
        self.assertTrue(ent_select.value.sum() == 1878858118.0)
        self.assertTrue(ent_select.region_id.min() == 208)
        self.assertTrue(ent_select.region_id.max() == 208)

    def test_suriname_pass(self):
        """Test country Suriname for default parameters:"""
        country_name = 'Suriname'
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name)
        self.assertIn('Lat. range: +1.875 to +5.958.', cm.output[0])
        self.assertIn('Lon. range: -58.042 to -54.042.', cm.output[1])
        self.assertIn("Total V_agg TA Suriname: 78879225.2 USD.", cm.output[2])
        self.assertTrue(ent.gdf.region_id.min() == 740)
        self.assertTrue(ent.gdf.region_id.max() == 740)

    def test_zurich_pass(self):
        """Test admin 1 Zurich for default parameters:"""
        country_name = 'CHE'
        adm1 = 'Zurich'
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name, name_adm1=adm1)
        self.assertIn('Lat. range: +47.208 to +47.625.', cm.output[0])
        self.assertIn('Lon. range: +8.375 to +8.875.', cm.output[1])
        self.assertIn("Total V_agg TA CHE Zurich: 56644555.1 USD.", cm.output[2])
        self.assertTrue(ent.gdf.region_id.min() == 756)
        self.assertTrue(ent.gdf.region_id.max() == 756)

class TestOtherVar(unittest.TestCase):
    """Test SPAM exposures based on other variables."""
    def test_switzerland_pass(self):
        """Test country CHE for non-default parameters:"""
        country_name = 'CHE'
        tech = 'TI'  # irrigated
        var = 'H'  # harvest area
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name, spam_variable=var,
                                spam_technology=tech)
        self.assertIn('Lat. range: +45.875 to +47.792.', cm.output[0])
        self.assertIn('Lon. range: +6.042 to +10.375.', cm.output[1])
        self.assertIn("Total H TI CHE: 28427.1 Ha.", cm.output[2])
        self.assertTrue(ent.gdf.region_id.min() == 756)
        self.assertTrue(ent.gdf.region_id.max() == 756)

    def test_ucayali_pass(self):
        """Test admin 2 region Ucayali for non-default parameters:"""
        adm2 = 'Ucayali'
        tech = 'TA'  # all
        var = 'Y'  # yield
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(name_adm2=adm2, spam_variable=var,
                                spam_technology=tech)
        self.assertIn('Lat. range: -8.625 to -6.042.', cm.output[0])
        self.assertIn('Lon. range: -76.125 to -74.208.', cm.output[1])
        self.assertIn("Total Y TA  Ucayali: 12298441.3 kg/Ha.", cm.output[2])

class TestInvalidInput(unittest.TestCase):
    """Test SPAM exposures based on invalid inputs."""
    def test_invalid_country(self):
        """Invalid country or admin input returns global entity:"""
        country_name = 'Utopia'
        ent = SpamAgrar()
        with self.assertLogs('climada.entity.exposures.spam_agrar', level='INFO') as cm:
            ent.init_spam_agrar(country=country_name)
        self.assertIn('Country name not found in data: Utopia', cm.output[0])
        self.assertIn('Lat. range: -55.375 to +71.125.', cm.output[1])
        self.assertIn('Lon. range: -179.125 to +179.958.', cm.output[2])
        self.assertIn("Total V_agg TA global: 1301919384722.2 USD.", cm.output[3])

    def test_invalid_parameter(self):
        """Invalid techonology or variable input returns error:"""
        tech = 'XY'  # does not exist
        ent = SpamAgrar()
        with self.assertRaises(ValueError) as cm:
            ent.init_spam_agrar(spam_technology=tech)
        self.assertIn('Invalid input parameter(s).', str(cm.exception))

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestDefault)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOtherVar))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestInvalidInput))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
