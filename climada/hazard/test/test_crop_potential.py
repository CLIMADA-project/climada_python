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

Test crop potential module.
"""
import os
import unittest
import numpy as np
from climada.hazard.crop_potential import CropPotential
from climada.util.constants import DATA_DIR

INPUT_DIR = os.path.join(DATA_DIR, 'demo')
FN_STR_DEMO = 'annual_FR_DE_DEMO'


class TestCropPotential(unittest.TestCase):
    """Test for defining crop potential event"""
    def test_load_EU_all(self):
        """Test defining crop potential hazard from complete demo file (Central Europe)"""
        haz = CropPotential()
        haz.set_from_single_run(input_dir=INPUT_DIR,yearrange=(2001, 2005), \
                        ag_model='lpjml', cl_model='ipsl-cm5a-lr', scenario='historical', \
                        soc='2005soc', co2='co2', crop='whe', irr='noirr', fn_str_var=FN_STR_DEMO)

        self.assertEqual(haz.centroids.lon.min(), -4.75)
        self.assertEqual(haz.centroids.lon.max(), 15.75)
        self.assertEqual(haz.centroids.lat.min(), 42.25)
        self.assertEqual(haz.centroids.lat.max(), 54.75)
        self.assertEqual(haz.intensity.shape, (5, 1092))
        self.assertEqual(haz.event_id.size, 5)
        self.assertAlmostEqual(haz.intensity.max(), 9.803154)

    def test_set_rel_yield(self):
        """Test setting intensity to relativ yield"""
        haz = CropPotential()
        haz.set_from_single_run(input_dir=INPUT_DIR,yearrange=(2001, 2005),ag_model='lpjml', \
                        cl_model='ipsl-cm5a-lr', scenario='historical', soc='2005soc', \
                        co2='co2', crop='whe', irr='noirr', fn_str_var=FN_STR_DEMO)
        hist_mean = haz.calc_mean()
        haz.set_rel_yield_to_int(hist_mean)
        self.assertAlmostEqual(np.max(hist_mean), 8.394599)
        self.assertEqual(haz.intensity.shape, (5, 1092))
        self.assertAlmostEqual(np.nanmax(haz.intensity.toarray()), 2.6180155277252197)
        self.assertAlmostEqual(haz.intensity.max, 2.6180155277252197)
        self.assertAlmostEqual(haz.intensity.min, 0.0)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCropPotential)
unittest.TextTestRunner(verbosity=2).run(TESTS)
