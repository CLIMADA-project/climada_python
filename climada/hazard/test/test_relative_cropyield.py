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

Test crop potential module.
"""
import unittest
import numpy as np
from climada.hazard.relative_cropyield import RelativeCropyield
from climada.util.constants import DEMO_DIR as INPUT_DIR

FN_STR_DEMO = 'annual_FR_DE_DEMO'


class TestRelativeCropyield(unittest.TestCase):
    """Test for defining crop potential event"""
    def test_load_EU_all(self):
        """Test defining crop potential hazard from complete demo file (Central Europe)"""
        haz = RelativeCropyield()
        haz.set_from_isimip_netcdf(input_dir=INPUT_DIR, yearrange=(2001, 2005),
                                ag_model='lpjml', cl_model='ipsl-cm5a-lr', scenario='historical',
                                soc='2005soc', co2='co2', crop='whe', irr='noirr',
                                fn_str_var=FN_STR_DEMO)

        self.assertEqual(haz.crop, 'whe')
        self.assertEqual(haz.tag.haz_type, 'RC')
        self.assertIn('lpjml', haz.tag.file_name)
        self.assertIn('ipsl-cm5a-lr', haz.tag.file_name)
        self.assertIn('hist', haz.tag.file_name)
        self.assertIn('2005soc', haz.tag.file_name)
        self.assertIn('noirr', haz.tag.file_name)

        self.assertEqual(haz.centroids.lon.min(), -4.75)
        self.assertEqual(haz.centroids.lon.max(), 15.75)
        self.assertEqual(haz.centroids.lat.min(), 42.25)
        self.assertEqual(haz.centroids.lat.max(), 54.75)
        self.assertEqual(haz.intensity.shape, (5, 1092))
        self.assertEqual(haz.event_id.size, 5)
        self.assertAlmostEqual(haz.intensity.max(), 10.176164, places=5)

    def test_set_rel_yield(self):
        """Test setting intensity to relativ yield"""
        haz = RelativeCropyield()
        haz.set_from_isimip_netcdf(input_dir=INPUT_DIR, yearrange=(2001, 2005), ag_model='lpjml',
                                cl_model='ipsl-cm5a-lr', scenario='historical', soc='2005soc',
                                co2='co2', crop='whe', irr='noirr', fn_str_var=FN_STR_DEMO)
        hist_mean = haz.calc_mean(np.array([2001, 2005]))

        self.assertEqual(haz.intensity_def, 'Yearly Yield')
        haz.set_rel_yield_to_int(hist_mean)
        self.assertEqual(haz.intensity_def, 'Relative Yield')

        self.assertEqual(np.shape(hist_mean), (1092,))
        self.assertAlmostEqual(np.max(hist_mean), 8.397826, places=5)
        self.assertEqual(haz.intensity.shape, (5, 1092))
        self.assertAlmostEqual(np.nanmax(haz.intensity.toarray()), 4.0, places=5)
        self.assertAlmostEqual(haz.intensity.max(), 4.0, places=5)
        self.assertAlmostEqual(haz.intensity.min(), -1.0, places=5)

    def test_set_percentile_to_int(self):
        """Test setting intensity to percentile of the yield"""
        haz = RelativeCropyield()
        haz.set_from_isimip_netcdf(input_dir=INPUT_DIR, yearrange=(2001, 2005), ag_model='lpjml',
                                cl_model='ipsl-cm5a-lr', scenario='historical', soc='2005soc',
                                co2='co2', crop='whe', irr='noirr', fn_str_var=FN_STR_DEMO)
        haz.set_percentile_to_int()
        self.assertEqual(haz.intensity_def, 'Percentile')

        self.assertEqual(haz.intensity.shape, (5, 1092))
        self.assertAlmostEqual(haz.intensity.max(), 1.0, places=5)
        self.assertAlmostEqual(haz.intensity.min(), 0.2, places=5)
        self.assertAlmostEqual(haz.intensity.data[10], 0.6, places=5)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestRelativeCropyield)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
