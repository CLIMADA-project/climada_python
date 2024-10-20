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

Test ImpactFunc class.
"""

import unittest

import numpy as np

from climada.entity.impact_funcs.base import ImpactFunc


class TestInterpolation(unittest.TestCase):
    """Impact function interpolation test"""

    def test_calc_mdr_pass(self):
        """Compute mdr interpolating values."""
        intensity = np.arange(0, 100, 10)
        paa = np.arange(0, 1, 0.1)
        mdd = np.arange(0, 1, 0.1)
        imp_fun = ImpactFunc(intensity=intensity, paa=paa, mdd=mdd)
        new_inten = 17.2
        self.assertEqual(imp_fun.calc_mdr(new_inten), 0.029583999999999996)

    def test_from_step(self):
        """Check default impact function: step function"""
        inten = (0, 5, 10)
        imp_fun = ImpactFunc.from_step_impf(intensity=inten, haz_type="TC", impf_id=2)
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones(4)))
        self.assertTrue(np.array_equal(imp_fun.mdd, np.array([0, 0, 1, 1])))
        self.assertTrue(np.array_equal(imp_fun.intensity, np.array([0, 5, 5, 10])))
        self.assertEqual(imp_fun.haz_type, "TC")
        self.assertEqual(imp_fun.id, 2)

    def test_from_sigmoid(self):
        """Check default impact function: sigmoid function"""
        inten = (0, 100, 5)
        imp_fun = ImpactFunc.from_sigmoid_impf(
            inten, L=1.0, k=2.0, x0=50.0, haz_type="RF", impf_id=2
        )
        self.assertTrue(np.array_equal(imp_fun.paa, np.ones(20)))
        self.assertEqual(imp_fun.mdd[10], 0.5)
        self.assertEqual(imp_fun.mdd[-1], 1.0)
        self.assertTrue(np.array_equal(imp_fun.intensity, np.arange(0, 100, 5)))
        self.assertEqual(imp_fun.haz_type, "RF")
        self.assertEqual(imp_fun.id, 2)

    def test_from_poly_s_shape(self):
        """Check default impact function: polynomial s-shape"""

        haz_type = "RF"
        threshold = 0.2
        half_point = 1
        scale = 0.8
        exponent = 4
        impf_id = 2
        unit = "m"
        intensity = (0, 5, 5)

        def test_aux_vars(impf):
            self.assertTrue(np.array_equal(impf.paa, np.ones(5)))
            self.assertTrue(np.array_equal(impf.intensity, np.linspace(0, 5, 5)))
            self.assertEqual(impf.haz_type, haz_type)
            self.assertEqual(impf.id, impf_id)
            self.assertEqual(impf.intensity_unit, unit)

        impf = ImpactFunc.from_poly_s_shape(
            intensity=intensity,
            threshold=threshold,
            half_point=half_point,
            scale=scale,
            exponent=exponent,
            haz_type=haz_type,
            impf_id=impf_id,
            intensity_unit=unit,
        )
        # True value can easily be computed with a calculator
        correct_mdd = np.array([0, 0.59836395, 0.78845941, 0.79794213, 0.79938319])
        np.testing.assert_array_almost_equal(impf.mdd, correct_mdd)
        test_aux_vars(impf)

        # If threshold > half_point, mdd should all be 0
        impf = ImpactFunc.from_poly_s_shape(
            intensity=intensity,
            threshold=half_point * 2,
            half_point=half_point,
            scale=scale,
            exponent=exponent,
            haz_type=haz_type,
            impf_id=impf_id,
            intensity_unit=unit,
        )
        np.testing.assert_array_almost_equal(impf.mdd, np.zeros(5))
        test_aux_vars(impf)

        # If exponent = 0, mdd should be constant
        impf = ImpactFunc.from_poly_s_shape(
            intensity=intensity,
            threshold=threshold,
            half_point=half_point,
            scale=scale,
            exponent=0,
            haz_type=haz_type,
            impf_id=impf_id,
            intensity_unit=unit,
        )
        np.testing.assert_array_almost_equal(impf.mdd, np.ones(5) * scale / 2)
        test_aux_vars(impf)

        # If exponent < 0, raise error.
        with self.assertRaisesRegex(ValueError, "Exponent value"):
            ImpactFunc.from_poly_s_shape(
                intensity=intensity,
                threshold=half_point,
                half_point=half_point,
                scale=scale,
                exponent=-1,
                haz_type=haz_type,
                impf_id=impf_id,
                intensity_unit=unit,
            )


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInterpolation)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
