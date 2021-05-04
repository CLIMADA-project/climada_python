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

Unit Tests for LitPop class.
"""

import numpy as np
import unittest
from climada.entity.exposures import litpop_new as lp # TODO: replace litpop_new


def data_arrays_demo(number_of_arrays=2):
    data_arrays = list()
    if number_of_arrays > 0:
        data_arrays.append(np.array([[0,1,2], [3,4,5]]))
        # array([[0, 1, 2],
        #       [3, 4, 5]])
    if number_of_arrays > 1:
        data_arrays.append(np.array([[10,10,10], [1,1,1]]))
        # array([[10, 10, 10],
        #       [1, 1, 1]])
    if number_of_arrays > 2:
        data_arrays.append(np.array([[0,1,10], [0,1,10]]))
        # array([[0, 1, 10],
        #       [0, 1, 10]])
    if number_of_arrays > 3:
        data_arrays.append([[0,1,10,100], [0,1,10,100]])
        # [[0, 1, 10, 100],
        #  [0, 1, 10, 100]]
    return data_arrays


class TestLitPop(unittest.TestCase):
    """Test LitPop Class methods and functions"""

    def test_gridpoints_core_calc_input_errors(self):
        """test for ValueErrors and TypeErrors due to wrong input to function
        gridpoints_core_calc"""
        data = data_arrays_demo(2)
        # negative offset:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data, offsets=[2,-1])
        # negative exponents:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data, exponents=[2,-1])

        # different shapes:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data_arrays_demo(4))

        # wrong format:
        with self.assertRaises(TypeError):
            lp.gridpoints_core_calc(data, exponents=['a', 'b'])
        data.append('hello i am a string')
        with self.assertRaises(TypeError):
            lp.gridpoints_core_calc(data)
        with self.assertRaises(TypeError):
            lp.gridpoints_core_calc(777)


    def test_gridpoints_core_calc_default_1(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets - 1 array"""
        data_arrays = data_arrays_demo(1) # get list with 1 demo array
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[1,1], results_check[1,1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_default_2(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets- 2 arrays"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0] * data_arrays[1]

        self.assertEqual(result_array[0,0], results_check[0,0])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)


    def test_gridpoints_core_calc_default_3(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets- 3 arrays"""
        data_arrays = data_arrays_demo(3)
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0] * data_arrays[1] * data_arrays[2]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[1,1], results_check[1,1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)
        #self.assertEqual(result_array, data_arrays[0] * data_arrays[1])

    def test_gridpoints_core_calc_exp(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed exponents"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        exp = [2, 1]
        result_array = lp.gridpoints_core_calc(data_arrays, exponents=exp)
        results_check = data_arrays[0] * data_arrays[0] * data_arrays[1]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0,2], results_check[0,2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

        exp = [2, .1]
        result_array = lp.gridpoints_core_calc(data_arrays, exponents=exp)
        results_check = data_arrays[0] * data_arrays[0] * (data_arrays[1] ** .1)
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        offsets = [1, 10]
        result_array = lp.gridpoints_core_calc(data_arrays, offsets=offsets)
        results_check = (data_arrays[0]+1) * (10 + data_arrays[1])

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0,2], results_check[0,2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets_exp(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets and exponents"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        offsets = [0, 10]
        exp = [2, 1]
        result_array = lp.gridpoints_core_calc(data_arrays, offsets=offsets,
                                              exponents=exp)
        results_check = (data_arrays[0]) * (data_arrays[0]) * (10+data_arrays[1])
        results_check2 = np.array([[0, 20, 80],[99, 176, 275]])

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0,2], results_check[0,2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)
        self.assertEqual(result_array[1,2], results_check2[1,2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check2)

    def test_gridpoints_core_calc_rescale(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with rescaling (default exponents and offsets)"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        result_array = lp.gridpoints_core_calc(data_arrays, total_val_rescale=2.5)
        results_check = (data_arrays[0]*data_arrays[1]) * 2.5/np.sum(data_arrays[0]*data_arrays[1])

        self.assertAlmostEqual(result_array.sum(), 2.5)
        self.assertEqual(result_array[0,1], results_check[0,1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets_exp_rescale(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets and exponents and rescaling"""
        data_arrays = data_arrays_demo(2) # get list with 2 demo arrays
        offsets = [0.2, 3]
        exp = [.5, 1.7]
        tot = -7
        result_array = lp.gridpoints_core_calc(data_arrays, offsets=offsets,
                                              exponents=exp, total_val_rescale=tot)
        results_check = np.array(data_arrays[0]+.2, dtype=float)**exp[0] * \
            (np.array(data_arrays[1]+3., dtype=float)**exp[1])
        results_check = results_check * tot / results_check.sum()
        self.assertEqual(result_array.shape, results_check.shape)
        self.assertAlmostEqual(result_array.sum(), tot)
        self.assertEqual(result_array[1,2], results_check[1,2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)



if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLitPop)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertainty))
    unittest.TextTestRunner(verbosity=2).run(TESTS)