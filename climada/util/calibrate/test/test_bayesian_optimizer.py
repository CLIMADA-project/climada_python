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
Tests for calibration module
"""

import unittest
from unittest.mock import patch, MagicMock

import numpy.testing as npt
import pandas as pd

from climada.util.calibrate import Input, BayesianOptimizer, BayesianOptimizerController

from .test_base import hazard, exposure


def input():
    """Return a mocked input"""
    return Input(
        hazard=hazard(),
        exposure=exposure(),
        data=pd.DataFrame(data={"col1": [1, 2], "col2": [2, 3]}, index=[0, 1]),
        cost_func=MagicMock(),
        impact_func_creator=MagicMock(),
        impact_to_dataframe=MagicMock(),
    )


class TestBayesianOptimizerController(unittest.TestCase):
    """Tests for the controller of the BayesianOptimizer"""

    def test_kappa_decay(self):
        """Check correct values for kappa_decay"""
        contr = BayesianOptimizerController(kappa=3, kappa_min=3, n_iter=10)
        self.assertAlmostEqual(contr.kappa_decay, 1.0)

        contr = BayesianOptimizerController(kappa=3, kappa_min=1, n_iter=10)
        self.assertAlmostEqual(contr.kappa * (contr.kappa_decay**10), 1.0)

    def test_from_input(self):
        """Check if input data is used correctly to set up controller"""
        inp = input()
        inp.bounds = {"a": (0, 1), "b": (1, 2)}

        contr = BayesianOptimizerController.from_input(inp, sampling_base=3, kappa=3)
        self.assertEqual(contr.kappa, 3)
        self.assertEqual(contr.init_points, 3**2)
        self.assertEqual(contr.n_iter, 3**2)


class TestBayesianOptimizer(unittest.TestCase):
    """Tests for the optimizer based on bayes_opt.BayesianOptimization"""

    def setUp(self):
        """Mock the input"""
        self.input = input()

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_kwargs_to_impact_func_creator(self, _):
        """Test transform of minimize func arguments to impact_func_gen arguments

        We test the method '_kwargs_to_impact_func_creator' through 'run' because it is
        private.
        """
        # Create stubs
        self.input.bounds = {"x_2": (0, 1), "x 1": (1, 2)}
        self.input.cost_func.return_value = 1.0
        self.optimizer = BayesianOptimizer(self.input)
        self.controller = BayesianOptimizerController(
            init_points=2, n_iter=1, max_iterations=1
        )

        # Call 'run'
        with patch.object(self.input, "impact_to_aligned_df") as align:
            align.return_value = (None, None)
            self.optimizer.run(self.controller)

        # Check call to '_kwargs_to_impact_func_gen'
        call_args = self.input.impact_func_creator.call_args_list
        self.assertEqual(len(call_args), 3)
        for args in call_args:
            self.assertSequenceEqual(args.kwargs.keys(), self.input.bounds.keys())

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_target_func(self, _):
        """Test if cost function is transformed correctly

        We test the method '_target_func' through 'run' because it is private
        """
        self.input.bounds = {"x_2": (0, 1), "x 1": (1, 2)}
        self.input.cost_func.side_effect = [1.0, -1.0]
        self.optimizer = BayesianOptimizer(self.input)
        self.controller = BayesianOptimizerController(
            init_points=1, n_iter=1, max_iterations=1
        )

        # Call 'run'
        with patch.object(self.input, "impact_to_aligned_df") as align:
            align.return_value = (None, None)
            output = self.optimizer.run(self.controller)

        # Check target space
        npt.assert_array_equal(output.p_space.target, [-1.0, 1.0])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimizer)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
