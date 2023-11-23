"""Tests for calibration module"""

import unittest
from unittest.mock import patch, MagicMock

import numpy.testing as npt
import pandas as pd

from climada.util.calibrate import Input, BayesianOptimizer

from .test_base import hazard, exposure

class TestBayesianOptimizer(unittest.TestCase):
    """Tests for the optimizer based on bayes_opt.BayesianOptimization"""

    def setUp(self):
        """Mock the input"""
        self.input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(data={"col1": [1, 2], "col2": [2, 3]}, index=[0, 1]),
            cost_func=MagicMock(),
            impact_func_creator=MagicMock(),
            impact_to_dataframe=MagicMock(),
        )

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

        # Call 'run'
        with patch.object(self.input, "impact_to_aligned_df") as align:
            align.return_value = (None, None)
            self.optimizer.run(init_points=2, n_iter=1)

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

        # Call 'run'
        with patch.object(self.input, "impact_to_aligned_df") as align:
            align.return_value = (None, None)
            output = self.optimizer.run(init_points=1, n_iter=1)

        # Check target space
        npt.assert_array_equal(output.p_space.target, [-1.0, 1.0])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimizer)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
