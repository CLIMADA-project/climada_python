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
from typing import List, Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.optimize import OptimizeResult

from climada.util.calibrate import Input, ScipyMinimizeOptimizer

from .test_base import exposure, hazard


class TestScipyMinimizeOptimizer(unittest.TestCase):
    """Tests for the optimizer based on scipy.optimize.minimize"""

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

        self.optimizer = ScipyMinimizeOptimizer(self.input)

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_kwargs_to_impact_func_creator(self, _):
        """Test transform of minimize func arguments to impact_func_gen arguments

        We test the method '_kwargs_to_impact_func_creator' through 'run' because it is
        private.
        """
        # Create stubs
        self.input.constraints = None
        self.input.bounds = None
        self.input.cost_func.return_value = 1.0

        # Call 'run', make sure that 'minimize' is only with these parameters
        params_init = {"x_2": 1, "x 1": 2, "x_3": 3}  # NOTE: Also works with whitespace
        with patch.object(self.input, "impact_to_aligned_df") as align:
            align.return_value = (None, None)
            self.optimizer.run(params_init=params_init, options={"maxiter": 1})

        # Check call to '_kwargs_to_impact_func_creator'
        first_call = self.input.impact_func_creator.call_args_list[0]
        self.assertEqual(first_call, call(**params_init))

        # Check error on missing kwargs
        with self.assertRaisesRegex(
            RuntimeError, "ScipyMinimizeOptimizer.run requires 'params_init'"
        ):
            self.optimizer.run(options={"maxiter": 1})

    def test_output(self):
        """Check output reporting"""
        params_init = {"x_2": 1, "x 1": 2, "x_3": 3}
        target_func_value = 1.12
        self.input.constraints = None
        self.input.bounds = None

        # Mock the optimization function and call 'run'
        with patch.object(self.optimizer, "_opt_func") as opt_func_mock:
            opt_func_mock.return_value = target_func_value
            output = self.optimizer.run(params_init=params_init, options={"maxiter": 1})

        # Assert output
        self.assertListEqual(list(output.params.keys()), list(params_init.keys()))
        npt.assert_allclose(list(output.params.values()), list(params_init.values()))
        self.assertEqual(output.target, target_func_value)
        self.assertIsInstance(output.result, OptimizeResult)

        # NOTE: For scipy.optimize, this means no error
        self.assertTrue(output.result.success)

    @patch("climada.util.calibrate.scipy_optimizer.minimize", autospec=True)
    def test_bounds_select(self, minimize_mock):
        """Test the _select_by_param_names method

        We test the method '_select_by_param_names' through 'run' because it is private.
        """

        def assert_bounds_in_call(bounds: Optional[List]):
            """Check if scipy.optimize.minimize was called with the expected kwargs"""
            call_kwargs = minimize_mock.call_args.kwargs
            print(minimize_mock.call_args)

            if bounds is None:
                self.assertIsNone(call_kwargs["bounds"])
            else:
                self.assertListEqual(call_kwargs["bounds"], bounds)

        # Initialize params and mock return value
        params_init = {"x_2": 1, "x_1": 2, "x_3": 3}
        minimize_mock.return_value = OptimizeResult(
            x=np.array(list(params_init.values())), fun=0, success=True
        )

        # Set constraints and bounds to None (default)
        self.input.bounds = None

        # Call the optimizer (constraints and bounds are None)
        self.optimizer.run(params_init=params_init)
        self.assertListEqual(self.optimizer._param_names, list(params_init.keys()))
        minimize_mock.assert_called_once()
        assert_bounds_in_call(None)
        minimize_mock.reset_mock()

        # Set new bounds and constraints
        self.input.bounds = {"x_1": "a", "x_4": "b", "x_3": (1, 2)}
        self.input.constraints = {"x_5": [1], "x_2": 2}

        # Call the optimizer
        self.optimizer.run(params_init=params_init)
        self.assertListEqual(self.optimizer._param_names, list(params_init.keys()))
        minimize_mock.assert_called_once()
        assert_bounds_in_call(bounds=[None, "a", (1, 2)])


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestScipyMinimizeOptimizer)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
