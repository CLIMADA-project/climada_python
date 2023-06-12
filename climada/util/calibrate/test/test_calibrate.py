"""Tests for calibration module"""

import unittest
from unittest.mock import create_autospec, patch, MagicMock
from typing import Optional, List

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.optimize import OptimizeResult

from climada.entity import Exposures, ImpactFuncSet
from climada.hazard import Hazard, Centroids

from climada.util.calibrate import Input, ScipyMinimizeOptimizer, BayesianOptimizer


def hazard():
    """Create a dummy hazard instance"""
    lat = [1, 2]
    lon = [0, 1]
    centroids = Centroids.from_lat_lon(lat=lat, lon=lon)
    event_id = np.array([1, 3, 10])
    intensity = csr_matrix([[1, 0.1], [2, 0.2], [3, 2]])
    return Hazard(event_id=event_id, centroids=centroids, intensity=intensity)


def exposure():
    """Create a dummy exposure instance"""
    return Exposures(
        data=dict(
            longitude=[0, 1, 100],
            latitude=[1, 2, 50],
            value=[1, 0.1, 1e6],
            impf_=[1, 1, 1],
        )
    )


class TestInputPostInit(unittest.TestCase):
    """Test the post_init dunder method of Input"""

    def setUp(self):
        """Create default input instance"""
        # Create the hazard instance
        self.hazard = hazard()

        # Create the exposure instance
        self.exposure = exposure()

        # Create some data
        self.data_events = [10, 3]
        self.data = pd.DataFrame(data={"a": [1, 2]}, index=self.data_events)

        # Create dummy funcs
        self.cost_func = lambda impact, data: 1.0
        self.impact_func_gen = lambda **kwargs: ImpactFuncSet()

    @patch("climada.util.calibrate.impact_func.np.setdiff1d")
    def test_post_init_calls(self, setdiff1d_mock):
        """Test if post_init calls stuff correctly using mocks"""
        # Create mocks
        hazard_mock_1 = create_autospec(Hazard())
        event_id = [10]
        hazard_mock_1.event_id = event_id
        hazard_mock_2 = create_autospec(Hazard())
        exposure_mock = create_autospec(Exposures())
        setdiff1d_mock.return_value = np.array([])

        # Make first hazard mock return another instance
        hazard_mock_1.select.return_value = hazard_mock_2

        # Create input
        input = Input(
            hazard=hazard_mock_1,
            exposure=exposure_mock,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_gen=self.impact_func_gen,
        )

        # Query checks
        npt.assert_array_equal(setdiff1d_mock.call_args.args[0], self.data_events)
        npt.assert_array_equal(setdiff1d_mock.call_args.args[1], event_id)
        hazard_mock_1.select.assert_called_once_with(event_id=self.data_events)
        self.assertNotEqual(input.hazard, hazard_mock_1)
        self.assertEqual(input.hazard, hazard_mock_2)
        exposure_mock.assign_centroids.assert_called_once_with(hazard_mock_2)

    def test_post_init(self):
        """Test if post_init results in a sensible hazard and exposure"""
        # Create input
        input = Input(
            hazard=self.hazard,
            exposure=self.exposure,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_gen=self.impact_func_gen,
        )

        # Check hazard and exposure
        npt.assert_array_equal(input.hazard.event_id, self.data.index)
        self.assertIn("centr_", input.exposure.gdf)
        npt.assert_array_equal(input.exposure.gdf["centr_"], [0, 1, -1])

    def test_non_matching_events(self):
        """Test if non-matching events result in errors"""
        data = pd.DataFrame(data={"a": [1, 2, 3]}, index=[9, 3, 12])
        input_kwargs = {
            "hazard": self.hazard,
            "exposure": self.exposure,
            "data": data,
            "cost_func": self.cost_func,
            "impact_func_gen": self.impact_func_gen,
            "align": False,
        }

        # No error without alignment
        Input(**input_kwargs)

        # Error with alignment
        input_kwargs.update(align=True)
        with self.assertRaises(RuntimeError) as cm:
            Input(**input_kwargs)

        self.assertIn(
            "Event IDs in 'data' do not match event IDs in 'hazard'", str(cm.exception)
        )
        self.assertIn("9", str(cm.exception))
        self.assertIn("12", str(cm.exception))
        self.assertNotIn("3", str(cm.exception))


class TestOptimizer(unittest.TestCase):
    """Base class for testing optimizers. Creates an input mock"""

    def setUp(self):
        """Mock the input"""
        self.input = MagicMock(
            spec_set=Input(
                hazard=create_autospec(Hazard, instance=True),
                exposure=create_autospec(Exposures, instance=True),
                data=create_autospec(pd.DataFrame, instance=True),
                cost_func=MagicMock(),
                impact_func_gen=MagicMock(),
                align=False,
            )
        )


class TestScipyMinimizeOptimizer(TestOptimizer):
    """Tests for the optimizer based on scipy.optimize.minimize"""

    def setUp(self):
        """Create the input and optimizer"""
        super().setUp()
        self.optimizer = ScipyMinimizeOptimizer(self.input)

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_kwargs_to_impact_func_gen(self, _):
        """Test transform of minimize func arguments to impact_func_gen arguments

        We test the method '_kwargs_to_impact_func_gen' through 'run' because it is
        private.
        """
        # Create stubs
        self.input.constraints = None
        self.input.bounds = None
        self.input.cost_func.return_value = 1.0

        # Call 'run', make sure that 'minimize' is only with these parameters
        params_init = {"x_2": 1, "x 1": 2, "x_3": 3}  # NOTE: Also works with whitespace
        self.optimizer.run(params_init=params_init, options={"maxiter": 1})

        # Check call to '_kwargs_to_impact_func_gen'
        self.input.impact_func_gen.assert_any_call(**params_init)

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


class TestBayesianOptimizer(TestOptimizer):
    """Tests for the optimizer based on bayes_opt.BayesianOptimization"""

    def setUp(self):
        """Create the input and optimizer"""
        super().setUp()

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_kwargs_to_impact_func_gen(self, _):
        """Test transform of minimize func arguments to impact_func_gen arguments

        We test the method '_kwargs_to_impact_func_gen' through 'run' because it is
        private.
        """
        # Create stubs
        self.input.bounds = {"x_2": (0, 1), "x 1": (1, 2)}
        self.input.cost_func.return_value = 1.0
        self.optimizer = BayesianOptimizer(self.input)

        # Call 'run'
        self.optimizer.run(init_points=2, n_iter=1)

        # Check call to '_kwargs_to_impact_func_gen'
        call_args = self.input.impact_func_gen.call_args_list
        self.assertEqual(len(call_args), 3)
        for args in call_args:
            self.assertSequenceEqual(args.kwargs.keys(), self.input.bounds.keys())


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInputPostInit)
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestScipyMinimizeOptimizer)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
