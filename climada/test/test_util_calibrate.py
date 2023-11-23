"""Integration tests for calibration utility module"""

import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt
from scipy.optimize import NonlinearConstraint
from sklearn.metrics import mean_squared_error

from climada.entity import ImpactFuncSet, ImpactFunc

from climada.util.calibrate import Input, ScipyMinimizeOptimizer, BayesianOptimizer

from climada.util.calibrate.test.test_base import hazard, exposure


class TestScipyMinimizeOptimizer(unittest.TestCase):
    """Test the TestScipyMinimizeOptimizer"""

    def setUp(self) -> None:
        """Prepare input for optimization"""
        self.hazard = hazard()
        self.hazard.frequency = np.ones_like(self.hazard.event_id)
        self.hazard.date = self.hazard.frequency
        self.hazard.event_name = ["event"] * len(self.hazard.event_id)
        self.exposure = exposure()
        self.events = [10, 1]
        self.hazard = self.hazard.select(event_id=self.events)
        self.data = pd.DataFrame(
            data={"a": [3, 1], "b": [0.2, 0.01]}, index=self.events
        )
        self.impact_to_dataframe = lambda impact: impact.impact_at_reg(["a", "b"])
        self.impact_func_creator = lambda slope: ImpactFuncSet(
            [
                ImpactFunc(
                    intensity=np.array([0, 10]),
                    mdd=np.array([0, 10 * slope]),
                    paa=np.ones(2),
                    id=1,
                )
            ]
        )
        self.input = Input(
            self.hazard,
            self.exposure,
            self.data,
            self.impact_func_creator,
            self.impact_to_dataframe,
            mean_squared_error,
        )

    def test_single(self):
        """Test with single parameter"""
        optimizer = ScipyMinimizeOptimizer(self.input)
        output = optimizer.run(params_init={"slope": 0.1})

        # Result should be nearly exact
        self.assertTrue(output.result.success)
        self.assertAlmostEqual(output.params["slope"], 1.0)
        self.assertAlmostEqual(output.target, 0.0)

    def test_bound(self):
        """Test with single bound"""
        self.input.bounds = {"slope": (-1.0, 0.91)}
        optimizer = ScipyMinimizeOptimizer(self.input)
        output = optimizer.run(params_init={"slope": 0.1})

        # Result should be very close to the bound
        self.assertTrue(output.result.success)
        self.assertGreater(output.params["slope"], 0.89)
        self.assertAlmostEqual(output.params["slope"], 0.91, places=2)

    def test_multiple_constrained(self):
        """Test with multiple constrained parameters"""
        # Set new generator
        self.input.impact_func_creator = lambda intensity_1, intensity_2: ImpactFuncSet(
            [
                ImpactFunc(
                    intensity=np.array([0, intensity_1, intensity_2]),
                    mdd=np.array([0, 1, 3]),
                    paa=np.ones(3),
                    id=1,
                )
            ]
        )

        # Constraint: param[0] < param[1] (intensity_1 < intensity_2)
        self.input.constraints = NonlinearConstraint(
            lambda params: params[0] - params[1], -np.inf, 0.0
        )
        self.input.bounds = {"intensity_1": (0, 3.1), "intensity_2": (0, 3.1)}

        # Run optimizer
        optimizer = ScipyMinimizeOptimizer(self.input)
        output = optimizer.run(
            params_init={"intensity_1": 2, "intensity_2": 2},
            options=dict(gtol=1e-5, xtol=1e-5),
        )

        # Check results (low accuracy)
        self.assertTrue(output.result.success)
        print(output.result.message)
        print(output.result.status)
        self.assertAlmostEqual(output.params["intensity_1"], 1.0, places=2)
        self.assertGreater(output.params["intensity_2"], 2.8)  # Should be 3.0
        self.assertAlmostEqual(output.target, 0.0, places=3)


class TestBayesianOptimizer(unittest.TestCase):
    """Integration tests for the BayesianOptimizer"""

    def setUp(self) -> None:
        """Prepare input for optimization"""
        self.hazard = hazard()
        self.hazard.frequency = np.ones_like(self.hazard.event_id)
        self.hazard.date = self.hazard.frequency
        self.hazard.event_name = ["event"] * len(self.hazard.event_id)
        self.exposure = exposure()
        self.events = [10, 1]
        self.hazard = self.hazard.select(event_id=self.events)
        self.data = pd.DataFrame(
            data={"a": [3, 1], "b": [0.2, 0.01]}, index=self.events
        )
        self.impact_to_dataframe = lambda impact: impact.impact_at_reg(["a", "b"])
        self.impact_func_creator = lambda slope: ImpactFuncSet(
            [
                ImpactFunc(
                    intensity=np.array([0, 10]),
                    mdd=np.array([0, 10 * slope]),
                    paa=np.ones(2),
                    id=1,
                )
            ]
        )
        self.input = Input(
            self.hazard,
            self.exposure,
            self.data,
            self.impact_func_creator,
            self.impact_to_dataframe,
            mean_squared_error,
        )

    def test_single(self):
        """Test with single parameter"""
        self.input.bounds = {"slope": (-1, 3)}
        optimizer = BayesianOptimizer(self.input)
        output = optimizer.run(init_points=10, n_iter=20, random_state=1)

        # Check result (low accuracy)
        self.assertAlmostEqual(output.params["slope"], 1.0, places=2)
        self.assertAlmostEqual(output.target, 0.0, places=3)
        self.assertEqual(output.p_space.dim, 1)
        self.assertTupleEqual(output.p_space_to_dataframe().shape, (30, 2))

    def test_multiple_constrained(self):
        """Test with multiple constrained parameters"""
        # Set new generator
        self.input.impact_func_creator = lambda intensity_1, intensity_2: ImpactFuncSet(
            [
                ImpactFunc(
                    intensity=np.array([0, intensity_1, intensity_2]),
                    mdd=np.array([0, 1, 3]),
                    paa=np.ones(3),
                    id=1,
                )
            ]
        )

        # Constraint: param[0] < param[1] (intensity_1 < intensity_2)
        self.input.constraints = NonlinearConstraint(
            lambda params: params[0] - params[1], -np.inf, 0.0
        )
        self.input.bounds = {"intensity_1": (-1, 4), "intensity_2": (-1, 4)}
        # Run optimizer
        optimizer = BayesianOptimizer(self.input)
        output = optimizer.run(n_iter=200, random_state=1)

        # Check results (low accuracy)
        self.assertEqual(output.p_space.dim, 2)
        self.assertAlmostEqual(output.params["intensity_1"], 1.0, places=2)
        self.assertAlmostEqual(output.params["intensity_2"], 3.0, places=1)
        self.assertAlmostEqual(output.target, 0.0, places=3)

        # Check constraints in parameter space
        p_space = output.p_space_to_dataframe()
        self.assertSetEqual(
            set(p_space.columns.to_list()),
            {"intensity_1", "intensity_2", "Cost Function"},
        )
        self.assertTupleEqual(p_space.shape, (300, 3))
