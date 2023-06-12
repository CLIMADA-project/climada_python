"""Integration tests for calibration utility module"""

import unittest
from functools import partial

import pandas as pd
import numpy as np
from scipy.optimize import NonlinearConstraint

from climada.entity import ImpactFuncSet, ImpactFunc

from climada.util.calibrate import Input, ScipyMinimizeOptimizer
from climada.util.calibrate.impact_func import cost_func_rmse

from climada.util.calibrate.test.test_calibrate import hazard, exposure


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
        self.data = pd.DataFrame(
            data={"a": [3, 1], "b": [0.2, 0.01]}, index=self.events
        )
        self.cost_func = partial(
            cost_func_rmse, impact_proc=lambda impact: impact.impact_at_reg(["a", "b"])
        )
        self.impact_func_gen = lambda slope: ImpactFuncSet(
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
            self.hazard, self.exposure, self.data, self.cost_func, self.impact_func_gen
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
        self.input.impact_func_gen = lambda intensity_1, intensity_2: ImpactFuncSet(
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
        )

        # Check results (low accuracy)
        self.assertTrue(output.result.success)
        self.assertAlmostEqual(output.params["intensity_1"], 1.0, places=3)
        self.assertAlmostEqual(output.params["intensity_2"], 3.0, places=3)
        self.assertAlmostEqual(output.target, 0.0, places=3)
