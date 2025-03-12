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
Integration tests for calibration module
"""

import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd
from matplotlib.axes import Axes
from scipy.optimize import NonlinearConstraint
from sklearn.metrics import mean_squared_error

from climada.entity import ImpactFunc, ImpactFuncSet
from climada.util.calibrate import (
    BayesianOptimizer,
    BayesianOptimizerController,
    BayesianOptimizerOutputEvaluator,
    Input,
    OutputEvaluator,
    ScipyMinimizeOptimizer,
)
from climada.util.calibrate.test.test_base import exposure, hazard


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
                    haz_type="TEST",
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
                    haz_type="TEST",
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
                    haz_type="TEST",
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
        controller = BayesianOptimizerController(
            init_points=10, n_iter=20, max_iterations=1
        )
        optimizer = BayesianOptimizer(self.input, random_state=1)
        output = optimizer.run(controller)

        # Check result (low accuracy)
        self.assertAlmostEqual(output.params["slope"], 1.0, places=2)
        self.assertAlmostEqual(output.target, 0.0, places=3)
        self.assertEqual(output.p_space.dim, 1)
        self.assertTupleEqual(output.p_space_to_dataframe().shape, (30, 2))
        self.assertEqual(controller.iterations, 1)

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
                    haz_type="TEST",
                )
            ]
        )

        # Constraint: param[0] < param[1] (intensity_1 < intensity_2)
        self.input.constraints = NonlinearConstraint(
            lambda intensity_1, intensity_2: intensity_1 - intensity_2, -np.inf, 0.0
        )
        self.input.bounds = {"intensity_1": (-1, 4), "intensity_2": (-1, 4)}
        # Run optimizer
        optimizer = BayesianOptimizer(self.input, random_state=1)
        controller = BayesianOptimizerController.from_input(
            self.input, sampling_base=5, max_iterations=3
        )
        output = optimizer.run(controller)

        # Check results (low accuracy)
        self.assertEqual(output.p_space.dim, 2)
        self.assertAlmostEqual(output.params["intensity_1"], 1.0, places=2)
        self.assertAlmostEqual(output.params["intensity_2"], 3.0, places=1)
        self.assertAlmostEqual(output.target, 0.0, places=3)
        self.assertGreater(controller.iterations, 1)

        # Check constraints in parameter space
        p_space = output.p_space_to_dataframe()
        self.assertSetEqual(
            set(p_space.columns.to_list()),
            {
                ("Parameters", "intensity_1"),
                ("Parameters", "intensity_2"),
                ("Calibration", "Cost Function"),
                ("Calibration", "Constraints Function"),
                ("Calibration", "Allowed"),
            },
        )
        self.assertGreater(p_space.shape[0], 50)  # Two times random iterations
        self.assertEqual(p_space.shape[1], 5)
        p_allowed = p_space.loc[p_space["Calibration", "Allowed"], "Parameters"]
        npt.assert_array_equal(
            (p_allowed["intensity_1"] < p_allowed["intensity_2"]).to_numpy(),
            np.full_like(p_allowed["intensity_1"].to_numpy(), True),
        )

    def test_plots(self):
        """Check if executing the default plots works"""
        self.input.bounds = {"slope": (-1, 3)}
        optimizer = BayesianOptimizer(self.input, random_state=1)
        controller = BayesianOptimizerController.from_input(
            self.input, max_iterations=1
        )
        output = optimizer.run(controller)

        output_eval = OutputEvaluator(self.input, output)
        output_eval.impf_set.plot()
        output_eval.plot_at_event()
        output_eval.plot_at_region()
        output_eval.plot_event_region_heatmap()

        output_eval = BayesianOptimizerOutputEvaluator(self.input, output)
        ax = output_eval.plot_impf_variability()
        self.assertIsInstance(ax, Axes)
