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
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from bayes_opt import BayesianOptimization, Events
from matplotlib.axes import Axes
from scipy.optimize import NonlinearConstraint

from climada.util.calibrate import BayesianOptimizer, BayesianOptimizerController, Input
from climada.util.calibrate.bayesian_optimizer import (
    BayesianOptimizerOutput,
    Improvement,
    StopEarly,
)

from .test_base import exposure, hazard


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

    def setUp(self):
        """Create a optimization instance"""
        self.bayes_opt = BayesianOptimization(
            f=lambda x: -(x**2),
            pbounds={"x": (-10, 10)},
            constraint=NonlinearConstraint(fun=lambda x: x, lb=-0.5, ub=np.inf),
            verbose=0,
            allow_duplicate_points=True,
        )

    def _make_step(self, x, controller):
        self.bayes_opt.probe({"x": x}, lazy=False)
        controller.update(Events.OPTIMIZATION_STEP, self.bayes_opt)

    def test_kappa_decay(self):
        """Check correct values for kappa_decay"""
        contr = BayesianOptimizerController(0, kappa=3, kappa_min=3, n_iter=10)
        self.assertAlmostEqual(contr.kappa_decay, 1.0)

        contr = BayesianOptimizerController(0, kappa=3, kappa_min=1, n_iter=10)
        self.assertAlmostEqual(contr.kappa * (contr.kappa_decay**10), 1.0)

    def test_from_input(self):
        """Check if input data is used correctly to set up controller"""
        inp = input()
        inp.bounds = {"a": (0, 1), "b": (1, 2)}

        contr = BayesianOptimizerController.from_input(inp, sampling_base=3, kappa=3)
        self.assertEqual(contr.kappa, 3)
        self.assertEqual(contr.init_points, 3**2)
        self.assertEqual(contr.n_iter, 3**2)

    def test_optimizer_params(self):
        """Test BayesianOptimizerController.optimizer_params"""
        contr = BayesianOptimizerController(
            1, 2, kappa=3, utility_func_kwargs={"xi": 1.11, "kind": "ei"}
        )
        result = contr.optimizer_params()

        self.assertEqual(result.get("init_points"), 1)
        self.assertEqual(result.get("n_iter"), 2)

        util_func = result["acquisition_function"]
        self.assertEqual(util_func.kappa, 3)
        self.assertEqual(util_func._kappa_decay, contr._calc_kappa_decay())
        self.assertEqual(util_func.xi, 1.11)
        self.assertEqual(util_func.kind, "ei")

    def test_update_step(self):
        """Test the update for STEP events"""
        contr = BayesianOptimizerController(3, 2)

        # Regular step
        self._make_step(3.0, contr)
        self.assertEqual(contr.steps, 1)
        best = Improvement(
            iteration=0, sample=0, random=True, target=-9.0, improvement=np.inf
        )
        self.assertEqual(len(contr._improvements), 1)
        self.assertTupleEqual(contr._improvements[-1], best)

        # Step that has no effect due to constraints
        self._make_step(-2.0, contr)
        self.assertEqual(contr.steps, 2)
        self.assertEqual(len(contr._improvements), 1)
        self.assertTupleEqual(contr._improvements[-1], best)

        # Step that is not new max
        self._make_step(4.0, contr)
        self.assertEqual(contr.steps, 3)
        self.assertEqual(len(contr._improvements), 1)
        self.assertTupleEqual(contr._improvements[-1], best)

        # Two minimal increments, therefore we should see a StopEarly
        self._make_step(2.999, contr)
        self.assertEqual(contr.steps, 4)
        self.assertEqual(len(contr._improvements), 2)

        with self.assertRaises(StopEarly):
            self._make_step(2.998, contr)
        self.assertEqual(contr.steps, 5)
        self.assertEqual(len(contr._improvements), 3)

    def test_update_end(self):
        """Test the update for END events"""
        contr = BayesianOptimizerController(1, 1)

        # One step with improvement, then stop
        self._make_step(3.0, contr)
        contr.update(Events.OPTIMIZATION_END, self.bayes_opt)
        self.assertEqual(contr._last_it_improved, 0)
        self.assertEqual(contr._last_it_end, 1)

        # One step with no more improvement
        self._make_step(4.0, contr)
        with self.assertRaises(StopIteration):
            contr.update(Events.OPTIMIZATION_END, self.bayes_opt)

    def test_improvements(self):
        """Test ouput of BayesianOptimizerController.improvements"""
        contr = BayesianOptimizerController(1, 1)
        self._make_step(3.0, contr)
        self._make_step(2.0, contr)
        contr.update(Events.OPTIMIZATION_END, self.bayes_opt)
        self._make_step(2.5, contr)  # Not better
        self._make_step(1.0, contr)
        contr.update(Events.OPTIMIZATION_END, self.bayes_opt)
        self._make_step(-0.9, contr)  # Constrained

        df = contr.improvements()
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame.from_dict(
                data={
                    "iteration": [0, 0, 1],
                    "sample": [0, 1, 3],
                    "random": [True, False, False],
                    "target": [-9.0, -4.0, -1.0],
                    "improvement": [np.inf, 9.0 / 4.0 - 1, 3.0],
                }
            ).set_index("sample"),
        )


class TestBayesianOptimizerOutput(unittest.TestCase):
    """Tests for the output class of BayesianOptimizer"""

    def setUp(self):
        """Create a default output"""
        bayes_opt = BayesianOptimization(
            f=lambda x: -(x**2),
            pbounds={"x": (-10, 10)},
            constraint=NonlinearConstraint(fun=lambda x: x, lb=-0.5, ub=np.inf),
            verbose=0,
            allow_duplicate_points=True,
        )
        bayes_opt.probe({"x": 2.0}, lazy=False)
        bayes_opt.probe({"x": 1.0}, lazy=False)
        bayes_opt.probe({"x": -0.9}, lazy=False)

        self.output = BayesianOptimizerOutput(
            params=bayes_opt.max["params"],
            target=bayes_opt.max["target"],
            p_space=bayes_opt.space,
        )

    def test_p_space_to_dataframe(self):
        """"""
        self.assertDictEqual(self.output.params, {"x": 1.0})
        self.assertEqual(self.output.target, -1.0)

        idx = pd.MultiIndex.from_tuples(
            [
                ("Parameters", "x"),
                ("Calibration", "Cost Function"),
                ("Calibration", "Constraints Function"),
                ("Calibration", "Allowed"),
            ]
        )
        df = pd.DataFrame(data=None, columns=idx)
        df["Parameters", "x"] = [2.0, 1.0, -0.9]
        df["Calibration", "Cost Function"] = [4.0, 1.0, 0.9**2]
        df["Calibration", "Constraints Function"] = df["Parameters", "x"]
        df["Calibration", "Allowed"] = [True, True, False]
        df.index.rename("Iteration", inplace=True)
        pd.testing.assert_frame_equal(self.output.p_space_to_dataframe(), df)

    def test_cycle(self):
        """Check if the output can be cycled to produce the same p_space_df"""
        with TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir, "file.h5")
            self.output.to_hdf5(outpath)
            self.assertTrue(outpath.is_file())

            output = BayesianOptimizerOutput.from_hdf5(outpath)
        pd.testing.assert_frame_equal(
            self.output.p_space_to_dataframe(), output.p_space_to_dataframe()
        )

    def test_plot_p_space(self):
        """Test plotting of different parameter combinations"""
        # Mock data
        mock_df = pd.DataFrame(
            {
                ("Parameters", "param1"): [1, 2, 3],
                ("Parameters", "param2"): [4, 5, 6],
                ("Parameters", "param3"): [7, 8, 9],
                ("Calibration", "Cost Function"): [10, 15, 20],
            }
        )

        # Create instance of BayesianOptimizerOutput
        output = BayesianOptimizerOutput(params=None, target=None, p_space=None)

        # Mock the p_space_to_dataframe method
        with patch.object(
            BayesianOptimizerOutput, "p_space_to_dataframe", return_value=mock_df
        ):
            # Plot all
            axes = output.plot_p_space()
            self.assertEqual(len(axes), 3)
            for ax in axes:
                self.assertIsInstance(ax, Axes)
                self.assertTrue(ax.has_data())

            # # Keep x fixed
            axes = output.plot_p_space(x="param2")
            self.assertEqual(len(axes), 2)
            for ax in axes:
                self.assertEqual(ax.get_xlabel(), "(Parameters, param2)")

            # # Keep y fixed
            axes = output.plot_p_space(y="param1")
            self.assertEqual(len(axes), 2)
            for ax in axes:
                self.assertEqual(ax.get_ylabel(), "(Parameters, param1)")

            # # Plot single combination
            ax = output.plot_p_space(x="param2", y="param1")
            self.assertIsInstance(ax, Axes)
            self.assertEqual(ax.get_xlabel(), "(Parameters, param2)")
            self.assertEqual(ax.get_ylabel(), "(Parameters, param1)")

        # Plot single parameter
        ax = output.plot_p_space(
            pd.DataFrame(
                {
                    ("Parameters", "param1"): [1, 2, 3],
                    ("Calibration", "Cost Function"): [10, 15, 20],
                }
            )
        )
        self.assertIsInstance(ax, Axes)
        self.assertEqual(ax.get_xlabel(), "(Parameters, param1)")
        self.assertEqual(ax.get_ylabel(), "(Parameters, none)")


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
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimizerOutput)
    )
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestBayesianOptimizerController)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
