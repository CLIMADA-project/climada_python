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
Calibration with Bayesian Optimization
"""

from dataclasses import dataclass, InitVar, field
from typing import Mapping, Optional, Any, Union, List, Tuple
from numbers import Number
from itertools import combinations, repeat
from collections import deque, namedtuple
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from bayes_opt import BayesianOptimization, Events, UtilityFunction
from bayes_opt.target_space import TargetSpace

from .base import Input, Output, Optimizer, OutputEvaluator


LOGGER = logging.getLogger(__name__)


@dataclass
class BayesianOptimizerOutput(Output):
    """Output of a calibration with :py:class:`BayesianOptimizer`

    Attributes
    ----------
    p_space : bayes_opt.target_space.TargetSpace
        The parameter space sampled by the optimizer.
    """

    p_space: TargetSpace

    def p_space_to_dataframe(self):
        """Return the sampled parameter space as pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Data frame whose columns are the parameter values and the associated cost
            function value (``Cost Function``) and whose rows are the optimizer
            iterations.
        """
        # Build MultiIndex for columns
        index = pd.MultiIndex.from_tuples(
            [("Parameters", p) for p in self.p_space.keys]
            + [("Calibration", "Cost Function")]
        )

        # Create DataFrame and fill
        data = pd.DataFrame(data=None, columns=index)
        for i in range(self.p_space.dim):
            data["Parameters", self.p_space.keys[i]] = self.p_space.params[..., i]
        data["Calibration", "Cost Function"] = -self.p_space.target

        # Constraints
        if self.p_space.constraint is not None:
            data["Calibration", "Constraints Function"] = self.p_space.constraint_values
            data["Calibration", "Allowed"] = self.p_space.constraint.allowed(
                self.p_space.constraint_values
            )

        # Rename index and return
        data.index.rename("Iteration", inplace=True)
        return data

    def plot_p_space(
        self,
        p_space_df: Optional[pd.DataFrame] = None,
        x: Optional[str] = None,
        y: Optional[str] = None,
        min_def: Optional[Union[str, Tuple[str, str]]] = "Cost Function",
        min_fmt: str = "x",
        min_color: str = "r",
        **plot_kwargs,
    ) -> Union[maxes.Axes, List[maxes.Axes]]:
        """Plot the parameter space as scatter plot(s)

        Produce a scatter plot where each point represents a parameter combination
        sampled by the optimizer. The coloring represents the cost function value.
        If there are more than two parameters in the input data frame, this method will
        produce one plot for each combination of two parameters.
        Explicit parameter names to plot can be given via the ``x`` and ``y`` arguments.
        If no data frame is provided as argument, the output of
        :py:meth:`p_space_to_dataframe` is used.

        Parameters
        ----------
        p_space_df : pd.DataFrame, optional
            The parameter space to plot. Defaults to the one returned by
            :py:meth:`p_space_to_dataframe`
        x : str, optional
            The parameter to plot on the x-axis. If ``y`` is *not* given, this will plot
            ``x`` against all other parameters.
        y : str, optional
            The parameter to plot on the y-axis. If ``x`` is *not* given, this will plot
            ``y`` against all other parameters.
        min_def : str, optional
            The name of the column in ``p_space_df`` defining which parameter set
            represents the minimum, which is plotted separately. Defaults to
            ``"Cost Function"``. Set to ``None`` to avoid plotting the minimum.
        min_fmt : str, optional
            Plot format string for plotting the minimum. Defaults to ``"x"``.
        min_color : str, optional
            Color for plotting the minimum. Defaults to ``"r"`` (red).
        """
        # pylint: disable=invalid-name

        if p_space_df is None:
            p_space_df = self.p_space_to_dataframe()

        if min_def is not None and not isinstance(min_def, tuple):
            min_def = ("Calibration", min_def)

        # Plot defaults
        cmap = plot_kwargs.pop("cmap", "viridis_r")
        s = plot_kwargs.pop("s", 40)
        c = ("Calibration", plot_kwargs.pop("c", "Cost Function"))

        def plot_single(x, y):
            """Plot a single combination of parameters"""
            x = ("Parameters", x)
            y = ("Parameters", y)

            # Plot scatter
            ax = p_space_df.plot(
                kind="scatter",
                x=x,
                y=y,
                c=c,
                s=s,
                cmap=cmap,
                **plot_kwargs,
            )

            # Plot the minimum
            if min_def is not None:
                best = p_space_df.loc[p_space_df.idxmin()[min_def]]
                ax.plot(best[x], best[y], min_fmt, color=min_color)

            return ax

        # Option 0: Only one parameter
        params = p_space_df["Parameters"].columns.to_list()
        if len(params) < 2:
            return plot_single(x=params[0], y=repeat(0))

        # Option 1: Only a single plot
        if x is not None and y is not None:
            return plot_single(x, y)

        # Option 2: Combination of all
        iterable = combinations(params, 2)
        # Option 3: Fix one and iterate over all others
        if x is not None:
            params.remove(x)
            iterable = zip(repeat(x), params)
        elif y is not None:
            params.remove(y)
            iterable = zip(params, repeat(y))

        # Iterate over parameter combinations
        return [plot_single(p_first, p_second) for p_first, p_second in iterable]


Improvement = namedtuple(
    "Improvement", ["iteration", "sample", "random", "target", "improvement"]
)


class StopEarly(Exception):
    """An exception for stopping an optimization iteration early"""

    pass


@dataclass(eq=False)
class BayesianOptimizerController(object):
    """A class for controlling the iterations of a :py:class:`BayesianOptimizer`.

    Each iteration in the optimizer consists of a random sampling of the parameter space
    with :py:attr:`init_points` steps, followed by a Gaussian process sampling with
    :py:attr:`n_iter` steps. During the latter, the :py:attr:`kappa` parameter is
    reduced to reach :py:attr:`kappa_min` at the end of the iteration. The iteration is
    stopped prematurely if improvements of the buest guess are below
    :py:attr:`min_improvement` for :py:attr:`min_improvement_count` consecutive times.
    At the beginning of the next iteration, :py:attr:`kappa` is reset to its original
    value.

    Optimization stops if :py:attr:`max_iterations` is reached or if an entire iteration
    saw now improvement.

    Attributes
    ----------
    init_points : int
        Number of randomly sampled points during each iteration.
    n_iter : int
        Maximum number of points using Gaussian process sampling during each iteration.
    min_improvement : float
        Minimal relative improvement. If improvements are below this value
        :py:attr:`min_improvement_count` times, the iteration is stopped.
    min_improvement_count : int
        Number of times the :py:attr:`min_improvement` must be undercut to stop the
        iteration.
    kappa : float
        Parameter controlling exploration of the upper-confidence-bound acquisition
        function of the sampling algorithm. Lower values mean less exploration of the
        parameter space and more exploitation of local information. This value is
        reduced throughout one iteration, reaching :py:attr:`kappa_min` at the
        last iteration step.
    kappa_min : float
        Minimal value of :py:attr:`kappa` after :py:attr:`n_iter` steps.
    max_iterations : int
        Maximum number of iterations before optimization is stopped, irrespective of
        convergence.
    utility_func_kwargs
        Further keyword arguments to the ``bayes_opt.UtilityFunction``.
    """

    # Init attributes
    init_points: int = 0
    n_iter: int = 0
    min_improvement: float = 1e-3
    min_improvement_count: int = 2
    kappa: float = 2.576
    kappa_min: float = 0.1
    max_iterations: int = 10
    utility_func_kwargs: dict[str, Union[int, float, str]] = field(default_factory=dict)

    # Other attributes
    kappa_decay: float = field(init=False, default=0.96)
    steps: int = field(init=False, default=0)
    iterations: int = field(init=False, default=0)
    _improvements: deque[Improvement] = field(init=False, default_factory=deque)
    _last_it_improved: int = 0
    _last_it_end: int = 0

    def __post_init__(self):
        """Set the decay factor for :py:attr:`kappa`."""
        self.kappa_decay = np.exp(
            (np.log(self.kappa_min) - np.log(self.kappa)) / self.n_iter
        )

    @classmethod
    def from_input(cls, inp: Input, sampling_base: float = 4, **kwargs):
        """Create a controller from a calibration input

        This uses the number of parameters to determine the appropriate values for
        :py:attr:`init_points` and :py:attr:`n_iter`. Both values are set to
        :math:`b^N`, where :math:`b` is the ``sampling_base`` parameter and :math:`N`
        is the number of estimated parameters.

        Parameters
        ----------
        inp : Input
            Input to the calibration
        sampling_base : float, optional
            Base for determining the sample size. Increase this for denser sampling.
            Defaults to 4.
        kwargs
            Keyword argument for the default constructor.
        """
        num_params = len(inp.bounds)
        init_points = round(sampling_base**num_params)
        n_iter = round(sampling_base**num_params)
        return cls(init_points=init_points, n_iter=n_iter, **kwargs)

    @property
    def _previous_max(self):
        """Return the maximum target value observed"""
        if not self._improvements:
            return -np.inf
        return self._improvements[-1].target

    def is_converged(self) -> bool:
        """Check if convergence criteria are met"""
        return True

    def optimizer_params(self) -> dict[str, Union[int, float, str, UtilityFunction]]:
        """Return parameters for the optimizer"""
        return {
            "init_points": self.init_points,
            "n_iter": self.n_iter,
            "acquisition_function": UtilityFunction(
                kappa=self.kappa,
                kappa_decay=self.kappa_decay,
                **self.utility_func_kwargs,
            ),
        }

    def _is_random_step(self):
        """Return true if we sample randomly instead of Bayesian"""
        return (self._last_it_end + self.steps) < self.init_points

    def _append_improvement(self, target):
        """Append a new improvement to the deque"""
        impr = np.inf
        if self._improvements:
            impr = (self._improvements[-1].target / target) - 1

        self._improvements.append(
            Improvement(
                sample=self.steps,
                iteration=self.iterations,
                target=target,
                improvement=impr,
                random=self._is_random_step(),
            )
        )

    def _is_new_max(self, instance):
        """Determine if a guessed value is the new maximum"""
        instance_max = instance.max
        if not instance_max or instance_max.get("target") is None:
            # During constrained optimization, there might not be a maximum
            # value since the optimizer might've not encountered any points
            # that fulfill the constraints.
            return False

        if instance_max["target"] > self._previous_max:
            return True

        return False

    def _maybe_stop_early(self, instance):
        """Throw if we want to stop this iteration early"""
        # Create sequence of last improvements
        last_improvements = [
            self._improvements[-idx]
            for idx in np.arange(
                min(self.min_improvement_count, len(self._improvements))
            )
            + 1
        ]
        if (
            # Same iteration
            np.unique([impr.iteration for impr in last_improvements]).size == 1
            # Not random
            and not any(impr.random for impr in last_improvements)
            # Less than min improvement
            and all(
                impr.improvement < self.min_improvement for impr in last_improvements
            )
        ):
            LOGGER.info("Minimal improvement. Stop iteration.")
            instance.dispatch(Events.OPTIMIZATION_END)
            raise StopEarly()

    def update(self, event, instance):
        """Update the step tracker of this instance.

        For step events, check if the latest guess is the new maximum. Also check if the
        iteration will be stopped early.

        For end events, check if any improvement occured. If not, stop the optimization.

        Parameters
        ----------
        event : bayes_opt.Events
            The event descriptor
        instance : bayes_opt.BayesianOptimization
            Optimization instance triggering the event

        Raises
        ------
        StopEarly
            If the optimization only achieves minimal improvement, stop the iteration
            early with this exception.
        StopIteration
            If an entire iteration did not achieve improvement, stop the optimization.
        """
        if event == Events.OPTIMIZATION_STEP:
            new_max = self._is_new_max(instance)
            if new_max:
                self._append_improvement(instance.max["target"])

            self.steps += 1

            # NOTE: Must call this after incrementing the step
            if new_max:
                self._maybe_stop_early(instance)

        if event == Events.OPTIMIZATION_END:
            self.iterations += 1
            # Stop if we do not improve anymore
            if (
                self._last_it_end > 0
                and self._last_it_improved == self._improvements[-1].iteration
            ):
                LOGGER.info("No improvement. Stop optimization.")
                raise StopIteration()

            self._last_it_improved = self._improvements[-1].iteration
            self._last_it_end = self.steps

    def improvements(self) -> pd.DataFrame:
        """Return improvements as nice data

        Returns
        -------
        improvements : pd.DataFrame
        """
        return pd.DataFrame.from_records(
            data=[impr._asdict() for impr in self._improvements]
        ).set_index("sample")


@dataclass
class BayesianOptimizer(Optimizer):
    """An optimization using ``bayes_opt.BayesianOptimization``

    This optimizer reports the target function value for each parameter set and
    *maximizes* that value. Therefore, a higher target function value is better.
    The cost function, however, is still minimized: The target function is defined as
    the inverse of the cost function.

    For details on the underlying optimizer, see
    https://github.com/bayesian-optimization/BayesianOptimization.

    Parameters
    ----------
    input : Input
        The input data for this optimizer. See the Notes below for input requirements.
    verbose : int, optional
        Verbosity of the optimizer output. Defaults to 1.
    random_state : int, optional
        Seed for initializing the random number generator. Defaults to 1.
    allow_duplicate_points : bool, optional
        Allow the optimizer to sample the same points in parameter space multiple times.
        This may happen if the parameter space is tightly bound or constrained. Defaults
        to ``True``.
    bayes_opt_kwds : dict
        Additional keyword arguments passed to the ``BayesianOptimization`` constructor.

    Notes
    -----
    The following requirements apply to the parameters of
    :py:class:`~climada.util.calibrate.base.Input` when using this class:

    bounds
        Setting :py:attr:`~climada.util.calibrate.base.Input.bounds` is required
        because the optimizer first "explores" the bound parameter space and then
        narrows its search to regions where the cost function is low.
    constraints
        Must be an instance of ``scipy.minimize.LinearConstraint`` or
        ``scipy.minimize.NonlinearConstraint``. See
        https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/constraints.ipynb
        for further information. Supplying contraints is optional.

    Attributes
    ----------
    optimizer : bayes_opt.BayesianOptimization
        The optimizer instance of this class.
    """

    verbose: InitVar[int] = 1
    random_state: InitVar[int] = 1
    allow_duplicate_points: InitVar[bool] = True
    bayes_opt_kwds: InitVar[Optional[Mapping[str, Any]]] = None

    def __post_init__(
        self, verbose, random_state, allow_duplicate_points, bayes_opt_kwds
    ):
        """Create optimizer"""
        if bayes_opt_kwds is None:
            bayes_opt_kwds = {}

        if self.input.bounds is None:
            raise ValueError("Input.bounds is required for this optimizer")

        self.optimizer = BayesianOptimization(
            f=self._opt_func,
            pbounds=self.input.bounds,
            constraint=self.input.constraints,
            verbose=verbose,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
            **bayes_opt_kwds,
        )
        self.optimizer.space._allow_duplicate_points = allow_duplicate_points

    def _target_func(self, data: pd.DataFrame, predicted: pd.DataFrame) -> Number:
        """Invert the cost function because BayesianOptimization maximizes the target"""
        return -self.input.cost_func(data, predicted)

    def run(
        self,
        controller: BayesianOptimizerController,
    ) -> BayesianOptimizerOutput:
        """Execute the optimization

        ``BayesianOptimization`` *maximizes* a target function. Therefore, this class
        inverts the cost function and used that as target function. The cost function is
        still minimized.

        Parameters
        ----------
        controller : BayesianOptimizerController
            The controller instance used to set the optimization iteration parameters.
        opt_kwargs
            Further keyword arguments passed to ``BayesianOptimization.maximize``.

        Returns
        -------
        output : BayesianOptimizerOutput
            Optimization output. :py:attr:`BayesianOptimizerOutput.p_space` stores data
            on the sampled parameter space.
        """
        for event in (Events.OPTIMIZATION_STEP, Events.OPTIMIZATION_END):
            self.optimizer.subscribe(event, controller)

        while controller.iterations < controller.max_iterations:
            try:
                LOGGER.info(f"Optimization iteration: {controller.iterations}")
                self.optimizer.maximize(**controller.optimizer_params())
            except StopEarly:
                continue
            except StopIteration:
                # Exit the loop
                break

        # Return output
        opt = self.optimizer.max
        return BayesianOptimizerOutput(
            params=opt["params"],
            target=opt["target"],
            p_space=self.optimizer.space,
        )


@dataclass
class BayesianOptimizerOutputEvaluator(OutputEvaluator):
    """Evaluate the output of :py:class:`BayesianOptimizer`.

    Parameters
    ----------
    input : Input
        The input object for the optimization task.
    output : BayesianOptimizerOutput
        The output object returned by the Bayesian optimization task.

    Raises
    ------
    TypeError
        If :py:attr:`output` is not of type :py:class:`BayesianOptimizerOutput`
    """

    output: BayesianOptimizerOutput

    def __post_init__(self):
        """Check output type and call base class post_init"""
        if not isinstance(self.output, BayesianOptimizerOutput):
            raise TypeError("'output' must be type BayesianOptimizerOutput")

        super().__post_init__()

    def plot_impf_variability(
        self,
        cost_func_diff: float = 0.1,
        p_space_df: Optional[pd.DataFrame] = None,
        plot_haz: bool = True,
        plot_impf_kws: Optional[dict] = None,
        plot_hist_kws: Optional[dict] = None,
    ):
        """Plot impact function variability with parameter combinations of
        almost equal cost function values

        Args:
            cost_func_diff (float, optional): Max deviation from optimal cost
                function value (as fraction). Defaults to 0.1 (i.e. 10%).
            p_space_df (pd.DataFrame, optional): parameter space. Defaults to None.
            plot_haz (bool, optional): Whether or not to plot hazard intensity
                distibution. Defaults to False.
            plot_impf_kws (dict, optional): Keyword arguments for impact
                function plot. Defaults to None.
            plot_hist_kws (dict, optional): Keyword arguments for hazard
                intensity distribution plot. Defaults to None.
        """

        # Initialize plot keyword arguments
        if plot_impf_kws is None:
            plot_impf_kws = {}
        if plot_hist_kws is None:
            plot_hist_kws = {}

        # Retrieve hazard type and parameter space
        haz_type = self.input.hazard.haz_type
        if p_space_df is None:
            p_space_df = self.output.p_space_to_dataframe()

        # Retrieve parameters of impact functions with cost function values
        # within 'cost_func_diff' % of the best estimate
        params_within_range = p_space_df["Parameters"]
        plot_space_label = "Parameter space"
        if cost_func_diff is not None:
            max_cost_func_val = p_space_df["Calibration", "Cost Function"].min() * (
                1 + cost_func_diff
            )
            params_within_range = params_within_range.loc[
                p_space_df["Calibration", "Cost Function"] <= max_cost_func_val
            ]
            plot_space_label = (
                f"within {int(cost_func_diff*100)} percent " f"of best fit"
            )

        # Set plot defaults
        color = plot_impf_kws.pop("color", "tab:blue")
        lw = plot_impf_kws.pop("lw", 2)
        zorder = plot_impf_kws.pop("zorder", 3)
        label = plot_impf_kws.pop("label", "best fit")

        # get number of impact functions and create a plot for each
        n_impf = len(self.impf_set.get_func(haz_type=haz_type))
        axes = []

        for impf_idx in range(n_impf):
            _, ax = plt.subplots()

            # Plot best-fit impact function
            best_impf = self.impf_set.get_func(haz_type=haz_type)[impf_idx]
            ax.plot(
                best_impf.intensity,
                best_impf.mdd * best_impf.paa * 100,
                color=color,
                lw=lw,
                zorder=zorder,
                label=label,
                **plot_impf_kws,
            )

            # Plot all impact functions within 'cost_func_diff' % of best estimate
            for row in range(params_within_range.shape[0]):
                label_temp = plot_space_label if row == 0 else None

                sel_params = params_within_range.iloc[row, :].to_dict()
                temp_impf_set = self.input.impact_func_creator(**sel_params)
                temp_impf = temp_impf_set.get_func(haz_type=haz_type)[impf_idx]

                ax.plot(
                    temp_impf.intensity,
                    temp_impf.mdd * temp_impf.paa * 100,
                    color="grey",
                    alpha=0.4,
                    label=label_temp,
                )

            # Plot hazard intensity value distributions
            if plot_haz:
                haz_vals = self.input.hazard.intensity[
                    :, self.input.exposure.gdf[f"centr_{haz_type}"]
                ]

                # Plot defaults
                color_hist = plot_hist_kws.pop("color", "tab:orange")
                alpha_hist = plot_hist_kws.pop("alpha", 0.3)

                ax2 = ax.twinx()
                ax2.hist(
                    haz_vals.data,
                    bins=40,
                    color=color_hist,
                    alpha=alpha_hist,
                    label="Hazard intensity\noccurence",
                )
                ax2.set(ylabel="Hazard intensity occurence (#Exposure points)")
                ax.axvline(
                    x=haz_vals.max(), label="Maximum hazard value", color="tab:orange"
                )
                ax2.legend(loc="lower right")

            ax.set(
                xlabel=f"Intensity ({self.input.hazard.units})",
                ylabel="Mean Damage Ratio (MDR) in %",
                xlim=(min(best_impf.intensity), max(best_impf.intensity)),
            )
            ax.legend()
            axes.append(ax)

        if n_impf > 1:
            return axes

        return ax
