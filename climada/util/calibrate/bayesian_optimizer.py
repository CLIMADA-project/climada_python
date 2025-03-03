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

import logging
from collections import deque, namedtuple
from dataclasses import InitVar, dataclass, field
from itertools import combinations, repeat
from numbers import Number
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.axes as maxes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, Events, ScreenLogger, UtilityFunction
from bayes_opt.target_space import TargetSpace

from .base import Input, Optimizer, Output, OutputEvaluator

LOGGER = logging.getLogger(__name__)


@dataclass
class _FakeConstraint:
    """Fake the behavior of the constrait for cycling the BayesianOutputOptimizer"""

    results: np.ndarray

    @property
    def lb(self):
        """Return the lower bound"""
        return np.array([0])

    def allowed(self, values):
        """Return if the values are allowed. This only mocks the true behavior"""
        if self.results.shape != values.shape:
            raise ValueError("Inserting wrong constraint values")
        return self.results


def select_best(
    p_space_df: pd.DataFrame,
    cost_limit: float,
    absolute: bool = True,
    cost_col=("Calibration", "Cost Function"),
) -> pd.DataFrame:
    """Select the best parameter space samples defined by a cost function limit

    The limit is a factor of the minimum value relative to itself (``absolute=True``) or
    to the range of cost function values (``absolute=False``). A ``cost_limit`` of 0.1
    will select all rows where the cost function is within

    - 110% of the minimum value if ``absolute=True``.
    - 10% of the range between minimum and maximum cost function value if
        ``absolute=False``.

    Parameters
    ----------
    p_space_df : pd.DataFrame
        The parameter space to select from.
    cost_limit : float
        The limit factor used for selection.
    absolute : bool, optional
        Whether the limit factor is applied to the minimum value (``True``) or the range
        of values (``False``). Defaults to ``True``.
    cost_col : Column specifier, optional
        The column indicating cost function values. Defaults to
        ``("Calibration", "Cost Function")``.

    Returns
    -------
    pd.DataFrame
        A subselection of the input data frame.
    """
    min_val = p_space_df[cost_col].min()
    cost_range = min_val if absolute else p_space_df[cost_col].max() - min_val
    max_val = min_val + cost_range * cost_limit
    return p_space_df.loc[p_space_df[cost_col] <= max_val]


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

    def to_hdf5(self, filepath: Union[Path, str], mode: str = "x"):
        """Write this output to an H5 file"""
        # Write base class information
        super().to_hdf5(filepath=filepath, mode=mode)

        # Write parameter space
        p_space_df = self.p_space_to_dataframe()
        p_space_df.to_hdf(filepath, mode="a", key="p_space")

    @classmethod
    def from_hdf5(cls, filepath: Union[Path, str]):
        """Read BayesianOptimizerOutput from an H5 file

        Warning
        -------
        This results in an object with broken :py:attr:`p_space` object. Do not further
        modify this parameter space. This function is only intended to load the
        parameter space again for analysis/plotting.
        """
        output = Output.from_hdf5(filepath)
        p_space_df = pd.read_hdf(filepath, mode="r", key="p_space")
        p_space_df["Calibration", "Target"] = -p_space_df[
            "Calibration", "Cost Function"
        ]

        # Reorganize data
        bounds = {param: (np.nan, np.nan) for param in p_space_df["Parameters"].columns}
        constraint = None
        if "Constraints Function" in p_space_df["Calibration"].columns:
            constraint = _FakeConstraint(
                p_space_df["Calibration", "Allowed"].to_numpy()
            )

        p_space = TargetSpace(
            target_func=lambda x: x,
            pbounds=bounds,
            constraint=constraint,
            allow_duplicate_points=True,
        )
        for _, row in p_space_df.iterrows():
            constraint_value = (
                None
                if constraint is None
                else row["Calibration", "Constraints Function"]
            )
            p_space.register(
                params=row["Parameters"].to_numpy(),
                target=row["Calibration", "Target"],
                constraint_value=constraint_value,
            )

        return cls(params=output.params, target=output.target, p_space=p_space)

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
            # Add zeros for scatter plot
            p_space_df["Parameters", "none"] = np.zeros_like(
                p_space_df["Parameters", params[0]]
            )
            return plot_single(x=params[0], y="none")

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
    init_points: int
    n_iter: int
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
        if self.init_points < 0 or self.n_iter < 0:
            raise ValueError("'init_points' and 'n_iter' must be 0 or positive")
        self.kappa_decay = self._calc_kappa_decay()

    def _calc_kappa_decay(self):
        """Compute the decay factor for :py:attr:`kappa`."""
        return np.exp((np.log(self.kappa_min) - np.log(self.kappa)) / self.n_iter)

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

    def optimizer_params(self) -> dict[str, Union[int, float, str, UtilityFunction]]:
        """Return parameters for the optimizer

        In the current implementation, these do not change.
        """
        return {
            "init_points": self.init_points,
            "n_iter": self.n_iter,
            "acquisition_function": UtilityFunction(
                kappa=self.kappa,
                kappa_decay=self.kappa_decay,
                **self.utility_func_kwargs,
            ),
        }

    def _is_random_step(self) -> bool:
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

    def update(self, event: str, instance: BayesianOptimization):
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
        """Return improvements as nicely formatted data

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
        Verbosity of the optimizer output. Defaults to 0. The output is *not* affected
        by the CLIMADA logging settings.
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

    verbose: int = 0
    random_state: InitVar[int] = 1
    allow_duplicate_points: InitVar[bool] = True
    bayes_opt_kwds: InitVar[Optional[Mapping[str, Any]]] = None

    def __post_init__(self, random_state, allow_duplicate_points, bayes_opt_kwds):
        """Create optimizer"""
        if bayes_opt_kwds is None:
            bayes_opt_kwds = {}

        if self.input.bounds is None:
            raise ValueError("Input.bounds is required for this optimizer")

        self.optimizer = BayesianOptimization(
            f=self._opt_func,
            pbounds=self.input.bounds,
            constraint=self.input.constraints,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
            **bayes_opt_kwds,
        )

    def _target_func(self, data: pd.DataFrame, predicted: pd.DataFrame) -> Number:
        """Invert the cost function because BayesianOptimization maximizes the target"""
        return -self.input.cost_func(data, predicted)

    def run(self, controller: BayesianOptimizerController) -> BayesianOptimizerOutput:
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
        # Register the controller
        for event in (Events.OPTIMIZATION_STEP, Events.OPTIMIZATION_END):
            self.optimizer.subscribe(event, controller)

        # Register the logger
        if self.verbose > 0:
            log = ScreenLogger(
                verbose=self.verbose, is_constrained=self.optimizer.is_constrained
            )
            for event in (
                Events.OPTIMIZATION_START,
                Events.OPTIMIZATION_STEP,
                Events.OPTIMIZATION_END,
            ):
                self.optimizer.subscribe(event, log)

        # Run the optimization
        while controller.iterations < controller.max_iterations:
            try:
                LOGGER.info(f"Optimization iteration: {controller.iterations}")
                self.optimizer.maximize(**controller.optimizer_params())
            except StopEarly:
                # Start a new iteration
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
        p_space_df: Optional[pd.DataFrame] = None,
        plot_haz: bool = True,
        plot_opt_kws: Optional[dict] = None,
        plot_impf_kws: Optional[dict] = None,
        plot_hist_kws: Optional[dict] = None,
        plot_axv_kws: Optional[dict] = None,
    ):
        """Plot impact function variability with parameter combinations of
        almost equal cost function values

        Args:
            p_space_df (pd.DataFrame, optional): Parameter space to plot functions from.
                If ``None``, this uses the space returned by
                :py:meth:`~BayesianOptimizerOutput.p_space_to_dataframe`. Use
                :py:func:`select_best` for a convenient subselection of parameters close
                to the optimum.
            plot_haz (bool, optional): Whether or not to plot hazard intensity
                distibution. Defaults to False.
            plot_opt_kws (dict, optional): Keyword arguments for optimal impact
                function plot. Defaults to None.
            plot_impf_kws (dict, optional): Keyword arguments for all impact
                function plots. Defaults to None.
            plot_hist_kws (dict, optional): Keyword arguments for hazard
                intensity histogram plot. Defaults to None.
            plot_axv_kws (dict, optional): Keyword arguments for hazard intensity range
                plot (axvspan).
        """

        # Initialize plot keyword arguments
        if plot_opt_kws is None:
            plot_opt_kws = {}
        if plot_impf_kws is None:
            plot_impf_kws = {}
        if plot_hist_kws is None:
            plot_hist_kws = {}
        if plot_axv_kws is None:
            plot_axv_kws = {}

        # Retrieve hazard type and parameter space
        haz_type = self.input.hazard.haz_type
        if p_space_df is None:
            p_space_df = self.output.p_space_to_dataframe()

        # Set plot defaults
        colors = mpl.colormaps["tab20"].colors
        lw = plot_opt_kws.pop("lw", 2)
        label_opt = plot_opt_kws.pop("label", "Optimal Function")
        color_opt = plot_opt_kws.pop("color", colors[0])
        zorder_opt = plot_opt_kws.pop("zorder", 4)

        label_impf = plot_impf_kws.pop("label", "All Functions")
        color_impf = plot_impf_kws.pop("color", colors[1])
        alpha_impf = plot_impf_kws.pop("alpha", 0.5)
        zorder_impf = plot_impf_kws.pop("zorder", 3)

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
                color=color_opt,
                lw=lw,
                zorder=zorder_opt,
                label=label_opt,
                **plot_opt_kws,
            )

            # Plot all impact functions within 'cost_func_diff' % of best estimate
            for idx, (_, row) in enumerate(p_space_df.iterrows()):
                label_temp = label_impf if idx == 0 else None

                temp_impf_set = self.input.impact_func_creator(**row["Parameters"])
                temp_impf = temp_impf_set.get_func(haz_type=haz_type)[impf_idx]

                ax.plot(
                    temp_impf.intensity,
                    temp_impf.mdd * temp_impf.paa * 100,
                    color=color_impf,
                    alpha=alpha_impf,
                    zorder=zorder_impf,
                    label=label_temp,
                )

            handles, _ = ax.get_legend_handles_labels()
            ax.set(
                xlabel=f"Intensity ({self.input.hazard.units})",
                ylabel="Mean Damage Ratio (MDR)",
                xlim=(min(best_impf.intensity), max(best_impf.intensity)),
            )
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

            # Plot hazard intensity value distributions
            if plot_haz:
                haz_vals = self.input.hazard.intensity[
                    :, self.input.exposure.gdf[f"centr_{haz_type}"]
                ].data

                # Plot defaults
                bins = plot_hist_kws.pop("bins", 40)
                label_hist = plot_hist_kws.pop("label", "Hazard Intensity")
                color_hist = plot_hist_kws.pop("color", colors[2])
                color_axv = plot_axv_kws.pop("color", colors[3])
                alpha_axv = plot_axv_kws.pop("alpha", 0.5)

                # Histogram plot
                ax2 = ax.twinx()
                ax.set_facecolor("none")
                ax.set_zorder(2)
                ax2.set_zorder(1)
                ax2.axvspan(
                    haz_vals.min(), haz_vals.max(), color=color_axv, alpha=alpha_axv
                )
                ax2.hist(
                    haz_vals,
                    bins=bins,
                    color=color_hist,
                    label=label_hist,
                    **plot_hist_kws,
                )
                ax2.set_ylabel("Exposure Points", color=color_hist)

                handles = handles + [
                    mpatches.Patch(color=color_hist, label=label_hist),
                    mpatches.Patch(color=color_axv, label=f"{label_hist} Range"),
                ]
                ax.yaxis.label.set_color(color_opt)
                ax.tick_params(axis="y", colors=color_opt)
                ax2.tick_params(axis="y", colors=color_hist)

            ax.legend(handles=handles)
            axes.append(ax)

        if n_impf > 1:
            return axes

        return ax
