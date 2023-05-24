"""Module for calibrating impact functions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Callable, Mapping, Optional, Tuple, Union, Any, Dict, List
from numbers import Number

import numpy as np
import pandas as pd
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    minimize,
)
from bayes_opt import BayesianOptimization

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.engine import Impact, ImpactCalc


def cost_func_rmse(
    impact: Impact,
    data: pd.DataFrame,
    impact_proc: Callable[[Impact], pd.DataFrame] = lambda x: x.impact_at_reg(),
) -> Number:
    return np.sqrt(np.mean(((impact_proc(impact) - data) ** 2).to_numpy()))


# TODO: haz_type has to be set from outside!
def impf_step_generator(threshold: Number, paa: Number) -> ImpactFuncSet:
    return ImpactFuncSet(
        [
            ImpactFunc.from_step_impf(
                haz_type="RF", intensity=(0, threshold, 100), paa=(0, paa)
            )
        ]
    )


ConstraintType = Union[LinearConstraint, NonlinearConstraint, Mapping]


@dataclass
class Input:
    """Define the static input for a calibration task

    Attributes
    ----------
    hazard : climada.Hazard
        Hazard object to compute impacts from
    exposure : climada.Exposures
        Exposures object to compute impacts from
    data : pandas.Dataframe
        The data to compare computed impacts to. Index: Event IDs matching the IDs of
        ``hazard``. Columns: Arbitrary columns.
    cost_func : Callable
        Function that takes an ``Impact`` object and a ``pandas.Dataframe`` as argument
        and returns a single number. The optimization algorithm will try to minimize this
        number. See this module for a suggestion of cost functions.
    impact_func_gen : Callable
        Function that takes the parameters as keyword arguments and returns an impact
        function set. This will be called each time the optimization algorithm updates
        the parameters.
    bounds : Mapping (str, {Bounds, tuple(float, float)}), optional
        The bounds for the parameters. Keys: parameter names. Values:
        ``scipy.minimize.Bounds`` instance or tuple of minimum and maximum value.
        Unbounded parameters need not be specified here. See the documentation for
        the selected optimization algorithm on which data types are supported.
    constraints : Constraint or list of Constraint, optional
        One or multiple instances of ``scipy.minimize.LinearConstraint``,
        ``scipy.minimize.NonlinearConstraint``, or a mapping. See the documentation for
        the selected optimization algorithm on which data types are supported.
    impact_calc_kwds : Mapping (str, Any), optional
        Keyword arguments to :py:meth:`climada.engine.impact_calc.ImpactCalc.impact`.
        Defaults to ``{"assign_centroids": False}`` (by default, centroids are assigned
        here via the ``align`` parameter, to avoid assigning them each time the impact is
        calculated).
    align : bool, optional
        Match event IDs from ``hazard`` and ``data``, and assign the centroids from
        ``hazard`` to ``exposure``. Defaults to ``True``.
    """

    hazard: Hazard
    exposure: Exposures
    data: pd.DataFrame
    cost_func: Callable[[Impact, pd.DataFrame], Number]
    impact_func_gen: Callable[..., ImpactFuncSet]
    bounds: Optional[Mapping[str, Union[Bounds, Tuple[Number, Number]]]] = None
    constraints: Optional[Union[ConstraintType, list[ConstraintType]]] = None
    impact_calc_kwds: Mapping[str, Any] = field(
        default_factory=lambda: {"assign_centroids": False}
    )
    align: InitVar[bool] = True

    def __post_init__(self, align):
        """Prepare input data"""
        if align:
            event_diff = np.setdiff1d(self.data.index, self.hazard.event_id)
            if event_diff.size > 0:
                raise RuntimeError(
                    "Event IDs in 'data' do not match event IDs in 'hazard': \n"
                    f"{event_diff}"
                )
            self.hazard = self.hazard.select(event_id=self.data.index.tolist())
            self.exposure.assign_centroids(self.hazard)


@dataclass
class Output:
    """Generic output of a calibration task

    Attributes
    ----------
    params : Mapping (str, Number)
        The optimal parameters
    target : Number
        The target function value for the optimal parameters
    success : bool
        If the calibration succeeded. The definition depends on the actual optimization
        algorithm used.
    result
        A result object specific to the optimization algorithm used. See the optimizer
        documentation for details.
    """

    params: Mapping[str, Number]
    target: Number
    success: bool
    result: Optional[Any] = None


@dataclass
class Optimizer(ABC):
    """Abstract base class (interface) for an optimization

    This defines the interface for optimizers in CLIMADA. New optimizers can be created
    by deriving from this class and overriding at least the :py:meth:`run` method.

    Attributes
    ----------
    input : Input
        The input object for the optimization task. See :py:class:`Input`.
    """

    input: Input

    def _target_func(self, impact: Impact, data: pd.DataFrame) -> Number:
        """Target function for the optimizer

        The default version of this function simply returns the value of the cost
        function evaluated on the arguments.

        Paramters
        ---------
        impact : climada.engine.Impact
            The impact object returned by the impact calculation.
        data : pandas.DataFrame
            The data used for calibration. See :py:attr:`Input.data`.

        Returns
        -------
        The value of the target function for the optimizer.
        """
        return self.input.cost_func(impact, data)

    def _kwargs_to_impact_func_gen(self, *_, **kwargs) -> Dict[str, Any]:
        """Define how the parameters to 'opt_func' must be transformed

        Optimizers may implement different ways of representing the parameters (e.g.,
        key-value pairs, arrays, etc.). Depending on this representation, the parameters
        must be transformed to match the syntax of the impact function generator used,
        see :py:attr:`Input.impact_func_gen`.

        In this default version, the method simply returns its keyword arguments as
        mapping. Override this method if the optimizer used *does not* represent
        parameters as key-value pairs.

        Parameters
        ----------
        kwargs
            The parameters as key-value pairs.

        Returns
        -------
        The parameters as key-value pairs.
        """
        return kwargs

    def _opt_func(self, *args, **kwargs) -> Number:
        """The optimization function iterated by the optimizer

        This function takes arbitrary arguments from the optimizer, generates a new set
        of impact functions from it, computes the impact, and finally calculates the
        target function value and returns it.

        Parameters
        ----------
        args, kwargs
            Arbitrary arguments from the optimizer, including parameters

        Returns
        -------
        Target function value for the given arguments
        """
        params = self._kwargs_to_impact_func_gen(*args, **kwargs)
        impf_set = self.input.impact_func_gen(**params)
        impact = ImpactCalc(
            exposures=self.input.exposure,
            impfset=impf_set,
            hazard=self.input.hazard,
        ).impact(**self.input.impact_calc_kwds)
        return self._target_func(impact, self.input.data)

    @abstractmethod
    def run(self, **opt_kwargs) -> Output:
        """Execute the optimization"""


@dataclass
class ScipyMinimizeOptimizer(Optimizer):
    """An optimization using scipy.optimize.minimize

    By default, this optimizer uses the ``"trust-constr"`` method. This
    is advertised as the most general minimization method of the ``scipy`` pacjage and
    supports bounds and constraints on the parameters. Users are free to choose
    any method of the catalogue, but must be aware that they might require different
    input parameters. These can be supplied via additional keyword arguments to
    :py:meth:`run`.

    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    for details.

    Parameters
    ----------
    input : Input
        The input data for this optimizer. Supported data types for
        :py:attr:`constraint` might vary depending on the minimization method used.
    """

    def __post_init__(self):
        """Create a private attribute for storing the parameter names"""
        self._param_names: List[str] = list()

    def _kwargs_to_impact_func_gen(self, *args, **_) -> Dict[str, Any]:
        """Transform the array of parameters into key-value pairs"""
        return dict(zip(self._param_names, args[0].flat))

    def _select_by_param_names(self, mapping: Mapping[str, Any]) -> List[Any]:
        """Return a list of entries from a map with matching keys or ``None``"""
        return [mapping.get(key) for key in self._param_names]

    def run(self, **opt_kwargs) -> Output:
        """Execute the optimization

        Parameters
        ----------
        params_init : Mapping (str, Number)
            The initial guess for all parameters as key-value pairs.
        method : str, optional
            The minimization method applied. Defaults to ``"trust-constr"``.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for details.
        kwargs
            Additional keyword arguments passed to ``scipy.optimize.minimize``.

        Returns
        -------
        output : Output
            The output of the optimization. The :py:attr:`Output.result` attribute
            stores the associated ``scipy.optimize.OptimizeResult`` instance.
        """
        # Parse kwargs
        params_init = opt_kwargs.pop("params_init")
        method = opt_kwargs.pop("method", "trust-constr")

        # Store names to rebuild dict when the minimize iterator returns an array
        self._param_names = list(params_init.keys())

        # Transform bounds to match minimize input
        bounds = (
            self._select_by_param_names(self.input.bounds)
            if self.input.bounds is not None
            else None
        )

        x0 = np.array(list(params_init.values()))
        res = minimize(
            fun=self._opt_func,
            x0=x0,
            bounds=bounds,
            constraints=self.input.constraints,
            method=method,
            **opt_kwargs,
        )

        params = dict(zip(self._param_names, res.x.flat))
        return Output(params=params, target=res.fun, success=res.success, result=res)


@dataclass
class BayesianOptimizer(Optimizer):
    """An optimization using bayes_opt.BayesianOptimization

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
    The following requirements apply to the parameters of :py:class:`Input` when using
    this class:

    bounds
        Setting ``bounds`` in the ``Input`` is required because the optimizer first
        "explores" the bound parameter space and then narrows its search to regions
        where the cost function is low.
    constraints
        Must be an instance of ``scipy.minimize.LinearConstraint`` or
        ``scipy.minimize.NonlinearConstraint``. See
        https://github.com/bayesian-optimization/BayesianOptimization/blob/master/examples/constraints.ipynb
        for further information.

    Attributes
    ----------
    optimizer : bayes_opt.BayesianOptimization
        The optimizer instance of this class. Will be returned by :py:meth:`run` in
        :py:attr:`Output.result`.
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
            verbose=verbose,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
            **bayes_opt_kwds,
        )

    def _target_func(self, impact: Impact, data: pd.DataFrame) -> Number:
        """Invert the cost function because BayesianOptimization maximizes the target"""
        return 1 / self.input.cost_func(impact, data)

    def run(self, **opt_kwargs):
        """Execute the optimization

        Implementation detail: ``BayesianOptimization`` *maximizes* a target function.
        Therefore, this class inverts the cost function and used that as target
        function. The cost function is still minimized.

        Parameters
        ----------
        init_points : int, optional
            Number of initial samples taken from the parameter space. Defaults to 10^N,
            where N is the number of parameters.
        n_iter : int, optional
            Number of iteration steps after initial sampling. Defaults to 10^N, where N
            is the number of parameters.
        opt_kwargs
            Further keyword arguments passed to ``BayesianOptimization.maximize``.

        Returns
        -------
        output : Output
            Optimization output. :py:attr:`Output.result` will be the
            :py:attr:`optimizer` used by this class instance.
        """
        # Retrieve parameters
        num_params = len(self.input.bounds)
        init_points = opt_kwargs.pop("init_points", 10**num_params)
        n_iter = opt_kwargs.pop("n_iter", 10**num_params)

        # Run optimizer
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, **opt_kwargs)

        # Return output
        opt = self.optimizer.max
        return Output(
            params=opt["params"],
            target=opt["target"],
            success=True,
            result=self.optimizer,
        )
