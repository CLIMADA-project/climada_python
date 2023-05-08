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

    Parameters
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
    """Define the output of a calibration task"""

    params: Mapping[str, Number]
    target: Number
    success: bool
    result: Optional[Any] = None


@dataclass
class Optimizer(ABC):
    """Define the basic interface for an optimization"""

    input: Input

    def _target_func(self, impact: Impact, data: pd.DataFrame):
        return self.input.cost_func(impact, data)

    def _kwargs_to_impact_func_gen(self, *_, **kwargs) -> Dict[str, Any]:
        """Define how the parameters to 'opt_func' must be transformed"""
        return kwargs

    def _opt_func(self, *args, **kwargs):
        """The optimization function that is iterated"""
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
        pass


@dataclass
class ScipyMinimizeOptimizer(Optimizer):
    """An optimization using scipy.optimize.minimize"""

    def __post_init__(self):
        """Create a private attribute for storing the parameter names"""
        self._param_names: List[str] = list()

    def _kwargs_to_impact_func_gen(self, *args, **_) -> Dict[str, Any]:
        return dict(zip(self._param_names, args[0].flat))

    def _select_by_param_names(self, mapping: Mapping[str, Any]) -> List[Any]:
        """Return a list of entries from a map with matching keys or ``None``"""
        return [mapping.get(key) for key in self._param_names]

    def run(self, **opt_kwargs):
        """Execute the optimization"""
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
    """An optimization using bayes_opt.BayesianOptimization"""

    verbose: InitVar[int] = 1
    random_state: InitVar[int] = 1
    allow_duplicate_points: InitVar[bool] = True
    bayes_opt_kwds: InitVar[Optional[Mapping[str, Any]]] = None

    def __post_init__(
        self, verbose, random_state, allow_duplicate_points, bayes_opt_kwds
    ):
        """Create optimizer"""
        self.optimizer = BayesianOptimization(
            f=lambda **kwargs: self._opt_func(**kwargs),
            pbounds=self.input.bounds,
            verbose=verbose,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
            **bayes_opt_kwds,
        )

    def run(self, init_points: int = 100, n_iter: int = 200, **opt_kwargs):
        """Execute the optimization"""
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, **opt_kwargs)
        opt = self.optimizer.max
        return Output(
            params=opt["params"],
            target=opt["target"],
            success=True,
            result=self.optimizer,
        )
