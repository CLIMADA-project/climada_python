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


def cost_func_rmse(impact: Impact, data: pd.DataFrame) -> Number:
    return np.sqrt(((impact - data) ** 2).mean(axis=None))


def impf_step_generator(threshold: Number, paa: Number) -> ImpactFuncSet:
    return ImpactFuncSet(
        [
            ImpactFunc.from_step_impf(
                haz_type="RF", intensity=(0, threshold, 100), paa=(0, paa)
            )
        ]
    )


@dataclass
class Input:
    """Define the static input for a calibration task"""

    hazard: Hazard
    exposure: Exposures
    data: pd.DataFrame
    cost_func: Callable[[Impact, pd.DataFrame], Number]
    impact_func_gen: Callable[..., ImpactFuncSet]
    bounds: Optional[Mapping[str, Union[Bounds, Tuple[Number, Number]]]] = None
    constraints: Optional[
        Mapping[str, Union[LinearConstraint, NonlinearConstraint, Mapping]]
    ] = None
    impact_calc_kwds: Mapping[str, Any] = field(
        default_factory=lambda: dict(assign_centroids=False)
    )

    def __post_init__(self):
        """Prepare input data"""
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

    def _kwargs_to_impact_func_gen(self, *args, **kwargs) -> Dict[str, Any]:
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
        ).impact(assign_centroids=False, **self.input.impact_calc_kwds)
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

    def _kwargs_to_impact_func_gen(self, *args, **kwargs) -> Dict[str, Any]:
        return dict(zip(self._param_names, args[0].flat))

    def _select_by_param_names(self, mapping: Mapping[str, Any]) -> List[Any]:
        """Return a list of entries from a map with matching keys or ``None``"""
        return [mapping.get(key) for key in self._param_names]

    def run(self, params_init: Mapping[str, Number], **opt_kwargs):
        """Execute the optimization"""
        self._param_names = list(params_init.keys())

        # Transform data to match minimize input
        bounds = (
            self._select_by_param_names(self.input.bounds)
            if self.input.bounds is not None
            else None
        )
        constraints = (
            self._select_by_param_names(self.input.constraints)
            if self.input.constraints is not None
            else None
        )

        x0 = np.array(list(params_init.values()))
        res = minimize(
            fun=lambda x: self._opt_func(x),
            x0=x0,
            bounds=bounds,
            constraints=constraints,
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
