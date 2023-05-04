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
    OptimizeResult,
    minimize,
)
from bayes_opt import BayesianOptimization

from ...hazard import Hazard
from ...entity import Exposures, ImpactFunc, ImpactFuncSet
from ...engine import Impact, ImpactCalc


@dataclass
class Input:
    """Define the static input for a calibration task"""

    hazard: Hazard
    exposure: Exposures
    data: pd.DataFrame
    cost_func: Callable[[Impact, pd.DataFrame], float]
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
        self.hazard = self.hazard.select(event_id=self.data.index)
        self.exposure.assign_centroids(self.hazard)


@dataclass
class Output:
    """Define the output of a calibration task"""

    params: Mapping[str, Number]
    target: Number
    success: bool
    result: Optional[OptimizeResult] = None


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

    _param_names: List[str] = field(default_factory=list)

    def _kwargs_to_impact_func_gen(self, *args, **kwargs) -> Dict[str, Any]:
        return dict(zip(self._param_names, args[0].flat))

    def run(self, params_init: Mapping[str, Number], **opt_kwargs):
        """Execute the optimization"""
        self._param_names = list(params_init.keys())

        # Transform data to match minimize input
        bounds = self.input.bounds
        if bounds is not None:
            bounds = [bounds.get(name) for name in self._param_names]

        constraints = self.input.constraints
        if constraints is not None:
            constraints = [constraints.get(name) for name in self._param_names]

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
    init_kwds: InitVar[Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self, **kwargs):
        """Create optimizer"""
        init_kwds = kwargs.pop("init_kwds")
        self.optimizer = BayesianOptimization(
            f=lambda **kwargs: self._opt_func(**kwargs),
            pbounds=self.input.bounds,
            **kwargs,
            **init_kwds,
        )

    def run(self, init_points: int = 100, n_iter: int = 200, **opt_kwargs):
        """Execute the optimization"""
        opt_kwargs.update(init_points=init_points, n_iter=n_iter)
        self.optimizer.maximize(**opt_kwargs)
        opt = self.optimizer.max
        return Output(params=opt["params"], target=opt["target"], success=True)
