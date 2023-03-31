"""Module for calibrating impact functions"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Tuple, Union, Any
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

from ....climada.hazard import Hazard
from ....climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from ....climada.engine import Impact, ImpactCalc


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

    @abstractmethod
    def run(
        self, opt_kwds: Mapping[str, Any], impact_calc_kwds: Mapping[str, Any]
    ) -> Output:
        """Execute the optimization"""
        pass

    @property
    @abstractmethod
    def optimize_func(self) -> Callable:
        """The function used for optimizing"""


@dataclass
class ScipyMinimizeOptimizer(Optimizer):
    """An optimization using scipy.optimize.minimize"""

    def run(
        self,
        params_init: Mapping[str, Number],
        opt_kwds: Mapping[str, Any],
        impact_calc_kwds: Mapping[str, Any],
    ):
        """Execute the optimization"""
        param_names = list(params_init.keys())

        # Transform data to match minimize input
        bounds = self.input.bounds
        if bounds is not None:
            bounds = [bounds[name] for name in param_names]

        constraints = self.input.constraints
        if constraints is not None:
            constraints = [constraints[name] for name in param_names]

        def fun(params: np.ndarray):
            """Calculate impact and return cost"""
            param_dict = {name: value for name, value in zip(param_names, params.flat)}
            impf_set = self.input.impact_func_gen(**param_dict)
            impact = ImpactCalc(
                exposures=self.input.exposure,
                impfset=impf_set,
                hazard=self.input.hazard,
            ).impact(assign_centroids=False, **impact_calc_kwds)
            return self.input.cost_func(impact, self.input.data)

        x0 = np.array(list(params_init.values()))
        res = minimize(
            fun=fun, x0=x0, bounds=bounds, constraints=constraints, **opt_kwds
        )

        params = {name: value for name, value in zip(param_names, res.x.flat)}
        return Output(params=params, target=res.fun, success=res.success, result=res)
