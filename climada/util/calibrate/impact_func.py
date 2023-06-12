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
    OptimizeResult,
)
from bayes_opt import BayesianOptimization
from bayes_opt.target_space import TargetSpace

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


