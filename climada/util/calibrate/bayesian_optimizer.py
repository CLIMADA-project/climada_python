"""Calibration with Bayesian Optimization"""

from dataclasses import dataclass, InitVar
from typing import Mapping, Optional, Any
from numbers import Number

import pandas as pd
from bayes_opt import BayesianOptimization
from bayes_opt.target_space import TargetSpace

from climada.engine import Impact
from .base import Output, Optimizer


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
            verbose=verbose,
            random_state=random_state,
            allow_duplicate_points=allow_duplicate_points,
            **bayes_opt_kwds,
        )

    def _target_func(self, impact: pd.DataFrame, data: pd.DataFrame) -> Number:
        """Invert the cost function because BayesianOptimization maximizes the target"""
        return -self.input.cost_func(impact, data)

    def run(self, **opt_kwargs):
        """Execute the optimization

        ``BayesianOptimization`` *maximizes* a target function. Therefore, this class
        inverts the cost function and used that as target function. The cost function is
        still minimized.

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
        output : BayesianOptimizerOutput
            Optimization output. :py:attr:`BayesianOptimizerOutput.p_space` stores data
            on the sampled parameter space.
        """
        # Retrieve parameters
        num_params = len(self.input.bounds)
        init_points = opt_kwargs.pop("init_points", 10**num_params)
        n_iter = opt_kwargs.pop("n_iter", 10**num_params)

        # Run optimizer
        self.optimizer.maximize(init_points=init_points, n_iter=n_iter, **opt_kwargs)

        # Return output
        opt = self.optimizer.max
        return BayesianOptimizerOutput(
            params=opt["params"],
            target=opt["target"],
            p_space=self.optimizer.space,
        )


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
            Data frame whose columns are the parameter values and the associated target
            function value (``target``) and whose rows are the optimizer iterations.
        """
        data = {
            self.p_space.keys[i]: self.p_space.params[..., i]
            for i in range(self.p_space.dim)
        }
        data["target"] = self.p_space.target
        data = pd.DataFrame.from_dict(data)
        data.index.rename("Iteration", inplace=True)
        return data
