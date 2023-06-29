"""Calibration with scipy Optimization"""

from dataclasses import dataclass
from typing import Mapping, Any, Dict, List

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .base import Output, Optimizer


@dataclass
class ScipyMinimizeOptimizer(Optimizer):
    """An optimization using scipy.optimize.minimize

    By default, this optimizer uses the ``"trust-constr"`` method. This
    is advertised as the most general minimization method of the ``scipy`` package and
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

    def _kwargs_to_impact_func_creator(self, *args, **_) -> Dict[str, Any]:
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
        output : ScipyMinimizeOptimizerOutput
            The output of the optimization. The
            :py:attr:`ScipyMinimizeOptimizerOutput.result` attribute stores the
            associated ``scipy.optimize.OptimizeResult`` instance.
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
        return ScipyMinimizeOptimizerOutput(params=params, target=res.fun, result=res)


@dataclass
class ScipyMinimizeOptimizerOutput(Output):
    """Output of a calibration with :py:class:`ScipyMinimizeOptimizer`

    Attributes
    ----------
    result : scipy.minimize.OptimizeResult
        The OptimizeResult instance returned by ``scipy.optimize.minimize``.
    """

    result: OptimizeResult
