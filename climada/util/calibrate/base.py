"""Calibration Base Classes and Interfaces"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Callable, Mapping, Optional, Tuple, Union, Any, Dict
from numbers import Number

import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
import seaborn as sns

from climada.hazard import Hazard
from climada.entity import Exposures, ImpactFuncSet
from climada.engine import Impact, ImpactCalc
import climada.util.coordinates as u_coord

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
    impact_func_creator : Callable
        Function that takes the parameters as keyword arguments and returns an impact
        function set. This will be called each time the optimization algorithm updates
        the parameters.
    impact_to_dataframe : Callable
        Function that takes an impact object as input and transforms its data into a
        pandas.DataFrame that is compatible with the format of :py:attr:`data`.
        The return value of this function will be passed to the :py:attr`cost_func`
        as first argument.
    cost_func : Callable
        Function that takes two ``pandas.Dataframe`` objects and returns the scalar
        "cost" between them. The optimization algorithm will try to minimize this
        number.
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
        here via the ``assign_centroids`` parameter, to avoid assigning them each time
        the impact is calculated).
    assign_centroids : bool, optional
        If ``True`` (default), assign the hazard centroids to the exposure.
    """

    hazard: Hazard
    exposure: Exposures
    data: pd.DataFrame
    impact_func_creator: Callable[..., ImpactFuncSet]
    impact_to_dataframe: Callable[[Impact], pd.DataFrame]
    cost_func: Callable[[pd.DataFrame, pd.DataFrame], Number]
    bounds: Optional[Mapping[str, Union[Bounds, Tuple[Number, Number]]]] = None
    constraints: Optional[Union[ConstraintType, list[ConstraintType]]] = None
    impact_calc_kwds: Mapping[str, Any] = field(
        default_factory=lambda: {"assign_centroids": False}
    )
    assign_centroids: InitVar[bool] = True

    def __post_init__(self, assign_centroids):
        """Prepare input data"""
        if assign_centroids:
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
    """

    params: Mapping[str, Number]
    target: Number


@dataclass
class OutputEvaluator:
    """Evaluate the output of a calibration task

    Parameters
    ----------
    input : Input
        The input object for the optimization task.
    output : Output
        The output object returned by the optimization task.

    Attributes
    ----------
    impf_set : climada.entity.ImpactFuncSet
        The impact function set built from the optimized parameters
    impact : climada.engine.Impact
        An impact object calculated using the optimal :py:attr:`impf_set`
    """

    input: Input
    output: Output

    def __post_init__(self):
        """Compute the impact for the optimal parameters"""
        self.impf_set = self.input.impact_func_creator(**self.output.params)
        self.impact = ImpactCalc(
            exposures=self.input.exposure,
            impfset=self.impf_set,
            hazard=self.input.hazard,
        ).impact(assign_centroids=True, save_mat=True)
        self._impact_label = f"Impact [{self.input.exposure.value_unit}]"

    def plot_impf_set(self, **plot_kwargs):
        """Plot the optimized impact functions"""
        return self.impf_set.plot(**plot_kwargs)

    def plot_at_event(self, **plot_kwargs):
        data = (
            pd.concat(
                [
                    pd.Series([self.impact.at_event]),
                    self.input.data.sum(axis="columns"),
                ],
                ignore_index=True,
                axis=1,
            )
            .rename(columns={0: "Model", 1: "Data"})
            .set_index(self.input.hazard.event_name)
        )
        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return data.plot.bar(ylabel=ylabel, **plot_kwargs)

    def plot_at_region(self, agg_regions=None, **plot_kwargs):
        data = pd.concat(
            [
                self.impact.impact_at_reg(agg_regions).sum(axis="index"),
                self.input.data.sum(axis="index"),
            ],
            axis=1,
        ).rename(columns={0: "Model", 1: "Data"})

        # Use nice country names if no agg_regions were given
        if agg_regions is None:
            data = data.rename(
                index=lambda x: u_coord.country_to_iso(x, representation="name")
            )

        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return data.plot.bar(ylabel=ylabel, **plot_kwargs)

    def plot_event_region_heatmap(self, agg_regions=None, **plot_kwargs):
        # Data preparation
        agg = self.impact.impact_at_reg(agg_regions)
        data = (agg + 1) / (self.input.data + 1)
        data = data.transform(np.log10).replace(0, np.nan)
        data = data.where((agg < 1) & (self.input.data < 1))

        # Use nice country names if no agg_regions were given
        if agg_regions is None:
            data = data.rename(
                index=lambda x: u_coord.country_to_iso(x, representation="name")
            )

        # Default plot settings
        annot = plot_kwargs.pop("annot", True)
        vmax = plot_kwargs.pop("vmax", 3)
        vmin = plot_kwargs.pop("vmin", -vmax)
        center = plot_kwargs.pop("center", 0)
        fmt = plot_kwargs.pop("fmt", ".1f")
        cmap = plot_kwargs.pop("cmap", "RdBu_r")
        cbar_kws = plot_kwargs.pop(
            "cbar_kws", {"label": r"Model Error $\log_{10}(\mathrm{Impact})$"}
        )

        return sns.heatmap(
            data,
            annot=annot,
            vmin=vmin,
            vmax=vmax,
            center=center,
            fmt=fmt,
            cmap=cmap,
            cbar_kws=cbar_kws,
            **plot_kwargs,
        )


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

    def _target_func(self, impact: pd.DataFrame, data: pd.DataFrame) -> Number:
        """Target function for the optimizer

        The default version of this function simply returns the value of the cost
        function evaluated on the arguments.

        Parameters
        ----------
        impact : climada.engine.Impact
            The impact object returned by the impact calculation.
        data : pandas.DataFrame
            The data used for calibration. See :py:attr:`Input.data`.

        Returns
        -------
        The value of the target function for the optimizer.
        """
        return self.input.cost_func(impact, data)

    def _kwargs_to_impact_func_creator(self, *_, **kwargs) -> Dict[str, Any]:
        """Define how the parameters to :py:meth:`_opt_func` must be transformed

        Optimizers may implement different ways of representing the parameters (e.g.,
        key-value pairs, arrays, etc.). Depending on this representation, the parameters
        must be transformed to match the syntax of the impact function generator used,
        see :py:attr:`Input.impact_func_creator`.

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
        params = self._kwargs_to_impact_func_creator(*args, **kwargs)
        impf_set = self.input.impact_func_creator(**params)
        impact = ImpactCalc(
            exposures=self.input.exposure,
            impfset=impf_set,
            hazard=self.input.hazard,
        ).impact(**self.input.impact_calc_kwds)
        impact_df = self.input.impact_to_dataframe(impact)
        return self._target_func(impact_df, self.input.data)

    @abstractmethod
    def run(self, **opt_kwargs) -> Output:
        """Execute the optimization"""
