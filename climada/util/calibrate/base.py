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
Calibration Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from climada.engine import Impact, ImpactCalc
from climada.entity import Exposures, ImpactFuncSet
from climada.hazard import Hazard

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
        ``hazard``. Columns: Arbitrary columns. NaN values in the data frame have
        special meaning: Corresponding impact values computed by the model are ignored
        in the calibration.
    impact_func_creator : Callable
        Function that takes the parameters as keyword arguments and returns an impact
        function set. This will be called each time the optimization algorithm updates
        the parameters.
    impact_to_dataframe : Callable
        Function that takes an impact object as input and transforms its data into a
        pandas.DataFrame that is compatible with the format of :py:attr:`data`.
        The return value of this function will be passed to the :py:attr:`cost_func`
        as first argument.
    cost_func : Callable
        Function that takes two ``pandas.Dataframe`` objects and returns the scalar
        "cost" between them. The optimization algorithm will try to minimize this
        number. The first argument is the true/correct values (:py:attr:`data`), and the
        second argument is the estimated/predicted values.
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
    missing_data_value : float, optional
        If the impact model returns impact data for which no values exist in
        :py:attr:`data`, insert this value. Defaults to NaN, in which case the impact
        from the model is ignored. Set this to zero to explicitly calibrate to zero
        impacts in these cases.
    assign_centroids : bool, optional
        If ``True`` (default), assign the hazard centroids to the exposure when this
        object is created.
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
    missing_data_value: float = np.nan
    assign_centroids: InitVar[bool] = True

    def __post_init__(self, assign_centroids):
        """Prepare input data"""
        if not isinstance(self.data, pd.DataFrame):
            if isinstance(self.data, pd.Series):
                raise TypeError(
                    "You passed a pandas Series as 'data'. Please transform it into a "
                    "dataframe with Series.to_frame() and make sure that columns "
                    "correctly indicate locations and indexes events."
                )
            raise TypeError("'data' must be a pandas.DataFrame")

        if assign_centroids:
            self.exposure.assign_centroids(self.hazard)

    def impact_to_aligned_df(
        self, impact: Impact, fillna: float = np.nan
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a dataframe from an impact and align it with the data.

        When aligning, two general cases might occur, which are not mutually exclusive:

        1. There are data points for which no impact was computed. This will always be
           treated as an impact of zero.
        2. There are impacts for which no data points exist. For these points, the input
           data will be filled with the value of :py:attr:`Input.missing_data_value`.

        This method performs the following steps:

        * Transform the impact into a dataframe using :py:attr:`impact_to_dataframe`.
        * Align the :py:attr:`data` with the impact dataframe, using
          :py:attr:`missing_data_value` as fill value.
        * Align the impact dataframe with the data, using zeros as fill value.
        * In the aligned impact, set all values to zero where the data is NaN.
        * Fill remaining NaNs in data with ``fillna``.

        Parameters
        ----------
        impact_df : pandas.DataFrame
            The impact computed by the model, transformed into a dataframe by
            :py:attr:`Input.impact_to_dataframe`.

        Returns
        -------
        data_aligned : pd.DataFrame
            The data aligned to the impact dataframe
        impact_df_aligned : pd.DataFrame
            The impact transformed to a dataframe and aligned with the data
        """
        # Transform impact to  to dataframe
        impact_df = self.impact_to_dataframe(impact)
        if impact_df.isna().any(axis=None):
            raise ValueError("NaN values computed in impact!")

        # Align with different fill values
        data_aligned, _ = self.data.align(
            impact_df, axis=None, fill_value=self.missing_data_value, copy=True
        )
        impact_df_aligned, _ = impact_df.align(
            data_aligned, join="right", axis=None, fill_value=0.0, copy=False
        )

        # Set all impacts to zero for which data is NaN
        impact_df_aligned.where(data_aligned.notna(), 0.0, inplace=True)

        # NOTE: impact_df_aligned should not contain any NaNs at this point
        return data_aligned.fillna(fillna), impact_df_aligned.fillna(fillna)


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

    def to_hdf5(self, filepath: Union[Path, str], mode: str = "x"):
        """Write the output into an H5 file

        This stores the data as attributes because we only store single numbers, not
        arrays

        Parameters
        ----------
        filepath : Path or str
            The filepath to store the data.
        mode : str (optional)
            The mode for opening the file. Defaults to ``x`` (Create file, fail if
            exists).
        """
        with h5py.File(filepath, mode=mode) as file:
            # Store target
            grp = file.create_group("base")
            grp.attrs["target"] = self.target

            # Store params
            grp_params = grp.create_group("params")
            for p_name, p_val in self.params.items():
                grp_params.attrs[p_name] = p_val

    @classmethod
    def from_hdf5(cls, filepath: Union[Path, str]):
        """Create an output object from an H5 file"""
        with h5py.File(filepath) as file:
            target = file["base"].attrs["target"]
            params = dict(file["base"]["params"].attrs.items())
            return cls(params=params, target=target)


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

    def plot_at_event(
        self,
        data_transf: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        **plot_kwargs,
    ):
        """Create a bar plot comparing estimated model output and data per event.

        Every row of the :py:attr:`Input.data` is considered an event.
        The data to be plotted can be transformed with a generic function
        ``data_transf``.

        Parameters
        ----------
        data_transf : Callable (pd.DataFrame -> pd.DataFrame), optional
            A function that transforms the data to plot before plotting.
            It receives a dataframe whose rows represent events and whose columns
            represent the modelled impact and the calibration data, respectively.
            By default, the data is not transformed.
        plot_kwargs
            Keyword arguments passed to the ``DataFrame.plot.bar`` method.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plot axis returned by ``DataFrame.plot.bar``

        Note
        ----
        This plot does *not* include the ignored impact, see :py:attr:`Input.data`.
        """
        data, impact = self.input.impact_to_aligned_df(self.impact)
        values = pd.concat(
            [impact.sum(axis="columns"), data.sum(axis="columns")],
            axis=1,
        ).rename(columns={0: "Model", 1: "Data"})

        # Transform data before plotting
        values = data_transf(values)

        # Now plot
        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return values.plot.bar(ylabel=ylabel, **plot_kwargs)

    def plot_at_region(
        self,
        data_transf: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        **plot_kwargs,
    ):
        """Create a bar plot comparing estimated model output and data per event

        Every column of the :py:attr:`Input.data` is considered a region.
        The data to be plotted can be transformed with a generic function
        ``data_transf``.

        Parameters
        ----------
        data_transf : Callable (pd.DataFrame -> pd.DataFrame), optional
            A function that transforms the data to plot before plotting.
            It receives a dataframe whose rows represent regions and whose columns
            represent the modelled impact and the calibration data, respectively.
            By default, the data is not transformed.
        plot_kwargs
            Keyword arguments passed to the ``DataFrame.plot.bar`` method.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plot axis returned by ``DataFrame.plot.bar``.

        Note
        ----
        This plot does *not* include the ignored impact, see :py:attr:`Input.data`.
        """
        data, impact = self.input.impact_to_aligned_df(self.impact)
        values = pd.concat(
            [impact.sum(axis="index"), data.sum(axis="index")],
            axis=1,
        ).rename(columns={0: "Model", 1: "Data"})

        # Transform data before plotting
        values = data_transf(values)

        # Now plot
        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return values.plot.bar(ylabel=ylabel, **plot_kwargs)

    def plot_event_region_heatmap(
        self,
        data_transf: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        **plot_kwargs,
    ):
        """Plot a heatmap comparing all events per all regions

        Every column of the :py:attr:`Input.data` is considered a region, and every
        row is considered an event.
        The data to be plotted can be transformed with a generic function
        ``data_transf``.

        Parameters
        ----------
        data_transf : Callable (pd.DataFrame -> pd.DataFrame), optional
            A function that transforms the data to plot before plotting.
            It receives a dataframe whose rows represent events and whose columns
            represent the regions, respectively.
            By default, the data is not transformed.
        plot_kwargs
            Keyword arguments passed to the ``DataFrame.plot.bar`` method.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plot axis returned by ``DataFrame.plot.bar``.

        """
        # Data preparation
        data, impact = self.input.impact_to_aligned_df(self.impact)
        values = (impact + 1) / (data + 1)  # Avoid division by zero
        values = values.transform(np.log10)

        # Transform data
        values = data_transf(values)

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
            values,
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

    def _target_func(self, data: pd.DataFrame, predicted: pd.DataFrame) -> Number:
        """Target function for the optimizer

        The default version of this function simply returns the value of the cost
        function evaluated on the arguments.

        Parameters
        ----------
        data : pandas.DataFrame
            The reference data used for calibration. By default, this is
            :py:attr:`Input.data`.
        predicted : pandas.DataFrame
            The impact predicted by the data calibration after it has been transformed
            into a dataframe by :py:attr:`Input.impact_to_dataframe`.

        Returns
        -------
        The value of the target function for the optimizer.
        """
        return self.input.cost_func(data, predicted)

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
        # Create the impact function set from a new parameter estimate
        params = self._kwargs_to_impact_func_creator(*args, **kwargs)
        impf_set = self.input.impact_func_creator(**params)

        # Compute the impact
        impact = ImpactCalc(
            exposures=self.input.exposure,
            impfset=impf_set,
            hazard=self.input.hazard,
        ).impact(**self.input.impact_calc_kwds)

        # Transform to DataFrame, align, and compute target function
        data_aligned, impact_df_aligned = self.input.impact_to_aligned_df(
            impact, fillna=0
        )
        return self._target_func(data_aligned, impact_df_aligned)

    @abstractmethod
    def run(self, **opt_kwargs) -> Output:
        """Execute the optimization"""
