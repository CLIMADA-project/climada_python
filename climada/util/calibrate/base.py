"""Calibration Base Classes and Interfaces"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar
from typing import Callable, Mapping, Optional, Tuple, Union, Any, Dict
from numbers import Number

import pandas as pd
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
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
    align_kwds : Mapping (str, Any), optional
        Keyword arguments to ``pandas.DataFrame.align`` for aligning the :py:attr:`data`
        with the data frame returned by :py:attr:`impact_to_dataframe`. By default,
        both axes will be aligned and the fill value is zero
        (``"axis": None, "fill_value": 0}``). This assumes that if events and/or regions
        between both data frames do not align, the respective value is assumed to be
        zero and this will be incorporated into the estimation. If you want to require
        alignment, set ``"fill_value": None``. This will set non-aligned values to NaN,
        which typically results in a NaN target function, aborting the estimation.
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
    align_kwds: Mapping[str, Any] = field(
        default_factory=lambda: {"axis": None, "fill_value": 0}
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
        """Plot the optimized impact functions

        This calls the plot function of the respective impact function set.

        Parameters
        ----------
        plot_kwargs
            Plotting keyword arguments passed to the underlying plotting method.

        See Also
        --------
        :py:meth:`~climada.entity.impact_funcs.impact_func_set.ImpactFuncSet.plot`
        """
        return self.impf_set.plot(**plot_kwargs)
    
    def plot_impf_variability(
        self,
        cost_func_diff: float = 0.1,
        p_space_df: Optional[pd.DataFrame] = None,
        plot_haz: bool = False,
        haz_vals: Optional[np.array] = None,
        **plot_kwargs
    ):
        """Plot impact function variability with parameter combinations of almost equal 
        cost function values

        Args:
            cost_func_diff (float, optional): Max deviation from optimal cost function value 
                (as fraction). Defaults to 0.1 (i.e. 10%).
            p_space_df (pd.DataFrame, optional): parameter space. Defaults to None.
            plot_haz (bool, optional): Whether or not to plot hazard intensity distibution. 
                Defaults to False.
            haz_vals (np.array, optional): Hazard values at exposure points (if 
                pre-calculated). Defaults to None.
        """
        haz_type = self.input.hazard.haz_type
        if p_space_df is None:
            p_space_df = self.output.p_space_to_dataframe()

        # Ignore cost dimension
        params = p_space_df.columns.tolist()
        try:
            params.remove('Cost Function') #Hardcoded name i.o.?
        except ValueError:
            pass

        #determine cost function values within 'cost_func_diff' % of best estimate
        max_cost_func_val = p_space_df['Cost Function'].min()*(1+cost_func_diff)
        params_within_range = p_space_df.loc[p_space_df['Cost Function']<=max_cost_func_val,
                                             params]

        # Initialize figure
        fig,ax = plt.subplots()

        #Plot best-fit impact function
        best_impf = self.impf_set.get_func(haz_type=haz_type)[0]
        ax.plot(best_impf.intensity,best_impf.mdd*best_impf.paa*100,color='tab:blue',
                lw=3,zorder=2,label='best fit')

        #Plot all impact functions within 'cost_func_diff' % of best estimate
        for row in range(params_within_range.shape[0]):
            label = f'within {int(cost_func_diff*100)} percent of best fit' if row==0 else None

            sel_params = params_within_range.iloc[row,:].to_dict()
            temp_impf_set = self.input.impact_func_creator(**sel_params)
            temp_impf = temp_impf_set.get_func(haz_type=haz_type)[0]

            ax.plot(temp_impf.intensity,temp_impf.mdd*temp_impf.paa*100,color='grey',
                    alpha=0.4,label=label)
            
        # Plot hazard intensity value distributions
        if plot_haz:
            if haz_vals is None:
                haz_vals = self.input.hazard.intensity[:,self.input.exposure.gdf[f"centr_{haz_type}"]]

            ax2 = ax.twinx()
            ax2.hist(haz_vals[haz_vals.nonzero()].getA1(),bins=40,color='tab:orange',
                     alpha=0.3,label='Hazard intensity\noccurence')
            ax2.set(ylabel='Hazard intensity occurence (#Exposure points)')
            ax.axvline(x=haz_vals.max(),label='Maximum hazard value',color='tab:orange')
            ax2.legend(loc='lower right')

        ax.set(xlabel=f"Intensity ({self.input.hazard.units})",
               ylabel="Mean Damage Ratio (MDR) in %",
               xlim=(min(best_impf.intensity),max(best_impf.intensity)))
        ax.legend()
        return ax


    def plot_at_event(
        self,
        data_transf: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
        **plot_kwargs,
    ):
        """Create a bar plot comparing estimated model output and data per event

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
        """
        data = pd.concat(
            [
                self.input.impact_to_dataframe(self.impact).sum(axis="columns"),
                self.input.data.sum(axis="columns"),
            ],
            axis=1,
        ).rename(columns={0: "Model", 1: "Data"})

        # Transform data before plotting
        data = data_transf(data)

        # Now plot
        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return data.plot.bar(ylabel=ylabel, **plot_kwargs)

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
        """
        data = pd.concat(
            [
                self.input.impact_to_dataframe(self.impact).sum(axis="index"),
                self.input.data.sum(axis="index"),
            ],
            axis=1,
        ).rename(columns={0: "Model", 1: "Data"})

        # Transform data before plotting
        data = data_transf(data)

        # Now plot
        ylabel = plot_kwargs.pop("ylabel", self._impact_label)
        return data.plot.bar(ylabel=ylabel, **plot_kwargs)

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
        agg = self.input.impact_to_dataframe(self.impact)
        data = (agg + 1) / (self.input.data + 1)
        data = data.transform(np.log10).replace(0, np.nan)
        data = data.where((agg > 0) | (self.input.data > 0))

        # Transform data
        data = data_transf(data)

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

    def _target_func(self, true: pd.DataFrame, predicted: pd.DataFrame) -> Number:
        """Target function for the optimizer

        The default version of this function simply returns the value of the cost
        function evaluated on the arguments.

        Parameters
        ----------
        true : pandas.DataFrame
            The "true" data used for calibration. By default, this is
            :py:attr:`Input.data`.
        predicted : pandas.DataFrame
            The impact predicted by the data calibration after it has been transformed
            into a dataframe by :py:attr:`Input.impact_to_dataframe`.

        Returns
        -------
        The value of the target function for the optimizer.
        """
        return self.input.cost_func(true, predicted)

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
        impact_df = self.input.impact_to_dataframe(impact)
        data_aligned, impact_df_aligned = self.input.data.align(
            impact_df, **self.input.align_kwds
        )
        return self._target_func(data_aligned, impact_df_aligned)

    @abstractmethod
    def run(self, **opt_kwargs) -> Output:
        """Execute the optimization"""
