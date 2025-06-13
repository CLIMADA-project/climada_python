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
Ensemble calibration on top of the single-function calibration module
"""

import logging
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from dataclasses import InitVar, dataclass, field
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from climada.engine.unsequa.input_var import InputVar
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.util.coordinates import country_to_iso

from .base import Input, Optimizer, Output

LOGGER = logging.getLogger(__name__)


def sample_data(data: pd.DataFrame, sample: list[tuple[int, int]]):
    """
    Return a DataFrame containing only the sampled values from the input data.

    The resulting data frame has the same shape and indices ad ``data`` and is filled
    with NaNs, except for the row and column indices specified by ``sample``.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame from which values will be sampled.
    sample : list of tuple of int
        A list of (row, column) index pairs indicating which positions
        to copy from ``data`` into the returned DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of the same shape as ``data`` with NaNs in all positions
        except those specified in ``sample``, which contain the corresponding values
        from ``data``.
    """
    # Create all-NaN data
    data_sampled = pd.DataFrame(np.nan, columns=data.columns, index=data.index)

    # Extract sample values from data
    for row, col in sample:
        data_sampled.iloc[row, col] = data.iloc[row, col]

    return data_sampled


def sample_weights(weights: pd.DataFrame, sample: list[tuple[int, int]]):
    """
    Return an updated DataFrame containing the appropriate weights for a sample.

    Weights that are not in ``sample`` are set to zero, whereas weights that are sampled
    multiple times will effectively multiplied by their occurrence in ``sample``.

    Parameters
    ----------
    weights : pandas.DataFrame
        The original weights for the data
    sample : list of tuple of int
        A list of (row, column) index pairs indicating which weights will be used, and
        how often.

    Returns
    -------
    pandas.DataFrame
        Updated ``weights`` for ``sample``.
    """
    # Create all-zero weights
    weights_sampled = pd.DataFrame(0.0, columns=weights.columns, index=weights.index)

    # Add weights for each sample
    for row, col in sample:
        weights_sampled.iloc[row, col] += weights.iloc[row, col]

    return weights_sampled


def event_info_from_input(inp: Input) -> dict[str, Any]:
    """Get information on the event(s) for which we calibrated

    This tries to retrieve the event IDs, region IDs, and event names.

    Returns
    -------
    dict
        With keys ``event_id``, ``region_id``, ``event_name``
    """
    # Get region and event IDs
    data = inp.data.dropna(axis="columns", how="all").dropna(axis="index", how="all")
    event_ids = data.index
    region_ids = data.columns

    # Get event name
    try:
        event_names = inp.hazard.select(event_id=event_ids.to_list()).event_name
    except IndexError:
        event_names = []

    # Return data
    return {
        "event_id": event_ids.to_numpy(),
        "region_id": region_ids.to_numpy(),
        "event_name": event_names,
    }


@dataclass
class SingleEnsembleOptimizerOutput(Output):
    """Output for a single member of an ensemble optimizer

    This extends a regular :py:class:`~climada.util.calibrate.base.Output` by
    information on the particular event(s) this calibration was performed on.

    Attributes
    ----------
    event_info : dict(str, any)
        Information on the events for this calibration instance
    """

    event_info: dict[str, Any] = field(default_factory=dict)


def optimize(
    optimizer_type: type[Optimizer],
    inp: Input,
    opt_init_kwargs: Mapping[str, Any],
    opt_run_kwargs: Mapping[str, Any],
) -> SingleEnsembleOptimizerOutput:
    """Instantiate an optimizer, run it, and return its output

    Parameters
    ----------
    optimizer_type : type
        The type of the optimizer to use
    inp : Input
        The optimizer input
    opt_init_kwargs
        Keyword argument for initializing the optimizer
    opt_run_kwargs
        Keyword argument for running the optimizer

    Returns
    -------
    SingleEnsembleOptimizerOutput
        The output of the optimizer
    """
    opt = optimizer_type(inp, **opt_init_kwargs)
    out = opt.run(**opt_run_kwargs)
    return SingleEnsembleOptimizerOutput(
        params=out.params,
        target=out.target,
        event_info=event_info_from_input(inp),
    )


@dataclass
class EnsembleOptimizerOutput:
    """The collective output of an ensemble optimization"""

    data: pd.DataFrame

    @classmethod
    def from_outputs(cls, outputs: Sequence[SingleEnsembleOptimizerOutput]):
        """Build data from a list of outputs"""
        # Support empty sequences
        if not outputs:
            return cls(data=pd.DataFrame())

        # Derive column names
        cols = pd.MultiIndex.from_tuples(
            [("Parameters", p_name) for p_name in outputs[0].params.keys()]
            + [("Event", p_name) for p_name in outputs[0].event_info]
        )
        data = pd.DataFrame(columns=cols)

        # Fill with data
        data["Parameters"] = pd.DataFrame.from_records([out.params for out in outputs])
        data["Event"] = pd.DataFrame.from_records([out.event_info for out in outputs])

        return cls(data=data)

    def to_hdf(self, filepath: Path | str):
        """Store data to HDF5"""
        self.data.to_hdf(filepath, key="data")

    @classmethod
    def from_hdf(cls, filepath: Path | str):
        """Load data from HDF"""
        return cls(data=pd.read_hdf(filepath, key="data"))

    @classmethod
    def from_csv(cls, filepath: Path | str):
        """Load data from CSV"""
        LOGGER.warning(
            "Do not use CSV for storage, because it does not preserve data types. "
            "Use HDF instead."
        )
        return cls(data=pd.read_csv(filepath, header=[0, 1]))

    def to_csv(self, filepath: Path | str):
        """Store data as CSV"""
        LOGGER.warning(
            "Do not use CSV for storage, because it does not preserve data types. "
            "Use HDF instead."
        )
        self.data.to_csv(filepath, index=None)

    def _to_impf_sets(
        self, impact_func_creator: Callable[..., ImpactFuncSet]
    ) -> list[ImpactFuncSet]:
        """Return a list of impact functions created from the stored parameters"""
        return [
            impact_func_creator(**row["Parameters"]) for _, row in self.data.iterrows()
        ]

    def to_input_var(
        self, impact_func_creator: Callable[..., ImpactFuncSet], **impfset_kwargs
    ) -> InputVar:
        """Build Unsequa InputVar from the parameters stored in this object"""
        return InputVar.impfset(
            self._to_impf_sets(impact_func_creator), **impfset_kwargs
        )

    def plot(
        self, impact_func_creator: Callable[..., ImpactFuncSet], **impf_set_plot_kwargs
    ):
        """Plot all impact functions into the same plot

        This uses the basic plot functions of
        :py:class:`~climada.entity.impact_funcs.base.ImpactFuncSet`.
        """
        impf_set_list = self._to_impf_sets(impact_func_creator)

        # Create a single plot for the overall layout, then continue plotting into it
        axes = impf_set_list[0].plot(**impf_set_plot_kwargs)

        # 'axes' might be array or single instance
        ax_first = axes
        if isinstance(axes, np.ndarray):
            ax_first = axes.flat[0]

        # Legend is always the same
        handles, labels = ax_first.get_legend_handles_labels()

        # Plot remaining impact function sets
        for impf_set in impf_set_list[1:]:
            impf_set.plot(axis=axes, **impf_set_plot_kwargs)

        # Adjust legends
        for ax in np.asarray([axes]).flat:
            ax.legend(handles, labels)

        return axes

    def plot_shiny(
        self,
        impact_func_creator: Callable[..., ImpactFuncSet],
        haz_type: str,
        impf_id: int,
        inp: Input | None = None,
        impf_plot_kwargs: Mapping[str, Any] | None = None,
        hazard_plot_kwargs: Mapping[str, Any] | None = None,
        legend: bool = True,
    ):
        """Plot all impact functions with appropriate color coding and event data

        Parameters
        ----------
        impact_func_creator : Callable
            A function taking parameters and returning an
            :py:class:`~climada.entity.impact_funcs.base.ImpactFuncSet`.
        haz_type : str
            The hazard type of the impact function to plot.
        impf_id : int
            The ID of the impact function to plot.
        inp : Input, optional
            The input object used for the calibration. If provided, a histogram of the
            hazard intensity will be drawn behin the impact functions.
        impf_plot_kwargs
            Keyword arguments for the function plotting the impact functions.
        hazard_plot_kwargs
            Keyword arguments for the function plotting the hazard intensity histogram.
        legend : bool
            Whether to create a legend or not. The legend may become cluttered for
            results of
            :py:class:`~climada.util.calibrate.ensemble.AverageEnsembleOptimizer`,
            therefore it is advisable to disable it in these cases.
        """
        # Store data to plot
        data_plt = []
        for _, row in self.data.iterrows():
            impf = impact_func_creator(**row["Parameters"]).get_func(
                haz_type=haz_type, fun_id=impf_id
            )
            if not isinstance(impf, ImpactFunc):
                raise ValueError(
                    f"Cannot find a unique impact function for haz_type: {haz_type}, "
                    f"impf_id: {impf_id}"
                )

            def single_entry(arr):
                """If ``arr`` has a single entry, return it, else ``arr`` itself"""
                if len(arr) == 1:
                    return arr[0]
                return arr

            region_id = single_entry(country_to_iso(row[("Event", "region_id")]))
            event_name = single_entry(row[("Event", "event_name")])
            event_id = single_entry(row[("Event", "event_id")])

            label = f"{event_name}, {region_id}, {event_id}"
            if len(event_name) > 1 or len(event_id) > 1:
                label = label.replace("], [", "]\n[")  # Multiline label

            data_plt.append(
                {
                    "intensity": impf.intensity,
                    "mdr": impf.paa * impf.mdd,
                    "label": label,
                }
            )

        # Create plot
        _, ax = plt.subplots()
        legend_xpad = 0

        # Plot hazard histogram
        # NOTE: Actually requires selection by exposure, but this is not trivial!
        if inp is not None:
            # Create secondary axis
            ax2 = ax.twinx()
            ax2.set_zorder(1)
            ax.set_zorder(2)
            ax.set_facecolor("none")
            legend_xpad = 0.15

            # Draw histogram
            hist_kwargs = {"bins": 40, "color": "grey", "alpha": 0.5}
            if hazard_plot_kwargs is not None:
                hist_kwargs.update(hazard_plot_kwargs)
            ax2.hist(inp.hazard.intensity.data, **hist_kwargs)
            ax2.set_ylabel("Intensity Count", color=hist_kwargs["color"])
            ax2.tick_params(axis="y", colors=hist_kwargs["color"])

        elif hazard_plot_kwargs is not None:
            LOGGER.warning("No 'inp' parameter provided. Ignoring 'hazard_plot_kwargs'")

        # Sort data by final MDR value, then plot
        colors = plt.get_cmap("turbo")(np.linspace(0, 1, self.data.shape[0]))
        data_plt = sorted(data_plt, key=lambda x: x["mdr"][-1], reverse=True)
        impf_plot_kwargs = impf_plot_kwargs if impf_plot_kwargs is not None else {}
        for idx, data_dict in enumerate(data_plt):
            ax.plot(
                data_dict["intensity"],
                data_dict["mdr"],
                label=data_dict["label"],
                color=colors[idx],
                **impf_plot_kwargs,
            )

        # Cosmetics
        ax.set_xlabel(f"Intensity [{impf.intensity_unit}]")
        ax.set_ylabel("Impact")
        ax.set_title(f"{haz_type} {impf_id}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        if legend:
            ax.legend(
                bbox_to_anchor=(1.05 + legend_xpad, 1),
                borderaxespad=0,
                borderpad=0,
                loc="upper left",
                title="Event Name, Country, Event ID",
                frameon=False,
                fontsize="xx-small",
                title_fontsize="x-small",
            )

        return ax

    def plot_category(
        self,
        impact_func_creator: Callable[..., ImpactFuncSet],
        haz_type: str,
        impf_id: int,
        category: str,
        category_colors: Mapping[str, str | tuple] | None = None,
        **impf_set_plot_kwargs,
    ):
        """Plot impact functions with coloring according to a certain category

        Parameters
        ----------
        impact_func_creator : Callable
            A function taking parameters and returning an
            :py:class:`~climada.entity.impact_funcs.base.ImpactFuncSet`.
        haz_type : str
            The hazard type of the impact function to plot.
        impf_id : int
            The ID of the impact function to plot.
        category : str
            The event information on which to categorize (can be ``"region_id"``,
            ``"event_id"``, or ``"event_name"``)
        category_colors : dict(str, str or tuple), optional
            Specify which categories to plot (keys) and what colors to use for them
            (values). If ``None``, will categorize for unique values in the ``category``
            column and color automatically.
        """
        impf_set_arr = np.array(self._to_impf_sets(impact_func_creator))

        if category_colors is None:
            unique_categories = self.data[("Event", category)].unique()
            unique_colors = plt.get_cmap("turbo")(
                np.linspace(0, 1, len(unique_categories))
            )
        else:
            unique_categories = category_colors.keys()
            unique_colors = category_colors.values()

        _, ax = plt.subplots()
        for sel_category, color in zip(unique_categories, unique_colors):
            cat_idx = self.data[("Event", category)] == sel_category

            for i, impf_set in enumerate(impf_set_arr[cat_idx]):
                impf = impf_set.get_func(haz_type=haz_type, fun_id=impf_id)
                if not isinstance(impf, ImpactFunc):
                    raise ValueError(
                        "Cannot find a unique impact function for haz_type: "
                        f"{haz_type}, impf_id: {impf_id}"
                    )
                label = f"{sel_category}, {cat_idx.sum()}" if i == 0 else None
                ax.plot(
                    impf.intensity,
                    impf.paa * impf.mdd,
                    color=color,
                    label=label,
                    **impf_set_plot_kwargs,
                )

        ax.legend(
            title=f"{category}, count",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
            borderaxespad=0,
            borderpad=0,
        )
        # Cosmetics
        ax.set_xlabel(f"Intensity [{impf.intensity_unit}]")
        ax.set_ylabel("Impact")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        return ax


@dataclass
class EnsembleOptimizer(ABC):
    """Abstract base class for defining an ensemble optimizer.

    An ensemble optimizer uses a user-defined optimizer type to run multiple calibration
    tasks. The tasks are defined by the :py:attr:`samples` attribute: For each entry in
    :py:attr:`samples`, a new :py:class:`~climada.util.calibrate.base.Input` is created
    and passed to an instance of :py:attr:`optimizer_type`. Derived classes need to set
    the :py:attr:`samples` during initialization and define the
    :py:meth:`input_from_sample` method.

    The calibration tasks can be conducted in parallel by executing :py:meth:`run` with
    ``processes`` set to a value larger than 1.

    Attributes
    ----------
    input : Input
        The generic input for the optimization
    optimizer_type : type[Optimizer]
        The type of the optimizer to use for each calibration task
    optimizer_init_kwargs
        Keyword argument for initializing an instance of the chosen
        :py:attr:`optimizer_type`.
    samples : list of list of tuple(int, int)
        The samples for each calibration task. Each entry is a list of tuples that
        encode row and column indices of the ``Input``
        :py:attr:`~climada.util.calibrate.base.Input.data` that are selected for the
        particular calibration task. See :py:func:`sample_data`.
    """

    input: Input
    optimizer_type: type[Optimizer]
    optimizer_init_kwargs: dict[str, Any] = field(default_factory=dict)
    samples: list[list[tuple[int, int]]] = field(init=False, default_factory=list)

    def run(self, processes=1, **optimizer_run_kwargs) -> EnsembleOptimizerOutput:
        """Execute the ensemble optimization

        Parameters
        ----------
        processes : int, optional
            The number of processes to distribute the optimization tasks to. Defaults to
            1 (no parallelization)
        optimizer_run_kwargs
            Additional keywords arguments for the
            :py:func:`~climada.util.calibrate.base.Optimizer.run` method of the
            particular optimizer used.
        """
        if processes == 1:
            outputs = self._iterate_sequential(**optimizer_run_kwargs)
        else:
            outputs = self._iterate_parallel(processes, **optimizer_run_kwargs)
        return EnsembleOptimizerOutput.from_outputs(outputs)

    def _inputs(self):
        """Generator for input objects"""
        for sample in self.samples:
            yield self.input_from_sample(sample)

    def _opt_init_kwargs(self):
        """Generator for optimizer initialization keyword arguments"""
        for idx in range(len(self.samples)):
            yield self._update_init_kwargs(self.optimizer_init_kwargs, idx)

    def _iterate_sequential(
        self, **optimizer_run_kwargs
    ) -> list[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples sequentially"""
        return [
            optimize(
                self.optimizer_type, input, init_kwargs, deepcopy(optimizer_run_kwargs)
            )
            for input, init_kwargs in tqdm(
                zip(self._inputs(), self._opt_init_kwargs()), total=len(self.samples)
            )
        ]

    def _iterate_parallel(
        self, processes, **optimizer_run_kwargs
    ) -> list[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples in parallel"""
        iterations = len(self.samples)
        opt_run_kwargs = (deepcopy(optimizer_run_kwargs) for _ in range(iterations))
        with ProcessPool(nodes=processes) as pool:
            return list(
                tqdm(
                    pool.imap(
                        optimize,
                        repeat(self.optimizer_type),
                        self._inputs(),
                        self._opt_init_kwargs(),
                        opt_run_kwargs,
                        # chunksize=processes,
                    ),
                    total=iterations,
                )
            )

    @abstractmethod
    def input_from_sample(self, sample: list[tuple[int, int]]) -> Input:
        """Define how an input is created from a sample"""

    def _update_init_kwargs(
        self, init_kwargs: dict[str, Any], iteration: int
    ) -> dict[str, Any]:
        """Copy settings in the init_kwargs and update for each iteration"""
        kwargs = copy(init_kwargs)  # Maybe deepcopy?
        if "random_state" in kwargs:
            kwargs["random_state"] = kwargs["random_state"] + iteration
        return kwargs


@dataclass
class AverageEnsembleOptimizer(EnsembleOptimizer):
    """An optimizer for the "average ensemble".

    This optimizer samples a fraction of the original events in
    :py:attr:`~climada.util.calibrate.ensemble.EnsembleOptimizer.input`.
    :py:attr:`~climada.util.calibrate.base.Input.data`.

    Attributes
    ----------
    sample_fraction : float
        The fraction of data points to use for each calibration. For values > 1,
        :py:attr:`replace` must be ``True``.
    ensemble_size : int
        The number of calibration tasks to perform (and hence size of the ensemble).
    random_state : int
        The seed for the random number generator selecting the samples
    replace : bool
        If samples of the input data should be drawn with replacement
    """

    sample_fraction: InitVar[float] = 0.8
    ensemble_size: InitVar[int] = 20
    random_state: InitVar[int] = 1
    replace: InitVar[bool] = False

    def __post_init__(self, sample_fraction, ensemble_size, random_state, replace):
        """Create the samples"""
        if sample_fraction <= 0:
            raise ValueError("Sample fraction must be larger than 0")
        if sample_fraction > 1 and not replace:
            raise ValueError("Sample fraction must be <=1 or replace must be True")
        if ensemble_size < 1:
            raise ValueError("Ensemble size must be >=1")

        # Find out number of samples
        notna_idx = np.argwhere(self.input.data.notna().to_numpy())
        num_notna = notna_idx.shape[0]
        num_samples = int(np.rint(num_notna * sample_fraction))

        # Create samples
        rng = default_rng(random_state)
        self.samples = [
            rng.choice(notna_idx, size=num_samples, replace=replace)
            for _ in range(ensemble_size)
        ]

    def input_from_sample(self, sample: list[tuple[int, int]]):
        """Shallow-copy the input and update the data"""
        input = copy(self.input)  # NOTE: Shallow copy!

        # Sampling
        # NOTE: We always need samples to support `replace=True`
        input.data = sample_data(input.data, sample)
        weights = (
            input.data_weights
            if input.data_weights is not None
            else pd.DataFrame(1.0, index=input.data.index, columns=input.data.columns)
        )
        input.data_weights = sample_weights(weights, sample)

        return input


@dataclass
class TragedyEnsembleOptimizer(EnsembleOptimizer):
    """An optimizer for the "ensemble of tragedies".

    Each sample (and thus calibration task) of this optimizer only contains a single
    event from :py:attr:`~climada.util.calibrate.ensemble.EnsembleOptimizer.input`.
    :py:attr:`~climada.util.calibrate.base.Input.data`.

    Attributes
    ----------
    ensemble_size : int, optional
        The number of calibration tasks to perform. Defaults to ``None``, which means
        one for each data point. Must be smaller or equal to the number of data points.
        If smaller, random events will be left out from the ensemble calibration.
    random_state : int
        The seed for the random number generator selecting the samples
    """

    ensemble_size: InitVar[Optional[int]] = None
    random_state: InitVar[int] = 1

    def __post_init__(self, ensemble_size, random_state):
        """Create the single samples"""
        notna_idx = np.argwhere(self.input.data.notna().to_numpy())
        self.samples = notna_idx[:, np.newaxis].tolist()  # Must extend by one dimension

        # Subselection for a given ensemble size
        if ensemble_size is not None:
            if ensemble_size < 1:
                raise ValueError("Ensemble size must be >=1")
            if ensemble_size > len(self.samples):
                raise ValueError(
                    "Ensemble size must be smaller than maximum number of samples "
                    f"(here: {len(self.samples)})"
                )

            rng = default_rng(random_state)
            self.samples = rng.choice(self.samples, ensemble_size, replace=False)

    def input_from_sample(self, sample: list[tuple[int, int]]):
        """Subselect all input"""
        # Data
        input = copy(self.input)  # NOTE: Shallow copy!
        data = sample_data(input.data, sample)
        input.data = data.dropna(axis="columns", how="all").dropna(
            axis="index", how="all"
        )
        if input.data_weights is not None:
            input.data_weights, _ = input.data_weights.align(
                input.data,
                axis=None,
                join="right",
                copy=True,
                fill_value=input.missing_weights_value,
            )
            input.data_weights = sample_weights(input.data_weights, sample)

        # Select single hazard event
        input.hazard = input.hazard.select(event_id=input.data.index)

        # Select single region in exposure
        # NOTE: This breaks impact_at_reg with pre-defined region IDs!!
        # exp = input.exposure.copy(deep=False)
        # exp.gdf = exp.gdf[exp.gdf["region_id"] == input.data.columns[0]]
        # input.exposure = exp

        return input
