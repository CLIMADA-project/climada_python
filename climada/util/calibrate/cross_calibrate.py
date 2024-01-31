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
Cross-calibration on top of a single calibration module
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, InitVar, field
from typing import List, Any, Tuple, Sequence, Dict, Callable
from copy import copy
from itertools import repeat
import logging

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from ...engine.unsequa.input_var import InputVar
from ...entity.impact_funcs import ImpactFuncSet
from ..coordinates import country_to_iso
from .base import Output, Input

LOGGER = logging.getLogger(__name__)


def sample_data(data: pd.DataFrame, sample: List[Tuple[int, int]]):
    """Return a sample of the data"""
    # Create all-NaN data
    data_sampled = pd.DataFrame(np.nan, columns=data.columns, index=data.index)

    # Extract sample values from data
    for x, y in sample:
        data_sampled.iloc[x, y] = data.iloc[x, y]

    return data_sampled


def event_info_from_input(input: Input) -> Dict[str, Any]:
    """Get information on the event(s) for which we calibrated"""
    # Get region and event IDs
    data = input.data.dropna(axis="columns", how="all").dropna(axis="index", how="all")
    event_ids = data.index
    region_ids = data.columns

    # Get event name
    event_names = input.hazard.select(event_id=event_ids.to_list()).event_name

    # Return data
    return {
        "event_id": event_ids.to_numpy(),
        "region_id": region_ids.to_numpy(),
        "event_name": event_names,
    }


def optimize(optimizer_type, input, opt_init_kwargs, opt_run_kwargs):
    opt = optimizer_type(input, **opt_init_kwargs)
    out = opt.run(**opt_run_kwargs)
    return SingleEnsembleOptimizerOutput(
        params=out.params,
        target=out.target,
        event_info=event_info_from_input(input),
    )


@dataclass
class SingleEnsembleOptimizerOutput(Output):
    """Output for a single member of an ensemble optimizer

    Attributes
    ----------
    event_info : dict(str, any)
        Information on the events for this calibration instance
    """

    event_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleOptimizerOutput:
    data: pd.DataFrame

    @classmethod
    def from_outputs(cls, outputs: Sequence[SingleEnsembleOptimizerOutput]):
        """Build data from a list of outputs"""
        cols = pd.MultiIndex.from_tuples(
            [("Parameters", p_name) for p_name in outputs[0].params.keys()]
            + [("Event", p_name) for p_name in outputs[0].event_info]
        )
        data = pd.DataFrame(columns=cols)

        # Fill with data
        data["Parameters"] = pd.DataFrame.from_records([out.params for out in outputs])
        data["Event"] = pd.DataFrame.from_records([out.event_info for out in outputs])

        return cls(data=data)

    def to_hdf(self, filepath):
        """Store data to HDF5"""
        self.data.to_hdf(filepath, key="data")

    @classmethod
    def from_hdf(cls, filepath):
        """Load data from HDF"""
        return cls(data=pd.read_hdf(filepath, key="data"))

    @classmethod
    def from_csv(cls, filepath):
        """Load data from CSV"""
        LOGGER.warning(
            "Do not use CSV for storage, because it does not preserve data types. "
            "Use HDF instead."
        )
        return cls(data=pd.read_csv(filepath, header=[0, 1]))

    def to_csv(self, filepath):
        """Store data as CSV"""
        LOGGER.warning(
            "Do not use CSV for storage, because it does not preserve data types. "
            "Use HDF instead."
        )
        self.data.to_csv(filepath, index=None)

    def _to_impf_sets(self, impact_func_creator) -> List[ImpactFuncSet]:
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
        """Plot all impact functions into the same plot"""
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
        haz_type,
        impf_id,
    ):
        """Plot all impact functions with appropriate color coding and event data"""
        # Store data to plot
        data_plt = []
        for _, row in self.data.iterrows():
            impf = impact_func_creator(**row["Parameters"]).get_func(
                haz_type=haz_type, fun_id=impf_id
            )
            region_id = country_to_iso(row[("Event", "region_id")])
            event_name = row[("Event", "event_name")]
            event_id = row[("Event", "event_id")]

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
        colors = plt.get_cmap("turbo")(np.linspace(0, 1, self.data.shape[0]))

        # Sort data by final MDR value, then plot
        data_plt = sorted(data_plt, key=lambda x: x["mdr"][-1], reverse=True)
        for idx, data_dict in enumerate(data_plt):
            ax.plot(
                data_dict["intensity"],
                data_dict["mdr"],
                label=data_dict["label"],
                color=colors[idx],
            )

        # Cosmetics
        ax.set_xlabel(f"Intensity [{impf.intensity_unit}]")
        ax.set_ylabel("Impact")
        ax.set_title(f"{haz_type} {impf_id}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim(0, 1)
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            borderpad=0,
            loc="upper left",
            title="Event Name, Country, Event ID",
            frameon=False,
            fontsize="xx-small",
            title_fontsize="x-small",
        )

        return ax


@dataclass
class EnsembleOptimizer(ABC):
    """"""

    input: Input
    optimizer_type: Any
    optimizer_init_kwargs: Dict[str, Any] = field(default_factory=dict)
    samples: List[List[Tuple[int, int]]] = field(init=False)

    def __post_init__(self):
        """"""
        if self.samples is None:
            raise RuntimeError("Samples must be set!")

    def run(self, processes=1, **optimizer_run_kwargs) -> EnsembleOptimizerOutput:
        """Execute the ensemble optimization

        Parameters
        ----------
        processes : int, optional
            The number of processes to distribute the optimization tasks to. Defaults to
            1 (no parallelization)
        optimizer_run_kwargs
            Additional keywords arguments for the
            :py:func`~climada.util.calibrate.base.Optimizer.run` method of the
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
    ) -> List[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples sequentially"""
        return [
            optimize(self.optimizer_type, input, init_kwargs, optimizer_run_kwargs)
            for input, init_kwargs in tqdm(
                zip(self._inputs(), self._opt_init_kwargs()), total=len(self.samples)
            )
        ]

    def _iterate_parallel(
        self, processes, **optimizer_run_kwargs
    ) -> List[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples in parallel"""
        with ProcessPool(nodes=processes) as pool:
            return list(
                tqdm(
                    pool.imap(
                        optimize,
                        repeat(self.optimizer_type),
                        self._inputs(),
                        self._opt_init_kwargs(),
                        repeat(optimizer_run_kwargs),
                        # chunksize=processes,
                    ),
                    total=len(self.samples),
                )
            )

    @abstractmethod
    def input_from_sample(self, sample: List[Tuple[int, int]]) -> Input:
        """"""

    def _update_init_kwargs(
        self, init_kwargs: Dict[str, Any], iteration: int
    ) -> Dict[str, Any]:
        """Copy settings in the init_kwargs and update for each iteration"""
        kwargs = copy(init_kwargs)  # Maybe deepcopy?
        if "random_state" in kwargs:
            kwargs["random_state"] = kwargs["random_state"] + iteration
        return kwargs


@dataclass
class AverageEnsembleOptimizer(EnsembleOptimizer):
    """"""

    sample_fraction: InitVar[float] = 0.8
    ensemble_size: InitVar[int] = 20
    random_state: InitVar[int] = 1

    def __post_init__(self, sample_fraction, ensemble_size, random_state):
        """Create the samples"""
        if sample_fraction <= 0 or sample_fraction >= 1:
            raise ValueError("Sample fraction must be in (0, 1)")
        if ensemble_size < 1:
            raise ValueError("Ensemble size must be >=1")

        # Find out number of samples
        notna_idx = np.argwhere(self.input.data.notna().to_numpy())
        num_notna = notna_idx.shape[0]
        num_samples = int(np.rint(num_notna * sample_fraction))

        # Create samples
        rng = default_rng(random_state)
        self.samples = [
            rng.choice(notna_idx, size=num_samples, replace=False)
            for _ in range(ensemble_size)
        ]

        return super().__post_init__()

    def input_from_sample(self, sample: List[Tuple[int, int]]):
        """Shallow-copy the input and update the data"""
        input = copy(self.input)  # NOTE: Shallow copy!
        input.data = sample_data(input.data, sample)
        return input


@dataclass
class TragedyEnsembleOptimizer(EnsembleOptimizer):
    """"""

    def __post_init__(self):
        """Create the single samples"""
        notna_idx = np.argwhere(self.input.data.notna().to_numpy())
        self.samples = notna_idx[:, np.newaxis].tolist()  # Must extend by one dimension

        return super().__post_init__()

    def input_from_sample(self, sample: List[Tuple[int, int]]):
        """Subselect all input"""
        # Data
        input = copy(self.input)  # NOTE: Shallow copy!
        data = sample_data(input.data, sample)
        input.data = data.dropna(axis="columns", how="all").dropna(
            axis="index", how="all"
        )

        # Select single hazard event
        input.hazard = input.hazard.select(event_id=input.data.index)

        # Select single region in exposure
        # NOTE: This breaks impact_at_reg with pre-defined region IDs!!
        # exp = input.exposure.copy(deep=False)
        # exp.gdf = exp.gdf[exp.gdf["region_id"] == input.data.columns[0]]
        # input.exposure = exp

        return input


# @dataclass
# class CrossCalibration:
#     """A class for running multiple calibration tasks on data subsets"""

#     input: Input
#     optimizer_type: Any
#     sample_size: int = 1
#     ensemble_size: Optional[int] = None
#     random_state: InitVar[int] = 1
#     optimizer_init_kwargs: Mapping[str, Any] = field(default_factory=dict)

#     def __post_init__(self, random_state):
#         """"""
#         if self.sample_size < 1:
#             raise ValueError("Sample size must be >=1")
#         if self.sample_size > 1 and self.ensemble_size is None:
#             raise ValueError("Ensemble size must be set if sample size > 1")

#         # Copy the original data
#         self.data = self.input.data.copy()
#         notna_idx = np.argwhere(self.data.notna().to_numpy())

#         # Create the samples
#         if self.ensemble_size is not None:
#             rng = default_rng(random_state)
#             self.samples = [
#                 rng.choice(notna_idx, size=self.sample_size, replace=False)
#                 for _ in range(self.ensemble_size)
#             ]
#         else:
#             self.samples = notna_idx.tolist()

#         print("Samples:\n", self.samples)

#     def run(self, **optimizer_run_kwargs) -> List[Output]:
#         """Run the optimizer for the ensemble"""
#         outputs = []
#         for idx, sample in enumerate(self.samples):
#             # Select data samples
#             data_sample = self.data.copy()
#             data_sample.iloc[:, :] = np.nan  # Set all to NaN
#             for x, y in sample:
#                 data_sample.iloc[x, y] = self.data.iloc[x, y]

#             # Run the optimizer
#             input = deepcopy(self.input)
#             input.data = data_sample

#             # NOTE: NOO assign_centroids
#             opt = self.optimizer_type(input, **self.optimizer_init_kwargs)
#             out = opt.run(**optimizer_run_kwargs)
#             outputs.append(out)
#             print(f"Ensemble: {idx}, Params: {out.params}")

#         return outputs


# # TODO: Tragedy: Localize exposure and hazards!
# @dataclass
# class TragedyEnsembleCrossCalibration(CrossCalibration):
#     """Cross calibration for computing an ensemble of tragedies"""
