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
from typing import Optional, List, Mapping, Any, Tuple, Union, Sequence, Dict
from copy import copy, deepcopy
from pathlib import Path
from itertools import repeat

import numpy as np
from numpy.random import default_rng
import pandas as pd
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

from ...engine.unsequa.input_var import InputVar
from .base import Optimizer, Output, Input

# TODO: derived classes for average and tragedy


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
        "event_id": event_ids,
        "region_id": region_ids,
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

    @classmethod
    def from_csv(cls, filepath):
        """Load data from CSV"""
        return cls(data=pd.read_csv(filepath, header=[0, 1]))

    def to_csv(self, filepath):
        """Store data as CSV"""
        self.data.to_csv(filepath, index=None)

    def to_input_var(self, impact_func_creator, **impfset_kwargs):
        """Build Unsequa InputVar from the parameters stored in this object"""
        impf_set_list = [
            impact_func_creator(**row["Parameters"]) for _, row in self.data.iterrows()
        ]
        return InputVar.impfset(impf_set_list, **impfset_kwargs)


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
        if processes == 1:
            outputs = self._iterate_sequential(**optimizer_run_kwargs)
        else:
            outputs = self._iterate_parallel(processes, **optimizer_run_kwargs)
        return EnsembleOptimizerOutput.from_outputs(outputs)

    def _iterate_sequential(
        self, **optimizer_run_kwargs
    ) -> List[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples sequentially"""
        outputs = []
        for idx, sample in enumerate(tqdm(self.samples)):
            input = self.input_from_sample(sample)

            # Run optimizer
            opt = self.optimizer_type(
                input, **self._update_init_kwargs(self.optimizer_init_kwargs, idx)
            )
            out = opt.run(**optimizer_run_kwargs)
            out = SingleEnsembleOptimizerOutput(
                params=out.params,
                target=out.target,
                event_info=event_info_from_input(input),
            )

            outputs.append(out)

        return outputs

    def _iterate_parallel(
        self, processes, **optimizer_run_kwargs
    ) -> List[SingleEnsembleOptimizerOutput]:
        """Iterate over all samples in parallel"""
        inputs = (self.input_from_sample(sample) for sample in self.samples)
        opt_init_kwargs = (
            self._update_init_kwargs(self.optimizer_init_kwargs, idx)
            for idx in range(len(self.samples))
        )

        with ProcessPool(nodes=processes) as pool:
            return list(
                tqdm(
                    pool.imap(
                        optimize,
                        repeat(self.optimizer_type),
                        inputs,
                        opt_init_kwargs,
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
