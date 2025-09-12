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

Define Hazard Timeseries.
"""

__all__ = ["HazardTimeSeries"]

# import datetime as dt
# import itertools
import logging

# import h5py
import numpy as np

# import climada.util.constants as u_const
# import climada.util.coordinates as u_coord
# import climada.util.hdf5_handler as u_hdf5
# from climada.hazard.centroids.centr import Centroids
from climada.hazard.base import Hazard

# import pathlib
# import warnings
# from collections.abc import Collection
# from typing import Any, Dict, Optional, Union

# import pandas as pd
# import rasterio
# import xarray as xr
# from deprecation import deprecated
# from scipy import sparse


# from .xarray import HazardXarrayReader

LOGGER = logging.getLogger(__name__)


class HazardTimeSeries(Hazard):
    """
    Contains Hazard Time Series class
    """

    def __init__(
        self,
        timesteps: np.ndarray,
        id_timeseries,
        **kwargs,
    ):
        """Initialize values.

        Parameters
        ----------
        timesteps : np.ndarray of int
            array of timesteps. Each int represents the ordinal date of the
            beginning of the time step.
        **kwargs : Hazard properties, optional
            All other keyword arguments are passed to the Hazard constructor.
        """
        Hazard.__init__(self, **kwargs)
        self.timesteps = timesteps
        self.id_timeseries = id_timeseries

    def check_time_series(self):
        "check time series dimension"

        # check hazard structure
        self.check()
        # check if timesteps attribute exists
        if not hasattr(self, "timesteps"):
            raise ValueError("HazardTimeSeries must have timesteps attribute")
        # check if timesteps are equally spaced
        # TBD this does not work because monthly will results in different timesteps
        # if len(np.unique(np.diff(self.timesteps))) > 1:
        #     raise ValueError(
        #         "HazardTimeSeries must include timesteps with equal distance"
        #     )
        # check if event dates correspond to timesteps
        if not set(self.date).issubset(self.timesteps):
            raise ValueError("Event dates do not correspond to timesteps")
        # TBD check that timesteps are increasing
        if not np.all(np.diff(self.timesteps) > 0):
            raise ValueError("Timesteps must be inceasing dates")

    @classmethod
    def sample_from_hazard_set(
        cls,
        hazard,
        n_timeseries,
        timesteps,
        seasonality=None,
        seed=None,
        # time_correlation=0,
    ):
        """sample time series from probabilistic hazard event set

        Parameters
        ----------
        hazard : climada.hazard.base.Hazard
            Hazard object containing probabilistic event set
        n_timeseries : int
            Number of time series to be sampled
        timesteps : iterable[int]
            Array of int the oridnal date of each timestep
        seasonality : iterable[float] or None, optional
            Array of floats describing weights for each timestep to weight the sampling.
            If None, equal weights are used. Defaults to None.
        seed : int, optional
            random seed

        (future implementation) time_correlation : float
            Time correlation for sampling. 0 means no time correlation. 1 means perfect
            time correlation (event happen togther). -1 means no time correlation (events)
            are not anticorrelated in time.

        Returns
        -------
        Hazard.hazard.timeries.HazardTimeSeries
            sampled Hazard time series instance
        """

        if seed is not None:
            np.random.seed(seed)

        # select only nonzero hazard events
        hazard = hazard.select(
            event_id=hazard.event_id[np.where(hazard.intensity.sum(axis=1) > 0)[0]]
        )

        # compute mean frequency per timestep
        if hazard.frequency_unit not in ["1/year", "annual", "1/y", "1/a"]:
            raise ValueError(
                "Frequency unit  of Hazard object must be '1/year'. Please convert the frequncies"
                "of the object to yearly frequencies and adapt the frequency unit.",
                hazard.frequency_unit,
            )
        frequency_per_year = sum(hazard.frequency)
        # TBD does not work for only one timestep
        average_timestep_size = (timesteps[-1] - timesteps[0]) / (len(timesteps) - 1)
        mean_frequency_per_timestep = frequency_per_year / 365 * average_timestep_size

        # sample n_event per timestep
        n_events_per_timestep = _sample_independent_nevents_per_time_series_bin(
            n_timeseries,
            len(timesteps),
            mean_frequency_per_timestep,
            weights=seasonality,
        )

        # generate date arrays
        date = [
            list([timestep] * n_events_per_timestep[i_timeseries, i_timestep])
            for i_timeseries in range(n_timeseries)
            for i_timestep, timestep in enumerate(timesteps)
        ]
        date = np.array([item for sublist in date for item in sublist])

        # generate id_timeseries array
        n_events_per_timeseries = np.sum(n_events_per_timestep, axis=1)
        id_timeseries = [
            [i_timeseries] * n_events_per_timeseries[i_timeseries]
            for i_timeseries in range(n_timeseries)
        ]
        id_timeseries = np.array(
            [item for sublist in id_timeseries for item in sublist]
        )
        event_name = [
            f"timeseries{j}_hazard{event_id}"
            for j in range(n_timeseries)
            for event_id in range(n_events_per_timeseries[j])
        ]

        # sample from hazard events
        n_total_events = np.sum(n_events_per_timeseries)
        id_hazard = hazard.event_id[
            np.random.choice(
                range(hazard.size),
                n_total_events,
                p=hazard.frequency / hazard.frequency.sum(),
            )
        ]

        sampled_hazard = [hazard.select(event_id=[j]) for j in id_hazard]
        sampled_hazard = Hazard.concat(sampled_hazard)  # TBD check if ids coincide

        return cls(
            intensity=sampled_hazard.intensity,
            fraction=sampled_hazard.fraction,
            date=date,
            frequency=np.ones(n_total_events) / n_timeseries,
            event_name=event_name,
            event_id=np.arange(n_total_events),
            haz_type=sampled_hazard.haz_type,
            frequency_unit=sampled_hazard.frequency_unit,
            centroids=sampled_hazard.centroids,
            units=sampled_hazard.units,
            # timeseries attributes
            id_timeseries=id_timeseries,
            timesteps=timesteps,
        )


def _sample_independent_nevents_per_time_series_bin(
    n_timeseries,
    n_timesteps,
    mean_frequency,
    weights=None,
):
    """sample number of events per timestep including seasonaily, for n_timeseries time series

    Parameters
    ----------
    n_timeseries : int
        how many time series to sample
    n_timesteps : int
        how many timesteps per time series
    mean_frequency : float
        mean frequency of occurence per timestep
    weights : iterable[float] or None, optional
        1-D array of weights to adapt frequency of single time steps (to simulate seasonailty). Must
        have length n_timesteps (one weight per timestep). By default None, corresponding to balanced weights.

    Returns
    -------
    np.array
        2D array with the number of events per time series (0th dimension) and per timestep (1st dimension)
    """

    if weights is not None:
        if len(weights) != n_timesteps:
            raise ValueError(
                f"Number of timesteps {n_timesteps} must be equal to the length of weights {len(weights)}."
            )
        normalized_weights = np.array(weights) / sum(weights) * len(weights)
        frequency_per_step = normalized_weights * mean_frequency
    else:
        frequency_per_step = np.full(n_timesteps, mean_frequency)

    return np.random.poisson(
        lam=frequency_per_step, size=(n_timeseries, n_timesteps)
    ).astype("int")
