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

    def check_time_series(self):
        "check time series dimension"

        # check hazard structure
        self.check()
        # check if timesteps attribute exists
        if not hasattr(self, "timesteps"):
            raise ValueError("HazardTimeSeries must have timesteps attribute")
        # check if timesteps are equally spaced
        if len(np.unique(np.diff(self.timesteps))) > 1:
            raise ValueError(
                "HazardTimeSeries must include timesteps with equal distance"
            )
        # check if event dates correspond to timesteps
        if not set(self.date).issubset(self.timesteps):
            raise ValueError("Event dates do not correspond to timesteps")

    @classmethod
    def sample_from_hazard_set(
        cls,
        hazard,
        n_timeseries,
        timesteps,
        time_correlation,
    ):
        return None
