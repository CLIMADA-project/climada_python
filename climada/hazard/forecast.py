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

Define Forecast variant of Hazard.
"""

import logging
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

from climada.hazard.base import Hazard
from climada.hazard.xarray import HazardXarrayReader
from climada.util.forecast import Forecast

LOGGER = logging.getLogger(__name__)


class HazardForecast(Forecast, Hazard):
    """A hazard object with forecast information"""

    def __init__(
        self,
        lead_time: np.ndarray | None = None,
        member: np.ndarray | None = None,
        **hazard_kwargs,
    ):
        """
        Initialize a HazardForecast object.

        Parameters
        ----------
        lead_time : np.ndarray of np.timedelta64 or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers as integers. Default is empty array.
        **hazard_kwargs
            keyword arguments to pass to :py:class:`~climada.hazard.base.Hazard` See
            py:meth`~climada.hazard.base.Hazard.__init__` for details.
        """
        super().__init__(lead_time=lead_time, member=member, **hazard_kwargs)

    @classmethod
    def from_hazard(cls, hazard: Hazard, lead_time: np.ndarray, member: np.ndarray):
        """
        Create a HazardForecast object from a Hazard object.

        Parameters
        ----------
        hazard : climada.hazard.base.Hazard
            Hazard object to convert into a HazardForecast.
        lead_time : np.ndarray of np.timedelta64 or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers as integers. Default is empty array.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the same attributes as the input hazard,
            but with lead_time and member attributes set from instance of HazardForecast.
        """
        return cls(
            lead_time=lead_time,
            member=member,
            haz_type=hazard.haz_type,
            pool=hazard.pool,
            units=hazard.units,
            centroids=hazard.centroids,
            event_id=hazard.event_id,
            frequency=hazard.frequency,
            frequency_unit=hazard.frequency_unit,
            event_name=hazard.event_name,
            date=hazard.date,
            orig=hazard.orig,
            intensity=hazard.intensity,
            fraction=hazard.fraction,
        )

    @classmethod
    def from_xarray_raster(
        cls,
        data: xr.Dataset | pathlib.Path | str,
        hazard_type: str,
        intensity_unit: str,
        *,
        intensity: str = "intensity",
        coordinate_vars: Optional[Dict[str, str]] = None,
        data_vars: Optional[Dict[str, str]] = None,
        crs: str = "EPSG:4326",
        rechunk: bool = False,
        open_dataset_kws: dict[str, Any] | None = None,
    ):
        """Read forecast hazard data from an xarray Dataset

        This extends the parent :py:meth:`~climada.hazard.base.Hazard.from_xarray_raster`
        to handle forecast dimensions (lead_time and member). For forecast data, the
        "event" dimension is constructed from the Cartesian product of lead_time and
        member dimensions, so you don't need to specify an "event" coordinate.

        Parameters
        ----------
        data : xarray.Dataset or Path or str
            The filepath to read the data from or the already opened dataset
        hazard_type : str
            The type identifier of the hazard
        intensity_unit : str
            The physical units of the intensity
        intensity : str, optional
            Identifier of the DataArray containing the hazard intensity data
        coordinate_vars : dict(str, str), optional
            Mapping from default coordinate names to coordinate names in the data.
            For HazardForecast, should include:

            - ``"leadtime"``: name of the lead time coordinate (required)
            - ``"members"``: name of the ensemble member coordinate (required)
            - ``"longitude"``: name of longitude coordinate (default: "longitude")
            - ``"latitude"``: name of latitude coordinate (default: "latitude")

            Note: The "event" coordinate is automatically constructed from lead_time
            and member, so it should not be specified.
        data_vars : dict(str, str), optional
            Mapping from default variable names to variable names in the data
        crs : str, optional
            Coordinate reference system identifier. Defaults to "EPSG:4326"
        rechunk : bool, optional
            Whether to rechunk the dataset before processing. Defaults to False
        open_dataset_kws : dict, optional
            Keyword arguments passed to xarray.open_dataset if data is a file path

        Returns
        -------
        HazardForecast
            A forecast hazard object with lead_time and member attributes populated

        See Also
        --------
        :py:meth:`climada.hazard.base.Hazard.from_xarray_raster`
            Parent method documentation for standard hazard loading
        """

        # Open dataset if needed

        hazard_type = "PR"
        intensity_unit = "mm/h"
        coordinate_vars = {
            "longitude": "lon",
            "latitude": "lat",
            "lead_time": "lead_time",
            "member": "eps",
            "event": "event",
        }

        if isinstance(data, (pathlib.Path, str)):
            open_dataset_kws = open_dataset_kws or {}
            open_dataset_kws = {"chunks": "auto"} | open_dataset_kws
            dset = xr.open_dataset(data)  # , **open_dataset_kws
        else:
            dset = data

        # Dynamically extract the data variable name
        data_var_names = list(dset.data_vars.keys())
        if len(data_var_names) == 0:
            raise ValueError("Dataset has no data variables")
        intensity = data_var_names[0]  # Use first data variable name

        # Extract forecast coordinates
        coordinate_vars = coordinate_vars or {}
        for key in ["lead_time", "member"]:
            if key not in coordinate_vars:
                raise ValueError(
                    f"coordinate_vars must include '{key}' key. "
                    f"Available coordinates: {list(dset.coords.keys())}"
                )
        leadtime_var = coordinate_vars["lead_time"]
        member_var = coordinate_vars["member"]

        dset = ds.assign_coords(
            event=(
                ("lead_time", "eps"),
                np.zeros((len(ds["lead_time"]), len(ds["eps"]))),
            )
        )

        dset_squeezed = dset.squeeze()

        # Prepare coordinate_vars for parent call
        # Remove forecast-specific keys that the parent doesn't understand
        parent_coord_vars = {
            k: v for k, v in coordinate_vars.items() if k not in ["member", "lead_time"]
        }
        parent_coord_vars["event"] = "event"

        reader = HazardXarrayReader(
            data=dset_squeezed,
            coordinate_vars=parent_coord_vars,
            intensity=intensity,
        )

        kwargs = reader.get_hazard_kwargs() | {
            "haz_type": hazard_type,
            "units": intensity_unit,
            "lead_time": reader.data_stacked[leadtime_var].values,
            "member": reader.data_stacked[member_var].values,
        }

        # Convert to HazardForecast with forecast attributes
        return cls(**Hazard._check_and_cast_attrs(kwargs))
