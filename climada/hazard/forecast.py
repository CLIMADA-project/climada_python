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
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from climada.hazard.xarray import HazardXarrayReader

from ..util.checker import size
from ..util.forecast import Forecast
from .base import Hazard

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

        Notes
        -----
        If event_name or date are not provided in hazard_kwargs, they will
        be automatically generated from the Cartesian product of lead_time and member.
        Event names are formatted as "lt_{hours}h_m_{member}" where hours is the
        lead time converted to hours. Dates are set to 0 (undefined) by default
        since forecast events don't have historical dates. event_id is preserved
        from hazard_kwargs if provided.
        """
        auto_generate = (
            "event_name" not in hazard_kwargs and "date" not in hazard_kwargs
        )

        super().__init__(lead_time=lead_time, member=member, **hazard_kwargs)

        if auto_generate and len(self.lead_time) > 0 and len(self.member) > 0:
            self._set_event_attrs_from_forecast_dims()

    def _set_event_attrs_from_forecast_dims(self) -> None:
        """
        Set event_name and date from lead_time and member dimensions.

        This method generates event attributes based on the Cartesian product of
        lead_time and member arrays. It should have the same length as the number
        of events in the hazard intensity matrix. event_id is left unchanged.
        """
        n_events = len(self.lead_time)
        if n_events != len(self.member):
            raise ValueError(
                f"Length mismatch: lead_time has {len(self.lead_time)} elements "
                f"but member has {len(self.member)} elements. They should be equal "
                f"in a stacked forecast hazard."
            )

        self.event_name = [
            f"lt_{self._format_lead_time(lt)}_m_{m}"
            for lt, m in zip(self.lead_time, self.member)
        ]

        self.date = np.zeros(n_events, dtype=int)

    @staticmethod
    def _format_lead_time(lead_time: np.timedelta64) -> str:
        """
        Format lead_time as hours for event names.

        Parameters
        ----------
        lead_time : np.timedelta64
            Lead time to format

        Returns
        -------
        str
            Formatted lead time as "{hours}h"
        """
        # Convert to hours
        hours = lead_time / np.timedelta64(1, "h")
        return f"{hours:.0f}h"

        if auto_generate and len(self.lead_time) > 0 and len(self.member) > 0:
            self._set_event_attrs_from_forecast_dims()

    def _set_event_attrs_from_forecast_dims(self) -> None:
        """
        Set event_name and date from lead_time and member dimensions.

        This method generates event attributes based on the Cartesian product of
        lead_time and member arrays. It should have the same length as the number
        of events in the hazard intensity matrix. event_id is left unchanged.
        """
        n_events = len(self.lead_time)
        if n_events != len(self.member):
            raise ValueError(
                f"Length mismatch: lead_time has {len(self.lead_time)} elements "
                f"but member has {len(self.member)} elements. They should be equal "
                f"in a stacked forecast hazard."
            )

        self.event_name = [
            f"lt_{self._format_lead_time(lt)}_m_{m}"
            for lt, m in zip(self.lead_time, self.member)
        ]

        self.date = np.zeros(n_events, dtype=int)

    @staticmethod
    def _format_lead_time(lead_time: np.timedelta64) -> str:
        """
        Format lead_time as hours for event names.

        Parameters
        ----------
        lead_time : np.timedelta64
            Lead time to format

        Returns
        -------
        str
            Formatted lead time as "{hours}h"
        """
        # Convert to hours
        hours = lead_time / np.timedelta64(1, "h")
        return f"{hours:.0f}h"

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
        # Keep event_id from hazard but let event_name and date be auto-generated
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
            orig=hazard.orig,
            intensity=hazard.intensity,
            fraction=hazard.fraction,
        )

    def _check_sizes(self):
        """Check sizes of forecast data vs. hazard data.

        Raises
        ------
        ValueError
            If the sizes of the forecast data do not match the
            :py:attr:`~climada.hazard.base.Hazard.event_id`
        """
        num_entries = len(self.event_id)
        size(exp_len=num_entries, var=self.member, var_name="Forecast.member")
        size(exp_len=num_entries, var=self.lead_time, var_name="Forecast.lead_time")

    def select(
        self,
        member=None,
        lead_time=None,
        event_names=None,
        event_id=None,
        date=None,
        orig=None,
        reg_id=None,
        extent=None,
        reset_frequency=False,
    ):
        """Select entries based on the parameters and return a new instance.

        The selection will contain the intersection of all given parameters.

        Parameters
        ----------
        member : Sequence of ints
            Ensemble members to select
        lead_time : Sequence of numpy.timedelta64
            Lead times to select

        Returns
        -------
        HazardForecast

        See Also
        --------
        :py:meth:`~climada.hazard.base.Hazard.select`
        """
        if member is not None or lead_time is not None:
            mask_member = (
                self.idx_member(member)
                if member is not None
                else np.full_like(self.member, True, dtype=bool)
            )
            mask_lead_time = (
                self.idx_lead_time(lead_time)
                if lead_time is not None
                else np.full_like(self.lead_time, True, dtype=bool)
            )
            event_id_from_forecast_mask = np.asarray(self.event_id)[
                (mask_member & mask_lead_time)
            ]
            event_id = (
                np.intersect1d(event_id, event_id_from_forecast_mask)
                if event_id is not None
                else event_id_from_forecast_mask
            )

        return super().select(
            event_names=event_names,
            event_id=event_id,
            date=date,
            orig=orig,
            reg_id=reg_id,
            extent=extent,
            reset_frequency=reset_frequency,
        )

    @classmethod
    def from_xarray_raster(
        cls,
        data: xr.Dataset | pathlib.Path | str,
        hazard_type: str,
        intensity_unit: str,
        *,
        intensity: Optional[str] = None,
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
            - ``"lead_time"``: name of the lead time coordinate (required)
            - ``"member"``: name of the ensemble member coordinate (required)
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
        if isinstance(data, (pathlib.Path, str)):
            open_dataset_kws = open_dataset_kws or {}
            open_dataset_kws = {"chunks": "auto"} | open_dataset_kws
            dset = xr.open_dataset(data, **open_dataset_kws)
        else:
            dset = data

        if intensity is None:
            data_var_names = list(dset.data_vars.keys())
            if len(data_var_names) == 0:
                raise ValueError("Dataset has no data variables")
            intensity = data_var_names[0]
            LOGGER.info(
                "No intensity variable specified. "
                "Assuming intensity variable is '%s'",
                intensity,
            )

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

        dset = dset.assign_coords(
            event=(
                (leadtime_var, member_var),
                np.zeros((len(dset[leadtime_var]), len(dset[member_var]))),
            )
        )

        dset_squeezed = dset.squeeze()

        # Prepare coordinate_vars for parent call
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

        # Remove event_name and date so they get auto-generated from lead_time/member
        kwargs.pop("event_name", None)
        kwargs.pop("date", None)

        # Convert to HazardForecast with forecast attributes
        return cls(**Hazard._check_and_cast_attrs(kwargs))
