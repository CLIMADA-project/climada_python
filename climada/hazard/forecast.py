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
from typing import Any, Dict, Literal, Optional

import numpy as np
import xarray as xr
from packaging.version import Version
from scipy import sparse

import climada.util.constants as u_const
from climada.hazard.base import Hazard
from climada.hazard.xarray import HazardXarrayReader
from climada.util.checker import size
from climada.util.forecast import ForecastMixin, reduce_unique_selection

LOGGER = logging.getLogger(__name__)

XARRAY_TIMEDELTA_BUG_BEGIN = Version("2025.04.0")
XARRAY_TIMEDELTA_BUG_END = Version("2025.07.0")


def xarray_has_timedelta_bug() -> bool:
    """Return True if xarray contains the timedelta bug

    See https://docs.xarray.dev/en/stable/whats-new.html#id80
    """
    return (Version(xr.__version__) >= XARRAY_TIMEDELTA_BUG_BEGIN) and (
        Version(xr.__version__) < XARRAY_TIMEDELTA_BUG_END
    )


class HazardForecast(ForecastMixin, Hazard):
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
            py:meth:`~climada.hazard.base.Hazard.__init__` for details.
        """
        super().__init__(lead_time=lead_time, member=member, **hazard_kwargs)
        self._check_sizes()

    @classmethod
    def from_hazard(cls, hazard: Hazard, lead_time: np.ndarray, member: np.ndarray):
        """
        Create a HazardForecast object from a Hazard object.

        Parameters
        ----------
        hazard : Hazard
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
            orig=hazard.orig,
            event_name=hazard.event_name,
            date=hazard.date,
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

    def min(self, dim: Literal["member", "lead_time"] | None = None):
        """Reduce the intensity and fraction to the minimum value.

        Parameters
        ----------
        dim : "member", "lead_time", or None
            Dimension to reduce over. If None, reduce over all data.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the min intensity and fraction.

        Warning
        -------
        This reduces :py:attr:`~climada.hazard.base.Hazard.intensity`
        and :py:attr:`~climada.hazard.base.Hazard.fraction` independently, possibly
        leading to an inconsistent result: Reduced intensity and fraction values of a
        single centroid might then originate from different events!
        """
        if dim is not None:
            rdim = self._reduce_iter_dim(dim)
            return reduce_unique_selection(self, attr=rdim, reduce_method="min")

        red_intensity = self.intensity.min(axis=0).tocsr()
        red_fraction = self.fraction.min(axis=0).tocsr()
        return HazardForecast(
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            frequency_unit=self.frequency_unit,
            intensity=red_intensity,
            fraction=red_fraction,
            **self._reduce_attrs("min"),
        )

    def max(self, dim: Literal["member", "lead_time"] | None = None):
        """Reduce the intensity and fraction to the maximum value.

        Parameters
        ----------
        dim : "member", "lead_time", or None
            Dimension to reduce over. If None, reduce over all data.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the max intensity and fraction.

        Warning
        -------
        This reduces :py:attr:`~climada.hazard.base.Hazard.intensity`
        and :py:attr:`~climada.hazard.base.Hazard.fraction` independently, possibly
        leading to an inconsistent result: Reduced intensity and fraction values of a
        single centroid might then originate from different events!
        """
        if dim is not None:
            rdim = self._reduce_iter_dim(dim)
            return reduce_unique_selection(self, attr=rdim, reduce_method="max")

        red_intensity = self.intensity.max(axis=0).tocsr()
        red_fraction = self.fraction.max(axis=0).tocsr()
        return HazardForecast(
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            frequency_unit=self.frequency_unit,
            intensity=red_intensity,
            fraction=red_fraction,
            **self._reduce_attrs("max"),
        )

    def mean(self, dim: Literal["member", "lead_time"] | None = None):
        """Reduce the intensity and fraction to the mean value.

        Parameters
        ----------
        dim : "member", "lead_time", or None
            Dimension to reduce over. If None, reduce over all data.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the mean intensity and fraction.

        Warning
        -------
        This reduces :py:attr:`~climada.hazard.base.Hazard.intensity`
        and :py:attr:`~climada.hazard.base.Hazard.fraction` independently, possibly
        leading to an inconsistent result: Reduced intensity and fraction values of a
        single centroid might then originate from different events!
        """
        if dim is not None:
            rdim = self._reduce_iter_dim(dim)
            return reduce_unique_selection(self, attr=rdim, reduce_method="mean")

        red_intensity = sparse.csr_matrix(self.intensity.mean(axis=0))
        red_fraction = sparse.csr_matrix(self.fraction.mean(axis=0))
        return HazardForecast(
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            frequency_unit=self.frequency_unit,
            intensity=red_intensity,
            fraction=red_fraction,
            **self._reduce_attrs("mean"),
        )

    # TODO: Do not densify the entire matrix but compute quantiles column-wise!
    def _quantile(
        self,
        q: float,
        dim: Literal["member", "lead_time"] | None = None,
        event_name: str | None = None,
    ):
        """Reduce intensity and fraction to the quantile value."""
        if dim is not None:
            rdim = self._reduce_iter_dim(dim)
            return reduce_unique_selection(
                self,
                attr=rdim,
                reduce_method="quantile",
                q=q,
            )

        red_intensity = sparse.csr_matrix(
            np.quantile(self.intensity.toarray(), q, axis=0)
        )
        red_fraction = sparse.csr_matrix(
            np.quantile(self.fraction.toarray(), q, axis=0)
        )
        if event_name is None:
            event_name = f"quantile_{q}"
        return HazardForecast(
            haz_type=self.haz_type,
            pool=self.pool,
            units=self.units,
            centroids=self.centroids,
            frequency_unit=self.frequency_unit,
            intensity=red_intensity,
            fraction=red_fraction,
            **self._reduce_attrs(event_name),
        )

    def quantile(self, q: float, dim: Literal["member", "lead_time"] | None = None):
        """Reduce intensity and fraction to the quantile value.

        The quantile value is computed by taking the quantile of the impact matrix
        along the event dimension axis (axis=0) and then taking the quantile of the
        resulting array.

        Parameters
        ----------
        q : float
            The quantile to compute, between 0 and 1.
        dim : "member", "lead_time", or None
            The dimension to reduce when computing the quantile. If ``None`` (default),
            this computes the centroid-wise quantile over the entire matrix.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the quantile intensity and fraction.

        Warning
        -------
        This reduces :py:attr:`~climada.hazard.base.Hazard.intensity`
        and :py:attr:`~climada.hazard.base.Hazard.fraction` independently, possibly
        leading to an inconsistent result: Reduced intensity and fraction values of a
        single centroid might then originate from different events!
        """
        return self._quantile(q=q, dim=dim)

    def median(self, dim: Literal["member", "lead_time"] | None = None):
        """Reduce the intensity and fraction to the median value.

        The median value is computed by taking the median of the impact matrix along the
        event dimension axis (axis=0) and then taking the median of the resulting array.

        Parameters
        ----------
        dim : "member", "lead_time", or None
            The dimension to reduce when computing the median. If ``None`` (default),
            this computes the centroid-wise median over the entire matrix.

        Returns
        -------
        HazardForecast
            A HazardForecast object with the median intensity and fraction.

        Warning
        -------
        This reduces :py:attr:`~climada.hazard.base.Hazard.intensity`
        and :py:attr:`~climada.hazard.base.Hazard.fraction` independently, possibly
        leading to an inconsistent result: Reduced intensity and fraction values of a
        single centroid might then originate from different events!
        """
        return self._quantile(q=0.5, event_name="median", dim=dim)

    @classmethod
    def concat(cls, haz_list: list):
        """Concatenate multiple HazardForecast instances and return a new object"""
        if len(haz_list) == 0:
            return cls()
        hazard = Hazard.concat(haz_list)
        lead_time = np.concatenate(tuple(haz.lead_time for haz in haz_list))
        member = np.concatenate(tuple(haz.member for haz in haz_list))
        return cls.from_hazard(hazard, lead_time=lead_time, member=member)

    def select(
        self,
        event_names=None,
        event_id=None,
        date=None,
        orig=None,
        reg_id=None,
        extent=None,
        reset_frequency=False,
        member=None,
        lead_time=None,
    ):
        """Select entries based on the parameters and return a new instance.

        The selection will contain the intersection of all given parameters.

        Parameters
        ----------
        member : Sequence of ints
            Ensemble members to select
        lead_time : Sequence of numpy.timedelta64
            Lead times to select

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
            if event_id.size == 0:
                raise ValueError(
                    "Empty selection for 'member', 'lead_time', and/or 'event_id'"
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
        crs: str = u_const.DEF_CRS,
        rechunk: bool = False,
        open_dataset_kws: dict[str, Any] | None = None,
    ):
        """Read forecast hazard data from an xarray Dataset

        This extends the parent
        :py:meth:`~climada.hazard.io.HazardIO.from_xarray_raster` to handle forecast
        dimensions (lead_time and member). For forecast data, the "event" dimension is
        constructed from the Cartesian product of lead_time and member dimensions, so
        you don't need to specify an "event" coordinate.

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
            Mapping from default variable names to variable names used in the data
            to read. See :py:meth:`~climada.hazard.io.HazardIO.from_xarray_raster` for
            details. To load forecast dates, map ``date`` to the corresponding
            variable. Dates default to 0 when no date mapping is provided.
        crs : str, optional
            Coordinate reference system identifier. Defaults to "EPSG:4326".
        rechunk : bool, optional
            Rechunk the dataset before flattening. Defaults to ``False``. See
            :py:meth:`~climada.hazard.io.HazardIO.from_xarray_raster` for details.
        open_dataset_kws : dict, optional
            Keyword arguments passed to xarray.open_dataset if data is a file path

        Returns
        -------
        HazardForecast
            A forecast hazard object with lead_time and member attributes populated

        See Also
        --------
        :py:meth:`climada.hazard.io.HazardIO.from_xarray_raster`
            Parent method documentation for standard hazard loading
        """
        if xarray_has_timedelta_bug():
            LOGGER.warning(
                "xarray version %s contains a bug that prevents proper timedelta "
                "parsing. Consider updating to version %s or later, or downgrading to "
                "before version %s",
                xr.__version__,
                XARRAY_TIMEDELTA_BUG_END,
                XARRAY_TIMEDELTA_BUG_BEGIN,
            )

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
            intensity = str(data_var_names[0])
            LOGGER.info(
                "No intensity variable specified. Assuming intensity variable is '%s'",
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
            crs=crs,
            data_vars=data_vars,
            rechunk=rechunk,
        )

        kwargs = reader.get_hazard_kwargs() | {
            "haz_type": hazard_type,
            "units": intensity_unit,
            "lead_time": reader.data_stacked[leadtime_var].to_numpy(),
            "member": reader.data_stacked[member_var].to_numpy(),
        }

        # Generate from lead_time/member
        kwargs["event_name"] = [
            f"lt_{lt / np.timedelta64(1, 'h'):.0f}h_m_{m}"
            for lt, m in zip(kwargs["lead_time"], kwargs["member"])
        ]
        if not (data_vars or {}).get("date"):
            kwargs["date"] = np.zeros_like(kwargs["date"], dtype=int)

        # Convert to HazardForecast with forecast attributes
        return cls(**Hazard._check_and_cast_attrs(kwargs))
