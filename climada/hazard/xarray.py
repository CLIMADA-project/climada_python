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

Hazard input using xarray
"""

import copy
import itertools
import logging
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np
import pandas as pd
import sparse as sp
import xarray as xr
from scipy import sparse

import climada.util.constants as u_const
import climada.util.dates_times as u_dt
from climada.hazard.centroids.centr import Centroids

LOGGER = logging.getLogger(__name__)

DEF_COORDS = {"event": "time", "longitude": "longitude", "latitude": "latitude"}
"""Default coordinates when reading Hazard data from an xarray Dataset"""

DEF_DATA_VARS = ["fraction", "frequency", "event_id", "event_name", "date"]
"""Default keys for optional Hazard attributes when reading from an xarray Dataset"""


def to_csr_matrix(array: xr.DataArray) -> sparse.csr_matrix:
    """Store a numpy array as sparse matrix, optimizing storage space

    The CSR matrix stores NaNs explicitly, so we set them to zero.
    """
    array = array.where(lambda x: ~np.isnan(x), 0)
    array = xr.apply_ufunc(
        sp.COO.from_numpy,
        array,
        dask="parallelized",
        output_dtypes=[array.dtype],
    )
    sparse_coo = array.compute().data  # Load into memory
    return sparse_coo.tocsr()  # Convert sparse.COO to scipy.sparse.csr_matrix


# Define accessors for xarray DataArrays
def default_accessor(array: xr.DataArray) -> np.ndarray:
    """Take a DataArray and return its numpy representation"""
    return array.values


def strict_positive_int_accessor(array: xr.DataArray) -> np.ndarray:
    """Take a positive int DataArray and return its numpy representation

    Raises
    ------
    TypeError
        If the underlying data type is not integer
    ValueError
        If any value is zero or less
    """
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"'{array.name}' data array must be integers")
    if not (array > 0).all():
        raise ValueError(f"'{array.name}' data must be larger than zero")
    return array.to_numpy()


def date_to_ordinal_accessor(array: xr.DataArray, strict: bool = True) -> np.ndarray:
    """Take a DataArray and transform it into ordinals"""
    try:
        if np.issubdtype(array.dtype, np.integer):
            # Assume that data is ordinals
            return strict_positive_int_accessor(array)

        # Try transforming to ordinals
        return np.array(u_dt.datetime64_to_ordinal(array.to_numpy()))

    # Handle access errors
    except (ValueError, TypeError, AttributeError) as err:
        if strict:
            raise err

        LOGGER.warning(
            "Failed to read values of '%s' as dates or ordinals. Hazard.date "
            "will be ones only",
            array.name,
        )
        return np.ones(array.shape)


def year_month_day_accessor(array: xr.DataArray, strict: bool = True) -> np.ndarray:
    """Take an array and return am array of YYYY-MM-DD strings"""
    try:
        return array.dt.strftime("%Y-%m-%d").to_numpy()

    # Handle access errors
    except (ValueError, TypeError, AttributeError) as err:
        if strict:
            raise err

        LOGGER.warning(
            "Failed to read values of '%s' as dates. Hazard.event_name will be "
            "empty strings",
            array.name,
        )
        return np.full(array.shape, "")


def maybe_repeat(values: np.ndarray, times: int) -> np.ndarray:
    """Return the array or repeat a single-valued array

    If ``values`` has size 1, return an array that repeats this value ``times``
    times. If the size is different, just return the array.
    """
    if values.size == 1:
        return np.array(list(itertools.repeat(values.flat[0], times)))

    return values


def load_from_xarray_or_return_default(
    data: xr.Dataset,
    user_key: str | None,
    default_key: str,
    hazard_attr: str,
    accessor: Callable[[xr.DataArray], Any],
    default_value: Any,
) -> Any:
    """Load data for a single Hazard attribute or return the default value

    Does the following based on the ``user_key``:
    * If the key is an empty string, return the default value
    * If the key is a non-empty string, load the data for that key and return it.
    * If the key is ``None``, look for the ``default_key`` in the data. If it
        exists, return that data. If not, return the default value.

    Parameters
    ----------
    user_key : str or None
        The key set by the user to identify the DataArray to read data from.
    default_key : str
        The default key identifying the DataArray to read data from.
    hazard_attr : str
        The name of the attribute of ``Hazard`` where the data will be stored in.
    accessor : Callable
        A callable that takes the DataArray as argument and returns the data structure
        that is required by the ``Hazard`` attribute.
    default_value
        The default value/array in case the data could not be found.

    Returns
    -------
    The object that will be stored in the ``Hazard`` attribute ``hazard_attr``.

    Raises
    ------
    KeyError
        If ``user_key`` was a non-empty string but no such key was found in the data
    RuntimeError
        If the data structure loaded has a different shape than the default data
        structure
    """
    # User does not want to read data
    if user_key == "":
        LOGGER.debug("Using default values for Hazard.%s per user request", hazard_attr)
        return default_value

    if not pd.isna(user_key):
        # Read key exclusively
        LOGGER.debug(
            "Reading data for Hazard.%s from DataArray '%s'", hazard_attr, user_key
        )
        val = accessor(data[user_key])
    else:
        # Try default key
        try:
            val = accessor(data[default_key])
            LOGGER.debug(
                "Reading data for Hazard.%s from DataArray '%s'",
                hazard_attr,
                default_key,
            )
        except KeyError:
            LOGGER.debug(
                "Using default values for Hazard.%s. No data found", hazard_attr
            )
            return default_value

    def vshape(array):
        """Return a shape tuple for any array-like type we use"""
        if isinstance(array, list):
            return len(array)
        if isinstance(array, sparse.csr_matrix):
            return array.get_shape()
        return array.shape

    # Check size for read data
    if default_value is not None and not np.array_equal(
        vshape(val), vshape(default_value)
    ):
        raise RuntimeError(
            f"'{user_key if user_key else default_key}' must have shape "
            f"{vshape(default_value)}, but shape is {vshape(val)}"
        )

    # Return the data
    return val


@dataclass(repr=False, eq=False)
class HazardXarrayReader:
    """A helper class for creating a Hazard object from an xarray dataset

    Initialize this class, then use :py:meth:`get_hazard_kwargs` to retrieve the kwargs
    to be passed to the :py:class:`~climada.hazard.base.Hazard` initializer.

    Attributes
    ----------
    data : xr.Dataset
        The data to be read as hazard.
    intensity : str
        The name of the variable containing the hazard intensity information.
        Default: ``"intensity"``
    coordinate_vars : dict(str, str)
        Mapping from default coordinate names to coordinate names in the dataset.
    data_vars : dict(str, str)
        Mapping from default variable names to variable names in the dataset.
    crs : str
        Coordinate reference system of the data to be read. Defaults to ``"EPSG:4326"``
        (WGS 84).
    rechunk : bool
        If ``False``, automatically rechunk the data for more efficient reads from disk.
        Default: ``False``.
    """

    data: xr.Dataset
    intensity: str = "intensity"
    coordinate_vars: InitVar[dict[str, str] | None] = field(default=None, kw_only=True)
    data_vars: dict[str, str] | None = field(default=None, kw_only=True)
    crs: str = field(default=u_const.DEF_CRS, kw_only=True)
    rechunk: bool = field(default=False, kw_only=True)

    coords: dict[str, str] = field(
        init=False, default_factory=lambda: copy.deepcopy(DEF_COORDS)
    )
    data_dims: dict[str, tuple[Hashable, ...]] = field(init=False, default_factory=dict)
    hazard_kwargs: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self, coordinate_vars):
        """Update coordinate and dimension names"""
        # Update coordinate identifiers
        coordinate_vars = coordinate_vars or {}
        unknown_coords = [co for co in coordinate_vars if co not in self.coords]
        if unknown_coords:
            raise ValueError(
                f"Unknown coordinates passed: '{unknown_coords}'. Supported "
                f"coordinates are {list(self.coords.keys())}."
            )
        self.coords.update(coordinate_vars)

        # Retrieve dimensions of coordinates
        try:
            self.data_dims = {
                "event": self.data[self.coords["event"]].dims,
                "longitude": self.data[self.coords["longitude"]].dims,
                "latitude": self.data[self.coords["latitude"]].dims,
            }
        # Handle KeyError for better error message
        except KeyError as err:
            key = err.args[0]
            raise RuntimeError(
                f"Dataset is missing dimension/coordinate: {key}. Dataset dimensions: "
                f"{list(self.data.dims.keys())}"
            ) from err

        # Check for unexpected keys
        self.data_vars = self.data_vars or {}
        default_keys = copy.deepcopy(DEF_DATA_VARS)
        unknown_keys = [key for key in self.data_vars.keys() if key not in default_keys]
        if unknown_keys:
            raise ValueError(
                f"Unknown data variables passed: '{unknown_keys}'. Supported data "
                f"variables are {default_keys}."
            )

    @classmethod
    def from_file(cls, filename: Path | str, *args, open_dataset_kws=None, **kwargs):
        """Open reader from a file"""
        open_dataset_kws = open_dataset_kws or {}
        open_dataset_kws = {"chunks": "auto"} | open_dataset_kws
        with xr.open_dataset(filename, **open_dataset_kws) as dset:
            return cls(dset, *args, **kwargs)

    def rechunk_data(self, data: xr.Dataset) -> xr.Dataset:
        """Try to rechunk the data to optimize the stack operation afterwards."""
        chunks = (
            # We want one event to be contained in one chunk
            {dim: -1 for dim in self.data_dims["longitude"]}
            | {dim: -1 for dim in self.data_dims["latitude"]}
            # Automated chunking in the event dimensions (as many as fit)
            | {dim: "auto" for dim in self.data_dims["event"]}
        )
        return data.chunk(chunks=chunks)

    def get_hazard_kwargs(self) -> dict[str, Any]:
        """Return kwargs to initialize the hazard"""
        # Shallow copy of the data
        data = self.data.copy()

        # Try promoting single-value coordinates to dimensions
        for key, val in self.data_dims.items():
            if not val:
                coord = self.coords[key]
                LOGGER.debug("Promoting Dataset coordinate '%s' to dimension", coord)
                data = data.expand_dims(coord)
                self.data_dims[key] = data[coord].dims

        # Maybe rechunk
        if self.rechunk:
            data = self.rechunk_data(data)

        # Stack (vectorize) the entire dataset into 2D (time, lat/lon)
        # NOTE: We want the set union of the dimensions, but Python 'set' does not
        #       preserve order. However, we want longitude to run faster than latitude.
        #       So we use 'dict' without values, as 'dict' preserves insertion order
        #       (dict keys behave like a set).
        data = data.stack(
            event=self.data_dims["event"],
            lat_lon=list(
                dict.fromkeys(
                    self.data_dims["latitude"] + self.data_dims["longitude"]
                ).keys()
            ),
        )

        # Transform coordinates into centroids
        centroids = Centroids(
            lat=data[self.coords["latitude"]].values,
            lon=data[self.coords["longitude"]].values,
            crs=self.crs,
        )

        # Read the intensity data
        LOGGER.debug("Loading Hazard intensity from DataArray '%s'", self.intensity)
        intensity_matrix = to_csr_matrix(data[self.intensity])

        # Create a DataFrame storing access information for each of data_vars
        # NOTE: Each row will be passed as arguments to
        #       `load_from_xarray_or_return_default`, see its docstring for further
        #       explanation of the DataFrame columns / keywords.
        num_events = data.sizes["event"]
        data_ident = pd.DataFrame(
            data={
                # The attribute of the Hazard class where the data will be stored
                "hazard_attr": DEF_DATA_VARS,
                # The identifier and default key used in this method
                "default_key": DEF_DATA_VARS,
                # The key assigned by the user
                "user_key": None,
                # The default value for each attribute
                "default_value": [
                    None,
                    np.ones(num_events),
                    np.array(range(num_events), dtype=int) + 1,
                    list(
                        year_month_day_accessor(
                            data[self.coords["event"]], strict=False
                        ).flat
                    ),
                    date_to_ordinal_accessor(data[self.coords["event"]], strict=False),
                ],
                # The accessor for the data in the Dataset
                "accessor": [
                    to_csr_matrix,
                    lambda x: maybe_repeat(default_accessor(x), num_events),
                    strict_positive_int_accessor,
                    lambda x: list(maybe_repeat(default_accessor(x), num_events).flat),
                    lambda x: maybe_repeat(date_to_ordinal_accessor(x), num_events),
                ],
            }
        )

        # Update with keys provided by the user
        # NOTE: Keys in 'default_keys' missing from 'data_vars' will be set to 'None'
        #       (which is exactly what we want) and the result is written into
        #       'user_key'. 'default_keys' is not modified.
        data_ident["user_key"] = data_ident["default_key"].map(self.data_vars)

        # Set the Hazard attributes
        for _, ident in data_ident.iterrows():
            self.hazard_kwargs[ident["hazard_attr"]] = (
                load_from_xarray_or_return_default(data=data, **ident)
            )

        # Done!
        LOGGER.debug("Hazard successfully loaded. Number of events: %i", num_events)
        self.hazard_kwargs.update(centroids=centroids, intensity=intensity_matrix)
        return self.hazard_kwargs
