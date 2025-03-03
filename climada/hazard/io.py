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

Define Hazard IO Methods.
"""

import copy
import datetime as dt
import itertools
import logging
import pathlib
import warnings
from collections.abc import Collection
from typing import Any, Callable, Dict, Optional, Union

import h5py
import numpy as np
import pandas as pd
import rasterio
import sparse as sp
import xarray as xr
from scipy import sparse

import climada.util.constants as u_const
import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt
import climada.util.hdf5_handler as u_hdf5
from climada.hazard.centroids.centr import Centroids

LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {
    "sheet_name": {"inten": "hazard_intensity", "freq": "hazard_frequency"},
    "col_name": {
        "cen_id": "centroid_id/event_id",
        "even_id": "event_id",
        "even_dt": "event_date",
        "even_name": "event_name",
        "freq": "frequency",
        "orig": "orig_event_flag",
    },
    "col_centroids": {
        "sheet_name": "centroids",
        "col_name": {
            "cen_id": "centroid_id",
            "latitude": "lat",
            "longitude": "lon",
        },
    },
}
"""Excel variable names"""

DEF_VAR_MAT = {
    "field_name": "hazard",
    "var_name": {
        "per_id": "peril_ID",
        "even_id": "event_ID",
        "ev_name": "name",
        "freq": "frequency",
        "inten": "intensity",
        "unit": "units",
        "frac": "fraction",
        "comment": "comment",
        "datenum": "datenum",
        "orig": "orig_event_flag",
    },
    "var_cent": {
        "field_names": ["centroids", "hazard"],
        "var_name": {"cen_id": "centroid_ID", "lat": "lat", "lon": "lon"},
    },
}
"""MATLAB variable names"""

DEF_COORDS = dict(event="time", longitude="longitude", latitude="latitude")
"""Default coordinates when reading Hazard data from an xarray Dataset"""

DEF_DATA_VARS = ["fraction", "frequency", "event_id", "event_name", "date"]
"""Default keys for optional Hazard attributes when reading from an xarray Dataset"""


# pylint: disable=no-member


class HazardIO:
    """
    Contains all read/write methods of the Hazard class
    """

    def set_raster(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_raster."""
        LOGGER.warning(
            "The use of Hazard.set_raster is deprecated."
            "Use Hazard.from_raster instead."
        )
        self.__dict__ = self.__class__.from_raster(*args, **kwargs).__dict__

    @classmethod
    def from_raster(
        cls,
        files_intensity,
        files_fraction=None,
        attrs=None,
        band=None,
        haz_type=None,
        pool=None,
        src_crs=None,
        window=None,
        geometry=None,
        dst_crs=None,
        transform=None,
        width=None,
        height=None,
        resampling=rasterio.warp.Resampling.nearest,
    ):
        """Create Hazard with intensity and fraction values from raster files

        If raster files are masked, the masked values are set to 0.

        Files can be partially read using either window or geometry. Additionally, the data is
        reprojected when custom dst_crs and/or transform, width and height are specified.

        Parameters
        ----------
        files_intensity : list(str)
            file names containing intensity
        files_fraction : list(str)
            file names containing fraction
        attrs : dict, optional
            name of Hazard attributes and their values
        band : list(int), optional
            bands to read (starting at 1), default [1]
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
            Default: None, which will use the class default ('' for vanilla
            `Hazard` objects, and hard coded in some subclasses)
        pool : pathos.pool, optional
            Pool that will be used for parallel computation when applicable.
            Default: None
        src_crs : crs, optional
            source CRS. Provide it if error without it.
        window : rasterio.windows.Windows, optional
            window where data is
            extracted
        geometry : list of shapely.geometry, optional
            consider pixels only within these shapes
        dst_crs : crs, optional
            reproject to given crs
        transform : rasterio.Affine
            affine transformation to apply
        wdith : float, optional
            number of lons for transform
        height : float, optional
            number of lats for transform
        resampling : rasterio.warp.Resampling, optional
            resampling function used for reprojection to dst_crs

        Returns
        -------
        Hazard
        """
        if isinstance(files_intensity, (str, pathlib.Path)):
            files_intensity = [files_intensity]
        if isinstance(files_fraction, (str, pathlib.Path)):
            files_fraction = [files_fraction]
        if not attrs:
            attrs = {}
        else:
            attrs = cls._check_and_cast_attrs(attrs)
        if not band:
            band = [1]
        if files_fraction is not None and len(files_intensity) != len(files_fraction):
            raise ValueError(
                "Number of intensity files differs from fraction files:"
                f"{len(files_intensity)} != {len(files_fraction)}"
            )

        # List all parameters for initialization here (missing ones will be default)
        hazard_kwargs = dict()
        if haz_type is not None:
            hazard_kwargs["haz_type"] = haz_type

        centroids, meta = Centroids.from_raster_file(
            files_intensity[0],
            src_crs=src_crs,
            window=window,
            geometry=geometry,
            dst_crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
            return_meta=True,
        )

        if pool:
            chunksize = max(min(len(files_intensity) // pool.ncpus, 1000), 1)
            inten_list = pool.map(
                _values_from_raster_files,
                [[f] for f in files_intensity],
                itertools.repeat(meta),
                itertools.repeat(band),
                itertools.repeat(src_crs),
                itertools.repeat(window),
                itertools.repeat(geometry),
                itertools.repeat(dst_crs),
                itertools.repeat(transform),
                itertools.repeat(width),
                itertools.repeat(height),
                itertools.repeat(resampling),
                chunksize=chunksize,
            )
            intensity = sparse.vstack(inten_list, format="csr")
            if files_fraction is not None:
                fract_list = pool.map(
                    _values_from_raster_files,
                    [[f] for f in files_fraction],
                    itertools.repeat(meta),
                    itertools.repeat(band),
                    itertools.repeat(src_crs),
                    itertools.repeat(window),
                    itertools.repeat(geometry),
                    itertools.repeat(dst_crs),
                    itertools.repeat(transform),
                    itertools.repeat(width),
                    itertools.repeat(height),
                    itertools.repeat(resampling),
                    chunksize=chunksize,
                )
                fraction = sparse.vstack(fract_list, format="csr")
        else:
            intensity = _values_from_raster_files(
                files_intensity,
                meta=meta,
                band=band,
                src_crs=src_crs,
                window=window,
                geometry=geometry,
                dst_crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
            )
            if files_fraction is not None:
                fraction = _values_from_raster_files(
                    files_fraction,
                    meta=meta,
                    band=band,
                    src_crs=src_crs,
                    window=window,
                    geometry=geometry,
                    dst_crs=dst_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=resampling,
                )

        if files_fraction is None:
            fraction = intensity.copy()
            fraction.data.fill(1)

        hazard_kwargs.update(cls._attrs_to_kwargs(attrs, num_events=intensity.shape[0]))
        return cls(
            centroids=centroids, intensity=intensity, fraction=fraction, **hazard_kwargs
        )

    @classmethod
    def from_xarray_raster_file(
        cls, filepath: Union[pathlib.Path, str], *args, **kwargs
    ):
        """Read raster-like data from a file that can be loaded with xarray

        This wraps :py:meth:`~Hazard.from_xarray_raster` by first opening the target file
        as xarray dataset and then passing it to that classmethod. Use this wrapper as a
        simple alternative to opening the file yourself. The signature is exactly the
        same, except for the first argument, which is replaced by a file path here.

        Additional (keyword) arguments are passed to
        :py:meth:`~Hazard.from_xarray_raster`.

        Parameters
        ----------
        filepath : Path or str
            Path of the file to read with xarray. May be any file type supported by
            xarray. See https://docs.xarray.dev/en/stable/user-guide/io.html

        Returns
        -------
        hazard : climada.Hazard
            A hazard object created from the input data

        Examples
        --------

        >>> hazard = Hazard.from_xarray_raster_file("path/to/file.nc", "", "")

        Notes
        -----

        If you have specific requirements for opening a data file, prefer opening it
        yourself and using :py:meth:`~Hazard.from_xarray_raster`, following this pattern:

        >>> open_kwargs = dict(engine="h5netcdf", chunks=dict(x=-1, y="auto"))
        >>> with xarray.open_dataset("path/to/file.nc", **open_kwargs) as dset:
        ...     hazard = Hazard.from_xarray_raster(dset, "", "")
        """
        with xr.open_dataset(filepath, chunks="auto") as dset:
            return cls.from_xarray_raster(dset, *args, **kwargs)

    @classmethod
    def from_xarray_raster(
        cls,
        data: xr.Dataset,
        hazard_type: str,
        intensity_unit: str,
        *,
        intensity: str = "intensity",
        coordinate_vars: Optional[Dict[str, str]] = None,
        data_vars: Optional[Dict[str, str]] = None,
        crs: str = u_const.DEF_CRS,
        rechunk: bool = False,
    ):
        """Read raster-like data from an xarray Dataset

        This method reads data that can be interpreted using three coordinates: event,
        latitude, and longitude. The names of the coordinates to be read from the
        dataset can be specified via the ``coordinate_vars`` parameter. The data and the
        coordinates themselves may be organized in arbitrary dimensions (e.g. two
        dimensions 'year' and 'altitude' for the coordinate 'event').  See Notes and
        Examples if you want to load single-event data that does not contain an event
        dimension.

        The only required data is the intensity. For all other data, this method can
        supply sensible default values. By default, this method will try to find these
        "optional" data in the Dataset and read it, or use the default values otherwise.
        Users may specify the variables in the Dataset to be read for certain Hazard
        object entries, or may indicate that the default values should be used although
        the Dataset contains appropriate data. This behavior is controlled via the
        ``data_vars`` parameter.

        If this method succeeds, it will always return a "consistent" Hazard object,
        meaning that the object can be used in all CLIMADA operations without throwing
        an error due to missing data or faulty data types.

        Use :py:meth:`~Hazard.from_xarray_raster_file` to open a file on disk
        and load the resulting dataset with this method in one step.

        Parameters
        ----------
        data : xarray.Dataset
            The dataset to read from.
        hazard_type : str
            The type identifier of the hazard. Will be stored directly in the hazard
            object.
        intensity_unit : str
            The physical units of the intensity.
        intensity : str, optional
            Identifier of the `xarray.DataArray` containing the hazard intensity data.
        coordinate_vars : dict(str, str), optional
            Mapping from default coordinate names to coordinate names used in the data
            to read. The default is
            ``dict(event="time", longitude="longitude", latitude="latitude")``, as most
            of the commonly used hazard data happens to have a "time" attribute but no
            "event" attribute.
        data_vars : dict(str, str), optional
            Mapping from default variable names to variable names used in the data
            to read. The default names are ``fraction``, ``hazard_type``, ``frequency``,
            ``event_name``, ``event_id``, and ``date``. If these values are not set, the
            method tries to load data from the default names. If this fails, the method
            uses default values for each entry. If the values are set to empty strings
            (``""``), no data is loaded and the default values are used exclusively. See
            examples for details.

            Default values are:

            * ``date``: The ``event`` coordinate interpreted as date or ordinal, or
              ones if that fails (which will issue a warning).
            * ``fraction``: ``None``, which results in a value of 1.0 everywhere, see
              :py:meth:`Hazard.__init__` for details.
            * ``hazard_type``: Empty string
            * ``frequency``: 1.0 for every event
            * ``event_name``: String representation of the event date or empty strings
              if that fails (which will issue a warning).
            * ``event_id``: Consecutive integers starting at 1 and increasing with time

        crs : str, optional
            Identifier for the coordinate reference system of the coordinates. Defaults
            to ``EPSG:4326`` (WGS 84), defined by ``climada.util.constants.DEF_CRS``.
            See https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_user_input
            for further information on how to specify the coordinate system.
        rechunk : bool, optional
            Rechunk the dataset before flattening. This might have serious performance
            implications. Rechunking in general is expensive, but it might be less
            expensive than stacking a poorly-chunked array. One event being stored in
            one chunk would be the optimal configuration. If ``rechunk=True``, this will
            be forced by rechunking the data. Ideally, you would select the chunks in
            that manner when opening the dataset before passing it to this function.
            Defaults to ``False``.

        Returns
        -------
        hazard : climada.Hazard
            A hazard object created from the input data

        See Also
        --------
        :py:meth:`~Hazard.from_xarray_raster_file`
            Use this method if you want CLIMADA to open and read a file on disk for you.

        Notes
        -----
        * Single-valued coordinates given by ``coordinate_vars``, that are not proper
          dimensions of the data, are promoted to dimensions automatically. If one of the
          three coordinates does not exist, use ``Dataset.expand_dims`` (see
          https://docs.xarray.dev/en/stable/generated/xarray.Dataset.expand_dims.html
          and Examples) before loading the Dataset as Hazard.
        * Single-valued data for variables ``frequency``. ``event_name``, and
          ``event_date`` will be broadcast to every event.
        * The ``event`` coordinate may take arbitrary values. In case these values
          cannot be interpreted as dates or date ordinals, the default values for
          ``Hazard.date`` and ``Hazard.event_name`` are used, see the
          ``data_vars``` parameter documentation above.
        * To avoid confusion in the call signature, several parameters are keyword-only
          arguments.
        * The attributes ``Hazard.haz_type`` and ``Hazard.unit`` currently cannot be
          read from the Dataset. Use the method parameters to set these attributes.
        * This method does not read coordinate system metadata. Use the ``crs`` parameter
          to set a custom coordinate system identifier.

        Examples
        --------
        The use of this method is straightforward if the Dataset contains the data with
        expected names.

        >>> dset = xr.Dataset(
        ...     dict(
        ...         intensity=(
        ...             ["time", "latitude", "longitude"],
        ...             [[[0, 1, 2], [3, 4, 5]]],
        ...         )
        ...     ),
        ...     dict(
        ...         time=[datetime.datetime(2000, 1, 1)],
        ...         latitude=[0, 1],
        ...         longitude=[0, 1, 2],
        ...     ),
        ... )
        >>> hazard = Hazard.from_xarray_raster(dset, "", "")

        For non-default coordinate names, use the ``coordinate_vars`` argument.

        >>> dset = xr.Dataset(
        ...     dict(
        ...         intensity=(
        ...             ["day", "lat", "longitude"],
        ...             [[[0, 1, 2], [3, 4, 5]]],
        ...         )
        ...     ),
        ...     dict(
        ...         day=[datetime.datetime(2000, 1, 1)],
        ...         lat=[0, 1],
        ...         longitude=[0, 1, 2],
        ...     ),
        ... )
        >>> hazard = Hazard.from_xarray_raster(
        ...     dset, "", "", coordinate_vars=dict(event="day", latitude="lat")
        ... )

        Coordinates can be different from the actual dataset dimensions. The following
        loads the data with coordinates ``longitude`` and ``latitude`` (default names):

        >>> dset = xr.Dataset(
        ...     dict(intensity=(["time", "y", "x"], [[[0, 1, 2], [3, 4, 5]]])),
        ...     dict(
        ...         time=[datetime.datetime(2000, 1, 1)],
        ...         y=[0, 1],
        ...         x=[0, 1, 2],
        ...         longitude=(["y", "x"], [[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]),
        ...         latitude=(["y", "x"], [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
        ...     ),
        ... )
        >>> hazard = Hazard.from_xarray_raster(dset, "", "")

        Optional data is read from the dataset if the default keys are found. Users can
        specify custom variables in the data, or that the default keys should be ignored,
        with the ``data_vars`` argument.

        >>> dset = xr.Dataset(
        ...     dict(
        ...         intensity=(
        ...             ["time", "latitude", "longitude"],
        ...             [[[0, 1, 2], [3, 4, 5]]],
        ...         ),
        ...         fraction=(
        ...             ["time", "latitude", "longitude"],
        ...             [[[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]]],
        ...         ),
        ...         freq=(["time"], [0.4]),
        ...         event_id=(["time"], [4]),
        ...     ),
        ...     dict(
        ...         time=[datetime.datetime(2000, 1, 1)],
        ...         latitude=[0, 1],
        ...         longitude=[0, 1, 2],
        ...     ),
        ... )
        >>> hazard = Hazard.from_xarray_raster(
        ...     dset,
        ...     "",
        ...     "",
        ...     data_vars=dict(
        ...         # Load frequency from 'freq' array
        ...         frequency="freq",
        ...         # Ignore 'event_id' array and use default instead
        ...         event_id="",
        ...         # 'fraction' array is loaded because it has the default name
        ...     ),
        ... )
        >>> np.array_equal(hazard.frequency, [0.4]) and np.array_equal(
        ...     hazard.event_id, [1]
        ... )
        True

        If your read single-event data your dataset probably will not have a time
        dimension. As long as a time *coordinate* exists, however, this method will
        automatically promote it to a dataset dimension and load the data:

        >>> dset = xr.Dataset(
        ...     dict(
        ...         intensity=(
        ...             ["latitude", "longitude"],
        ...             [[0, 1, 2], [3, 4, 5]],
        ...         )
        ...     ),
        ...     dict(
        ...         time=[datetime.datetime(2000, 1, 1)],
        ...         latitude=[0, 1],
        ...         longitude=[0, 1, 2],
        ...     ),
        ... )
        >>> hazard = Hazard.from_xarray_raster(dset, "", "")  # Same as first example

        If one coordinate is missing altogehter, you must add it or expand the dimensions
        before loading the dataset:

        >>> dset = xr.Dataset(
        ...     dict(
        ...         intensity=(
        ...             ["latitude", "longitude"],
        ...             [[0, 1, 2], [3, 4, 5]],
        ...         )
        ...     ),
        ...     dict(
        ...         latitude=[0, 1],
        ...         longitude=[0, 1, 2],
        ...     ),
        ... )
        >>> dset = dset.expand_dims(time=[numpy.datetime64("2000-01-01")])
        >>> hazard = Hazard.from_xarray_raster(dset, "", "")
        """
        # Check data type for better error message
        if not isinstance(data, xr.Dataset):
            if isinstance(data, (pathlib.Path, str)):
                raise TypeError(
                    "Passing a path to this classmethod is not supported. "
                    "Use Hazard.from_xarray_raster_file instead."
                )

            raise TypeError("This method only supports xarray.Dataset as input data")

        # Initialize Hazard object
        hazard_kwargs = dict(haz_type=hazard_type, units=intensity_unit)

        # Update coordinate identifiers
        coords = copy.deepcopy(DEF_COORDS)
        coordinate_vars = coordinate_vars if coordinate_vars is not None else {}
        unknown_coords = [co for co in coordinate_vars if co not in coords]
        if unknown_coords:
            raise ValueError(
                f"Unknown coordinates passed: '{unknown_coords}'. Supported "
                f"coordinates are {list(coords.keys())}."
            )
        coords.update(coordinate_vars)

        # Retrieve dimensions of coordinates
        try:
            dims = dict(
                event=data[coords["event"]].dims,
                longitude=data[coords["longitude"]].dims,
                latitude=data[coords["latitude"]].dims,
            )
        # Handle KeyError for better error message
        except KeyError as err:
            key = err.args[0]
            raise RuntimeError(
                f"Dataset is missing dimension/coordinate: {key}. Dataset dimensions: "
                f"{list(data.dims.keys())}"
            ) from err

        # Try promoting single-value coordinates to dimensions
        for key, val in dims.items():
            if not val:
                coord = coords[key]
                LOGGER.debug("Promoting Dataset coordinate '%s' to dimension", coord)
                data = data.expand_dims(coord)
                dims[key] = data[coord].dims

        # Try to rechunk the data to optimize the stack operation afterwards.
        if rechunk:
            # We want one event to be contained in one chunk
            chunks = {dim: -1 for dim in dims["longitude"]}
            chunks.update({dim: -1 for dim in dims["latitude"]})

            # Chunks can be auto-sized along the event dimensions
            chunks.update({dim: "auto" for dim in dims["event"]})
            data = data.chunk(chunks=chunks)

        # Stack (vectorize) the entire dataset into 2D (time, lat/lon)
        # NOTE: We want the set union of the dimensions, but Python 'set' does not
        #       preserve order. However, we want longitude to run faster than latitude.
        #       So we use 'dict' without values, as 'dict' preserves insertion order
        #       (dict keys behave like a set).
        data = data.stack(
            event=dims["event"],
            lat_lon=dict.fromkeys(dims["latitude"] + dims["longitude"]),
        )

        # Transform coordinates into centroids
        centroids = Centroids(
            lat=data[coords["latitude"]].values,
            lon=data[coords["longitude"]].values,
            crs=crs,
        )

        def to_csr_matrix(array: xr.DataArray) -> sparse.csr_matrix:
            """Store a numpy array as sparse matrix, optimizing storage space

            The CSR matrix stores NaNs explicitly, so we set them to zero.
            """
            array = array.where(array.notnull(), 0)
            array = xr.apply_ufunc(
                sp.COO.from_numpy,
                array,
                dask="parallelized",
                output_dtypes=[array.dtype],
            )
            sparse_coo = array.compute().data  # Load into memory
            return sparse_coo.tocsr()  # Convert sparse.COO to scipy.sparse.csr_matrix

        # Read the intensity data
        LOGGER.debug("Loading Hazard intensity from DataArray '%s'", intensity)
        intensity_matrix = to_csr_matrix(data[intensity])

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
            return array.values

        def date_to_ordinal_accessor(
            array: xr.DataArray, strict: bool = True
        ) -> np.ndarray:
            """Take a DataArray and transform it into ordinals"""
            try:
                if np.issubdtype(array.dtype, np.integer):
                    # Assume that data is ordinals
                    return strict_positive_int_accessor(array)

                # Try transforming to ordinals
                return np.array(u_dt.datetime64_to_ordinal(array.values))

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

        def year_month_day_accessor(
            array: xr.DataArray, strict: bool = True
        ) -> np.ndarray:
            """Take an array and return am array of YYYY-MM-DD strings"""
            try:
                return array.dt.strftime("%Y-%m-%d").values

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

        # Create a DataFrame storing access information for each of data_vars
        # NOTE: Each row will be passed as arguments to
        #       `load_from_xarray_or_return_default`, see its docstring for further
        #       explanation of the DataFrame columns / keywords.
        num_events = data.sizes["event"]
        data_ident = pd.DataFrame(
            data=dict(
                # The attribute of the Hazard class where the data will be stored
                hazard_attr=DEF_DATA_VARS,
                # The identifier and default key used in this method
                default_key=DEF_DATA_VARS,
                # The key assigned by the user
                user_key=None,
                # The default value for each attribute
                default_value=[
                    None,
                    np.ones(num_events),
                    np.array(range(num_events), dtype=int) + 1,
                    list(
                        year_month_day_accessor(
                            data[coords["event"]], strict=False
                        ).flat
                    ),
                    date_to_ordinal_accessor(data[coords["event"]], strict=False),
                ],
                # The accessor for the data in the Dataset
                accessor=[
                    to_csr_matrix,
                    lambda x: maybe_repeat(default_accessor(x), num_events),
                    strict_positive_int_accessor,
                    lambda x: list(maybe_repeat(default_accessor(x), num_events).flat),
                    lambda x: maybe_repeat(date_to_ordinal_accessor(x), num_events),
                ],
            )
        )

        # Check for unexpected keys
        data_vars = data_vars if data_vars is not None else {}
        default_keys = data_ident["default_key"]
        unknown_keys = [
            key for key in data_vars.keys() if not default_keys.str.contains(key).any()
        ]
        if unknown_keys:
            raise ValueError(
                f"Unknown data variables passed: '{unknown_keys}'. Supported "
                f"data variables are {list(default_keys)}."
            )

        # Update with keys provided by the user
        # NOTE: Keys in 'default_keys' missing from 'data_vars' will be set to 'None'
        #       (which is exactly what we want) and the result is written into
        #       'user_key'. 'default_keys' is not modified.
        data_ident["user_key"] = default_keys.map(data_vars)

        def load_from_xarray_or_return_default(
            user_key: Optional[str],
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
                A callable that takes the DataArray as argument and returns the data
                structure that is required by the ``Hazard`` attribute.
            default_value
                The default value/array to return in case the data could not be found.

            Returns
            -------
            The object that will be stored in the ``Hazard`` attribute ``hazard_attr``.

            Raises
            ------
            KeyError
                If ``user_key`` was a non-empty string but no such key was found in the
                data
            RuntimeError
                If the data structure loaded has a different shape than the default data
                structure
            """
            # User does not want to read data
            if user_key == "":
                LOGGER.debug(
                    "Using default values for Hazard.%s per user request", hazard_attr
                )
                return default_value

            if not pd.isna(user_key):
                # Read key exclusively
                LOGGER.debug(
                    "Reading data for Hazard.%s from DataArray '%s'",
                    hazard_attr,
                    user_key,
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

        # Set the Hazard attributes
        for _, ident in data_ident.iterrows():
            hazard_kwargs[ident["hazard_attr"]] = load_from_xarray_or_return_default(
                **ident
            )

        hazard_kwargs = cls._check_and_cast_attrs(hazard_kwargs)

        # Done!
        LOGGER.debug("Hazard successfully loaded. Number of events: %i", num_events)
        return cls(centroids=centroids, intensity=intensity_matrix, **hazard_kwargs)

    @staticmethod
    def _check_and_cast_attrs(attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Check the validity of the hazard attributes given and cast to correct type if required and possible.

        The current purpose is to check that event_name is a list of string
        (and convert to string otherwise), although other checks and casting could be included here in the future.

        Parameters
        ----------

        attrs : dict
            Attributes for a new Hazard object

        Returns
        -------

        attrs : dict
            Attributes checked for type validity and casted otherwise (only event_name at the moment).

        Warns
        -----

        UserWarning
            Warns the user if any value casting happens.
        """

        def _check_and_cast_container(
            attr_value: Any, expected_container: Collection
        ) -> Any:
            """Check if the attribute is of the expected container type and cast if necessary.

            Parameters
            ----------
            attr_value : any
                The current value of the attribute.

            expected_container : type
                The expected type of the container (e.g., list, np.ndarray).

            Returns
            -------
            attr_value : any
                The value cast to the expected container type, if needed.
            """
            if not isinstance(attr_value, expected_container):
                warnings.warn(
                    f"Value should be of type {expected_container}. Casting it.",
                    UserWarning,
                )
                # Attempt to cast to the expected container type
                if expected_container is list:
                    return list(attr_value)
                elif expected_container is np.ndarray:
                    return np.array(attr_value)
                else:
                    raise TypeError(f"Unsupported container type: {expected_container}")
            return attr_value

        def _check_and_cast_elements(
            attr_value: Any, expected_dtype: Union[Any, None]
        ) -> Any:
            """Check if the elements of the container are of the expected dtype and cast if necessary,
            while preserving the original container type.

            Parameters
            ----------
            attr_value : any
                The current value of the attribute (a container).

            expected_dtype : type or None
                The expected type of the elements within the container. If None, no casting is done.

            Returns
            -------
            attr_value : any
                The value with elements cast to the expected type, preserving the original container type.
            """
            if expected_dtype is None:
                # No dtype enforcement required
                return attr_value

            container_type = type(attr_value)  # Preserve the original container type

            # Perform type checking and casting of elements
            if isinstance(attr_value, (list, np.ndarray)):
                if not all(isinstance(val, expected_dtype) for val in attr_value):
                    warnings.warn(
                        f"Not all values are of type {expected_dtype}. Casting values.",
                        UserWarning,
                    )
                    casted_values = [expected_dtype(val) for val in attr_value]
                    # Return the casted values in the same container type
                    if container_type is list:
                        return casted_values
                    elif container_type is np.ndarray:
                        return np.array(casted_values)
                    else:
                        raise TypeError(f"Unsupported container type: {container_type}")
            else:
                raise TypeError(
                    f"Expected a container (e.g., list or ndarray), got {type(attr_value)} instead."
                )

            return attr_value

        ## This should probably be defined as a CONSTANT?
        attrs_to_check = {"event_name": (list, str), "event_id": (np.ndarray, None)}

        for attr_name, (expected_container, expected_dtype) in attrs_to_check.items():
            attr_value = attrs.get(attr_name)

            if attr_value is not None:
                # Check and cast the container type
                attr_value = _check_and_cast_container(attr_value, expected_container)

                # Check and cast the element types (if applicable)
                attr_value = _check_and_cast_elements(attr_value, expected_dtype)

                # Update the attrs dictionary with the modified value
                attrs[attr_name] = attr_value

        return attrs

    @staticmethod
    def _attrs_to_kwargs(attrs: Dict[str, Any], num_events: int) -> Dict[str, Any]:
        """Transform attributes to init kwargs or use default values

        If attributes are missing from ``attrs``, this method will use a sensible default
        value.

        Parameters
        ----------
        attrs : dict
            Attributes for a new Hazard object
        num_events : int
            Number of events stored in a new Hazard object. Used for determining default
            values if Hazard object attributes are missing from ``attrs``.

        Returns
        -------
        kwargs : dict
            Keywords arguments to be passed to a Hazard constructor
        """

        kwargs = dict()

        if "event_id" in attrs:
            kwargs["event_id"] = attrs["event_id"]
        else:
            kwargs["event_id"] = np.arange(1, num_events + 1)
        if "frequency" in attrs:
            kwargs["frequency"] = attrs["frequency"]
        else:
            kwargs["frequency"] = np.ones(kwargs["event_id"].size)
        if "frequency_unit" in attrs:
            kwargs["frequency_unit"] = attrs["frequency_unit"]
        if "event_name" in attrs:
            kwargs["event_name"] = attrs["event_name"]
        else:
            kwargs["event_name"] = list(map(str, kwargs["event_id"]))
        if "date" in attrs:
            kwargs["date"] = np.array([attrs["date"]])
        else:
            kwargs["date"] = np.ones(kwargs["event_id"].size)
        if "orig" in attrs:
            kwargs["orig"] = np.array([attrs["orig"]])
        else:
            kwargs["orig"] = np.ones(kwargs["event_id"].size, bool)
        if "unit" in attrs:
            kwargs["units"] = attrs["unit"]

        return kwargs

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_excel."""
        LOGGER.warning(
            "The use of Hazard.read_excel is deprecated."
            "Use Hazard.from_excel instead."
        )
        self.__dict__ = self.__class__.from_excel(*args, **kwargs).__dict__

    @classmethod
    def from_excel(cls, file_name, var_names=None, haz_type=None):
        """Read climada hazard generated with the MATLAB code in Excel format.

        Parameters
        ----------
        file_name : str
            absolute file name
        var_names (dict, default): name of the variables in the file,
            default: DEF_VAR_EXCEL constant
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
            Default: None, which will use the class default ('' for vanilla `Hazard` objects, and
            hard coded in some subclasses)

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard object from the provided Excel file

        Raises
        ------
        KeyError
        """
        # pylint: disable=protected-access
        if not var_names:
            var_names = DEF_VAR_EXCEL
        LOGGER.info("Reading %s", file_name)
        hazard_kwargs = {}
        if haz_type is not None:
            hazard_kwargs["haz_type"] = haz_type
        try:
            centroids = Centroids._legacy_from_excel(
                file_name, var_names=var_names["col_centroids"]
            )
            attrs = cls._read_att_excel(file_name, var_names, centroids)
            attrs = cls._check_and_cast_attrs(attrs)
            hazard_kwargs.update(attrs)
        except KeyError as var_err:
            raise KeyError("Variable not in Excel file: " + str(var_err)) from var_err

        return cls(centroids=centroids, **hazard_kwargs)

    def write_raster(self, file_name, variable="intensity", output_resolution=None):
        """Write intensity or fraction as GeoTIFF file. Each band is an event.
        Output raster is always a regular grid (same resolution for lat/lon).

        Note that if output_resolution is not None, the data is rasterized to that
        resolution. This is an expensive operation. For hazards that are already
        a raster, output_resolution=None saves on this raster which is efficient.

        If you want to save both fraction and intensity, create two separate files.
        These can then be read together with the from_raster method.

        Parameters
        ----------
        file_name: str
            file name to write in tif format
        variable: str
            if 'intensity', write intensity, if 'fraction' write fraction.
            Default is 'intensity'
        output_resolution: int
            If not None, the data is rasterized to this resolution.
            Default is None (resolution is estimated from the data).

        See also
        --------
        from_raster:
            method to read intensity and fraction raster files.
        """

        if variable == "intensity":
            var_to_write = self.intensity
        elif variable == "fraction":
            var_to_write = self.fraction
        else:
            raise ValueError(
                f"The variable {variable} is not valid. Please use 'intensity' or 'fraction'."
            )

        meta = self.centroids.get_meta(resolution=output_resolution)
        meta.update(driver="GTiff", dtype=rasterio.float32, count=self.size)
        res = meta["transform"][0]  # resolution from lon coordinates

        if meta["height"] * meta["width"] == self.centroids.size:
            # centroids already in raster format
            u_coord.write_raster(file_name, var_to_write.toarray(), meta)
        else:
            geometry = self.centroids.get_pixel_shapes(res=res)
            with rasterio.open(file_name, "w", **meta) as dst:
                LOGGER.info("Writing %s", file_name)
                for i_ev in range(self.size):
                    raster = rasterio.features.rasterize(
                        (
                            (geom, value)
                            for geom, value in zip(
                                geometry, var_to_write[i_ev].toarray().flatten()
                            )
                        ),
                        out_shape=(meta["height"], meta["width"]),
                        transform=meta["transform"],
                        fill=0,
                        all_touched=True,
                        dtype=meta["dtype"],
                    )
                    dst.write(raster.astype(meta["dtype"]), i_ev + 1)

    def write_hdf5(self, file_name, todense=False):
        """Write hazard in hdf5 format.

        Parameters
        ----------
        file_name: str
            file name to write, with h5 format
        todense: bool
            if True write the sparse matrices as hdf5.dataset by converting them to dense format
            first. This increases readability of the file for other programs. default: False
        """
        LOGGER.info("Writing %s", file_name)
        with h5py.File(file_name, "w") as hf_data:
            str_dt = h5py.special_dtype(vlen=str)
            for var_name, var_val in self.__dict__.items():
                if var_name == "event_name":
                    if not all((isinstance(val, str) for val in var_val)):
                        raise TypeError("'event_name' must be a list of strings")
                if var_name == "centroids":
                    # Centroids have their own write_hdf5 method,
                    # which is invoked at the end of this method (s.b.)
                    continue
                elif isinstance(var_val, sparse.csr_matrix):
                    if todense:
                        hf_data.create_dataset(var_name, data=var_val.toarray())
                    else:
                        hf_csr = hf_data.create_group(var_name)
                        hf_csr.create_dataset("data", data=var_val.data)
                        hf_csr.create_dataset("indices", data=var_val.indices)
                        hf_csr.create_dataset("indptr", data=var_val.indptr)
                        hf_csr.attrs["shape"] = var_val.shape
                elif isinstance(var_val, str):
                    hf_str = hf_data.create_dataset(var_name, (1,), dtype=str_dt)
                    hf_str[0] = var_val
                elif (
                    isinstance(var_val, list)
                    and var_val
                    and isinstance(var_val[0], str)
                ):
                    hf_str = hf_data.create_dataset(
                        var_name, (len(var_val),), dtype=str_dt
                    )
                    for i_ev, var_ev in enumerate(var_val):
                        hf_str[i_ev] = var_ev
                elif var_val is not None and var_name != "pool":
                    try:
                        hf_data.create_dataset(var_name, data=var_val)
                    except TypeError:
                        LOGGER.warning(
                            "write_hdf5: the class member %s is skipped, due to its "
                            "type, %s, for which writing to hdf5 "
                            "is not implemented. Reading this H5 file will probably lead to "
                            "%s being set to its default value.",
                            var_name,
                            var_val.__class__.__name__,
                            var_name,
                        )
        self.centroids.write_hdf5(file_name, mode="a")

    def read_hdf5(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_hdf5."""
        LOGGER.warning(
            "The use of Hazard.read_hdf5 is deprecated." "Use Hazard.from_hdf5 instead."
        )
        self.__dict__ = self.__class__.from_hdf5(*args, **kwargs).__dict__

    @classmethod
    def from_hdf5(cls, file_name):
        """Read hazard in hdf5 format.

        Parameters
        ----------
        file_name: str
            file name to read, with h5 format

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard object from the provided MATLAB file

        """
        LOGGER.info("Reading %s", file_name)
        # NOTE: This is a stretch. We instantiate one empty object to iterate over its
        #       attributes. But then we create a new one with the attributes filled!
        haz = cls()
        hazard_kwargs = dict()
        with h5py.File(file_name, "r") as hf_data:
            for var_name, var_val in haz.__dict__.items():
                if var_name not in hf_data.keys():
                    continue
                if var_name == "centroids":
                    continue
                if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                    hazard_kwargs[var_name] = np.array(hf_data.get(var_name))
                elif isinstance(var_val, sparse.csr_matrix):
                    hf_csr = hf_data.get(var_name)
                    if isinstance(hf_csr, h5py.Dataset):
                        hazard_kwargs[var_name] = sparse.csr_matrix(hf_csr)
                    else:
                        hazard_kwargs[var_name] = sparse.csr_matrix(
                            (
                                hf_csr["data"][:],
                                hf_csr["indices"][:],
                                hf_csr["indptr"][:],
                            ),
                            hf_csr.attrs["shape"],
                        )
                elif isinstance(var_val, str):
                    hazard_kwargs[var_name] = u_hdf5.to_string(hf_data.get(var_name)[0])
                elif isinstance(var_val, list):
                    hazard_kwargs[var_name] = [
                        x
                        for x in map(
                            u_hdf5.to_string, np.array(hf_data.get(var_name)).tolist()
                        )
                    ]
                else:
                    hazard_kwargs[var_name] = hf_data.get(var_name)
        hazard_kwargs["centroids"] = Centroids.from_hdf5(file_name)
        hazard_kwargs = cls._check_and_cast_attrs(hazard_kwargs)
        # Now create the actual object we want to return!
        return cls(**hazard_kwargs)

    @staticmethod
    def _read_att_mat(data, file_name, var_names, centroids):
        """Read MATLAB hazard's attributes."""
        attrs = dict()
        attrs["frequency"] = np.squeeze(data[var_names["var_name"]["freq"]])
        try:
            attrs["frequency_unit"] = u_hdf5.get_string(
                data[var_names["var_name"]["freq_unit"]]
            )
        except KeyError:
            pass
        attrs["orig"] = np.squeeze(data[var_names["var_name"]["orig"]]).astype(bool)
        attrs["event_id"] = np.squeeze(
            data[var_names["var_name"]["even_id"]].astype(int, copy=False)
        )
        try:
            attrs["units"] = u_hdf5.get_string(data[var_names["var_name"]["unit"]])
        except KeyError:
            pass

        n_cen = centroids.size
        n_event = len(attrs["event_id"])
        try:
            attrs["intensity"] = u_hdf5.get_sparse_csr_mat(
                data[var_names["var_name"]["inten"]], (n_event, n_cen)
            )
        except ValueError as err:
            raise ValueError("Size missmatch in intensity matrix.") from err
        try:
            attrs["fraction"] = u_hdf5.get_sparse_csr_mat(
                data[var_names["var_name"]["frac"]], (n_event, n_cen)
            )
        except ValueError as err:
            raise ValueError("Size missmatch in fraction matrix.") from err
        except KeyError:
            attrs["fraction"] = sparse.csr_matrix(
                np.ones(attrs["intensity"].shape, dtype=float)
            )
        # Event names: set as event_id if no provided
        try:
            attrs["event_name"] = u_hdf5.get_list_str_from_ref(
                file_name, data[var_names["var_name"]["ev_name"]]
            )
        except KeyError:
            attrs["event_name"] = list(attrs["event_id"])

        try:
            datenum = data[var_names["var_name"]["datenum"]].squeeze()
            attrs["date"] = np.array(
                [
                    (
                        dt.datetime.fromordinal(int(date))
                        + dt.timedelta(days=date % 1)
                        - dt.timedelta(days=366)
                    ).toordinal()
                    for date in datenum
                ]
            )
        except KeyError:
            pass

        return attrs

    @staticmethod
    def _read_att_excel(file_name, var_names, centroids):
        """Read Excel hazard's attributes."""
        dfr = pd.read_excel(file_name, var_names["sheet_name"]["freq"])

        num_events = dfr.shape[0]
        attrs = dict()
        attrs["frequency"] = dfr[var_names["col_name"]["freq"]].values
        attrs["orig"] = dfr[var_names["col_name"]["orig"]].values.astype(bool)
        attrs["event_id"] = dfr[var_names["col_name"]["even_id"]].values.astype(
            int, copy=False
        )
        attrs["date"] = dfr[var_names["col_name"]["even_dt"]].values.astype(
            int, copy=False
        )
        attrs["event_name"] = dfr[var_names["col_name"]["even_name"]].values.tolist()

        dfr = pd.read_excel(file_name, var_names["sheet_name"]["inten"])
        # number of events (ignore centroid_ID column)
        # check the number of events is the same as the one in the frequency
        if dfr.shape[1] - 1 is not num_events:
            raise ValueError(
                "Hazard intensity is given for a number of events "
                "different from the number of defined in its frequency: "
                f"{dfr.shape[1] - 1} != {num_events}"
            )
        # check number of centroids is the same as retrieved before
        if dfr.shape[0] is not centroids.size:
            raise ValueError(
                "Hazard intensity is given for a number of centroids "
                "different from the number of centroids defined: "
                f"{dfr.shape[0]} != {centroids.size}"
            )

        attrs["intensity"] = sparse.csr_matrix(
            dfr.values[:, 1 : num_events + 1].transpose()
        )
        attrs["fraction"] = sparse.csr_matrix(
            np.ones(attrs["intensity"].shape, dtype=float)
        )

        return attrs


def _values_from_raster_files(
    file_names,
    meta,
    band=None,
    src_crs=None,
    window=None,
    geometry=None,
    dst_crs=None,
    transform=None,
    width=None,
    height=None,
    resampling=rasterio.warp.Resampling.nearest,
):
    """Read raster of bands and set 0 values to the masked ones.

    Each band is an event. Select region using window or geometry. Reproject input by proving
    dst_crs and/or (transform, width, height).

    The main purpose of this function is to read intensity/fraction values from raster files for
    use in Hazard.read_raster. It is implemented as a separate helper function (instead of a
    class method) to allow for parallel computing.

    Parameters
    ----------
    file_names : str
        path of the file
    meta : dict
        description of the centroids raster
    band : list(int), optional
        band number to read. Default: [1]
    src_crs : crs, optional
        source CRS. Provide it if error without it.
    window : rasterio.windows.Window, optional
        window to read
    geometry : list of shapely.geometry, optional
        consider pixels only within these shapes
    dst_crs : crs, optional
        reproject to given crs
    transform : rasterio.Affine
        affine transformation to apply
    wdith : float
        number of lons for transform
    height : float
        number of lats for transform
    resampling : rasterio.warp,.Resampling optional
        resampling function used for reprojection to dst_crs

    Raises
    ------
    ValueError

    Returns
    -------
    inten : scipy.sparse.csr_matrix
        Each row is an event.
    """
    if band is None:
        band = [1]

    values = []
    for file_name in file_names:
        tmp_meta, data = u_coord.read_raster(
            file_name,
            band,
            src_crs,
            window,
            geometry,
            dst_crs,
            transform,
            width,
            height,
            resampling,
        )
        if (
            tmp_meta["crs"] != meta["crs"]
            or tmp_meta["transform"] != meta["transform"]
            or tmp_meta["height"] != meta["height"]
            or tmp_meta["width"] != meta["width"]
        ):
            raise ValueError("Raster data is inconsistent with contained raster.")
        values.append(sparse.csr_matrix(data))

    return sparse.vstack(values, format="csr")
