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

Define Centroids class.
"""

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Literal, Union

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from deprecation import deprecated
from pyproj.crs.crs import CRS
from shapely.geometry.point import Point

import climada.util.coordinates as u_coord
from climada.util.constants import DEF_CRS

__all__ = ["Centroids"]

PROJ_CEA = CRS.from_user_input({"proj": "cea"})

LOGGER = logging.getLogger(__name__)

DEF_SHEET_NAME = "centroids"
EXP_SPECIFIC_COLS = ["value", "impf_", "centr_", "cover", "deductible"]


class Centroids:
    """Contains vector centroids as a GeoDataFrame

    Attributes
    ----------
    lat : np.array
        Latitudinal coordinates in the specified CRS (can be any unit).
    lon : np.array
        Longitudinal coordinates in the specified CRS (can be any unit).
    crs : pyproj.CRS
        Coordinate reference system. Default: EPSG:4326 (WGS84)
    region_id : np.array, optional
        Numeric country (or region) codes. Default: None
    on_land : np.array, optional
        Boolean array indicating on land (True) or off shore (False). Default: None
    """

    def __init__(
        self,
        *,
        lat: Union[np.ndarray, list[float]],
        lon: Union[np.ndarray, list[float]],
        crs: Any = DEF_CRS,
        region_id: Union[Literal["country"], None, np.ndarray, list[float]] = None,
        on_land: Union[Literal["natural_earth"], None, np.ndarray, list[bool]] = None,
        **kwargs,
    ):
        """Initialization

        Parameters
        ----------
        lat : np.array
            Latitudinal coordinates in the specified CRS (can be any unit).
        lon : np.array
            Longitudinal coordinates in the specified CRS (can be any unit).
        crs : str or anything accepted by pyproj.CRS.from_user_input()
            Coordinate reference system. Default: EPSG:4326 (WGS84)
        region_id : np.array or str, optional
            Array of numeric country (or region) codes. If the special value "country" is given
            admin-0 codes are automatically assigned. Default: None
        on_land : np.array or str, optional
            Boolean array indicating on land (True) or off shore (False). If the special value
            "natural_earth" is given, the property is automatically determined from NaturalEarth
            shapes. Default: None
        kwargs : dict
            Additional columns with data to store in the internal GeoDataFrame (gdf attribute).
        """

        self.gdf = gpd.GeoDataFrame(
            data={
                "geometry": gpd.points_from_xy(lon, lat, crs=crs),
                "region_id": region_id,
                "on_land": on_land,
                **kwargs,
            }
        )

        if isinstance(region_id, str):
            LOGGER.info("Setting region id to %s level.", region_id)
            self.set_region_id(level=region_id, overwrite=True)
        if isinstance(on_land, str):
            LOGGER.info("Setting on land from %s source.", on_land)
            self.set_on_land(source=on_land, overwrite=True)

    @property
    def lat(self):
        """Return latitudes"""
        return self.gdf.geometry.y.values

    @property
    def lon(self):
        """Return longitudes"""
        return self.gdf.geometry.x.values

    @property
    def geometry(self):
        """Return the geometry"""
        return self.gdf["geometry"]

    @property
    def on_land(self):
        """Get the on_land property"""
        if "on_land" not in self.gdf:
            return None
        if self.gdf["on_land"].isna().all():
            return None
        return self.gdf["on_land"].values

    @property
    def region_id(self):
        """Get the assigned region_id"""
        if "region_id" not in self.gdf:
            return None
        if self.gdf["region_id"].isna().all():
            return None
        return self.gdf["region_id"].values

    @property
    def crs(self):
        """Get the crs"""
        return self.gdf.crs

    @property
    def size(self):
        """Get size (number of lat/lon pairs)"""
        return self.gdf.shape[0]

    @property
    def shape(self):
        """Get shape [lat, lon] assuming rastered data."""
        return (np.unique(self.lat).size, np.unique(self.lon).size)

    @property
    def total_bounds(self):
        """Get total bounds (minx, miny, maxx, maxy)."""
        return self.gdf.total_bounds

    @property
    def coord(self):
        """Get [lat, lon] array."""
        return np.stack([self.lat, self.lon], axis=1)

    def __eq__(self, other):
        """dunder method for Centroids comparison.
        returns True if two centroids equal, False otherwise

        Parameters
        ----------
        other : Centroids
            object to compare with

        Returns
        -------
        bool
        """
        if not u_coord.equal_crs(self.crs, other.crs):
            return False

        try:
            pd.testing.assert_frame_equal(self.gdf, other.gdf, check_like=True)
            return True
        except AssertionError:
            return False

    def to_default_crs(self, inplace=True):
        """Project the current centroids to the default CRS (epsg4326)

        Parameters
        ----------
        inplace: bool
            if True, modifies the centroids in place.
            if False, return projected centroids object.
            Default is True.

        Returns
        -------
        Centroids or None (if inplace is True)

        """
        return self.to_crs(DEF_CRS, inplace=inplace)

    def to_crs(self, crs, inplace=False):
        """Project the current centroids to the desired crs

        Parameters
        ----------
        crs : str
            coordinate reference system
        inplace: bool, default False
            if True, modifies the centroids in place.
            if False, returns a copy.

        Returns
        -------
        Centroids or None (if inplace is True)
        """
        if inplace:
            self.gdf.to_crs(crs, inplace=True)
            return None
        return Centroids.from_geodataframe(self.gdf.to_crs(crs))

    @classmethod
    def from_geodataframe(cls, gdf):
        """Initialize centroids from a geodataframe

        Parameters
        ----------
        gdf : GeoDataFrame
            Input geodataframe with centroids as points
            in the geometry column. All other columns are
            attached to the centroids geodataframe.

        Returns
        -------
        Centroids
            Centroids built from the geodataframe.

        Raises
        ------
        ValueError
        """
        if (gdf.geom_type != "Point").any():
            raise ValueError(
                "The inpute geodataframe contains geometries that are not points."
            )

        # Don't forget to make a copy!!
        # This is a bit ugly, but avoids to recompute the geometries
        # in the init. For large datasets this saves computation time
        centroids = cls(lat=[1], lon=[1])  # make "empty" centroids
        centroids.gdf = gdf.copy(deep=True)
        if gdf.crs is None:
            centroids.gdf.set_crs(DEF_CRS, inplace=True)
        return centroids

    @classmethod
    def from_exposures(cls, exposures):
        """Generate centroids from the locations of exposures.

        The properties "region_id" and "on_land" are also extracted from the Exposures object if
        available. The columns "value", "impf_*", "centr_*", "cover", and "deductible" are not
        used.

        Parameters
        ----------
        exposures : Exposures
            Exposures from which to take the centroids locations (as well as region_id and on_land
            if available).

        Returns
        -------
        Centroids

        Raises
        ------
        ValueError
        """
        # exclude exposures specific columns
        col_names = [
            column
            for column in exposures.gdf.columns
            if not any(pattern in column for pattern in EXP_SPECIFIC_COLS)
        ]

        gdf = exposures.gdf[col_names]
        return cls.from_geodataframe(gdf)

    @classmethod
    def from_pnt_bounds(cls, points_bounds, res, crs=DEF_CRS):
        """Create Centroids object from coordinate bounds and resolution.

        The result contains all points from a regular raster with the given resolution and CRS,
        covering the given bounds. Note that the raster bounds are larger than the points' bounds
        by res/2.

        Parameters
        ----------
        points_bounds : tuple
            The bounds (lon_min, lat_min, lon_max, lat_max) of the point coordinates.
        res : float
            The desired resolution in same units as `points_bounds`.
        crs : dict() or rasterio.crs.CRS, optional
            Coordinate reference system. Default: DEF_CRS

        Returns
        -------
        Centroids
        """
        height, width, transform = u_coord.pts_to_raster_meta(
            points_bounds, (res, -res)
        )
        return cls.from_meta(
            {
                "crs": crs,
                "width": width,
                "height": height,
                "transform": transform,
            }
        )

    def append(self, *centr):
        """Append Centroids to the current centroid object for concatenation.

        This method checks that all centroids use the same CRS, appends the list of centroids to
        the initial Centroid object and eventually concatenates them to create a single centroid
        object with the union of all centroids.

        Note that the result might contain duplicate points if the object to append has an overlap
        with the current object. Remove duplicates by either using :py:meth:`union`
        or calling :py:meth:`remove_duplicate_points` after appending.

        Parameters
        ----------
        centr : Centroids
            Centroids to append. The centroids need to have the same CRS.

        Raises
        ------
        ValueError

        See Also
        --------
        union : Union of Centroid objects.
        remove_duplicate_points : Remove duplicate points in a Centroids object.
        """
        for other in centr:
            if not u_coord.equal_crs(self.crs, other.crs):
                raise ValueError(
                    f"The given centroids use different CRS: {self.crs}, {other.crs}. "
                    "The centroids are incompatible and cannot be concatenated."
                )
        self.gdf = pd.concat([self.gdf] + [other.gdf for other in centr])

    def union(self, *others):
        """Create the union of the current Centroids object with one or more other centroids
        objects by passing the list of centroids to :py:meth:`append` for concatenation and then
        removes duplicates.

        All centroids must have the same CRS. Points that are contained in more than one of the
        Centroids objects will only be contained once (i.e. duplicates are removed).

        Parameters
        ----------
        others : Centroids
            Centroids contributing to the union.

        Returns
        -------
        centroids : Centroids
            Centroids object containing the union of all Centroids.
        """
        centroids = copy.deepcopy(self)
        centroids.append(*others)

        return centroids.remove_duplicate_points()

    def remove_duplicate_points(self):
        """Return a copy of centroids with duplicate points removed

        Parameters
        ----------
        centr : Centroids
            Centroids with or without duplicate points

        Returns
        -------
        centroids : Centroids
            A new Centroids object that contains a subselection of the original centroids without
            duplicates. Note that a copy is returned even if there were no duplicates.
        """
        return self.from_geodataframe(self.gdf.drop_duplicates(subset=["geometry"]))

    def select(self, reg_id=None, extent=None, sel_cen=None):
        """Return new Centroids object containing points following certain criteria

        It is currently possible to filter by region (reg_id), by geographical extent (extent), or
        by an explicit list of indices/a mask (sel_cen). If more than one criterion is given, all
        of them must be satisfied for a point to be included in the selection.

        Parameters
        ----------
        reg_id : int or list of int, optional
            Numeric ID (or IDs) of the region (or regions) to restrict to, according to the values
            in the region_id property. Default: None
        extent : tuple, optional
            The geographical extent (min_lon, max_lon, min_lat, max_lat) to restrict to, including
            the boundary. If the value for min_lon is greater than max_lon, the extent is
            interpreted to cross the antimeridian ([max_lon, 180] and [-180, min_lon]).
            Default: None
        sel_cen : np.ndarray of int or bool, optional
            Boolean mask, or list of indices to restrict to. Default: None

        Returns
        -------
        centroids : Centroids
            Sub-selection of this object
        """
        sel_cen_bool = sel_cen
        if sel_cen is not None and sel_cen.dtype.kind == "i":
            # if needed, convert indices to bool
            sel_cen_bool = np.zeros(self.size, dtype=bool)
            sel_cen_bool[np.unique(sel_cen)] = True

        sel_cen_mask = self.select_mask(
            sel_cen=sel_cen_bool, reg_id=reg_id, extent=extent
        )
        return Centroids.from_geodataframe(self.gdf.iloc[sel_cen_mask])

    def select_mask(self, sel_cen=None, reg_id=None, extent=None):
        """Create mask of selected centroids

        Parameters
        ----------
        sel_cen: np.ndarray of bool, optional
            Boolean array, with size matching the number of centroids. Default: None
        reg_id : int or list of int, optional
            Numeric ID (or IDs) of the region (or regions) to restrict to, according to the values
            in the region_id property. Default: None
        extent : tuple, optional
            The geographical extent (min_lon, max_lon, min_lat, max_lat) to restrict to, including
            the boundary. If the value for min_lon is greater than lon_max, the extent is
            interpreted to cross the antimeridian ([lon_max, 180] and [-180, lon_min]).
            Default: None

        Returns
        -------
        sel_cen : np.ndarray of bool
            Boolean array (mask) with value True for centroids in selection.
        """
        if sel_cen is None:
            sel_cen = np.ones(self.size, dtype=bool)
        if reg_id is not None:
            sel_cen &= np.isin(self.region_id, reg_id)
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            lon_max += 360 if lon_min > lon_max else 0
            lon_normalized = u_coord.lon_normalize(
                self.lon.copy(), center=0.5 * (lon_min + lon_max)
            )
            sel_cen &= (
                (lon_normalized >= lon_min)
                & (lon_normalized <= lon_max)
                & (self.lat >= lat_min)
                & (self.lat <= lat_max)
            )
        return sel_cen

    def plot(self, *, axis=None, figsize=(9, 13), **kwargs):
        """Plot centroids geodataframe using geopandas and cartopy plotting functions.

        Parameters
        ----------
        axis: optional
            user-defined cartopy.mpl.geoaxes.GeoAxes instance
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        args : optional
            positional arguments for geopandas.GeoDataFrame.plot
        kwargs : optional
            keyword arguments for geopandas.GeoDataFrame.plot

        Returns
        -------
        ax : cartopy.mpl.geoaxes.GeoAxes instance
        """
        if axis == None:
            fig, axis = plt.subplots(
                figsize=figsize, subplot_kw={"projection": ccrs.PlateCarree()}
            )
        if type(axis) != cartopy.mpl.geoaxes.GeoAxes:
            raise AttributeError(
                f"The axis provided is of type: {type(axis)} "
                "The function requires a cartopy.mpl.geoaxes.GeoAxes."
            )

        axis.add_feature(cfeature.BORDERS)
        axis.add_feature(cfeature.COASTLINE)
        axis.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

        if self.gdf.crs != DEF_CRS:
            centroids_plot = self.to_default_crs(inplace=False)
            centroids_plot.gdf.plot(ax=axis, transform=ccrs.PlateCarree(), **kwargs)
        else:
            self.gdf.plot(ax=axis, transform=ccrs.PlateCarree(), **kwargs)
        return axis

    def set_region_id(self, level="country", overwrite=False):
        """Set region_id as country ISO numeric code attribute for every pixel or point.

        Parameters
        ----------
        level: str, optional
            The admin level on which to assign centroids. Currently only 'country' (admin0) is
            implemented. Default: 'country'
        overwrite : bool, optional
            If True, overwrite the existing region_id information. If False, region_id is set
            only if region_id is missing (None). Default: False
        """
        if overwrite or self.region_id is None:
            LOGGER.debug("Setting region_id %s points.", str(self.size))
            if level == "country":
                ne_geom = self._ne_crs_geom()
                self.gdf["region_id"] = u_coord.get_country_code(
                    ne_geom.y.values,
                    ne_geom.x.values,
                )
            else:
                raise NotImplementedError(
                    "The region id can only be assigned for countries so far"
                )

    def set_on_land(self, source="natural_earth", overwrite=False):
        """Set on_land attribute for every pixel or point.

        Parameters
        ----------
        source: str, optional
            The source of the on-land information. Currently, only 'natural_earth' (based on shapes
            from NaturalEarth, https://www.naturalearthdata.com/) is implemented.
            Default: 'natural_earth'.
        overwrite : bool, optional
            If True, overwrite the existing on_land information. If False, on_land is set
            only if on_land is missing (None). Default: False
        """
        if overwrite or self.on_land is None:
            LOGGER.debug("Setting on_land %s points.", str(self.lat.size))
            if source == "natural_earth":
                ne_geom = self._ne_crs_geom()
                self.gdf["on_land"] = u_coord.coord_on_land(
                    ne_geom.y.values, ne_geom.x.values
                )
            else:
                raise NotImplementedError(
                    "The on land variables can only be automatically assigned using natural earth."
                )

    def get_pixel_shapes(self, res=None, **kwargs):
        """Create a GeoSeries of the quadratic pixel shapes at the centroid locations

        Note that this assumes that the centroids define a regular grid of pixels.

        Parameters
        ----------
        res : float, optional
            The resolution of the regular grid the pixels are taken from. If not given, it is
            estimated using climada.util.coordinates.get_resolution. Default: None
        kwargs : optional
            Additional keyword arguments are passed to climada.util.coordinates.get_resolution.

        Returns
        -------
        GeoSeries

        See also
        --------
        climada.util.coordinates.get_resolution
        """
        if res is None:
            res = np.abs(u_coord.get_resolution(self.lat, self.lon, **kwargs)).min()
        geom = self.geometry.copy()
        # unset CRS to avoid warnings about geographic CRS when using `GeoSeries.buffer`
        geom.crs = None
        return geom.buffer(
            # resolution=1, cap_style=3: squared buffers
            # https://shapely.readthedocs.io/en/latest/manual.html#object.buffer
            distance=res / 2,
            resolution=1,
            cap_style=3,
            # reset CRS (see above)
        ).set_crs(self.crs)

    def get_area_pixel(self, min_resol=1.0e-8):
        """Compute the area per centroid in the CEA projection

        Note that this assumes that the centroids define a regular grid of pixels (area in m²).

        Parameters
        ----------
        min_resol : float, optional
            When estimating the grid resolution, use this as the minimum resolution in lat and lon.
            It is passed to climada.util.coordinates.get_resolution. Default: 1.0e-8

        Returns
        -------
        areapixels : np.array
            Area of each pixel in square meters.
        """
        LOGGER.debug("Computing pixel area for %d centroids.", self.size)
        xy_pixels = self.get_pixel_shapes(min_resol=min_resol)
        if PROJ_CEA != xy_pixels.crs:
            xy_pixels = xy_pixels.to_crs(crs={"proj": "cea"})
        return xy_pixels.area.values

    def get_closest_point(self, x_lon, y_lat):
        """Returns closest centroid and its index to a given point.

        Parameters
        ----------
        x_lon : float
            Longitudinal (x) coordinate.
        y_lat : float
            Latitudinal (y) coordinate.

        Returns
        -------
        x_close : float
            x-coordinate (longitude) of closest centroid.
        y_close : float
            y-coordinate (latitude) of closest centroid.
        idx_close : int
            Index of centroid in internal ordering of centroids.
        """
        close_idx = self.geometry.distance(Point(x_lon, y_lat)).values.argmin()
        return self.lon[close_idx], self.lat[close_idx], close_idx

    # NOT REALLY AN ELEVATION FUNCTION, JUST READ RASTER
    def get_elevation(self, topo_path):
        """Return elevation attribute for every pixel or point in meters.

        Parameters
        ----------
        topo_path : str
            Path to a raster file containing gridded elevation data.

        Returns
        -------
        values : np.array of shape (npoints,)
            Interpolated elevation values from raster file for each given coordinate point.
        """
        return u_coord.read_raster_sample(topo_path, self.lat, self.lon)

    def get_dist_coast(self, signed=False, precomputed=True):
        """Get dist_coast attribute for every pixel or point in meters.

        The distances are read from a raster file containing precomputed distances (from NASA) at
        0.01 degree (approximately 1 km) resolution.

        Parameters
        ----------
        signed : bool, optional
            If True, use signed distances (positive off shore and negative on land). Default: False
        precomputed : bool, optional
            Whether distances should be read from a pre-computed raster (True) or computed
            on-the-fly (False). Default: True.

            .. deprecated:: 5.0
               Argument is ignored, because distances are not computed on-the-fly anymore.

        Returns
        -------
        dist : np.array
            (Signed) distance to coast in meters.
        """
        if not precomputed:
            LOGGER.warning(
                "The `precomputed` argument is deprecated and will be removed in the future"
                " because `get_dist_coast` always uses precomputed distances."
            )
        ne_geom = self._ne_crs_geom()
        return u_coord.dist_to_coast_nasa(
            ne_geom.y.values,
            ne_geom.x.values,
            highres=True,
            signed=signed,
        )

    def get_meta(self, resolution=None):
        """Returns a meta raster based on the centroids bounds.

        Note that this function is not perfectly inverse with `from_meta` since `get_meta` enforces
        a grid with equal resolution in x- and y-direction with coordinates increasing in
        x-direction and decreasing in y-direction.

        Parameters
        ----------
        resolution : float, optional
            Resolution of the raster. If not given, the resolution is estimated from the centroids
            by assuming that they form a regular raster. Default: None

        Returns
        -------
        meta: dict
            meta raster representation of the centroids
        """
        if resolution is None:
            resolution = np.abs(u_coord.get_resolution(self.lat, self.lon)).min()
        xmin, ymin, xmax, ymax = self.gdf.total_bounds
        rows, cols, ras_trans = u_coord.pts_to_raster_meta(
            (xmin, ymin, xmax, ymax),
            (resolution, -resolution),
        )
        meta = {
            "crs": self.crs,
            "height": rows,
            "width": cols,
            "transform": ras_trans,
        }
        return meta

    ##
    # I/O methods
    ##

    @classmethod
    def from_raster_file(
        cls,
        file_name,
        src_crs=None,
        window=None,
        geometry=None,
        dst_crs=None,
        transform=None,
        width=None,
        height=None,
        resampling=rasterio.warp.Resampling.nearest,
        return_meta=False,
    ):
        """Create a new Centroids object from a raster file

        Select region using window or geometry. Reproject input by providing
        dst_crs and/or (transform, width, height).

        Parameters
        ----------
        file_name : str
            path of the file
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
        resampling : rasterio.warp.Resampling optional
            resampling function used for reprojection to dst_crs,
            default: nearest
        return_meta : bool, optional
            default: False

        Returns
        -------
        centr : Centroids
            Centroids according to the given raster file
        meta : dict, optional if return_meta is True
            Raster meta (height, width, transform, crs).
        """
        meta, _ = u_coord.read_raster(
            file_name,
            [1],
            src_crs,
            window,
            geometry,
            dst_crs,
            transform,
            width,
            height,
            resampling,
        )
        centr = cls.from_meta(meta)
        return (centr, meta) if return_meta else centr

    @classmethod
    def from_meta(cls, meta):
        """initiate centroids from meta raster definition

        Parameters
        ----------
        meta : dict
            meta description of raster

        Returns
        -------
        Centroid
            Centroids initialized for raster described by meta.
        """
        crs = meta["crs"]
        lat, lon = _meta_to_lat_lon(meta)
        return cls(lon=lon, lat=lat, crs=crs)

    @classmethod
    def from_vector_file(cls, file_name, dst_crs=None):
        """Create Centroids object from vector file (any format supported by fiona).

        Parameters
        ----------
        file_name : str
            vector file with format supported by fiona and 'geometry' field.
        dst_crs : crs, optional
            reproject to given crs
            If no crs is given in the file, simply sets the crs.

        Returns
        -------
        centr : Centroids
            Centroids with points according to the given vector file
        """

        centroids = cls.from_geodataframe(gpd.read_file(file_name))
        if dst_crs is not None:
            if centroids.crs:
                centroids.to_crs(dst_crs, inplace=True)
            else:
                centroids.gdf.set_crs(dst_crs, inplace=True)
        return centroids

    @classmethod
    def from_csv(cls, file_path, **kwargs):
        """Generate centroids from a CSV file with column names in var_names.

        Parameters
        ----------
        file_path : str
            path to CSV file to be read
        kwargs : dict
            Additional keyword arguments to pass on to pandas.read_csv.

        Returns
        -------
        Centroids
        """
        return cls._from_dataframe(pd.read_csv(file_path, **kwargs))

    def write_csv(self, file_path):
        """Save centroids as CSV file

        Parameters
        ----------
        file_path : str, Path
            absolute or relative file path and name to write to
        """
        file_path = Path(file_path).with_suffix(".csv")
        LOGGER.info("Writing %s", file_path)
        self._centroids_to_dataframe().to_csv(file_path, index=False)

    @classmethod
    def from_excel(cls, file_path, sheet_name=None):
        """Generate a new centroids object from an excel file with column names in var_names.

        Parameters
        ----------
        file_path : str
            absolute or relative file path
        sheet_name : str, optional
            name of sheet in excel file containing centroid information
            Default: "centroids"

        Returns
        -------
        centr : Centroids
            Centroids with data from the given excel file
        """
        if sheet_name is None:
            sheet_name = "centroids"
        df = pd.read_excel(file_path, sheet_name)
        return cls._from_dataframe(df)

    def write_excel(self, file_path):
        """Save centroids as excel file

        Parameters
        ----------
        file_path : str, Path
            absolute or relative file path and name to write to
        """
        file_path = Path(file_path).with_suffix(".xlsx")
        LOGGER.info("Writing %s", file_path)
        self._centroids_to_dataframe().to_excel(
            file_path,
            sheet_name=DEF_SHEET_NAME,
            index=False,
        )

    def write_hdf5(self, file_name, mode="w"):
        """Write data frame and metadata in hdf5 format

        Parameters
        ----------
        file_name : str
            (path and) file name to write to.
        """
        LOGGER.info("Writing %s", file_name)
        store = pd.HDFStore(file_name, mode=mode)
        pandas_df = pd.DataFrame(self.gdf)
        for col in pandas_df.columns:
            if str(pandas_df[col].dtype) == "geometry":
                pandas_df[col] = np.asarray(self.gdf[col])

        # Avoid pandas PerformanceWarning when writing HDF5 data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            # Write dataframe
            store.put("centroids", pandas_df)

        store.get_storer("centroids").attrs.metadata = {
            "crs": CRS.from_user_input(self.crs).to_wkt()
        }

        store.close()

    @classmethod
    def from_hdf5(cls, file_name):
        """Create a centroids object from a HDF5 file.

        Parameters
        ----------
        file_data : str or h5
            If string, path to read data. If h5 object, the datasets will be read from there.

        Returns
        -------
        centr : Centroids
            Centroids with data from the given file

        Raises
        ------
        FileNotFoundError
        """
        if not Path(file_name).is_file():
            raise FileNotFoundError(str(file_name))
        try:
            with pd.HDFStore(file_name, mode="r") as store:
                metadata = store.get_storer("centroids").attrs.metadata
                # in previous versions of CLIMADA and/or geopandas,
                # the CRS was stored in '_crs'/'crs'
                crs = metadata.get("crs")
                gdf = gpd.GeoDataFrame(store["centroids"], crs=crs)
        except TypeError:
            with h5py.File(file_name, "r") as data:
                gdf = cls._gdf_from_legacy_hdf5(data.get("centroids"))
        except KeyError:
            with h5py.File(file_name, "r") as data:
                gdf = cls._gdf_from_legacy_hdf5(data)

        return cls.from_geodataframe(gdf)

    ##
    # Private methods
    ##

    @classmethod
    def _from_dataframe(cls, df):
        if "crs" in df.columns:
            crs = df["crs"].iloc[0]
        else:
            LOGGER.info(
                "No 'crs' column provided in file, setting CRS to WGS84 default."
            )
            crs = DEF_CRS

        extra_values = {
            col: df[col] for col in df.columns if col not in ["lat", "lon", "crs"]
        }

        return cls(lat=df["lat"], lon=df["lon"], **extra_values, crs=crs)

    @staticmethod
    def _gdf_from_legacy_hdf5(data):
        crs = DEF_CRS
        if data.get("crs"):
            crs = u_coord.to_crs_user_input(data.get("crs")[0])
        if data.get("lat") and data.get("lat").size:
            latitude = np.array(data.get("lat"))
            longitude = np.array(data.get("lon"))
        elif data.get("latitude") and data.get("latitude").size:
            latitude = np.array(data.get("latitude"))
            longitude = np.array(data.get("longitude"))
        else:
            centr_meta = data.get("meta")
            meta = dict()
            meta["crs"] = crs
            for key, value in centr_meta.items():
                if key != "transform":
                    meta[key] = value[0]
                else:
                    meta[key] = rasterio.Affine(*value)
            latitude, longitude = _meta_to_lat_lon(meta)

        extra_values = {}
        for centr_name in data.keys():
            if centr_name not in ("crs", "lat", "lon", "meta", "latitude", "longitude"):
                values = np.array(data.get(centr_name))
                if latitude.size != 0 and values.size != 0:
                    extra_values[centr_name] = values

        return gpd.GeoDataFrame(
            extra_values,
            geometry=gpd.points_from_xy(x=longitude, y=latitude, crs=crs),
        )

    @classmethod
    def _legacy_from_excel(cls, file_name, var_names):
        LOGGER.info("Reading %s", file_name)
        try:
            df = pd.read_excel(file_name, var_names["sheet_name"])
            df = df.rename(columns=var_names["col_name"])
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err
        return cls._from_dataframe(df)

    def _centroids_to_dataframe(self):
        """Create dataframe from Centroids object to facilitate
        saving in different file formats.

        Returns
        -------
        df : DataFrame
        """
        df = pd.DataFrame(self.gdf)
        df["lon"] = self.gdf["geometry"].x
        df["lat"] = self.gdf["geometry"].y
        df["crs"] = CRS.from_user_input(self.crs).to_wkt()
        df = df.drop(["geometry"], axis=1)
        return df

    def _ne_crs_geom(self):
        """Return `geometry` attribute in the CRS of Natural Earth.

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if u_coord.equal_crs(self.gdf.crs, u_coord.NE_CRS):
            return self.gdf.geometry
        return self.to_crs(u_coord.NE_CRS, inplace=False).geometry

    ##
    # Deprecated methods
    ##

    @classmethod
    @deprecated(
        details="Reading Centroids data from matlab files is not supported anymore."
        "This method has been removed with climada 5.0"
    )
    def from_mat(cls, file_name, var_names=None):
        """Reading Centroids data from matlab files is not supported anymore.
        This method has been removed with climada 5.0"""
        raise NotImplementedError(
            "You are suggested to use an old version of climada (<=4.*) and"
            " convert the file to hdf5 format."
        )

    @staticmethod
    @deprecated(details="This method has been removed with climada 5.0")
    def from_base_grid(land=False, res_as=360, base_file=None):
        """This method has been removed with climada 5.0"""
        raise NotImplementedError(
            "Create the Centroids from a custom base file or from Natural"
            " Earth (files are available in Climada, look up ``climada.util"
            ".constants.NATEARTH_CENTROIDS`` for their location)"
        )

    @classmethod
    @deprecated(
        details="This method will be removed in a future version."
        " Simply use the constructor instead."
    )
    def from_lat_lon(cls, lat, lon, crs="EPSG:4326"):
        """deprecated, use the constructor instead"""
        return cls(lat=lat, lon=lon, crs=crs)

    @deprecated(
        details="This method is futile and will be removed in a future version."
        " `Centroids.get_area_pixel` can be run without initialization."
    )
    def set_area_pixel(self, min_resol=1e-08, scheduler=None):
        """deprecated, obsolete"""

    @deprecated(
        details="This method is futile and will be removed in a future version."
        " `Centroids.get_area_pixel` can be run without initialization."
    )
    def set_area_approx(self, min_resol=1e-08):
        """deprecated, obsolete"""

    @deprecated(
        details="This method is futile and will be removed in a future version."
        " `Centroids.get_dist_coast` can be run without initialization."
    )
    def set_dist_coast(self, signed=False, precomputed=False, scheduler=None):
        """deprecated, obsolete"""

    @deprecated(
        details="This method has no effect and will be removed in a future version."
        " In the current version of climada the geometry points of a `Centroids` object"
        " cannot be removed as they are the backbone of the Centroids' GeoDataFrame."
    )
    def empty_geometry_points(self):
        """ "deprecated, has no effect, which may be unexpected: no geometry points will be removed,
        the centroids' GeoDataFrame is built on them!
        """

    @deprecated(
        details="This method has no effect and will be removed in a future version."
    )
    def set_meta_to_lat_lon(self):
        """deprecated, has no effect"""

    @deprecated(
        details="This method has no effect and will be removed in a future version."
    )
    def set_lat_lon_to_meta(self, min_resol=1e-08):
        """deprecated, has no effect"""


def _meta_to_lat_lon(meta):
    """Compute lat and lon of every pixel center from meta raster.

    Parameters
    ----------
    meta : dict
        meta description of raster

    Returns
    -------
    latitudes : np.ndarray
        Latitudinal coordinates of pixel centers.
    longitudes : np.ndarray
        Longitudinal coordinates of pixel centers.
    """
    xgrid, ygrid = u_coord.raster_to_meshgrid(
        meta["transform"], meta["width"], meta["height"]
    )
    return ygrid.ravel(), xgrid.ravel()
