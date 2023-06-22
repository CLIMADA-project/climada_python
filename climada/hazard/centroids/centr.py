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
from pathlib import Path
from typing import Optional, Dict, Any

import cartopy.crs as ccrs
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj.crs.crs import CRS
import rasterio
from scipy import sparse
from rasterio.warp import Resampling
from shapely.geometry.point import Point

from climada.util.constants import (DEF_CRS,
                                    ONE_LAT_KM,
                                    NATEARTH_CENTROIDS)
import climada.util.coordinates as u_coord
import climada.util.hdf5_handler as u_hdf5
import climada.util.plot as u_plot

__all__ = ['Centroids']

PROJ_CEA = CRS.from_user_input({'proj': 'cea'})

DEF_VAR_MAT = {
    'field_names': ['centroids', 'hazard'],
    'var_name': {
        'lat': 'lat',
        'lon': 'lon',
        'dist_coast': 'distance2coast_km',
        'admin0_name': 'admin0_name',
        'admin0_iso3': 'admin0_ISO3',
        'comment': 'comment',
        'region_id': 'NatId'
    }
}
"""MATLAB variable names"""

DEF_VAR_EXCEL = {
    'sheet_name': 'centroids',
    'col_name': {
        'region_id': 'region_id',
        'lat': 'latitude',
        'lon': 'longitude',
    }
}
"""Excel variable names"""

LOGGER = logging.getLogger(__name__)


class Centroids():
    """Contains raster or vector centroids.

    Attributes
    ----------
    lat : np.array
        latitudes
    lon : np.array
        longitudes
    crs : str, optional
        coordinate reference system, default is WGS84
    area_pixel : np.array, optional
        areas
    dist_coast : np.array, optional
        distances to coast
    on_land : np.array, optional
        on land (True) and on sea (False)
    region_id : np.array, optional
        region numeric codes
    elevation : np.array, optional
        elevations
    kwargs: dicts of np.arrays, optional
        any further desired properties of centroids. Is passed to the
        GeoDataFrame constructor
    """

    def __init__(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        crs: str = DEF_CRS,
        region_id: Optional[np.ndarray] = None,
        on_land: Optional[np.ndarray] = None,
        dist_coast: Optional[np.ndarray] = None,
        elevation: Optional[np.ndarray] = None,
        area_pixel: Optional[np.ndarray] = None,
        **kwargs
    ):
        """Initialization

        Parameters
        ----------
        lat : np.array
            latitude of size size. Defaults to empty array
        lon : np.array
            longitude of size size. Defaults to empty array
        crs : str
            coordinate reference system
        area_pixel : np.array, optional
            area of size size. Defaults to empty array
        on_land : np.array, optional
            on land (True) and on sea (False) of size size. Defaults to empty array
        region_id : np.array, optional
            country region code of size size, Defaults to empty array
        elevation : np.array, optional
            elevation of size size. Defaults to empty array
        dist_coast : np.array, optional
            distance to coast of size size. Defaults to empty array
        """
        attr_dict = {
            'geometry': gpd.points_from_xy(lon, lat, crs=crs),
            'region_id' : region_id,
            'on_land' : on_land,
            'dist_coast' : dist_coast,
            'elevation' : elevation,
            'area_pixel' : area_pixel,
        }
        if kwargs:
            attr_dict = dict(**attr_dict, **kwargs)
        self.gdf = gpd.GeoDataFrame(data=attr_dict, crs=crs)

    @property
    def lat(self):
        return self.gdf.geometry.y.values

    @property
    def lon(self):
        return self.gdf.geometry.x.values

    @property
    def geometry(self):
        return self.gdf['geometry']

    @property
    def on_land(self):
        return self.gdf['on_land']

    @property
    def region_id(self):
        return self.gdf['region_id']

    @property
    def elevation(self):
        return self.gdf['elevation']

    @property
    def area_pixel(self):
        return self.gdf['area_pixel']

    @property
    def dist_coast(self):
        return self.gdf['dist_coast']

    @property
    def crs(self):
        return self.gdf.crs

    @property
    def size(self):
        return self.gdf.shape[0]

    @property
    def shape(self):
        """Get shape assuming rastered data."""
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
        """Return True if two centroids equal, False otherwise

        Parameters
        ----------
        other : Centroids
            centroids to compare

        Returns
        -------
        eq : bool
        """
        return self.gdf.equals(other.gdf) & u_coord.equal_crs(self.crs, other.crs)

    @classmethod
    def from_geodataframe(cls, gdf):
        return cls(lon=gdf.geometry.x.values, lat=gdf.geometry.y.values, crs=gdf.crs, **gdf.drop(columns=['geometry']).to_dict(orient='list'))


    @classmethod
    def from_pnt_bounds(cls, points_bounds, res, crs=DEF_CRS):
        """Create Centroids object with meta attribute according to points border data.

        raster border = point border + res/2

        Parameters
        ----------
        points_bounds : tuple
            points' lon_min, lat_min, lon_max, lat_max
        res : float
            desired resolution in same units as points_bounds
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS

        Returns
        -------
        centr : Centroids
            Centroids with meta according to given points border data.
        """
        rows, cols, ras_trans = u_coord.pts_to_raster_meta(points_bounds, (res, -res))
        x_grid, y_grid = u_coord.raster_to_meshgrid(ras_trans, cols, rows)
        return cls(lat=y_grid, lon=x_grid, crs=crs)


    def append(self, centr):
        """Append centroids points.

        Parameters
        ----------
        centr : Centroids
            Centroids to append. The centroids need to have the same CRS.

        See Also
        --------
        union : Union of Centroid objects.
        """
        self.gdf = pd.concat([self.gdf, centr.gdf])

    def union(self, *others):
        """
        Create the union of centroids from the inputs.
        The centroids are combined together point by point.
        All centroids must have the same CRS.

        When at least one centroids has one of the following property
        defined, it is also computed for all others.
        .area_pixel, .dist_coast, .on_land, .region_id, .elevetaion'

        Parameters
        ----------
        others : any number of climada.hazard.Centroids()
            Centroids to form the union with

        Returns
        -------
        centroids : Centroids
            Centroids containing the union of the centroids in others.

        Raises
        ------
        ValueError
        """
        # restrict to non-empty centroids
        cent_list = [c for c in (self,) + others if c.size > 0] # pylint: disable=no-member
        if len(cent_list) == 0 or len(others) == 0:
            return copy.deepcopy(self)

        # check if all centroids are identical
        if all([cent_list[0] == cent for cent in cent_list[1:]]):
            return copy.deepcopy(cent_list[0])

        # make sure that all Centroids have the same CRS
        for cent in cent_list:
            if not u_coord.equal_crs(cent.crs, cent_list[0].crs):
                raise ValueError('In a union, all Centroids need to have the same CRS: '
                                 f'{cent.crs} != {cent_list[0].crs}')

        # set attributes that are missing in some but defined in others
        for attr in ["geometry", "area_pixel", "dist_coast", "on_land", "region_id", "elevation"]:
            if np.any([getattr(cent, attr).size > 0 for cent in cent_list]):
                for cent in cent_list:
                    if not getattr(cent, attr).size > 0:
                        fun_name = f"set_{attr}{'_points' if attr == 'geometry' else ''}"
                        getattr(Centroids, fun_name)(cent)

        # create new Centroids object and set concatenated attributes
        centroids = Centroids(None, None)
        for attr_name, attr_val in vars(cent_list[0]).items():
            if isinstance(attr_val, np.ndarray) and attr_val.ndim == 1:
                attr_val_list = [getattr(cent, attr_name) for cent in cent_list]
                setattr(centroids, attr_name, np.hstack(attr_val_list))
            elif isinstance(attr_val, gpd.GeoSeries):
                attr_val_list = [getattr(cent, attr_name) for cent in cent_list]
                setattr(centroids, attr_name, pd.concat(attr_val_list, ignore_index=True))

        # finally, remove duplicate points
        return centroids.remove_duplicate_points()

    def get_closest_point(self, x_lon, y_lat, scheduler=None):
        """Returns closest centroid and its index to a given point.

        Parameters
        ----------
        x_lon : float
            x coord (lon)
        y_lat : float
            y coord (lat)
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        x_close : float
            x-coordinate (longitude) of closest centroid.
        y_close : float
            y-coordinate (latitude) of closest centroids.
        idx_close : int
            Index of centroid in internal ordering of centroids.
        """
        close_idx = self.geometry.distance(Point(x_lon, y_lat)).values.argmin()
        return self.lon[close_idx], self.lat[close_idx], close_idx

    def set_region_id(self, scheduler=None):
        """Set region_id as country ISO numeric code attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        ne_geom = self._ne_crs_geom(scheduler)
        LOGGER.debug('Setting region_id %s points.', str(self.size))
        self.gdf.region_id = u_coord.get_country_code(
            ne_geom.geometry[:].y.values, ne_geom.geometry[:].x.values)

    def set_area_pixel(self, min_resol=1.0e-8, scheduler=None):
        """Set `area_pixel` attribute for every pixel or point (area in m*m).

        Parameters
        ----------
        min_resol : float, optional
            if centroids are points, use this minimum resolution in lat and lon. Default: 1.0e-8
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """

        res = u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol)
        res = np.abs(res).min()
        LOGGER.debug('Setting area_pixel %s points.', str(self.lat.size))
        xy_pixels = self.geometry.buffer(res / 2).envelope
        if PROJ_CEA == self.geometry.crs:
            self.gdf.area_pixel = xy_pixels.area.values
        else:
            self.gdf.area_pixel = xy_pixels.to_crs(crs={'proj': 'cea'}).area.values

    def set_area_approx(self, min_resol=1.0e-8):
        """Set `area_pixel` attribute for every pixel or point (approximate area in m*m).

        Values are differentiated per latitude. Faster than `set_area_pixel`.

        Parameters
        ----------
        min_resol : float, optional
            if centroids are points, use this minimum resolution in lat and lon. Default: 1.0e-8
        """
        if self.meta:
            if hasattr(self.meta['crs'], 'linear_units') and \
            str.lower(self.meta['crs'].linear_units) in ['m', 'metre', 'meter']:
                self.area_pixel = np.zeros((self.meta['height'], self.meta['width']))
                self.area_pixel *= abs(self.meta['transform'].a) * abs(self.meta['transform'].e)
                return
            res_lat, res_lon = self.meta['transform'].e, self.meta['transform'].a
            lat_unique = np.arange(self.meta['transform'].f + res_lat / 2,
                                   self.meta['transform'].f + self.meta['height'] * res_lat,
                                   res_lat)
            lon_unique_len = self.meta['width']
            res_lat = abs(res_lat)
        else:
            res_lat, res_lon = np.abs(
                u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol))
            lat_unique = np.array(np.unique(self.lat))
            lon_unique_len = len(np.unique(self.lon))
            if PROJ_CEA == self.geometry.crs:
                self.area_pixel = np.repeat(res_lat * res_lon, lon_unique_len)
                return

        LOGGER.debug('Setting area_pixel approx %s points.', str(self.lat.size))
        res_lat = res_lat * ONE_LAT_KM * 1000
        res_lon = res_lon * ONE_LAT_KM * 1000 * np.cos(np.radians(lat_unique))
        area_approx = np.repeat(res_lat * res_lon, lon_unique_len)
        if area_approx.size == self.size:
            self.area_pixel = area_approx
        else:
            raise ValueError('Pixel area of points can not be computed.')

    def set_elevation(self, topo_path):
        """Set elevation attribute for every pixel or point in meters.

        Parameters
        ----------
        topo_path : str
            Path to a raster file containing gridded elevation data.
        """
        self.gdf.elevation = u_coord.read_raster_sample(topo_path, self.lat, self.lon)

    def set_dist_coast(self, signed=False, precomputed=False, scheduler=None):
        """Set dist_coast attribute for every pixel or point in meters.

        Parameters
        ----------
        signed : bool
            If True, use signed distances (positive off shore and negative on land). Default: False.
        precomputed : bool
            If True, use precomputed distances (from NASA). Default: False.
        scheduler : str
            Used for dask map_partitions. "threads", "synchronous" or "processes"
        """
        if precomputed:
            self.gdf.dist_coast = u_coord.dist_to_coast_nasa(
                self.lat, self.lon, highres=True, signed=signed)
        else:
            ne_geom = self._ne_crs_geom(scheduler)
            LOGGER.debug('Computing distance to coast for %s centroids.', str(self.size))
            self.gdf.dist_coast = u_coord.dist_to_coast(ne_geom, signed=signed)

    def set_on_land(self, scheduler=None):
        """Set on_land attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        ne_geom = self._ne_crs_geom(scheduler)
        LOGGER.debug('Setting on_land %s points.', str(self.lat.size))
        self.gdf.on_land = u_coord.coord_on_land(
            ne_geom.geometry[:].y.values, ne_geom.geometry[:].x.values)

    @classmethod
    def remove_duplicate_points(cls, centroids):
        """Return a copy of centroids with removed duplicated points

        Returns
        -------
         : Centroids
            Sub-selection of centroids withtout duplicates
        """
        return cls().from_gdf(centroids.gdf.drop_duplicates())

    def select(self, reg_id=None, extent=None, sel_cen=None):
        """Return Centroids with points in the given reg_id or within mask

        Parameters
        ----------
        reg_id : int
            region to filter according to region_id values
        extent : tuple
            Format (min_lon, max_lon, min_lat, max_lat) tuple.
            If min_lon > lon_max, the extend crosses the antimeridian and is
            [lon_max, 180] + [-180, lon_min]
            Borders are inclusive.
        sel_cen : np.array
            1-dim mask, overrides reg_id and extent

        Returns
        -------
        cen : Centroids
            Sub-selection of this object
        """
        if sel_cen is None:
            sel_cen = self.select_mask(reg_id=reg_id, extent=extent)

        centr = Centroids.from_lat_lon(self.lat[sel_cen], self.lon[sel_cen], self.geometry.crs)
        if self.area_pixel.size:
            centr.area_pixel = self.area_pixel[sel_cen]
        if self.region_id.size:
            centr.region_id = self.region_id[sel_cen]
        if self.on_land.size:
            centr.on_land = self.on_land[sel_cen]
        if self.dist_coast.size:
            centr.dist_coast = self.dist_coast[sel_cen]
        return centr

    def select_mask(self, reg_id=None, extent=None):
        """
        Make mask of selected centroids

        Parameters
        ----------
        reg_id : int
            region to filter according to region_id values
        extent : tuple
            Format (min_lon, max_lon, min_lat, max_lat) tuple.
            If min_lon > lon_max, the extend crosses the antimeridian and is
            [lon_max, 180] + [-180, lon_min]
            Borders are inclusive.

        Returns
        -------
        sel_cen : 1d array of booleans
            1d mask of selected centroids

        """
        sel_cen = np.ones(self.size, dtype=bool)
        if reg_id is not None:
            sel_cen &= np.isin(self.region_id, reg_id)
        if extent is not None:
            lon_min, lon_max, lat_min, lat_max = extent
            lon_max += 360 if lon_min > lon_max else 0
            lon_normalized = u_coord.lon_normalize(
                self.lon.copy(), center=0.5 * (lon_min + lon_max))
            sel_cen &= (
              (lon_normalized >= lon_min) & (lon_normalized <= lon_max) &
              (self.lat >= lat_min) & (self.lat <= lat_max)
            )
        return sel_cen

    def plot(self, axis=None, figsize=(9, 13), **kwargs):
        """Plot centroids scatter points over earth.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        kwargs : optional
            arguments for scatter matplotlib function

        Returns
        -------
        axis : matplotlib.axes._subplots.AxesSubplot
        """
        if self.meta and not self.coord.size:
            self.set_meta_to_lat_lon()
        pad = np.abs(u_coord.get_resolution(self.lat, self.lon)).min()

        proj_data, _ = u_plot.get_transformation(self.crs)
        proj_plot = proj_data
        if isinstance(proj_data, ccrs.PlateCarree):
            # use different projections for plot and data to shift the central lon in the plot
            xmin, ymin, xmax, ymax = u_coord.latlon_bounds(self.lat, self.lon, buffer=pad)
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        else:
            xmin, ymin, xmax, ymax = (self.lon.min() - pad, self.lat.min() - pad,
                                      self.lon.max() + pad, self.lat.max() + pad)

        if not axis:
            _, axis, _fontsize = u_plot.make_map(proj=proj_plot, figsize=figsize)

        axis.set_extent((xmin, xmax, ymin, ymax), crs=proj_data)
        u_plot.add_shapes(axis)
        axis.scatter(self.lon, self.lat, transform=proj_data, **kwargs)
        plt.tight_layout()
        return axis

    def calc_pixels_polygons(self, scheduler=None):
        """Return a gpd.GeoSeries with a polygon for every pixel

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if not self.meta:
            self.set_lat_lon_to_meta()
        if abs(abs(self.meta['transform'].a) -
               abs(self.meta['transform'].e)) > 1.0e-5:
            raise ValueError('Area can not be computed for not squared pixels.')
        self.set_geometry_points(scheduler)
        return self.geometry.buffer(self.meta['transform'].a / 2).envelope


    '''
    I/O methods
    '''


    def set_raster_file(self, file_name, band=None, **kwargs):
        """This function is deprecated, use Centroids.from_raster_file
        and Centroids.values_from_raster_files instead."""
        LOGGER.warning("The use of Centroids.set_raster_file is deprecated. "
                       "Use Centroids.from_raster_file and "
                       "Centroids.values_from_raster_files instead.")
        if not self.meta:
            self.__dict__ = Centroids.from_raster_file(file_name, **kwargs).__dict__
        return self.values_from_raster_files([file_name], band=band, **kwargs)

    @classmethod
    def from_raster_file(cls, file_name, src_crs=None, window=None,
                         geometry=None, dst_crs=None, transform=None, width=None,
                         height=None, resampling=Resampling.nearest):
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
        resampling : rasterio.warp,.Resampling optional
            resampling function used for reprojection to dst_crs

        Returns
        -------
        centr : Centroids
            Centroids with meta attribute according to the given raster file
        """
        meta, _ = u_coord.read_raster(
            file_name, [1], src_crs, window, geometry, dst_crs,
            transform, width, height, resampling)
        return cls(meta=meta)

    def values_from_raster_files(self, file_names, band=None, src_crs=None, window=None,
                                 geometry=None, dst_crs=None, transform=None, width=None,
                                 height=None, resampling=Resampling.nearest):
        """Read raster of bands and set 0 values to the masked ones.

        Each band is an event. Select region using window or geometry. Reproject input by proving
        dst_crs and/or (transform, width, height).

        Parameters
        ----------
        file_names : str
            path of the file
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
                file_name, band, src_crs, window, geometry, dst_crs,
                transform, width, height, resampling)
            if (tmp_meta['crs'] != self.meta['crs']
                    or tmp_meta['transform'] != self.meta['transform']
                    or tmp_meta['height'] != self.meta['height']
                    or tmp_meta['width'] != self.meta['width']):
                raise ValueError('Raster data is inconsistent with contained raster.')
            values.append(sparse.csr_matrix(data))

        return sparse.vstack(values, format='csr')

    @classmethod
    def from_vector_file(cls, file_name, dst_crs=None):
        """Create Centroids object from vector file (any format supported by fiona).

        Parameters
        ----------
        file_name : str
            vector file with format supported by fiona and 'geometry' field.
        dst_crs : crs, optional
            reproject to given crs

        Returns
        -------
        centr : Centroids
            Centroids with points according to the given vector file
        """
        data_frame = gpd.read_file(file_name)
        if dst_crs is None:
            geometry = data_frame.geometry
        else:
            geometry = data_frame.geometry.to_crs(dst_crs)
        lat, lon = geometry[:].y.values, geometry[:].x.values
        return cls(lat=lat, lon=lon, crs=dst_crs)

    def values_from_vector_files(self, file_names, val_names=None, dst_crs=None):
        """Read intensity or other data from vector files, making sure that geometry is compatible.

        If the geometry of the shapes in any of the given files does not agree with the
        geometry of this Centroids instance, a ValueError is raised.

        Parameters
        ----------
        file_names : list(str)
            vector files with format supported by fiona and 'geometry' field.
        val_names : list(str), optional
            list of names of the columns of the values. Default: ['intensity']
        dst_crs : crs, optional
            reproject to given crs

        Raises
        ------
        ValueError

        Returns
        -------
        values : scipy.sparse.csr_matrix
            Sparse array of shape (len(val_name), len(geometry)).
        """
        if val_names is None:
            val_names = ['intensity']

        values = []
        for file_name in file_names:
            tmp_lat, tmp_lon, tmp_geometry, data = u_coord.read_vector(
                file_name, val_names, dst_crs=dst_crs)
            if not (u_coord.equal_crs(tmp_geometry.crs, self.geometry.crs)
                    and np.allclose(tmp_lat, self.lat)
                    and np.allclose(tmp_lon, self.lon)):
                raise ValueError('Vector data inconsistent with contained vector.')
            values.append(sparse.csr_matrix(data))

        return sparse.vstack(values, format='csr')

    @classmethod
    def from_excel(cls, file_name, var_names=None):
        """Generate a new centroids object from an excel file with column names in var_names.

        Parameters
        ----------
        file_name : str
            absolute or relative file name
        var_names : dict, default
            name of the variables

        Raises
        ------
        KeyError

        Returns
        -------
        centr : Centroids
            Centroids with data from the given file
        """
        LOGGER.info('Reading %s', file_name)
        if var_names is None:
            var_names = DEF_VAR_EXCEL

        try:
            dfr = pd.read_excel(file_name, var_names['sheet_name'])
            centr = cls.from_lat_lon(dfr[var_names['col_name']['lat']],
                                     dfr[var_names['col_name']['lon']])
            try:
                centr.region_id = dfr[var_names['col_name']['region_id']]
            except KeyError:
                pass

        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err

        return centr

    def write_hdf5(self, file_data):
        """Write centroids attributes into hdf5 format.

        Parameters
        ----------
        file_data : str or h5
            If string, path to write data. If h5 object, the datasets will be generated there.
        """
        if isinstance(file_data, str):
            LOGGER.info('Writing %s', file_data)
            with h5py.File(file_data, 'w') as data:
                self._write_hdf5(data)
        else:
            self._write_hdf5(file_data)

    def _write_hdf5(self, data):
        str_dt = h5py.special_dtype(vlen=str)
        for centr_name, centr_val in self.__dict__.items():
            if isinstance(centr_val, np.ndarray):
                data.create_dataset(centr_name, data=centr_val, compression="gzip")
            elif centr_name == 'meta' and centr_val:
                centr_meta = data.create_group(centr_name)
                for key, value in centr_val.items():
                    if value is None:
                        LOGGER.info("Skip writing Centroids.meta['%s'] for it is None.", key)
                    elif key not in ('crs', 'transform'):
                        if not isinstance(value, str):
                            centr_meta.create_dataset(key, (1,), data=value, dtype=type(value))
                        else:
                            hf_str = centr_meta.create_dataset(key, (1,), dtype=str_dt)
                            hf_str[0] = value
                    elif key == 'transform':
                        centr_meta.create_dataset(
                            key, (6,),
                            data=[value.a, value.b, value.c, value.d, value.e, value.f],
                            dtype=float)
            elif centr_name == 'geometry':
                LOGGER.debug("Skip writing Centroids.geometry")
            else:
                LOGGER.info("Skip writing Centroids.%s:%s, it's neither an array nor a non-empty"
                            " meta object", centr_name, centr_val.__class__.__name__)
        hf_str = data.create_dataset('crs', (1,), dtype=str_dt)
        hf_str[0] = CRS.from_user_input(self.crs).to_wkt()

    def read_hdf5(self, *args, **kwargs):
        """This function is deprecated, use Centroids.from_hdf5 instead."""
        LOGGER.warning("The use of Centroids.read_hdf5 is deprecated."
                       "Use Centroids.from_hdf5 instead.")
        self.__dict__ = Centroids.from_hdf5(*args, **kwargs).__dict__

    @classmethod
    def from_hdf5(cls, file_data):
        """Create a centroids object from a HDF5 file.

        Parameters
        ----------
        file_data : str or h5
            If string, path to read data. If h5 object, the datasets will be read from there.

        Returns
        -------
        centr : Centroids
            Centroids with data from the given file
        """
        if isinstance(file_data, (str, Path)):
            LOGGER.info('Reading %s', file_data)
            with h5py.File(file_data, 'r') as data:
                return cls._from_hdf5(data)
        else:
            return cls._from_hdf5(file_data)

    @classmethod
    def _from_hdf5(cls, data):
        centr = None
        crs = DEF_CRS
        if data.get('crs'):
            crs = u_coord.to_crs_user_input(data.get('crs')[0])
        if data.get('lat') and data.get('lat').size:
            centr = cls.from_lat_lon(
                np.array(data.get('lat')),
                np.array(data.get('lon')),
                crs=crs)
        elif data.get('latitude') and data.get('latitude').size:
            centr = cls.from_lat_lon(
                np.array(data.get('latitude')),
                np.array(data.get('longitude')),
                crs=crs)
        else:
            centr_meta = data.get('meta')
            meta = dict()
            meta['crs'] = crs
            for key, value in centr_meta.items():
                if key != 'transform':
                    meta[key] = value[0]
                else:
                    meta[key] = rasterio.Affine(*value)
            centr = cls(meta=meta)

        for centr_name in data.keys():
            if centr_name not in ('crs', 'lat', 'lon', 'meta'):
                setattr(centr, centr_name, np.array(data.get(centr_name)))
        return centr

    def _ne_crs_geom(self, scheduler=None):
        """Return `geometry` attribute in the CRS of Natural Earth.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if u_coord.equal_crs(self.gdfgeometry.crs, u_coord.NE_CRS):
            return self.gdf.geometry
        return self.gdf.geometry.to_crs(u_coord.NE_CRS)
