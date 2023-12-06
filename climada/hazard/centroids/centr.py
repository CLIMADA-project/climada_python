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
from typing import Optional
import warnings

import h5py
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj.crs.crs import CRS
import rasterio
from shapely.geometry.point import Point

from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coord
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
        'on_land': 'on_land',
        'dist_coast': 'dist_coast',
        'elevation': 'elevation',
       ' area_pixel': 'area_pixel'
    }
}
"""Excel variable names"""

DEF_VAR_CSV = {
    'lat': 'latitude',
    'lon': 'longitude',
}
"""CSV variable names"""

LOGGER = logging.getLogger(__name__)


class Centroids():
    """Contains raster or vector centroids.

    Attributes
    ----------
    lat : np.array
        latitudes in the chosen crs (can be any unit)
    lon : np.array
        longitudes in the chosen crs (can be any unit)
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
        longitude: np.ndarray,
        latitude: np.ndarray,
        crs: str = DEF_CRS,
        region_id: Optional[np.ndarray] = None,
        on_land: Optional[np.ndarray] = None,
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
        kwargs:
            columns of values to passed to the geodataframe constructor
        """
        attr_dict = {
            'geometry': gpd.points_from_xy(longitude, latitude, crs=crs),
        }
        if region_id is not None:
            attr_dict['region_id'] = region_id
        if on_land is not None:
            attr_dict['on_land'] = on_land
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
        if 'on_land' in self.gdf.columns.values:
            return self.gdf['on_land']
        return None

    @property
    def region_id(self):
        if 'region_id' in self.gdf.columns.values:
            return self.gdf['region_id']
        return None

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

    def to_default_crs(self):
        """
        Project the current centroids to the default CRS (epsg4326)
        Modifies the object in place.
        """
        self.gdf.to_crs(DEF_CRS, inplace=True)

    def to_crs(self, crs):
        """
        Project the current centroids to the default CRS (epsg4326)
        Modifies the object in place.

        Parameters
        ----------
        crs : str
            coordinate reference system

        Returns
        -------
        Centroids
            Centroids in the new crs
        """
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
        """
        return cls(
            longitude=gdf.geometry.x.values,
            latitude=gdf.geometry.y.values,
            crs=gdf.crs,
            **gdf.drop(columns=['geometry', 'latitude', 'longitude'],
                       errors='ignore').to_dict(orient='list')
            )

    @classmethod
    def from_exposures(cls, exposures):
        """
        Generate centroids from the location of an exposures.

        Parameters
        ----------
        exposures : Exposure
            exposures from which to take the centroids location
            and region_id (if defined)

        Returns
        -------
        Centroids
            Centroids built from the exposures
        """
        gdf = exposures.gdf[['geometry', 'region_id']]
        return cls.from_geodataframe(gdf)

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
        return cls(latitude=y_grid.flatten(), longitude=x_grid.flatten(), crs=crs)

    def append(self, centr):
        """Append centroids points.

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
        """
        if not u_coord.equal_crs(self.crs, centr.crs):
            raise ValueError(
                "The centroids have different Coordinate-Reference-Systems (CRS)")
        self.gdf = pd.concat([self.gdf, centr.gdf])

    def union(self, *others):
        """
        Create the union of centroids from the inputs.
        The centroids are combined together point by point.
        All centroids must have the same CRS.

        Parameters
        ----------
        others : any number of climada.hazard.Centroids()
            Centroids to form the union with

        Returns
        -------
        centroids : Centroids
            Centroids containing the union of the centroids in others.

        """
        centroids = copy.deepcopy(self)
        for cent in others:
            centroids.append(cent)

        # remove duplicate points
        return Centroids.remove_duplicate_points(centroids)

    def get_closest_point(self, x_lon, y_lat):
        """Returns closest centroid and its index to a given point.

        Parameters
        ----------
        x_lon : float
            x coord (lon)
        y_lat : float
            y coord (lat)

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

    def set_region_id(self, level='country'):
        """Set region_id as country ISO numeric code attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        LOGGER.debug('Setting region_id %s points.', str(self.size))
        if level == 'country':
            if u_coord.equal_crs(self.crs, 'epsg:4326'):
                self.gdf['region_id'] = u_coord.get_country_code(
                    self.lat, self.lon)
            else:
                raise NotImplementedError(
                    'The region id can only be assigned if the crs is epsg:4326'
                    'Please use .to_default_crs to change to epsg:4326'
                    )
        else:
            raise NotImplementedError(
                'The region id can only be assigned for countries so far'
                )


    # NOT REALLY AN ELEVATION FUNCTION, JUST READ RASTER
    def get_elevation(self, topo_path):
        """Return elevation attribute for every pixel or point in meters.

        Parameters
        ----------
        topo_path : str
            Path to a raster file containing gridded elevation data.
        """
        return u_coord.read_raster_sample(topo_path, self.lat, self.lon)

    def get_dist_coast(self, signed=False, precomputed=False):
        """Get dist_coast attribute for every pixel or point in meters.

        Parameters
        ----------
        signed : bool
            If True, use signed distances (positive off shore and negative on land). Default: False.
        precomputed : bool
            If True, use precomputed distances (from NASA). Works only for crs=epsg:4326
            Default: False.
        """
        if not u_coord.equal_crs(self.crs, 'epsg:4326'):
            raise NotImplementedError(
                'The distance to coast can only be assigned if the crs is epsg:4326'
                'Please use .to_default_crs to change to epsg:4326'
                )
        if precomputed:
            return u_coord.dist_to_coast_nasa(
                self.lat, self.lon, highres=True, signed=signed)
        else:
            ne_geom = self._ne_crs_geom()
            LOGGER.debug('Computing distance to coast for %s centroids.', str(self.size))
            return u_coord.dist_to_coast(ne_geom, signed=signed)

    def set_on_land(self,):
        """Set on_land attribute for every pixel or point.

        Parameters
        ----------
        scheduler : str
            used for dask map_partitions. “threads”, “synchronous” or “processes”
        """
        if not u_coord.equal_crs(self.crs, 'epsg:4326'):
            raise NotImplementedError(
                'The on land property can only be assigned if the crs is epsg:4326'
                'Please use .to_default_crs to change to epsg:4326'
                )

        LOGGER.debug('Setting on_land %s points.', str(self.lat.size))
        self.gdf['on_land'] = u_coord.coord_on_land(
            self.lat, self.lon
        )

    @classmethod
    def remove_duplicate_points(cls, centroids):
        """Return a copy of centroids with removed duplicated points

        Returns
        -------
         : Centroids
            Sub-selection of centroids withtout duplicates
        """
        return cls.from_geodataframe(centroids.gdf.drop_duplicates())

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
            1-dim mask or 1-dim centroids indices, complements reg_id and extent

        Returns
        -------
        cen : Centroids
            Sub-selection of this object
        """
        sel_cen_bool = sel_cen
        #if needed, convert indices to bool
        if sel_cen is not None:
            if sel_cen.dtype.kind == 'i':  #is integer
                sel_cen_bool = np.zeros(self.size, dtype=bool)
                sel_cen_bool[np.unique(sel_cen)] = True

        sel_cen_mask = self.select_mask(sel_cen=sel_cen_bool, reg_id=reg_id, extent=extent)
        return Centroids.from_geodataframe(self.gdf[sel_cen_mask])


    def select_mask(self, sel_cen=None, reg_id=None, extent=None):
        """
        Make mask of selected centroids

        Parameters
        ----------
        sel_cen: np.array(bool)
            boolean array mask for centroids
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
        if sel_cen is None:
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

    #TODO replace with nice Geodataframe util plot method.
    def plot(self, ax=None, figsize=(9, 13), latlon_bounds_buffer=0.0, **kwargs):
        """Plot centroids scatter points over earth.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        latlon_bounds_buffer : float, optional
            Buffer to add to all sides of the bounding box. Default: 0.0.
        kwargs : optional
            arguments for scatter matplotlib function

        Returns
        -------
        axis : matplotlib.axes._subplots.AxesSubplot
        """
        proj_data, _ = u_plot.get_transformation(self.crs)
        proj_plot = proj_data
        if isinstance(proj_data, ccrs.PlateCarree):
            # use different projections for plot and data to shift the central lon in the plot
            xmin, _ymin, xmax, _ymax = u_coord.latlon_bounds(self.lat, self.lon,
                                                           buffer=latlon_bounds_buffer)
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))

        if ax is None:
            ax = self.gdf.copy().to_crs(proj_plot).plot(figsize=figsize, **kwargs)
        else:
            self.gdf.copy().to_crs(proj_plot).plot(figsize=figsize, **kwargs)

        u_plot.add_shapes(ax)
        plt.tight_layout()
        return ax

    def get_area_pixel(self, min_resol=1.0e-8):
        """Computes the area per centroid in the CEA projection
        assuming that the centroids define a regular grid of pixels
        (area in m*m).

        Parameters
        ----------
        min_resol : float, optional
            Use this minimum resolution in lat and lon. Is passed to the
            method climada.util.coordinates.get_resolution.
            Default: 1.0e-8

        See also
        --------
        climada.util.coordinates.get_resolution
        """

        res = u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol)
        res = np.abs(res).min()
        LOGGER.debug('Setting area_pixel %s points.', str(self.lat.size))
        xy_pixels = self.geometry.buffer(res / 2).envelope
        if PROJ_CEA == self.geometry.crs:
            area_pixel = xy_pixels.area.values
        else:
            area_pixel = xy_pixels.to_crs(crs={'proj': 'cea'}).area.values
        return area_pixel


    '''
    I/O methods
    '''

    @classmethod
    def from_raster_file(cls, file_name, src_crs=None, window=None, geometry=None,
                         dst_crs=None, transform=None, width=None, height=None,
                         resampling=rasterio.warp.Resampling.nearest, return_meta=False):
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

        Returns
        -------
        centr : Centroids
            Centroids with meta attribute according to the given raster file
        """
        meta, _ = u_coord.read_raster(
            file_name, [1], src_crs, window, geometry, dst_crs,
            transform, width, height, resampling)
        lat, lon = _meta_to_lat_lon(meta)
        if return_meta:
            return cls(longitude=lon, latitude=lat, crs=meta['crs']), meta
        return cls(longitude=lon, latitude=lat, crs=meta['crs'])

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
        crs = meta['crs']
        lat, lon = _meta_to_lat_lon(meta)
        return cls(longitude=lon, latitude=lat, crs=crs)

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

        centroids = cls.from_geodataframe(gpd.read_file(file_name))
        if dst_crs is not None:
            centroids.gdf.to_crs(dst_crs, inplace=True)
        return centroids

    #TODO: Check whether other variables are necessary, e.g. dist to coast
    @classmethod
    def from_csv(cls, file_path, crs=None, var_names=None):
        """
        Generate centroids from a csv file with column names in var_names.

        Parameters
        ----------
        file_path : str
            path to csv file to be read
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS
        var_names : dict, default
            name of the variables. Default: DEF_VAR_CSV

        Returns
        -------
        Centroids
            Centroids with data from the given csv file
        """
        if crs is None:
            crs = DEF_CRS

        if var_names is None:
            var_names = DEF_VAR_CSV

        df = pd.read_csv(file_path)
        geometry = gpd.points_from_xy(df[var_names['lon']], df[var_names['lat']])
        gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        return cls.from_geodataframe(gdf)

    #TODO: this method is badly written but kept for backwards compatibility. It should be improved.
    @classmethod
    def from_excel(cls, file_name, crs=None, var_names=None):
        """Generate a new centroids object from an excel file with column names in var_names.

        Parameters
        ----------
        file_name : str
            absolute or relative file name
        crs : dict() or rasterio.crs.CRS, optional
            CRS. Default: DEF_CRS
        var_names : dict, default
            name of the variables

        Raises
        ------
        KeyError

        Returns
        -------
        centr : Centroids
            Centroids with data from the given excel file
        """
        if crs is None:
            crs = DEF_CRS

        if var_names is None:
            var_names = DEF_VAR_EXCEL

        try:
            df = pd.read_excel(file_name, var_names['sheet_name'])

        except KeyError as err:
            raise KeyError(f"Not existing variable: {err}") from err

        geometry = gpd.points_from_xy(df[var_names['col_name']['lon']],
                                      df[var_names['col_name']['lat']])
        gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        region_column = var_names['col_name'].get('region_id')
        if region_column and region_column in df.columns.values:
            gdf['region_id'] = df[region_column]
        return cls.from_geodataframe(gdf)

    def write_hdf5(self, file_name, mode='w'):
        """Write data frame and metadata in hdf5 format

        Parameters
        ----------
        file_name : str
            (path and) file name to write to.
        """
        LOGGER.info('Writing %s', file_name)
        store = pd.HDFStore(file_name, mode=mode)
        pandas_df = pd.DataFrame(self.gdf)
        for col in pandas_df.columns:
            if str(pandas_df[col].dtype) == "geometry":
                pandas_df[col] = np.asarray(self.gdf[col])

        # Avoid pandas PerformanceWarning when writing HDF5 data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
            # Write dataframe
            store.put('centroids', pandas_df)

        store.get_storer('centroids').attrs.metadata = {
            'crs': CRS.from_user_input(self.crs).to_wkt()
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
        """
        if not Path(file_name).is_file():
            raise FileNotFoundError(str(file_name))
        try:
            with pd.HDFStore(file_name, mode='r') as store:
                metadata = store.get_storer('centroids').attrs.metadata
                # in previous versions of CLIMADA and/or geopandas,
                # the CRS was stored in '_crs'/'crs'
                crs = metadata.get('crs')
                gdf = gpd.GeoDataFrame(store['centroids'], crs=crs)
        except TypeError:
            with h5py.File(file_name, 'r') as data:
                gdf = cls._legacy_from_hdf5(data.get('centroids'))
        except KeyError:
            with h5py.File(file_name, 'r') as data:
                gdf = cls._legacy_from_hdf5(data)

        return cls.from_geodataframe(gdf)

    @classmethod
    def _legacy_from_hdf5(cls, data):
        crs = DEF_CRS
        if data.get('crs'):
            crs = u_coord.to_crs_user_input(data.get('crs')[0])
        if data.get('lat') and data.get('lat').size:
            latitude = np.array(data.get('lat'))
            longitude = np.array(data.get('lon'))
        elif data.get('latitude') and data.get('latitude').size:
            latitude = np.array(data.get('latitude'))
            longitude = np.array(data.get('longitude'))
        else:
            centr_meta = data.get('meta')
            meta = dict()
            meta['crs'] = crs
            for key, value in centr_meta.items():
                if key != 'transform':
                    meta[key] = value[0]
                else:
                    meta[key] = rasterio.Affine(*value)
            latitude, longitude = _meta_to_lat_lon(meta)

        extra_values = {}
        for centr_name in data.keys():
            if centr_name not in ('crs', 'lat', 'lon', 'meta', 'latitude', 'longitude'):
                values = np.array(data.get(centr_name))
                if latitude.size != 0 and values.size != 0 :
                    extra_values[centr_name] = values

        return gpd.GeoDataFrame(
            extra_values,
            geometry=gpd.points_from_xy(x=longitude, y=latitude, crs=crs)
        )


    def _ne_crs_geom(self):
        """Return `geometry` attribute in the CRS of Natural Earth.

        Returns
        -------
        geo : gpd.GeoSeries
        """
        if u_coord.equal_crs(self.gdf.geometry.crs, u_coord.NE_CRS):
            return self.gdf.geometry
        return self.gdf.geometry.to_crs(u_coord.NE_CRS)

    def get_meta(self, resolution=None):
        """
        Returns a meta raster based on the centroids bounds.

        When resolution is None it is estimated from the centroids
        by assuming that they form a regular raster.

        Parameters
        ----------
        resolution : int, optional
            Resolution of the raster.
            By default None (resolution is estimated from centroids)

        Returns
        -------
        meta: dict
            meta raster representation of the centroids
        """
        if resolution is None:
            resolution = np.abs(u_coord.get_resolution(self.lat, self.lon)).min()
        xmin, ymin, xmax, ymax = self.gdf.total_bounds
        rows, cols, ras_trans = u_coord.pts_to_raster_meta(
            (xmin, ymin, xmax, ymax), (resolution, -resolution)
            )
        meta = {
            'crs': self.crs,
            'height': rows,
            'width': cols,
            'transform': ras_trans,
        }
        return meta

def _meta_to_lat_lon(meta):
    """Compute lat and lon of every pixel center from meta raster.

    Parameters
    ----------
    meta : dict
        meta description of raster

    Returns
    -------
    (np.ndarray, np.ndarray)
        latitudes, longitudes
    """
    xgrid, ygrid = u_coord.raster_to_meshgrid(
        meta['transform'], meta['width'], meta['height']
        )
    return ygrid.flatten(), xgrid.flatten()
