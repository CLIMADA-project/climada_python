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
from typing import Union, Literal
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

LOGGER = logging.getLogger(__name__)

DEF_COLS = ['region_id', 'on_land']
DEF_SHEET_NAME = 'centroids'


class Centroids():
    """Contains vector centroids as a GeoDataFrame

    Attributes
    ----------
    lat : np.array
        latitudes in the chosen crs (can be any unit)
    lon : np.array
        longitudes in the chosen crs (can be any unit)
    crs : str, optional
        coordinate reference system, default is WGS84
    region_id : np.array, optional
        region numeric codes
        (can be any values, admin0, admin1, custom values)
    on_land : np.array, optional
        on land (True) and on sea (False)
    """

    def __init__(
        self,
        *,
        lat: Union[np.ndarray, list[float]],
        lon: Union[np.ndarray, list[float]],
        crs: str = DEF_CRS,
        region_id: Union[Literal["country"], None, np.ndarray, list[float]] = None,
        on_land: Union[Literal["natural_earth"], None, np.ndarray, list[bool]] = None,
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
        region_id : np.array, optional
            country region code of size size, Defaults to None array
        on_land : np.array, optional
            on land (True) and on sea (False) of size size. Defaults to None array
        """

        self.gdf = gpd.GeoDataFrame(
            data={
                'geometry': gpd.points_from_xy(lon, lat, crs=crs),
                'region_id': region_id,
                'on_land': on_land,
            }
        )

        if isinstance(region_id, str):
            LOGGER.info(f'Setting region id to {region_id} level.')
            self._set_region_id(level=region_id, overwrite=True)
        if isinstance(on_land, str):
            LOGGER.info(f'Setting on land from {on_land} source.')
            self._set_on_land(source=on_land, overwrite=True)

    @property
    def lat(self):
        """ Return latitudes """
        return self.gdf.geometry.y.values

    @property
    def lon(self):
        """ Return longitudes """
        return self.gdf.geometry.x.values

    @property
    def geometry(self):
        """ Return the geometry """
        return self.gdf['geometry']

    @property
    def on_land(self):
        """ Get the on_land property """
        if self.gdf.on_land.isna().all():
            return None
        return self.gdf['on_land'].values

    @property
    def region_id(self):
        """ Get the assigned region_id """
        if self.gdf.region_id.isna().all():
            return None
        return self.gdf['region_id'].values

    @property
    def crs(self):
        """ Get the crs"""
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
        """ dunder method for Centroids comparison. 
        returns True if two centroids equal, False otherwise

        Parameters
        ----------
        other : Centroids
            object to compare with

        Returns
        -------
        eq : bool
        """
        eq_crs = u_coord.equal_crs(self.crs, other.crs)
        try:
            pd.testing.assert_frame_equal(
                self.gdf, other.gdf, check_like=True
                )
            eq_df = True
        except AssertionError:
            eq_df = False

        return eq_crs & eq_df

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
        Centroids
            Centroids in the new crs

        """
        return self.to_crs(DEF_CRS, inplace=inplace)

    def to_crs(self, crs, inplace=False):
        """ Project the current centroids to the desired crs

        Parameters
        ----------
        crs : str
            coordinate reference system
        inplace: bool, default False
            if True, modifies the centroids in place.
            if False, returns a copy.

        Returns
        -------
        Centroids
            Centroids in the new crs
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
        if np.any(gdf.geom_type != 'Point'):
            raise ValueError(
                'The inpute geodataframe contains geometries'
                ' that are not points.'
            )

        # This is a bit ugly, but avoids to recompute the geometries
        # in the init. For large datasets this saves computation time
        centroids = cls(lat=[1], lon=[1]) #make "empty" centroids
        columns = [col for col in gdf.columns if col in DEF_COLS]
        columns.insert(0, 'geometry') #Same order as init
        centroids.gdf = gdf[columns]
        if not gdf.crs:
            centroids.gdf.set_crs(DEF_CRS, inplace=True)
        return centroids

    @classmethod
    def from_exposures(cls, exposures):
        """Generate centroids from the location of an exposures.

        Parameters
        ----------
        exposures : Exposure
            exposures from which to take the centroids location
            and region_id (if defined) and on_land (if defined)

        Returns
        -------
        Centroids
            Centroids built from the exposures geodataframe

        Raises
        ------
        ValueError
        """
        col_names = [
            column
            for column in exposures.gdf.columns
            if column in DEF_COLS
            ]

        # Legacy behaviour
        # Exposures can be without geometry column
        #TODO: remove once exposures is real geodataframe with geometry.
        if 'geometry' in exposures.gdf.columns:
            col_names.append('geometry')
            gdf = exposures.gdf[col_names]
            return cls.from_geodataframe(gdf)

        if 'latitude' in exposures.gdf.columns and 'longitude' in exposures.gdf.columns:
            gdf = exposures.gdf[col_names]
            return cls(
                lat = exposures.gdf['latitude'],
                lon = exposures.gdf['longitude'],
                crs = exposures.crs,
                **dict(gdf.items())
            )

        raise ValueError(
            "The given exposures object has no coordinates information."
            "The exposures' geodataframe must have either point geometries"
            " or latitude and longitude values.")

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
        return cls(lat=y_grid.flatten(), lon=x_grid.flatten(), crs=crs)

    def append(self, centr):
        """Append Centroids

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
        """Create the union of centroids from the inputs.
        The centroids are combined together point by point.
        All centroids must have the same CRS.

        Parameters
        ----------
        others : any number of climada.hazard.Centroids()
            Centroids to form the union with

        Returns
        -------
        centroids : Centroids
            Centroids containing the union of all Centroids.

        """
        centroids = copy.deepcopy(self)
        for cent in others:
            centroids.append(cent)

        # remove duplicate points
        return Centroids.remove_duplicate_points(centroids)

    @classmethod
    def remove_duplicate_points(cls, centr):
        """Return a copy of centroids with removed duplicated points

        Parameters
        ----------
        centr : Centroids
            Centroids with or without duplicate points      

        Returns
        -------
        centroids : Centroids
            Sub-selection of centroids without duplicates
        """
        return cls.from_geodataframe(centr.gdf.drop_duplicates())

    def select(self, reg_id=None, extent=None, sel_cen=None):
        """Return Centroids with points in the given reg_id and/or in an
        spatial extent and/or in an index based list

        Parameters
        ----------
        reg_id : int, optional
            region to filter according to region_id values
        extent : tuple, optional
            Format (min_lon, max_lon, min_lat, max_lat) tuple.
            If min_lon > lon_max, the extend crosses the antimeridian and is
            [lon_max, 180] + [-180, lon_min]
            Borders are inclusive.
        sel_cen : np.array, optional
            1-dim mask or 1-dim centroids indices, complements reg_id and extent

        Returns
        -------
        centroids : Centroids
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
        """Return mask of selected centroids

        Parameters
        ----------
        sel_cen: np.array(bool), optional
            boolean array mask for centroids
        reg_id : int, optional
            region to filter according to region_id values
        extent : tuple, optional
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
    def plot(self, ax=None, figsize=(9, 13), latlon_bounds_buffer=0.0, shapes=True, **kwargs):
        """Plot centroids scatter points over earth

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: (float, float), optional
            figure size for plt.subplots
            The default is (9, 13)
        latlon_bounds_buffer : float, optional
            Buffer to add to all sides of the bounding box. Default: 0.0.
        shapes : bool, optional
            overlay axis with coastlines. Default: True
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

        if shapes:
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

        Returns
        -------
        areapixels : np.array
            area values in m*m

        See also
        --------
        climada.util.coordinates.get_resolution
        """

        res = np.abs(u_coord.get_resolution(self.lat, self.lon, min_resol=min_resol)).min()
        LOGGER.debug('Setting area_pixel %s points.', str(self.lat.size))
        xy_pixels = self.geometry.buffer(res / 2, resolution=1, cap_style=3).envelope
        if PROJ_CEA == self.geometry.crs:
            area_pixel = xy_pixels.area.values
        else:
            area_pixel = xy_pixels.to_crs(crs={'proj': 'cea'}).area.values
        return area_pixel

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

    def get_dist_coast(self, signed=False, precomputed=False):
        """Get dist_coast attribute for every pixel or point in meters.

        Parameters
        ----------
        signed : bool, optional
            If True, use signed distances (positive off shore and negative on land). Default: False.
        precomputed : bool, optional
            If True, use precomputed distances (from NASA). Works only for crs=epsg:4326
            Default: False.

        Returns
        -------
        dist : np.array
            (Signed) distance to coast in meters.
        """
        ne_geom = self._ne_crs_geom()
        if precomputed:
            return u_coord.dist_to_coast_nasa(
                ne_geom.y.values, ne_geom.x.values, highres=True, signed=signed)
        else:
            LOGGER.debug('Computing distance to coast for %s centroids.', str(self.size))
            return u_coord.dist_to_coast(ne_geom, signed=signed)

    def get_meta(self, resolution=None):
        """Returns a meta raster based on the centroids bounds.

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
            file_name, [1], src_crs, window, geometry, dst_crs,
            transform, width, height, resampling)
        lat, lon = _meta_to_lat_lon(meta)
        if return_meta:
            return cls(lon=lon, lat=lat, crs=meta['crs']), meta
        return cls(lon=lon, lat=lat, crs=meta['crs'])

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
    def from_csv(cls, file_path):
        """Generate centroids from a CSV file with column names in var_names.

        Parameters
        ----------
        file_path : str
            path to CSV file to be read

        Returns
        -------
        Centroids
            Centroids with data from the given CSV file
        """
        df = pd.read_csv(file_path)
        return cls._from_dataframe(df)

    def write_csv(self, file_path):
        """Save centroids as CSV file

        Parameters
        ----------
        file_path : str, Path
            absolute or relative file path and name to write to
        """
        LOGGER.info('Writing %s', file_path)
        df = self._centroids_to_df()
        df.to_csv(Path(file_path).with_suffix('.csv'), index=False)


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
            sheet_name = 'centroids'
        df = pd.read_excel(file_path, sheet_name)
        return cls._from_dataframe(df)

    def write_excel(self, file_path):
        """Save centroids as excel file

        Parameters
        ----------
        file_path : str, Path
            absolute or relative file path and name to write to
        """
        LOGGER.info('Writing %s', file_path)
        df = self._centroids_to_df()
        df.to_excel(
            Path(file_path).with_suffix('.xlsx'),
            sheet_name=DEF_SHEET_NAME, index=False
            )

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

        Raises 
        ------    
        FileNotFoundError
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

    '''
    Private methods
    '''
    @classmethod
    def _from_dataframe(cls, df):
        if 'crs' in df.columns:
            crs = df['crs'].iloc[0]
        else:
            LOGGER.info(
                'No \'crs\' column provided in file,'
                'setting CRS to WGS84 default.'
                )
            crs = DEF_CRS

        extra_values = {
            col: df[col]
            for col in df.columns
            if col in DEF_COLS
            }

        return cls(
            lat=df['lat'], lon=df['lon'],
            **extra_values, crs=crs
            )

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

    @classmethod
    def _legacy_from_excel(cls, file_name, var_names):
        LOGGER.info('Reading %s', file_name)
        try:
            df = pd.read_excel(file_name, var_names['sheet_name'])
            df = df.rename(columns=var_names['col_name'])
        except KeyError as err:
            raise KeyError("Not existing variable: %s" % str(err)) from err
        return cls._from_dataframe(df)

    def _centroids_to_df(self):
        """Create dataframe from Centroids object to facilitate
        saving in different file formats.

        Returns
        -------
        df : DataFrame
        """

        df = pd.DataFrame(self.gdf)
        df['lon'] = self.gdf['geometry'].x
        df['lat'] = self.gdf['geometry'].y
        df = df.drop(['geometry'], axis=1)
        crs = CRS.from_user_input(self.crs).to_wkt()
        df['crs'] = crs
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

    def _set_region_id(self, level='country', overwrite=False):
        """Set region_id as country ISO numeric code attribute for every pixel or point.

        Parameters
        ----------
        level: str
            defines the admin level on which to assign centroids. Currently
            only 'country' (admin0) is implemented. Default is 'country'.
        overwrite : bool, optional
            if True, overwrites the existing region_id information.
            if False and region_id is None region_id is computed.
        """
        if overwrite or self.region_id is None:
            LOGGER.debug('Setting region_id %s points.', str(self.size))
            if level == 'country':
                ne_geom = self._ne_crs_geom()
                self.gdf['region_id'] = u_coord.get_country_code(
                    ne_geom.y.values, ne_geom.x.values
                    )
            else:
                raise NotImplementedError(
                    'The region id can only be assigned for countries so far'
                    )

    def _set_on_land(self, source='natural_earth', overwrite=False):
        """Set on_land attribute for every pixel or point.

        natural_earth: https://www.naturalearthdata.com/

        Parameters
        ----------
        source: str
            defines the source of the coastlines. Currently
            only 'natural_earth' is implemented.
            Default is 'natural_earth'.
        overwrite : bool
            if True, overwrites the existing on_land information.
            if False and on_land is None on_land is computed.
        """
        if overwrite or self.on_land is None:
            LOGGER.debug('Setting on_land %s points.', str(self.lat.size))
            if source=='natural_earth':
                ne_geom = self._ne_crs_geom()
                self.gdf['on_land'] = u_coord.coord_on_land(
                    ne_geom.y.values, ne_geom.x.values
                )
            else:
                raise NotImplementedError(
                    'The on land variables can only be assigned'
                    'using natural earth.'
                    )


def _meta_to_lat_lon(meta):
    """Compute lat and lon of every pixel center from meta raster.

    Parameters
    ----------
    meta : dict
        meta description of raster

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        latitudes, longitudes
    """
    xgrid, ygrid = u_coord.raster_to_meshgrid(
        meta['transform'], meta['width'], meta['height']
        )
    return ygrid.flatten(), xgrid.flatten()
