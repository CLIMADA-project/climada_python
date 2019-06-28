"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Centroids class.
"""
import copy
import logging
import numpy as np
from scipy import sparse
import pandas as pd
from rasterio import Affine
from rasterio.warp import Resampling
from geopandas import GeoSeries
from shapely.geometry.point import Point
from shapely.vectorized import contains
import climada.util.plot as u_plot

from climada.util.constants import DEF_CRS, ONE_LAT_KM
import climada.util.hdf5_handler as hdf5
from climada.util.coordinates import get_country_geometries, dist_to_coast, \
get_resolution, coord_on_land, pts_to_raster_meta, read_raster, read_vector, NE_CRS, \
equal_crs

__all__ = ['Centroids']

DEF_VAR_MAT = {'field_names': ['centroids', 'hazard'],
               'var_name': {'lat' : 'lat',
                            'lon' : 'lon',
                            'dist_coast': 'distance2coast_km',
                            'admin0_name': 'admin0_name',
                            'admin0_iso3': 'admin0_ISO3',
                            'comment': 'comment',
                            'region_id': 'NatId'
                           }
              }
""" MATLAB variable names """

DEF_VAR_EXCEL = {'sheet_name': 'centroids',
                 'col_name': {'region_id' : 'region_id',
                              'lat' : 'latitude',
                              'lon' : 'longitude',
                             }
                }
""" Excel variable names """

LOGGER = logging.getLogger(__name__)

class Centroids():
    """ Contains raster or vector centroids. Raster data can be set with
    set_raster_file() or set_meta(). Vector data can be set with set_lat_lon()
    or set_vector_file().

    Attributes:
        meta (dict, optional): rasterio meta dictionary containing raster
            properties: width, height, crs and transform must be present
            at least (transform needs to contain upper left corner!)
        lat (np.array, optional): latitude of size size
        lon (np.array, optional): longitude of size size
        geometry (GeoSeries, optional): contains lat and lon crs. Might contain
            geometry points for lat and lon
        area_pixel (np.array, optional): area of size size
        dist_coast (np.array, optional): distance to coast of size size
        on_land (np.array, optional): on land (True) and on sea (False) of size size
        region_id (np.array, optional): country region code of size size
    """

    vars_check = {'lat', 'lon', 'geometry', 'area_pixel', 'dist_coast',
                  'on_land', 'region_id'}
    """ Variables whose size will be checked """

    def __init__(self):
        """ Initialize to None raster and vector """
        self.meta = dict()
        self.geometry = GeoSeries()
        self.lat = np.array([])
        self.lon = np.array([])
        self.area_pixel = np.array([])
        self.dist_coast = np.array([])
        self.on_land = np.array([])
        self.region_id = np.array([])

    def check(self):
        """ Check that either raster meta attribute is set or points lat, lon
        and geometry.crs. Check attributes sizes """
        n_centr = self.lat.size
        for var_name, var_val in self.__dict__.items():
            if var_name in self.vars_check:
                if var_val.size > 0 and var_val.size != n_centr:
                    LOGGER.error('Wrong %s size: %s != %s.', var_name,
                                 str(n_centr), str(var_val.size))
                    raise ValueError
        if self.meta:
            if 'width' not in self.meta.keys() or 'height' not in self.meta.keys() or \
            'crs' not in self.meta.keys() or 'transform' not in self.meta.keys():
                LOGGER.error('Missing meta information: width, height,'\
                             + 'crs or transform')
                raise ValueError
            if self.meta['transform'][4] > 0:
                LOGGER.error('Meta does not contain upper left corner data.')
                raise ValueError

    def equal(self, centr):
        """ Return true if two centroids equal, false otherwise

        Parameters:
            centr (Centroids): centroids to compare

        Returns:
            bool
        """
        if self.meta and centr.meta:
            return equal_crs(self.meta['crs'], centr.meta['crs']) and \
            self.meta['height'] == centr.meta['height'] and \
            self.meta['width'] == centr.meta['width'] and \
            self.meta['transform'] == centr.meta['transform']
        return equal_crs(self.geometry.crs, centr.geometry.crs) and \
        self.lat.shape == centr.lat.shape and self.lon.shape == centr.lon.shape and \
        np.allclose(self.lat, centr.lat) and np.allclose(self.lon, centr.lon)

    def set_raster_from_pix_bounds(self, xf_lat, xo_lon, d_lat, d_lon, n_lat,
                                   n_lon, crs=DEF_CRS):
        """ Set raster metadata (meta attribute) from pixel border data

        Parameters:
            xf_lat (float): upper latitude (top)
            xo_lon (float): left longitude
            d_lat (float): latitude step
            d_lon (float): longitude step
            n_lat (int): number of latitude points
            n_lon (int): number of longitude points
            crs (dict() or rasterio.crs.CRS, optional): CRS. Default: DEF_CRS
        """
        self.__init__()
        self.meta = {'dtype':'float32', 'width':n_lon, 'height':n_lat,
                     'crs':crs, 'transform':Affine(d_lon, 0.0, xo_lon,
                                                   0.0, d_lat, xf_lat)}

    def set_raster_from_pnt_bounds(self, points_bounds, res, crs=DEF_CRS):
        """ Set raster metadata (meta attribute) from points border data.
        Raster border = point_border + res/2

        Parameters:
            points_bounds (tuple): points' lon_min, lat_min, lon_max, lat_max
            res (float): desired resolution in same units as points_bounds
            crs (dict() or rasterio.crs.CRS, optional): CRS. Default: DEF_CRS
        """
        self.__init__()
        rows, cols, ras_trans = pts_to_raster_meta(points_bounds, res)
        self.set_raster_from_pix_bounds(ras_trans[5], ras_trans[2], ras_trans[4],
                                        ras_trans[0], rows, cols, crs)

    def set_lat_lon(self, lat, lon, crs=DEF_CRS):
        """ Set Centroids points from given latitude, longitude and CRS.

        Parameters:
            lat (np.array): latitude
            lon (np.array): longitude
            crs (dict() or rasterio.crs.CRS, optional): CRS. Default: DEF_CRS
        """
        self.__init__()
        self.lat, self.lon, self.geometry = lat, lon, GeoSeries(crs=crs)

    def set_raster_file(self, file_name, band=[1], src_crs=None, window=False,
                        geometry=False, dst_crs=False, transform=None, width=None,
                        height=None, resampling=Resampling.nearest):
        """ Read raster of bands and set 0 values to the masked ones. Each
        band is an event. Select region using window or geometry. Reproject
        input by proving dst_crs and/or (transform, width, height).

        Parameters:
            file_pth (str): path of the file
            band (int, optional): band number to read. Default: 1
            src_crs (crs, optional): source CRS. Provide it if error without it.
            window (rasterio.windows.Window, optional): window to read
            geometry (shapely.geometry, optional): consider pixels only in shape
            dst_crs (crs, optional): reproject to given crs
            transform (rasterio.Affine): affine transformation to apply
            wdith (float): number of lons for transform
            height (float): number of lats for transform
            resampling (rasterio.warp,.Resampling optional): resampling
                function used for reprojection to dst_crs

        Raises:
            ValueError

        Returns:
            np.array
        """
        if not self.meta:
            self.meta, inten = read_raster(file_name, band, src_crs, window,
                                           geometry, dst_crs, transform, width,
                                           height, resampling)
            return sparse.csr_matrix(inten)

        tmp_meta, inten = read_raster(file_name, band, src_crs, window, geometry,
                                      dst_crs, transform, width, height, resampling)
        if (tmp_meta['crs'] != self.meta['crs']) or \
        (tmp_meta['transform'] != self.meta['transform']) or \
        (tmp_meta['height'] != self.meta['height']) or \
        (tmp_meta['width'] != self.meta['width']):
            LOGGER.error('Raster data inconsistent with contained raster.')
            raise ValueError
        return sparse.csr_matrix(inten)

    def set_vector_file(self, file_name, inten_name=['intensity'], dst_crs=None):
        """ Read vector file format supported by fiona. Each intensity name is
        considered an event. Returns intensity array with shape
        (len(inten_name), len(geometry)).

        Parameters:
            file_name (str): vector file with format supported by fiona and
                'geometry' field.
            inten_name (list(str)): list of names of the columns of the
                intensity of each event.
            dst_crs (crs, optional): reproject to given crs

        Returns:
            np.array
        """
        if not self.geometry.crs:
            self.lat, self.lon, self.geometry, inten = read_vector(file_name, \
                inten_name, dst_crs)
            return sparse.csr_matrix(inten)
        tmp_lat, tmp_lon, tmp_geometry, inten = read_vector(file_name, \
            inten_name, dst_crs)
        if not equal_crs(tmp_geometry.crs, self.geometry.crs) or \
        not np.allclose(tmp_lat, self.lat) or\
        not np.allclose(tmp_lon, self.lon):
            LOGGER.error('Vector data inconsistent with contained vector.')
            raise ValueError
        return sparse.csr_matrix(inten)

    def read_mat(self, file_name, var_names=DEF_VAR_MAT):
        """ Read centroids from CLIMADA's MATLAB version

        Parameters:
            file_name (str): absolute or relative file name
            var_names (dict, default): name of the variables

        Raises:
            KeyError
        """
        LOGGER.info('Reading %s', file_name)
        if var_names is None:
            var_names = DEF_VAR_MAT

        cent = hdf5.read(file_name)
        # Try open encapsulating variable FIELD_NAMES
        num_try = 0
        for field in var_names['field_names']:
            try:
                cent = cent[field]
                break
            except KeyError:
                num_try += 1
        if num_try == len(var_names['field_names']):
            LOGGER.warning("Variables are not under: %s.", var_names['field_names'])

        try:
            cen_lat = np.squeeze(cent[var_names['var_name']['lat']])
            cen_lon = np.squeeze(cent[var_names['var_name']['lon']])
            self.set_lat_lon(cen_lat, cen_lon)

            try:
                self.dist_coast = np.squeeze(cent[var_names['var_name']['dist_coast']])
            except KeyError:
                pass
            try:
                self.region_id = np.squeeze(cent[var_names['var_name']['region_id']])
            except KeyError:
                pass
        except KeyError as err:
            LOGGER.error("Not existing variable: %s", str(err))
            raise err

    def read_excel(self, file_name, var_names=DEF_VAR_EXCEL):
        """ Read centroids from excel file with column names in var_names

        Parameters:
            file_name (str): absolute or relative file name
            var_names (dict, default): name of the variables

        Raises:
            KeyError
        """
        LOGGER.info('Reading %s', file_name)
        if var_names is None:
            var_names = DEF_VAR_EXCEL

        try:
            dfr = pd.read_excel(file_name, var_names['sheet_name'])
            self.set_lat_lon(dfr[var_names['col_name']['lat']],
                             dfr[var_names['col_name']['lon']])
            try:
                self.region_id = dfr[var_names['col_name']['region_id']]
            except KeyError:
                pass

        except KeyError as err:
            LOGGER.error("Not existing variable: %s", str(err))
            raise err

    def clear(self):
        """ Clear vector and raster data """
        self.__init__()

    def append(self, centr):
        """ Append raster or points. Raster needs to have the same resolution """
        if self.meta and centr.meta:
            LOGGER.debug('Appending raster')
            if centr.meta['crs'] != self.meta['crs']:
                LOGGER.error('Different CRS not accepted.')
                raise ValueError
            if self.meta['transform'][0] != centr.meta['transform'][0] or \
            self.meta['transform'][4] != centr.meta['transform'][4]:
                LOGGER.error('Different raster resolutions.')
                raise ValueError
            left = min(self.total_bounds[0], centr.total_bounds[0])
            bottom = min(self.total_bounds[1], centr.total_bounds[1])
            right = max(self.total_bounds[2], centr.total_bounds[2])
            top = max(self.total_bounds[3], centr.total_bounds[3])
            crs = self.meta['crs']
            width = (right - left)/self.meta['transform'][0]
            height = (bottom - top)/self.meta['transform'][4]
            self.meta = {'dtype':'float32', 'width':width, 'height':height,
                         'crs':crs, 'transform':Affine(self.meta['transform'][0], \
                         0.0, left, 0.0, self.meta['transform'][4], top)}
        else:
            LOGGER.debug('Appending points')
            if not equal_crs(centr.geometry.crs, self.geometry.crs):
                LOGGER.error('Different CRS not accepted.')
                raise ValueError
            lat = np.append(self.lat, centr.lat)
            lon = np.append(self.lon, centr.lon)
            crs = self.geometry.crs
            self.__init__()
            self.set_lat_lon(lat, lon, crs)

    def get_closest_point(self, x_lon, y_lat, scheduler=None):
        """ Returns closest centroid and its index to a given point.

        Parameters:
            x_lon (float): x coord (lon)
            y_lat (float): y coord (lat)
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

        Returns:
            x_close (float), y_close (float), idx_close (int)
        """
        if self.meta:
            if not self.lat.size or not self.lon.size:
                self.set_meta_to_lat_lon()
            i_lat = np.floor((self.meta['transform'][5]- y_lat)/abs(self.meta['transform'][4]))
            i_lon = np.floor((x_lon - self.meta['transform'][2])/abs(self.meta['transform'][0]))
            close_idx = int(i_lat*self.meta['width'] + i_lon)
        else:
            self.set_geometry_points(scheduler)
            close_idx = self.geometry.distance(Point(x_lon, y_lat)).values.argmin()
        return self.lon[close_idx], self.lat[close_idx], close_idx

    def set_region_id(self, scheduler=None):
        """ Set region_id attribute for every pixel or point

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        lon_ne, lat_ne = self._ne_crs_xy(scheduler)
        LOGGER.debug('Setting region_id %s points.', str(self.lat.size))
        countries = get_country_geometries(extent=(lon_ne.min(), lon_ne.max(),
                                                   lat_ne.min(), lat_ne.max()))
        self.region_id = np.zeros(lon_ne.size, dtype=int)
        for geom in zip(countries.geometry, countries.ISO_N3):
            select = contains(geom[0], lon_ne, lat_ne)
            self.region_id[select] = int(geom[1])

    def set_area_pixel(self, scheduler=None):
        """ Set area_pixel attribute for every pixel or point. area in m*m

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        if self.meta:
            if hasattr(self.meta['crs'], 'linear_units') and \
            str.lower(self.meta['crs'].linear_units) in ['m', 'metre', 'meter']:
                self.area_pixel = np.zeros((self.meta['height'], self.meta['width']))
                self.area_pixel *= abs(self.meta['transform'].a)*abs(self.meta['transform'].e)
                return
            if abs(abs(self.meta['transform'].a) -
                   abs(self.meta['transform'].e)) > 1.0e-5:
                LOGGER.error('Area can not be computed for not squared pixels.')
                raise ValueError
            res = self.meta['transform'].a
        else:
            res = min(get_resolution(self.lat, self.lon))
        self.set_geometry_points(scheduler)
        LOGGER.debug('Setting area_pixel %s points.', str(self.lat.size))
        xy_pixels = self.geometry.buffer(res/2).envelope
        if ('units' in self.geometry.crs and \
        self.geometry.crs['units'] in ['m', 'metre', 'meter']) or \
        equal_crs(self.geometry.crs, {'proj':'cea'}):
            self.area_pixel = xy_pixels.area.values
        else:
            self.area_pixel = xy_pixels.to_crs(crs={'proj':'cea'}).area.values

    def set_area_approx(self):
        """ Computes approximated area_pixel values: differentiated per latitude.
        area in m*m. Faster than set_area_pixel """
        if self.meta:
            if hasattr(self.meta['crs'], 'linear_units') and \
            str.lower(self.meta['crs'].linear_units) in ['m', 'metre', 'meter']:
                self.area_pixel = np.zeros((self.meta['height'], self.meta['width']))
                self.area_pixel *= abs(self.meta['transform'].a)*abs(self.meta['transform'].e)
                return
            res_lat, res_lon = self.meta['transform'].e, self.meta['transform'].a
            lat_unique = np.arange(self.meta['transform'].f + res_lat/2, \
                self.meta['transform'].f + self.meta['height'] * res_lat, res_lat)
            lon_unique_len = self.meta['width']
            res_lat = abs(res_lat)
        else:
            res_lat, res_lon = get_resolution(self.lat, self.lon)
            lat_unique = np.array(np.unique(self.lat))
            lon_unique_len = len(np.unique(self.lon))
            if ('units' in self.geometry.crs and \
            self.geometry.crs['units'] in ['m', 'metre', 'meter']) or \
            equal_crs(self.geometry.crs, {'proj':'cea'}):
                self.area_pixel = np.repeat(res_lat * res_lon, lon_unique_len)
                return

        LOGGER.debug('Setting area_pixel approx %s points.', str(self.lat.size))
        res_lat = res_lat * ONE_LAT_KM * 1000
        res_lon = res_lon * ONE_LAT_KM * 1000 * np.cos(np.radians(lat_unique))
        area_approx = np.repeat(res_lat * res_lon, lon_unique_len)
        if area_approx.size == self.size:
            self.area_pixel = area_approx
        else:
            LOGGER.error('Pixel area of points can not be computed.')
            raise ValueError

    def set_dist_coast(self, scheduler=None):
        """ Set dist_coast attribute for every pixel or point. Distan to
        coast is computed in meters

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        lon, lat = self._ne_crs_xy(scheduler)
        LOGGER.debug('Setting dist_coast %s points.', str(self.lat.size))
        self.dist_coast = dist_to_coast(lat, lon)

    def set_on_land(self, scheduler=None):
        """ Set on_land attribute for every pixel or point

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        lon, lat = self._ne_crs_xy(scheduler)
        LOGGER.debug('Setting on_land %s points.', str(self.lat.size))
        self.on_land = coord_on_land(lat, lon)

    def remove_duplicate_points(self, scheduler=None):
        """ Return Centroids with removed duplicated points

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

        Returns:
            Centroids
        """
        self.set_geometry_points(scheduler)
        geom_wkb = self.geometry.apply(lambda geom: geom.wkb)
        sel_cen = geom_wkb.drop_duplicates().index
        return self.select(sel_cen=sel_cen)

    def select(self, reg_id=None, sel_cen=None):
        """ Return Centroids with points in the given reg_id or within mask

        Parameters:
            reg_id (int): region to filter according to region_id values
            sel_cen (np.array): 1-dim mask

        Returns:
            Centroids
        """
        if sel_cen is None:
            sel_cen = np.isin(self.region_id, reg_id)

        if not self.lat.size or not self.lon.size:
            self.set_meta_to_lat_lon()

        centr = Centroids()
        centr.set_lat_lon(self.lat[sel_cen], self.lon[sel_cen], self.geometry.crs)
        if self.area_pixel.size:
            centr.area_pixel = self.area_pixel[sel_cen]
        if self.region_id.size:
            centr.region_id = self.region_id[sel_cen]
        if self.on_land.size:
            centr.on_land = self.on_land[sel_cen]
        if self.dist_coast.size:
            centr.dist_coast = self.dist_coast[sel_cen]
        return centr

    def set_lat_lon_to_meta(self):
        """ Compute meta from lat and lon values. To match the existing lat
        and lon, lat and lon need to start from the upper left corner!!"""
        self.meta = dict()
        res = min(get_resolution(self.lat, self.lon))
        rows, cols, ras_trans = pts_to_raster_meta(self.total_bounds, res)
        LOGGER.debug('Resolution points: %s', str(res))
        self.meta = {'width':cols, 'height':rows, 'crs':self.crs, 'transform':ras_trans}

    def set_meta_to_lat_lon(self):
        """ Compute lat and lon of every pixel center from meta raster """
        ulx, xres, _, uly, _, yres = self.meta['transform'].to_gdal()
        lrx = ulx + (self.meta['width'] * xres)
        lry = uly + (self.meta['height'] * yres)
        x_grid, y_grid = np.meshgrid(np.arange(ulx+xres/2, lrx, xres),
                                     np.arange(uly+yres/2, lry, yres))
        self.lon = x_grid.flatten()
        self.lat = y_grid.flatten()
        self.geometry = GeoSeries(crs=self.meta['crs'])

    def plot(self, **kwargs):
        """ Plot centroids scatter points over earth.

        Parameters:
            kwargs (optional): arguments for scatter matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if 's' not in kwargs:
            kwargs['s'] = 1
        fig, axis = u_plot.make_map()
        axis = axis[0][0]
        u_plot.add_shapes(axis)
        if self.meta and not self.coord.size:
            self.set_meta_to_lat_lon()
        axis.scatter(self.lon, self.lat, **kwargs)
        return fig, axis

    def calc_pixels_polygons(self, scheduler=None):
        """ Return a GeoSeries with a polygon for every pixel

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

        Returns:
            GeoSeries
        """
        if not self.meta:
            self.set_lat_lon_to_meta()
        if abs(abs(self.meta['transform'].a) -
               abs(self.meta['transform'].e)) > 1.0e-5:
            LOGGER.error('Area can not be computed for not squared pixels.')
            raise ValueError
        self.set_geometry_points(scheduler)
        return self.geometry.buffer(self.meta['transform'].a/2).envelope

    def empty_geometry_points(self):
        """ Removes points in geometry. Useful when centroids is used in
        multiprocessing function """
        self.geometry = GeoSeries(crs=self.geometry.crs)

    @property
    def crs(self):
        """ Get CRS of raster or vector """
        if self.meta:
            return self.meta['crs']
        return self.geometry.crs

    @property
    def size(self):
        """ Get size of pixels or points"""
        if self.meta:
            return self.meta['height']*self.meta['width']
        return self.lat.size

    @property
    def shape(self):
        """ Get shape of rastered data """
        try:
            if self.meta:
                return (self.meta['height'], self.meta['width'])
            return (np.unique(self.lat).size, np.unique(self.lon).size)
        except AttributeError:
            return ()

    @property
    def total_bounds(self):
        """ Get total bounds (left, bottom, right, top)"""
        if self.meta:
            left = self.meta['transform'].xoff
            right = left + self.meta['transform'][0]*self.meta['width']
            top = self.meta['transform'].yoff
            bottom = top + self.meta['transform'][4]*self.meta['height']
            return left, bottom, right, top
        return self.lon.min(), self.lat.min(), self.lon.max(), self.lat.max()

    @property
    def coord(self):
        """ Get [lat, lon] array. Might take some time. """
        return np.array([self.lat, self.lon]).transpose()

    def set_geometry_points(self, scheduler=None):
        """ Set geometry attribute of GeoSeries with Points from latitude and
        longitude attributes if geometry not present.

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        LOGGER.info('Setting geometry points.')
        def apply_point(df_exp):
            return df_exp.apply((lambda row: Point(row.longitude, row.latitude)), axis=1)
        if not self.geometry.size:
            if not self.lat.size or not self.lon.size:
                self.set_meta_to_lat_lon()
            if not scheduler:
                self.geometry = GeoSeries(list(zip(self.lon, self.lat)),
                                          crs=self.geometry.crs)
                self.geometry = self.geometry.apply(Point)
            else:
                import dask.dataframe as dd
                from multiprocessing import cpu_count
                ddata = dd.from_pandas(self, npartitions=cpu_count())
                self.geometry = ddata.map_partitions(apply_point, meta=Point).\
                compute(scheduler=scheduler)

    def _ne_crs_xy(self, scheduler=None):
        """ Return x (lon) and y (lat) in the CRS of Natural Earth

        Parameter:
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

        Returns:
            np.array, np.array
        """
        if not self.lat.size or not self.lon.size:
            self.set_meta_to_lat_lon()
        if equal_crs(self.geometry.crs, NE_CRS):
            return self.lon, self.lat
        self.set_geometry_points(scheduler)
        xy_points = self.geometry.to_crs(NE_CRS)
        return xy_points.geometry[:].x.values, xy_points.geometry[:].y.values

    def __deepcopy__(self, memo):
        """ Avoid error deep copy in GeoSeries by setting only the crs """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key == 'geometry':
                setattr(result, key, GeoSeries(crs=self.geometry.crs))
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result
