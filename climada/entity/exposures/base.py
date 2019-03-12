"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Exposures class.
"""

__all__ = ['Exposures', 'add_sea']

import logging
import numpy as np
import pandas as pd
from shapely.geometry import Point
import cartopy.crs as ccrs
from geopandas import GeoDataFrame
from rasterio.features import rasterize
from rasterio.transform import from_origin
import rasterio
import matplotlib.pyplot as plt

from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5
from climada.util.constants import ONE_LAT_KM
from climada.util.coordinates import coord_on_land
from climada.util.interpolation import interpol_index
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

INDICATOR_IF = 'if_'
""" Name of the column containing the impact functions id of specified hazard"""

INDICATOR_CENTR = 'centr_'
""" Name of the column containing the centroids id of specified hazard """

DEF_REF_YEAR = 2018
""" Default reference year """

DEF_VALUE_UNIT = 'USD'
""" Default reference year """

DEF_CRS = {'init': 'epsg:4326'}
""" Default coordinate reference system WGS 84 """

DEF_VAR_MAT = {'sup_field_name': 'entity',
               'field_name': 'assets',
               'var_name': {'lat' : 'lat',
                            'lon' : 'lon',
                            'val' : 'Value',
                            'ded' : 'Deductible',
                            'cov' : 'Cover',
                            'imp' : 'DamageFunID',
                            'cat' : 'Category_ID',
                            'reg' : 'Region_ID',
                            'uni' : 'Value_unit',
                            'ass' : 'centroid_index',
                            'ref' : 'reference_year'
                           }
              }
""" MATLAB variable names """

class Exposures(GeoDataFrame):
    """geopandas GeoDataFrame with metada and columns (pd.Series) defined in
    Attributes.

    Attributes:
        tag (Tag): metada - information about the source data
        ref_year (int): metada - reference year
        value_unit (str): metada - unit of the exposures values
        latitude (pd.Series): latitude
        longitude (pd.Series): longitude
        value (pd.Series): a value for each exposure
        if_ (pd.Series, optional): e.g. if_TC. impact functions id for hazard TC.
            There might be different hazards defined: if_TC, if_FL, ...
            If not provided, set to default 'if_' with ids 1 in check().
        geometry (pd.Series, optional): geometry of type Point of each instance.
            Computed in method set_geometry_points().
        deductible (pd.Series, optional): deductible value for each exposure
        cover (pd.Series, optional): cover value for each exposure
        category_id (pd.Series, optional): category id for each exposure
        region_id (pd.Series, optional): region id for each exposure
        centr_ (pd.Series, optional): e.g. centr_TC. centroids index for hazard
            TC. There might be different hazards defined: centr_TC, centr_FL, ...
            Computed in method assign_centroids().
    """
    _metadata = GeoDataFrame._metadata + ['tag', 'ref_year', 'value_unit']

    vars_oblig = ['value', 'latitude', 'longitude']
    """Name of the variables needed to compute the impact."""

    vars_def = [INDICATOR_IF]
    """Name of variables that can be computed."""

    vars_opt = [INDICATOR_CENTR, 'deductible', 'cover', 'category_id',
                'region_id', 'geometry']
    """Name of the variables that aren't need to compute the impact."""

    @property
    def _constructor(self):
        return Exposures

    def check(self):
        """ Check which variables are present """
        # check metadata
        for var in self._metadata:
            if var[0] == '_':
                continue
            try:
                if getattr(self, var) is None and var == 'crs':
                    self.crs = DEF_CRS
                    LOGGER.info('%s set to default value: %s', var, self.__dict__[var])
            except AttributeError:
                if var == 'tag':
                    self.tag = Tag()
                elif var == 'ref_year':
                    self.ref_year = DEF_REF_YEAR
                elif var == 'value_unit':
                    self.value_unit = DEF_VALUE_UNIT
                LOGGER.info('%s metadata set to default value: %s', var, self.__dict__[var])

        for var in self.vars_oblig:
            if not var in self.columns:
                LOGGER.error("%s missing.", var)
                raise ValueError

        for var in self.vars_def:
            if var == INDICATOR_IF:
                found = np.array([var in var_col for var_col in self.columns]).any()
                if INDICATOR_IF in self.columns:
                    LOGGER.info("Hazard type not set in %s", var)
            else:
                found = var in self.columns
            if not found and var == INDICATOR_IF:
                LOGGER.info("Setting %s to default impact functions ids 1.", var)
                self[INDICATOR_IF] = np.ones(self.shape[0], dtype=int)
            elif not found:
                LOGGER.info("%s not set.", var)

        for var in self.vars_opt:
            if var == INDICATOR_CENTR:
                found = np.array([var in var_col for var_col in self.columns]).any()
                if INDICATOR_CENTR in self.columns:
                    LOGGER.info("Hazard type not set in %s", var)
            else:
                found = var in self.columns
            if not found:
                LOGGER.info("%s not set.", var)
            elif var == 'geometry' and \
            (self.geometry.values[0].x != self.longitude.values[0] or \
            self.geometry.values[0].y != self.latitude.values[0]):
                LOGGER.error('Geometry values do not correspond to latitude ' +\
                'and longitude. Use set_geometry_points() or set_lat_lon().')
                raise ValueError

    def assign_centroids(self, hazard, method='NN', distance='haversine',
                         threshold=100):
        """ Assign for each exposure coordinate closest hazard coordinate

        Parameters:
            hazard (Hazard): hazard to match
            method (str, optional): interpolation method to use. Nearest
                neighbor (NN) default
            distance (str, optional): distance to use. Haversine default
            threshold (float): distance threshold in km over which no neighbor
                will be found. Those are assigned with a -1 index. Default 100
        """
        LOGGER.info('Matching %s exposures with %s centroids.',
                    str(self.shape[0]), str(hazard.centroids.size))

        coord = np.stack([self.latitude.values, self.longitude.values], axis=1)
        if np.array_equal(coord, hazard.centroids.coord):
            assigned = np.arange(self.shape[0])
        else:
            assigned = interpol_index(hazard.centroids.coord, coord, \
                method=method, distance=distance, threshold=threshold)

        self[INDICATOR_CENTR + hazard.tag.haz_type] = assigned

    def set_geometry_points(self):
        """ Set geometry attribute of GeoDataFrame from latitude and longitude
        attributes."""
        LOGGER.info('Setting geometry attribute.')
        self['geometry'] = list(zip(self.longitude, self.latitude))
        self['geometry'] = self['geometry'].apply(Point)

    def set_lat_lon(self):
        """ Set latitude and longitude attributes from geometry attribute. """
        LOGGER.info('Setting latitude and longitude attributes.')
        self['latitude'] = self.geometry[:].y
        self['longitude'] = self.geometry[:].x

    def plot_scatter(self, mask=None, ignore_zero=False, pop_name=True,
                     buffer=0.0, extend='neither', **kwargs):
        """Plot exposures geometry's value sum scattered over Earth's map.
        The plot will we projected according to the current crs.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'
         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        crs_epsg, _ = self._get_transformation()
        title = self.tag.description
        cbar_label = 'Value (%s)' % self.value_unit
        if mask is None:
            mask = np.ones((self.shape[0],), dtype=bool)
        if ignore_zero:
            pos_vals = self.value[mask].values > 0
        else:
            pos_vals = np.ones((self.value[mask].values.size,), dtype=bool)
        value = self.value[mask][pos_vals].values
        coord = np.stack([self.latitude[mask][pos_vals].values,
                          self.longitude[mask][pos_vals].values], axis=1)
        return u_plot.geo_scatter_from_array(value, coord, cbar_label, title, \
            pop_name, buffer, extend, proj=crs_epsg, **kwargs)

    def plot_hexbin(self, mask=None, ignore_zero=False, pop_name=True,
                    buffer=0.0, extend='neither', **kwargs):
        """Plot exposures geometry's value sum binned over Earth's map.
        An other function for the bins can be set through the key reduce_C_function.
        The plot will we projected according to the current crs.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for hexbin matplotlib function, e.g.
                reduce_C_function=np.average. Default: reduce_C_function=np.sum
         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        crs_epsg, _ = self._get_transformation()
        title = self.tag.description
        cbar_label = 'Value (%s)' % self.value_unit
        if 'reduce_C_function' not in kwargs:
            kwargs['reduce_C_function'] = np.sum
        if mask is None:
            mask = np.ones((self.shape[0],), dtype=bool)
        if ignore_zero:
            pos_vals = self.value[mask].values > 0
        else:
            pos_vals = np.ones((self.value[mask].values.size,), dtype=bool)
        value = self.value[mask][pos_vals].values
        coord = np.stack([self.latitude[mask][pos_vals].values,
                          self.longitude[mask][pos_vals].values], axis=1)
        return u_plot.geo_bin_from_array(value, coord, cbar_label, title, \
            pop_name, buffer, extend, proj=crs_epsg, **kwargs)

    def plot_raster(self, res=None, raster_res=None, save_tiff=None,
                    raster_f=lambda x: np.log10((np.fmax(x+1, 1))),
                    label='value (log10)', **kwargs):
        """ Generate raster from points geometry and plot it using log10 scale:
        np.log10((np.fmax(raster+1, 1))).

        Parameters:
            res (float, optional): resolution of current data in units of latitude
                and longitude, approximated if not provided.
            raster_res (float, optional): desired resolution of the raster
            save_tiff (str, optional): file name to save the raster in tiff
                format, if provided
            kwargs (optional): arguments for imshow matplotlib function

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        crs_epsg, crs_unit = self._get_transformation()
        if not res:
            res_lat, res_lon = np.diff(self.latitude.values), np.diff(self.longitude.values)
            res = min(res_lat[res_lat > 0].min(), res_lon[res_lon > 0].min())
        if not raster_res:
            raster_res = res
        LOGGER.info('Raster from resolution %s%s to %s%s.', res, crs_unit,
                    raster_res, crs_unit)

        # generate polygons of resolution
        if not 'geometry' in self.columns:
            self.set_geometry_points()
        exp_poly = self[['value']].set_geometry(self.buffer(res/2).envelope)
        # construct raster
        xmin, ymin, xmax, ymax = self.total_bounds
        rows = int(np.ceil((ymax-ymin) /  raster_res))
        cols = int(np.ceil((xmax-xmin) / raster_res))
        res_x, res_y = (xmax - xmin) / cols, (ymax - ymin) / rows
        ras_trans = from_origin(xmin - res_x / 2, ymax + res_y / 2, res_x, res_y)
        raster = rasterize([(x, val) for (x, val) in zip(exp_poly.geometry, exp_poly.value)],
                           out_shape=(rows, cols), transform=ras_trans, fill=0,
                           all_touched=True, dtype=rasterio.float32, )
        # save tiff
        if save_tiff is not None:
            ras_tiff = rasterio.open(save_tiff, 'w', driver='GTiff', \
                height=raster.shape[0], width=raster.shape[1], count=1, \
                dtype=np.float32, crs=self.crs, transform=ras_trans)
            ras_tiff.write(raster.astype(np.float32), 1)
            ras_tiff.close()
        # make plot
        fig, axis = u_plot.make_map(proj=crs_epsg)
        cbar_ax = fig.add_axes([0.99, 0.238, 0.03, 0.525])
        fig.subplots_adjust(hspace=0, wspace=0)
        axis[0, 0].set_extent([max(xmin, crs_epsg.x_limits[0]),
                               min(xmax, crs_epsg.x_limits[1]),
                               max(ymin, crs_epsg.y_limits[0]),
                               min(ymax, crs_epsg.y_limits[1])], crs_epsg)
        u_plot.add_shapes(axis[0, 0])
        imag = axis[0, 0].imshow(raster_f(raster), **kwargs, origin='upper',
                                 extent=[xmin, xmax, ymin, ymax], transform=crs_epsg)
        plt.colorbar(imag, cax=cbar_ax, label=label)
        plt.draw()
        posn = axis[0, 0].get_position()
        cbar_ax.set_position([posn.x0 + posn.width + 0.01, posn.y0, 0.04, posn.height])

        return fig, axis

    def write_hdf5(self, file_name):
        """ Write data frame and metadata in hdf5 format """
        store = pd.HDFStore(file_name)
        store.put('exposures', pd.DataFrame(self))

        var_meta = {}
        for var in self._metadata:
            var_meta[var] = getattr(self, var)

        store.get_storer('exposures').attrs.metadata = var_meta
        store.close()

    def read_hdf5(self, file_name):
        """ Read data frame and metadata in hdf5 format """
        with pd.HDFStore(file_name) as store:
            self.__init__(store['exposures'])
            metadata = store.get_storer('exposures').attrs.metadata
            for key, val in metadata.items():
                setattr(self, key, val)

    def read_mat(self, file_name, var_names=DEF_VAR_MAT):
        """Read MATLAB file and store variables in exposures.

        Parameters:
            file_name (str): absolute path file
            var_names (dict, optional): dictionary containing the name of the
                MATLAB variables. Default: DEF_VAR_MAT.
        """
        if var_names is None:
            var_names = DEF_VAR_MAT

        data = hdf5.read(file_name)
        try:
            data = data[var_names['sup_field_name']]
        except KeyError:
            pass

        try:
            data = data[var_names['field_name']]
            exposures = dict()

            _read_mat_obligatory(exposures, data, var_names)
            _read_mat_optional(exposures, data, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable: %s", str(var_err))
            raise var_err

        Exposures.__init__(self, data=exposures)
        _read_mat_metadata(self, data, file_name, var_names)

    #
    # Implement geopandas methods
    #

    def to_crs(self, crs=None, epsg=None, inplace=False):
        res = super(Exposures, self).to_crs(crs, epsg, inplace)
        if res is not None:
            res.set_lat_lon()
            return res

        self.set_lat_lon()
        return None

    to_crs.__doc__ = GeoDataFrame.to_crs.__doc__

    def copy(self, deep=True):
        """ Make a copy of this Exposures object.

        Parameters
        ----------
        deep (bool): Make a deep copy, i.e. also copy data. Default True.

        Returns
        -------
            GeoDataFrame
        """
        # FIXME: this will likely be unnecessary if removed from GeoDataFrame
        data = self._data
        if deep:
            data = data.copy()
        return Exposures(data).__finalize__(self)

    def _get_transformation(self):
        """ Get projection and its units to use in cartopy transforamtions from
        current crs

        Returns:
            ccrs.Projection, str
        """
        try:
            if self.crs['init'][-4:] == '3395':
                crs_epsg = ccrs.Mercator()
            else:
                crs_epsg = ccrs.epsg(self.crs['init'][-4:])
        except ValueError:
            crs_epsg = ccrs.PlateCarree()

        try:
            units = crs_epsg.proj4_params['units']
        except KeyError:
            units = '°'
        return crs_epsg, units

def add_sea(exposures, sea_res):
    """ Add sea to geometry's surroundings with given resolution. region_id
    set to -1 and other variables to 0.

    Parameters:
        sea_res (tuple): (sea_coast_km, sea_res_km), where first parameter
            is distance from coast to fill with water and second parameter
            is resolution between sea points

    Returns:
        Exposures
    """
    LOGGER.info("Adding sea at %s km resolution and %s km distance from coast.",
                str(sea_res[1]), str(sea_res[0]))

    sea_res = (sea_res[0]/ONE_LAT_KM, sea_res[1]/ONE_LAT_KM)

    min_lat = max(-90, float(exposures.latitude.min()) - sea_res[0])
    max_lat = min(90, float(exposures.latitude.max()) + sea_res[0])
    min_lon = max(-180, float(exposures.longitude.min()) - sea_res[0])
    max_lon = min(180, float(exposures.longitude.max()) + sea_res[0])

    lat_arr = np.arange(min_lat, max_lat+sea_res[1], sea_res[1])
    lon_arr = np.arange(min_lon, max_lon+sea_res[1], sea_res[1])

    lon_mgrid, lat_mgrid = np.meshgrid(lon_arr, lat_arr)
    lon_mgrid, lat_mgrid = lon_mgrid.ravel(), lat_mgrid.ravel()
    on_land = np.logical_not(coord_on_land(lat_mgrid, lon_mgrid))

    sea_exp = Exposures()
    sea_exp['latitude'] = lat_mgrid[on_land]
    sea_exp['longitude'] = lon_mgrid[on_land]
    sea_exp['region_id'] = np.zeros(sea_exp.latitude.size, int) - 1

    if 'geometry' in exposures.columns:
        sea_exp.set_geometry_points()

    for var_name in exposures.columns:
        if var_name not in ('latitude', 'longitude', 'region_id', 'geometry'):
            sea_exp[var_name] = np.zeros(sea_exp.latitude.size,
                                         exposures[var_name].dtype)

    return pd.concat([exposures, sea_exp], ignore_index=True, sort=False)

def _read_mat_obligatory(exposures, data, var_names):
    """Fill obligatory variables."""
    exposures['value'] = np.squeeze(data[var_names['var_name']['val']])

    exposures['latitude'] = data[var_names['var_name']['lat']].reshape(-1)
    exposures['longitude'] = data[var_names['var_name']['lon']].reshape(-1)

    exposures[INDICATOR_IF] = np.squeeze( \
        data[var_names['var_name']['imp']]).astype(int, copy=False)

def _read_mat_optional(exposures, data, var_names):
    """Fill optional parameters."""
    try:
        exposures['deductible'] = np.squeeze(data[var_names['var_name']['ded']])
    except KeyError:
        pass

    try:
        exposures['cover'] = np.squeeze(data[var_names['var_name']['cov']])
    except KeyError:
        pass

    try:
        exposures['category_id'] = \
        np.squeeze(data[var_names['var_name']['cat']]).astype(int, copy=False)
    except KeyError:
        pass

    try:
        exposures['region_id'] = \
        np.squeeze(data[var_names['var_name']['reg']]).astype(int, copy=False)
    except KeyError:
        pass

    try:
        assigned = np.squeeze(data[var_names['var_name']['ass']]).astype(int, copy=False)
        if assigned.size > 0:
            exposures[INDICATOR_CENTR] = assigned
    except KeyError:
        pass

def _read_mat_metadata(exposures, data, file_name, var_names):
    """ Fille metadata in DataFrame object """
    try:
        exposures.ref_year = int(np.squeeze(data[var_names['var_name']['ref']]))
    except KeyError:
        exposures.ref_year = DEF_REF_YEAR

    try:
        exposures.value_unit = hdf5.get_str_from_ref(file_name, \
            data[var_names['var_name']['uni']][0][0])
    except KeyError:
        exposures.value_unit = DEF_VALUE_UNIT

    exposures.tag = Tag(file_name)
