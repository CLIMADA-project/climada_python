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

Define auxiliary functions for plots.
"""

__all__ = ['geo_bin_from_array',
           'geo_im_from_array',
           'make_map',
           'add_shapes',
           'add_populated_places',
           'add_cntry_names'
          ]

import logging
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from rasterio.crs import CRS
import requests

from climada.util.files_handler import to_list
from climada.util.coordinates import grid_is_regular

LOGGER = logging.getLogger(__name__)

RESOLUTION = 250
"""Number of pixels in one direction in rendered image"""

BUFFER = 1.0
"""Degrees to add in the border"""

MAX_BINS = 2000
"""Maximum number of bins in geo_bin_from_array"""

def geo_bin_from_array(array_sub, geo_coord, var_name, title, pop_name=True,
                       buffer=BUFFER, extend='neither',
                       proj=ccrs.PlateCarree(), axes=None, **kwargs):
    """Plot array values binned over input coordinates.

    Parameters:
        array_sub (np.array(1d or 2d) or list(np.array)): Each array (in a row
            or in  the list) are values at each point in corresponding
            geo_coord that are binned in one subplot.
        geo_coord (2d np.array or list(2d np.array)): (lat, lon) for each
            point in a row. If one provided, the same grid is used for all
            subplots. Otherwise provide as many as subplots in array_sub.
        var_name (str or list(str)): label to be shown in the colorbar. If one
            provided, the same is used for all subplots. Otherwise provide as
            many as subplots in array_sub.
        title (str or list(str)): subplot title. If one provided, the same is
            used for all subplots. Otherwise provide as many as subplots in
            array_sub.
        pop_name (bool, optional): add names of the populated places.
        buffer (float, optional): border to add to coordinates
        extend (str, optional): extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        proj (ccrs): coordinate reference system used in coordinates
        kwargs (optional): arguments for hexbin matplotlib function

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    # Generate array of values used in each subplot
    num_im, list_arr = _get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, geo_coord, 'geo_coord')

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Wistia'
    if axes is None:
        _, axes = make_map(num_im, proj=proj)
    if not isinstance(axes, np.ndarray):
        axes_iter = np.array([[axes]])

    # Generate each subplot
    for array_im, axis, tit, name, coord in \
    zip(list_arr, axes_iter.flatten(), list_tit, list_name, list_coord):
        if coord.shape[0] != array_im.size:
            raise ValueError("Size mismatch in input array: %s != %s." %
                             (coord.shape[0], array_im.size))

        # Binned image with coastlines
        extent = _get_borders(coord, buffer=buffer, proj_limits=proj.x_limits + proj.y_limits)
        axis.set_extent((extent), proj)
        add_shapes(axis)
        if pop_name:
            add_populated_places(axis, extent, proj)

        if 'gridsize' not in kwargs:
            kwargs['gridsize'] = min(int(array_im.size / 2), MAX_BINS)
        hex_bin = axis.hexbin(coord[:, 1], coord[:, 0], C=array_im,
                              transform=proj, **kwargs)

        # Create colorbar in this axis
        cbax = make_axes_locatable(axis).append_axes('right', size="6.5%",
                                                     pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(hex_bin, cax=cbax, orientation='vertical',
                            extend=extend)
        cbar.set_label(name)
        axis.set_title(tit)

    return axes

def geo_scatter_from_array(array_sub, geo_coord, var_name, title,
                           pop_name=True, buffer=BUFFER, extend='neither',
                           proj=ccrs.PlateCarree(), shapes=True, axes=None, **kwargs):
    """Plot array values binned over input coordinates.

    Parameters:
        array_sub (np.array(1d or 2d) or list(np.array)): Each array (in a row
            or in  the list) are values at each point in corresponding
            geo_coord that are binned in one subplot.
        geo_coord (2d np.array or list(2d np.array)): (lat, lon) for each
            point in a row. If one provided, the same grid is used for all
            subplots. Otherwise provide as many as subplots in array_sub.
        var_name (str or list(str)): label to be shown in the colorbar. If one
            provided, the same is used for all subplots. Otherwise provide as
            many as subplots in array_sub.
        title (str or list(str)): subplot title. If one provided, the same is
            used for all subplots. Otherwise provide as many as subplots in
            array_sub.
        pop_name (bool, optional): add names of the populated places.
        buffer (float, optional): border to add to coordinates
        extend (str, optional): extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        proj (ccrs): coordinate reference system used in coordinates
        kwargs (optional): arguments for hexbin matplotlib function

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    # Generate array of values used in each subplot
    num_im, list_arr = _get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, geo_coord, 'geo_coord')

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Wistia'
    if axes is None:
        _, axes = make_map(num_im, proj=proj)
    axes_iter = axes
    if not isinstance(axes, np.ndarray):
        axes_iter = np.array([[axes]])
    # Generate each subplot
    for array_im, axis, tit, name, coord in \
    zip(list_arr, axes_iter.flatten(), list_tit, list_name, list_coord):
        if coord.shape[0] != array_im.size:
            raise ValueError("Size mismatch in input array: %s != %s." %
                             (coord.shape[0], array_im.size))
        # Binned image with coastlines
        extent = _get_borders(coord, buffer=buffer, proj_limits=proj.x_limits + proj.y_limits)
        axis.set_extent((extent), proj)
        if shapes:
            add_shapes(axis)
        if pop_name:
            add_populated_places(axis, extent, proj)
        hex_bin = axis.scatter(coord[:, 1], coord[:, 0], c=array_im,
                               transform=proj, **kwargs)
        # Create colorbar in this axis
        cbax = make_axes_locatable(axis).append_axes('right', size="6.5%",
                                                     pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(hex_bin, cax=cbax, orientation='vertical',
                            extend=extend)
        cbar.set_label(name)
        axis.set_title(tit)
    return axes

def geo_im_from_array(array_sub, coord, var_name, title,
                      proj=None, smooth=True, axes=None, **kwargs):
    """Image(s) plot defined in array(s) over input coordinates.

    Parameters:
        array_sub (np.array(1d or 2d) or list(np.array)): Each array (in a row
            or in  the list) are values at each point in corresponding
            geo_coord that are ploted in one subplot.
        coord (2d np.array): (lat, lon) for each point in a row. The same grid is used for all
            subplots.
        var_name (str or list(str)): label to be shown in the colorbar. If one
            provided, the same is used for all subplots. Otherwise provide as
            many as subplots in array_sub.
        title (str or list(str)): subplot title. If one provided, the same is
            used for all subplots. Otherwise provide as many as subplots in
            array_sub.
        proj (ccrs): coordinate reference system used in coordinates
        smooth (bool, optional): smooth plot to RESOLUTIONxRESOLUTION. Default:
            True.
        kwargs (optional): arguments for pcolormesh matplotlib function.

    Returns:
        cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    # Generate array of values used in each subplot
    num_im, list_arr = _get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')

    is_reg, height, width = grid_is_regular(coord)
    extent = _get_borders(coord, proj_limits=(-360, 360, -90, 90))
    mid_lon = 0
    if not proj:
        mid_lon = 0.5 * sum(extent[:2])
        proj = ccrs.PlateCarree(central_longitude=mid_lon)
    if 'vmin' not in kwargs:
        kwargs['vmin'] = np.nanmin(array_sub)
    if 'vmax' not in kwargs:
        kwargs['vmax'] = np.nanmax(array_sub)
    if axes is None:
        _, axes = make_map(num_im, proj=proj)
    axes_iter = axes
    if not isinstance(axes, np.ndarray):
        axes_iter = np.array([[axes]])

    # Generate each subplot
    for array_im, axis, tit, name in zip(list_arr, axes_iter.flatten(), list_tit, list_name):
        if coord.shape[0] != array_im.size:
            raise ValueError("Size mismatch in input array: %s != %s." %
                             (coord.shape[0], array_im.size))
        if smooth or not is_reg:
            # Create regular grid where to interpolate the array
            grid_x, grid_y = np.mgrid[
                extent[0]: extent[1]: complex(0, RESOLUTION),
                extent[2]: extent[3]: complex(0, RESOLUTION)]
            grid_im = griddata((coord[:, 1], coord[:, 0]), array_im,
                               (grid_x, grid_y))
        else:
            grid_x = coord[:, 1].reshape((width, height)).transpose()
            grid_y = coord[:, 0].reshape((width, height)).transpose()
            grid_im = np.array(array_im.reshape((width, height)).transpose())
            if grid_y[0, 0] > grid_y[0, -1]:
                grid_y = np.flip(grid_y)
                grid_im = np.flip(grid_im, 1)
            grid_im = np.resize(grid_im, (height, width, 1))
        axis.set_extent((extent[0] - mid_lon, extent[1] - mid_lon,
                         extent[2], extent[3]), crs=proj)

        # Add coastline to axis
        add_shapes(axis)
        # Create colormesh, colorbar and labels in axis
        cbax = make_axes_locatable(axis).append_axes('right', size="6.5%",
                                                     pad=0.1, axes_class=plt.Axes)
        img = axis.pcolormesh(grid_x - mid_lon, grid_y, np.squeeze(grid_im),
                              transform=proj, **kwargs)
        cbar = plt.colorbar(img, cax=cbax, orientation='vertical')
        cbar.set_label(name)
        axis.set_title(tit)

    return axes

def make_map(num_sub=1, figsize=(9, 13), proj=ccrs.PlateCarree()):
    """Create map figure with cartopy.

    Parameters:
        num_sub (int or tuple): number of total subplots in figure OR number of
            subfigures in row and column: (num_row, num_col).
        figsize (tuple): figure size for plt.subplots
        proj (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.

    Returns:
        matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
    """
    if isinstance(num_sub, int):
        num_row, num_col = _get_row_col_size(num_sub)
    else:
        num_row, num_col = num_sub

    fig, axis_sub = plt.subplots(num_row, num_col, figsize=figsize,
                                 subplot_kw=dict(projection=proj))
    axes_iter = axis_sub
    if not isinstance(axis_sub, np.ndarray):
        axes_iter = np.array([[axis_sub]])

    for axis in axes_iter.flatten():
        try:
            grid = axis.gridlines(draw_labels=True, alpha=0.2, transform=proj)
            grid.xlabels_top = grid.ylabels_right = False
            grid.xformatter = LONGITUDE_FORMATTER
            grid.yformatter = LATITUDE_FORMATTER
        except TypeError:
            pass

    fig.tight_layout()
    if num_col > 1:
        fig.subplots_adjust(wspace=0.3)
    if num_row > 1:
        fig.subplots_adjust(hspace=-0.5)

    return fig, axis_sub

def add_shapes(axis):
    """Overlay Earth's countries coastlines to matplotlib.pyplot axis.

    Parameters:
        axis (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy axis.
        projection (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.
    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')
    shp = shapereader.Reader(shp_file)
    for geometry in shp.geometries():
        axis.add_geometries([geometry], crs=ccrs.PlateCarree(), facecolor='',
                            edgecolor='black')

def add_populated_places(axis, extent, proj=ccrs.PlateCarree()):
    """Add city names.

    Parameters:
        axis (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy axis.
        extent (list): geographical limits [min_lon, max_lon, min_lat, max_lat]
        proj (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.

    """
    shp_file = shapereader.natural_earth(resolution='50m', category='cultural',
                                         name='populated_places_simple')

    shp = shapereader.Reader(shp_file)
    ext_pts = list(box(extent[0], extent[2], extent[1], extent[3]).exterior.coords)
    ext_trans = [ccrs.PlateCarree().transform_point(pts[0], pts[1], proj)
                 for pts in ext_pts]
    for rec, point in zip(shp.records(), shp.geometries()):
        if ext_trans[2][0] < point.x <= ext_trans[0][0]:
            if ext_trans[0][1] < point.y <= ext_trans[1][1]:
                axis.plot(point.x, point.y, 'ko', markersize=7,
                          transform=ccrs.PlateCarree(), markerfacecolor='None')
                axis.text(point.x, point.y, rec.attributes['name'],
                          horizontalalignment='right', verticalalignment='bottom',
                          transform=ccrs.PlateCarree(), fontsize=14)

def add_cntry_names(axis, extent, proj=ccrs.PlateCarree()):
    """Add country names.

    Parameters:
        axis (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy axis.
        extent (list): geographical limits [min_lon, max_lon, min_lat, max_lat]
        proj (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.

    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')

    shp = shapereader.Reader(shp_file)
    ext_pts = list(box(extent[0], extent[2], extent[1], extent[3]).exterior.coords)
    ext_trans = [ccrs.PlateCarree().transform_point(pts[0], pts[1], proj)
                 for pts in ext_pts]
    for rec, point in zip(shp.records(), shp.geometries()):
        point_x = point.centroid.xy[0][0]
        point_y = point.centroid.xy[1][0]
        if ext_trans[2][0] < point_x <= ext_trans[0][0]:
            if ext_trans[0][1] < point_y <= ext_trans[1][1]:
                axis.text(point_x, point_y, rec.attributes['NAME'],
                          horizontalalignment='center', verticalalignment='center',
                          transform=ccrs.PlateCarree(), fontsize=14)

def _get_collection_arrays(array_sub):
    """Get number of array rows and generate list of array if only one row

    Parameters:
        array_sub (np.array(1d or 2d) or list(np.array)): Each array (in a row
            or in  the list) are values at each point in corresponding

    Returns:
        num_im (int), list_arr (2d np.ndarray or list(1d np.array))
    """
    num_im = 1
    if not isinstance(array_sub, list):
        if len(array_sub.shape) == 1 or array_sub.shape[1] == 1:
            list_arr = list()
            list_arr.append(array_sub)
        else:
            list_arr = array_sub
            num_im = array_sub.shape[0]
    else:
        num_im = len(array_sub)
        list_arr = array_sub

    return num_im, list_arr

def _get_row_col_size(num_sub):
    """Compute number of rows and columns of subplots in figure.

    Parameters:
        num_sub (int): number of subplots

    Returns:
        num_row (int), num_col (int)
    """
    if num_sub <= 3:
        num_col = num_sub
        num_row = 1
    else:
        if num_sub % 3 == 0:
            num_col = 3
            num_row = int(num_sub / 3)
        else:
            num_col = 2
            num_row = int(num_sub / 2) + num_sub % 2
    return num_row, num_col

def _get_borders(geo_coord, buffer=0, proj_limits=(-180, 180, -90, 90)):
    """Get min and max longitude and min and max latitude (in this order).

    Parameters:
        geo_coord (2d np.array): (lat, lon) for each point in a row.
        buffer (float): border to add. Default: 0
        proj_limits (tuple, optional): limits of geographical projection
            (lon_min, lon_max, lat_min, lat_max)

    Returns:
        np.array
    """
    min_lon = max(np.min(geo_coord[:, 1]) - buffer, proj_limits[0])
    max_lon = min(np.max(geo_coord[:, 1]) + buffer, proj_limits[1])
    min_lat = max(np.min(geo_coord[:, 0]) - buffer, proj_limits[2])
    max_lat = min(np.max(geo_coord[:, 0]) + buffer, proj_limits[3])
    return [min_lon, max_lon, min_lat, max_lat]

def get_transformation(crs_in):
    """Get projection and its units to use in cartopy transforamtions from
    current crs

    Returns:
        ccrs.Projection, str
    """
    try:
        if CRS.from_user_input(crs_in) == CRS.from_user_input({'init': 'epsg:3395'}):
            crs_epsg = ccrs.Mercator()
        else:
            crs_epsg = ccrs.epsg(CRS.from_user_input(crs_in).to_epsg())
    except ValueError:
        crs_epsg = ccrs.PlateCarree()
    except requests.exceptions.ConnectionError:
        LOGGER.warning('No internet connection. Using projection PlateCarree in plot.')
        crs_epsg = ccrs.PlateCarree()

    try:
        units = crs_epsg.proj4_params['units']
    except KeyError:
        units = '°'
    return crs_epsg, units
