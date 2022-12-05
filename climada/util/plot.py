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

Define auxiliary functions for plots.
"""
# pylint: disable=abstract-class-instantiated

__all__ = ['geo_bin_from_array',
           'geo_im_from_array',
           'make_map',
           'add_shapes',
           'add_populated_places',
           'add_cntry_names'
          ]

import logging
from textwrap import wrap
import warnings

from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import box
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from rasterio.crs import CRS
import requests

from climada.util.constants import CMAP_EXPOSURES, CMAP_CAT, CMAP_RASTER
from climada.util.files_handler import to_list
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)

RESOLUTION = 250
"""Number of pixels in one direction in rendered image"""

BUFFER = 1.0
"""Degrees to add in the border"""

MAX_BINS = 2000
"""Maximum number of bins in geo_bin_from_array"""


def geo_bin_from_array(array_sub, geo_coord, var_name, title,
                       pop_name=True, buffer=BUFFER, extend='neither',
                       proj=ccrs.PlateCarree(), shapes=True, axes=None,
                       figsize=(9, 13), adapt_fontsize=True, **kwargs):
    """Plot array values binned over input coordinates.

    Parameters
    ----------
    array_sub : np.array(1d or 2d) or list(np.array)
        Each array (in a row or in  the list) are values at each point in corresponding
        geo_coord that are binned in one subplot.
    geo_coord : 2d np.array or list(2d np.array)
        (lat, lon) for each point in a row. If one provided, the same grid is used for all
        subplots. Otherwise provide as many as subplots in array_sub.
    var_name : str or list(str)
        label to be shown in the colorbar. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    title : str or list(str)
        subplot title. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    pop_name : bool, optional
        add names of the populated places, by default True.
    buffer : float, optional
        border to add to coordinates, by default BUFFER
    extend : str, optional
        extend border colorbar with arrows.
        [ 'neither' | 'both' | 'min' | 'max' ], by default 'neither'
    proj : ccrs, optional
        coordinate reference system of the given data, by default ccrs.PlateCarree()
    shapes : bool, optional
        Overlay Earth's countries coastlines to matplotlib.pyplot axis.
        The default is True
    axes : Axes or ndarray(Axes), optional
        by default None
    figsize : tuple, optional
        figure size for plt.subplots, by default (9, 13)
    adapt_fontsize : bool, optional
        If set to true, the size of the fonts will be adapted to the size of the figure. Otherwise
        the default matplotlib font size is used. Default is True.

    **kwargs
        arbitrary keyword arguments for hexbin matplotlib function

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises
    ------
    ValueError:
        Input array size missmatch
    """
    return _plot_scattered_data("hexbin", array_sub, geo_coord, var_name, title,
                                pop_name=pop_name, buffer=buffer, extend=extend,
                                proj=proj, shapes=shapes, axes=axes,
                                figsize=figsize, adapt_fontsize=adapt_fontsize, **kwargs)


def geo_scatter_from_array(array_sub, geo_coord, var_name, title,
                           pop_name=False, buffer=BUFFER, extend='neither',
                           proj=ccrs.PlateCarree(), shapes=True, axes=None,
                           figsize=(9, 13), adapt_fontsize=True, **kwargs):
    """Plot array values at input coordinates.

    Parameters
    ----------
    array_sub : np.array(1d or 2d) or list(np.array)
        Each array (in a row or in  the list) are values at each point in corresponding
        geo_coord that are binned in one subplot.
    geo_coord : 2d np.array or list(2d np.array)
        (lat, lon) for each point in a row. If one provided, the same grid is used for all
        subplots. Otherwise provide as many as subplots in array_sub.
    var_name : str or list(str)
        label to be shown in the colorbar. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    title : str or list(str)
        subplot title. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    pop_name : bool, optional
        add names of the populated places, by default False.
    buffer : float, optional
        border to add to coordinates, by default BUFFER
    extend : str, optional
        extend border colorbar with arrows.
        [ 'neither' | 'both' | 'min' | 'max' ], by default 'neither'
    proj : ccrs, optional
        coordinate reference system of the given data, by default ccrs.PlateCarree()
    shapes : bool, optional
        Overlay Earth's countries coastlines to matplotlib.pyplot axis.
        The default is True
    axes : Axes or ndarray(Axes), optional
        by default None
    figsize : tuple, optional
        figure size for plt.subplots, by default (9, 13)
    adapt_fontsize : bool, optional
        If set to true, the size of the fonts will be adapted to the size of the figure. Otherwise
        the default matplotlib font size is used. Default is True.
    **kwargs
        arbitrary keyword arguments for scatter matplotlib function

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises
    ------
    ValueError:
        Input array size missmatch
    """
    return _plot_scattered_data("scatter", array_sub, geo_coord, var_name, title,
                                pop_name=pop_name, buffer=buffer, extend=extend,
                                proj=proj, shapes=shapes, axes=axes,
                                figsize=figsize, adapt_fontsize=adapt_fontsize, **kwargs)


def _plot_scattered_data(method, array_sub, geo_coord, var_name, title,
                         pop_name=False, buffer=BUFFER, extend='neither',
                         proj=ccrs.PlateCarree(), shapes=True, axes=None,
                         figsize=(9, 13), adapt_fontsize=True, **kwargs):
    """Function for internal use in `geo_scatter_from_array` (when called with method="scatter")
    and `geo_bin_from_array` (when called with method="hexbin"). See the docstrings of the
    respective functions for more information on the parameters."""

    # Generate array of values used in each subplot
    num_im, list_arr = _get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, geo_coord, 'geo_coord')

    if 'cmap' not in kwargs:
        kwargs['cmap'] = CMAP_EXPOSURES

    if axes is None:
        proj_plot = proj
        if isinstance(proj, ccrs.PlateCarree):
            # for PlateCarree, center plot around data's central lon
            # without overwriting the data's original projection info
            xmin, xmax = u_coord.lon_bounds(np.concatenate([c[:, 1] for c in list_coord]))
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        _, axes, fontsize = make_map(num_im, proj=proj_plot, figsize=figsize,
                                     adapt_fontsize=adapt_fontsize)
    else:
        fontsize = None
    axes_iter = axes
    if not isinstance(axes, np.ndarray):
        axes_iter = np.array([[axes]])

    # Generate each subplot
    for array_im, axis, tit, name, coord in \
    zip(list_arr, axes_iter.flatten(), list_tit, list_name, list_coord):
        if coord.shape[0] != array_im.size:
            raise ValueError(f"Size mismatch in input array: {coord.shape[0]} != {array_im.size}.")

        # Binned image with coastlines
        if isinstance(proj, ccrs.PlateCarree):
            xmin, ymin, xmax, ymax = u_coord.latlon_bounds(coord[:, 0], coord[:, 1], buffer=buffer)
            extent = (xmin, xmax, ymin, ymax)
        else:
            extent = _get_borders(coord, buffer=buffer, proj_limits=proj.x_limits + proj.y_limits)
        axis.set_extent((extent), proj)

        if shapes:
            add_shapes(axis)
        if pop_name:
            add_populated_places(axis, extent, proj, fontsize)

        if method == "hexbin":
            if 'gridsize' not in kwargs:
                kwargs['gridsize'] = min(int(array_im.size / 2), MAX_BINS)
            mappable = axis.hexbin(coord[:, 1], coord[:, 0], C=array_im,
                                   transform=proj, **kwargs)
        else:
            mappable = axis.scatter(coord[:, 1], coord[:, 0], c=array_im,
                                    transform=proj, **kwargs)

        # Create colorbar in this axis
        cbax = make_axes_locatable(axis).append_axes(
            'right', size="6.5%", pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(mappable, cax=cbax, orientation='vertical', extend=extend)
        cbar.set_label(name)
        axis.set_title("\n".join(wrap(tit)))
        if fontsize:
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
            for item in [axis.title, cbar.ax.xaxis.label, cbar.ax.yaxis.label]:
                item.set_fontsize(fontsize)
    plt.tight_layout()
    return axes


def geo_im_from_array(array_sub, coord, var_name, title,
                      proj=None, smooth=True, axes=None, figsize=(9, 13), adapt_fontsize=True,
                      **kwargs):
    """Image(s) plot defined in array(s) over input coordinates.

    Parameters
    ----------
    array_sub : np.array(1d or 2d) or list(np.array)
        Each array (in a row or in  the list) are values at each point in corresponding
        geo_coord that are ploted in one subplot.
    coord : 2d np.array
        (lat, lon) for each point in a row. The same grid is used for all subplots.
    var_name : str or list(str)
        label to be shown in the colorbar. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    title : str or list(str)
        subplot title. If one provided, the same is used for all subplots.
        Otherwise provide as many as subplots in array_sub.
    proj : ccrs, optional
        coordinate reference system used in coordinates, by default None
    smooth : bool, optional
        smooth plot to RESOLUTIONxRESOLUTION, by default True
    axes : Axes or ndarray(Axes), optional
        by default None
    figsize : tuple, optional
        figure size for plt.subplots, by default (9, 13)
    adapt_fontsize : bool, optional
        If set to true, the size of the fonts will be adapted to the size of the figure. Otherwise
        the default matplotlib font size is used. Default is True.
    **kwargs
        arbitrary keyword arguments for pcolormesh matplotlib function

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises
    ------
    ValueError
    """

    # Generate array of values used in each subplot
    num_im, list_arr = _get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, coord, 'geo_coord')


    is_reg, height, width = u_coord.grid_is_regular(coord)
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
        proj_plot = proj
        if isinstance(proj, ccrs.PlateCarree):
            # for PlateCarree, center plot around data's central lon 
            # without overwriting the data's original projection info
            xmin, xmax = u_coord.lon_bounds(np.concatenate([c[:, 1] for c in list_coord]))
            proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        _, axes, fontsize = make_map(num_im, proj=proj_plot, figsize=figsize,
                                     adapt_fontsize=adapt_fontsize)
    else:
        fontsize = None
    axes_iter = axes
    if not isinstance(axes, np.ndarray):
        axes_iter = np.array([[axes]])

    if 'cmap' not in kwargs:
        kwargs['cmap'] = CMAP_RASTER

    # Generate each subplot
    for array_im, axis, tit, name in zip(list_arr, axes_iter.flatten(), list_tit, list_name):
        if coord.shape[0] != array_im.size:
            raise ValueError(f"Size mismatch in input array: {coord.shape[0]} != {array_im.size}.")
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
        axis.set_title("\n".join(wrap(tit)))
        if fontsize:
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
            for item in [axis.title, cbar.ax.xaxis.label, cbar.ax.yaxis.label]:
                item.set_fontsize(fontsize)

    plt.tight_layout()
    return axes


def geo_scatter_categorical(array_sub, geo_coord, var_name, title,
                            cat_name=None, adapt_fontsize=True, **kwargs):
    """
    Map plots for categorical data defined in array(s) over input
    coordinates. The categories must be a finite set of unique values
    as can be identified by np.unique() (mix of int, float, strings, ...).

    The categories are shared among all subplots, i.e. are obtained from
    np.unique(array_sub).
    Eg.:
        array_sub = [[1, 2, 1.0, 2], [1, 2, 'a', 'a']]
        -> categories mapping is [[0, 2, 1, 2], [0, 2, 3, 3]]

    Same category: 1 and '1'
    Different categories: 1 and 1.0

    This method wraps around util.geo_scatter_from_array and uses
    all its args and kwargs.

    Parameters
    ----------
    array_sub : np.array(1d or 2d) or list(np.array)
        Each array (in a row or in  the list) are values at each point
        in corresponding geo_coord that are binned in one subplot.
    geo_coord : 2d np.array or list(2d np.array)
        (lat, lon) for each point in a row. If one provided, the same grid
        is used for all subplots. Otherwise provide as many as subplots
        in array_sub.
    var_name : str or list(str)
        label to be shown in the colorbar. If one
        provided, the same is used for all subplots. Otherwise provide as
        many as subplots in array_sub.
    title : str or list(str)
        subplot title. If one provided, the same is
        used for all subplots. Otherwise provide as many as subplots in
        array_sub.
    cat_name : dict, optional
        Categories name for the colorbar labels.
        Keys are all the unique values in array_sub, values are their labels.
        The default is labels = unique values.
    adapt_fontsize : bool, optional
        If set to true, the size of the fonts will be adapted to the size of the figure. Otherwise
        the default matplotlib font size is used. Default is True.
    **kwargs
        Arbitrary keyword arguments for hexbin matplotlib function

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxesSubplot

    """

    # convert sorted categories to numeric array [0, 1, ...]
    array_sub = np.array(array_sub)
    array_sub_unique, array_sub_cat = np.unique(array_sub, return_inverse=True) #flattens array
    array_sub_cat = array_sub_cat.reshape(array_sub.shape)
    array_sub_n = array_sub_unique.size

    if 'cmap' in kwargs:
        # optional user defined colormap (can be continuous)
        cmap_arg = kwargs['cmap']
        if isinstance(cmap_arg, str):
            cmap_name = cmap_arg
            # for qualitative colormaps taking the first few colors is preferable
            # over jumping equal distances
            if cmap_name in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                    'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']:
                cmap = mpl.colors.ListedColormap(
                    mpl.cm.get_cmap(cmap_name).colors[:array_sub_n]
                )
            else:
                cmap = mpl.cm.get_cmap(cmap_arg, array_sub_n)
        elif isinstance(cmap_arg, mpl.colors.ListedColormap):
            # If a user brings their own colormap it's probably qualitative
            cmap_name = 'defined by the user'
            cmap = mpl.colors.ListedColormap(
                cmap_arg.colors[:array_sub_n]
            )
        else:
            raise TypeError("if cmap is given it must be either a str or a ListedColormap")
    else:
        # default qualitative colormap
        cmap_name = CMAP_CAT
        cmap = mpl.colors.ListedColormap(
            mpl.cm.get_cmap(cmap_name).colors[:array_sub_n]
        )

    if array_sub_n > cmap.N:
        LOGGER.warning("More than %d categories cannot be plotted accurately "
                       "using the colormap %s. Please specify "
                       "a different qualitative colormap using the `cmap` "
                       "attribute. For Matplotlib's built-in colormaps, see "
           "https://matplotlib.org/stable/tutorials/colors/colormaps.html",
                       cmap.N, cmap_name)

    # define the discrete colormap kwargs
    kwargs['cmap'] = cmap
    kwargs['vmin'] = -0.5
    kwargs['vmax'] = array_sub_n - 0.5

    # #create the axes
    axes = _plot_scattered_data(
        "scatter", array_sub_cat, geo_coord, var_name, title,
        adapt_fontsize=adapt_fontsize, **kwargs)

    #add colorbar labels
    if cat_name is None:
        cat_name = array_sub_unique.astype(str)
    if not isinstance(cat_name, dict):
        cat_name = dict(zip(array_sub_unique, cat_name))
    cat_name = {str(key): value for key, value in cat_name.items()}
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax in axes.ravel():
        cbar = [coll.colorbar for coll in ax.collections if coll.colorbar is not None]
        if len(cbar) > 0:
            cbar = cbar[-1]
            cbar.set_ticks(np.arange(array_sub_n))
            cbar.set_ticklabels([cat_name[str(val)] for val in array_sub_unique])

    return axes


def make_map(num_sub=1, figsize=(9, 13), proj=ccrs.PlateCarree(), adapt_fontsize=True):
    """
    Create map figure with cartopy.

    Parameters
    ----------
    num_sub : int or tuple
        number of total subplots in figure OR number of
        subfigures in row and column: (num_row, num_col).
    figsize : tuple
        figure size for plt.subplots
    proj : cartopy.crs projection, optional
        geographical projection,
        The default is PlateCarree default.
    adapt_fontsize : bool, optional
        If set to true, the size of the fonts will be adapted to the size of the figure. Otherwise
        the default matplotlib font size is used. Default is True.

    Returns
    -------
    fig, axis_sub : matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
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
            grid.top_labels = grid.right_labels = False
            grid.xformatter = LONGITUDE_FORMATTER
            grid.yformatter = LATITUDE_FORMATTER
            if adapt_fontsize:
                fontsize = axis.bbox.width/35
                if fontsize < 10:
                    fontsize = 10
                grid.xlabel_style = {'size': fontsize}
                grid.ylabel_style = {'size': fontsize}
            else:
                fontsize = None
        except TypeError:
            pass

    if num_col > 1:
        fig.subplots_adjust(wspace=0.3)
    if num_col > 2:
        fig.subplots_adjust(wspace=0.5)
    if num_row > 1:
        fig.subplots_adjust(hspace=-0.5)

    return fig, axis_sub, fontsize

def add_shapes(axis):
    """
    Overlay Earth's countries coastlines to matplotlib.pyplot axis.

    Parameters
    ----------
    axis : cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy axis
    projection : cartopy.crs projection, optional
        Geographical projection.
        The default is PlateCarree.
    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')
    shp = shapereader.Reader(shp_file)
    for geometry in shp.geometries():
        axis.add_geometries([geometry], crs=ccrs.PlateCarree(), facecolor='none',
                            edgecolor='dimgray')

def _ensure_utf8(val):
    # Without the `*.cpg` file present, the shape reader wrongly assumes latin-1 encoding:
    # https://github.com/SciTools/cartopy/issues/1282
    # https://github.com/SciTools/cartopy/commit/6d787b01e122eea68b67a9b2966e45877755a52d
    # As a workaround, we encode and decode again, unless this fails which means
    # that the `*.cpg` is present and the encoding is correct:
    try:
        return val.encode('latin-1').decode('utf-8')
    except (AttributeError, UnicodeDecodeError, UnicodeEncodeError):
        return val

def add_populated_places(axis, extent, proj=ccrs.PlateCarree(), fontsize=None):
    """
    Add city names.

    Parameters
    ----------
    axis : cartopy.mpl.geoaxes.GeoAxesSubplot
        cartopy axis.
    extent : list
        geographical limits [min_lon, max_lon, min_lat, max_lat]
    proj : cartopy.crs projection, optional
        geographical projection,
        The default is PlateCarree.
    fontsize : int, optional
        Size of the fonts. If set to None, the default matplotlib settings
        are used.

    """
    shp_file = shapereader.natural_earth(resolution='50m', category='cultural',
                                         name='populated_places_simple')

    shp = shapereader.Reader(shp_file)
    ext_pts = list(box(*u_coord.toggle_extent_bounds(extent)).exterior.coords)
    ext_trans = [ccrs.PlateCarree().transform_point(pts[0], pts[1], proj)
                 for pts in ext_pts]
    for rec, point in zip(shp.records(), shp.geometries()):
        if ext_trans[2][0] < point.x <= ext_trans[0][0]:
            if ext_trans[0][1] < point.y <= ext_trans[1][1]:
                axis.plot(point.x, point.y, color='navy', marker='o',
                          transform=ccrs.PlateCarree(), markerfacecolor='None')
                axis.text(point.x, point.y, _ensure_utf8(rec.attributes['name']),
                          horizontalalignment='right', verticalalignment='bottom',
                          transform=ccrs.PlateCarree(), color='navy', fontsize=fontsize)

def add_cntry_names(axis, extent, proj=ccrs.PlateCarree(), fontsize=None):
    """
    Add country names.

    Parameters
    ----------
    axis : cartopy.mpl.geoaxes.GeoAxesSubplot
        Cartopy axis.
    extent : list
        geographical limits [min_lon, max_lon, min_lat, max_lat]
    proj : cartopy.crs projection, optional
        Geographical projection.
        The default is PlateCarree.
     fontsize : int, optional
        Size of the fonts. If set to None, the default matplotlib settings
        are used.
    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')

    shp = shapereader.Reader(shp_file)
    ext_pts = list(box(*u_coord.toggle_extent_bounds(extent)).exterior.coords)
    ext_trans = [ccrs.PlateCarree().transform_point(pts[0], pts[1], proj)
                 for pts in ext_pts]
    for rec, point in zip(shp.records(), shp.geometries()):
        point_x = point.centroid.xy[0][0]
        point_y = point.centroid.xy[1][0]
        if ext_trans[2][0] < point_x <= ext_trans[0][0]:
            if ext_trans[0][1] < point_y <= ext_trans[1][1]:
                axis.text(point_x, point_y, _ensure_utf8(rec.attributes['NAME']),
                          horizontalalignment='center', verticalalignment='center',
                          transform=ccrs.PlateCarree(), fontsize=fontsize, color='navy')

def _get_collection_arrays(array_sub):
    """
    Get number of array rows and generate list of array if only one row

    Parameters
    ----------
    array_sub : np.array(1d or 2d) or list(np.array)
        Each array (in a row
        or in  the list) are values at each point in corresponding

    Returns
    -------
    num_im, list_arr : int, 2d np.ndarray or list(1d np.array)
        Number of array rows and list of array
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
    """
    Compute number of rows and columns of subplots in figure.

    Parameters
    ----------
    num_sub : int
        number of subplots

    Returns
    -------
    num_row, num_col : int, int
        Number of rows and columns
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
    """
    Get min and max longitude and min and max latitude (in this order).

    Parameters
    ----------
    geo_coord : 2d np.array
        (lat, lon) for each point in a row.
    buffer : float, optional
        border to add. The default is 0
    proj_limits  : tuple, optional
        limits of geographical projection (lon_min, lon_max, lat_min, lat_max)
        The default is (-180, 180, -90, 90)

    Returns
    -------
    extent : list [min_lon, max_lon, min_lat, max_lat]
    """
    min_lon = max(np.min(geo_coord[:, 1]) - buffer, proj_limits[0])
    max_lon = min(np.max(geo_coord[:, 1]) + buffer, proj_limits[1])
    min_lat = max(np.min(geo_coord[:, 0]) - buffer, proj_limits[2])
    max_lat = min(np.max(geo_coord[:, 0]) + buffer, proj_limits[3])
    return [min_lon, max_lon, min_lat, max_lat]

def get_transformation(crs_in):
    """
    Get projection and its units to use in cartopy transforamtions from current crs.

    Parameters
    ----------
    crs_in : str
        Current crs

    Returns
    ------
    crs_epsg : ccrs.Projection
    units : str
    """

    # projection
    try:
        epsg = CRS.from_user_input(crs_in).to_epsg()
        if epsg == 3395:
            crs = ccrs.Mercator()
        elif epsg == 4326:  # WSG 84
            crs = ccrs.PlateCarree()
        else:
            crs = ccrs.epsg(epsg)
    except ValueError:
        LOGGER.warning(
            "Error parsing coordinate system '%s'. Using projection PlateCarree in plot.", crs_in
        )
        crs = ccrs.PlateCarree()
    except requests.exceptions.ConnectionError:
        LOGGER.warning('No internet connection. Using projection PlateCarree in plot.')
        crs = ccrs.PlateCarree()

    # units
    with warnings.catch_warnings():
        # The method `to_dict` converts the crs into a string, which causes a user warning about
        # losing important information. Since we are only interested in its units at this point,
        # we may safely ignore it.
        warnings.simplefilter(action="ignore", category=UserWarning)
        try:
            units = (crs.proj4_params.get('units')
            # As of cartopy 0.20 the proj4_params attribute is {} for CRS from an EPSG number
            # (see issue raised https://github.com/SciTools/cartopy/issues/1974
            # and longterm discussion on https://github.com/SciTools/cartopy/issues/813).
            # In these cases the units can be fetched through the method `to_dict`.
            or crs.to_dict().get('units', '°'))
        except AttributeError:
            # This happens in setups with cartopy<0.20, where `to_dict` is not defined.
            # Officially, we require cartopy>=0.20, but there are still users around that
            # can't upgrade due to https://github.com/SciTools/iris/issues/4468
            units = '°'

    return crs, units


def multibar_plot(ax, data, colors=None, total_width=0.8, single_width=1,
                  legend=True, ticklabels=None, invert_axis=False):
    """
    Draws a bar plot with multiple bars per data point.
    https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x": [1, 2, 3],
            "y": [1, 2, 3],
            "z": [1, 2, 3],
        }
        fig, ax = plt.subplots()
        multibar_plot(ax, data, xticklabels=["a", "b", "c"])


    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.

    ticklabels: list, optional, default: None
        labels of the xticks (yticks if invert_axis=True)

    invert_axis: boolean, default: False
        Invert the x and y axis. By default, the bars are vertical.
        invert_axis=True gives horizontal bars.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (_name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            if invert_axis:
                lbar = ax.barh(x + x_offset, width=y, height=bar_width * single_width,
                              color=colors[i % len(colors)])
            else:
                lbar = ax.bar(x + x_offset, y, width=bar_width * single_width,
                             color=colors[i % len(colors)])

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(lbar[0])

    if ticklabels:
        if invert_axis:
            plt.setp(ax, yticks=range(len(data)), yticklabels=ticklabels)
        else:
            plt.setp(ax, xticks=range(len(data)), xticklabels=ticklabels)

    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())
