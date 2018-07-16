"""
Define auxiliary functions for plots.
"""

__all__ = ['Graph2D',
           'geo_bin_from_array',
           'geo_im_from_array',
           'make_map',
           'add_shapes'
          ]

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from climada.util.files_handler import to_list


# Number of pixels in one direction in rendered image
RESOLUTION = 250
# Degrees to add in the border
BUFFER_DEG = 1.0
# Maximum number of bins in geo_bin_from_array
MAX_BINS = 200

def geo_bin_from_array(array_sub, geo_coord, var_name, title, pop_name=True,
                       **kwargs):
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
        kwargs (optional): arguments for hexbin matplotlib function

    Returns:
        matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    # Generate array of values used in each subplot
    num_im, list_arr = get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, geo_coord, 'geo_coord')

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Wistia'

    # Generate each subplot
    fig, axis_sub = make_map(num_im)
    for array_im, axis, tit, name, coord in \
    zip(list_arr, axis_sub.flatten(), list_tit, list_name, list_coord):
        if coord.shape[0] != array_im.size:
            raise ValueError("Size mismatch in input array: %s != %s." % \
                             (coord.shape[0], array_im.size))
        # Binned image with coastlines
        extent = get_borders(coord)
        extent = ([extent[0] - BUFFER_DEG, extent[1] + BUFFER_DEG, extent[2] -\
                   BUFFER_DEG, extent[3] + BUFFER_DEG])
        axis.set_extent((extent))
        add_shapes(axis)
        if pop_name:
            add_populated(axis, extent)

        if 'gridsize' not in kwargs:
            kwargs['gridsize'] = min(int(array_im.size/2), MAX_BINS)
        hex_bin = axis.hexbin(coord[:, 1], coord[:, 0], C=array_im, \
            transform=ccrs.PlateCarree(), **kwargs)

        # Create colorbar in this axis
        cbax = make_axes_locatable(axis).append_axes('right', size="6.5%", \
            pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar(hex_bin, cax=cbax, orientation='vertical')
        cbar.set_label(name)
        axis.set_title(tit)

    return fig, axis_sub

def geo_im_from_array(array_sub, geo_coord, var_name, title, **kwargs):
    """Image(s) plot defined in array(s) over input coordinates.

    Parameters:
        array_sub (np.array(1d or 2d) or list(np.array)): Each array (in a row
            or in  the list) are values at each point in corresponding
            geo_coord that are ploted in one subplot.
        geo_coord (2d np.array or list(2d np.array)): (lat, lon) for each
            point in a row. If one provided, the same grid is used for all
            subplots. Otherwise provide as many as subplots in array_sub.
        var_name (str or list(str)): label to be shown in the colorbar. If one
            provided, the same is used for all subplots. Otherwise provide as
            many as subplots in array_sub.
        title (str or list(str)): subplot title. If one provided, the same is
            used for all subplots. Otherwise provide as many as subplots in
            array_sub.
        kwargs (optional): arguments for pcolormesh matplotlib function

    Returns:
        matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    # Generate array of values used in each subplot
    num_im, list_arr = get_collection_arrays(array_sub)
    list_tit = to_list(num_im, title, 'title')
    list_name = to_list(num_im, var_name, 'var_name')
    list_coord = to_list(num_im, geo_coord, 'geo_coord')

    # Generate each subplot
    fig, axis_sub = make_map(num_im)
    for array_im, axis, tit, name, coord in \
    zip(list_arr, axis_sub.flatten(), list_tit, list_name, list_coord):
        if coord.shape[0] != array_im.size:
            raise ValueError("Size mismatch in input array: %s != %s." % \
                             (coord.shape[0], array_im.size))
        # Create regular grid where to interpolate the array
        extent = get_borders(coord)
        grid_x, grid_y = np.mgrid[
            extent[0] : extent[1] : complex(0, RESOLUTION),
            extent[2] : extent[3] : complex(0, RESOLUTION)]
        grid_im = griddata((coord[:, 1], coord[:, 0]), array_im, \
                           (grid_x, grid_y))

        # Add coastline to axis
        axis.set_extent((extent))
        add_shapes(axis)
        # Create colormesh, colorbar and labels in axis
        cbax = make_axes_locatable(axis).append_axes('right', size="6.5%", \
            pad=0.1, axes_class=plt.Axes)
        cbar = plt.colorbar( \
                axis.pcolormesh(grid_x, grid_y, np.squeeze(grid_im), **kwargs),
                cax=cbax, orientation='vertical')
        cbar.set_label(name)
        axis.set_title(tit)

    return fig, axis_sub

class Graph2D(object):
    """2D graph object. Handles various subplots and curves."""
    def __init__(self, title='', num_subplots=1, num_row=None, num_col=None):

        if (num_row is None) or (num_col is None):
            self.num_row, self.num_col = get_row_col_size(num_subplots)
        else:
            self.num_row = num_row
            self.num_col = num_col
        self.fig = plt.figure(figsize=plt.figaspect(self.num_row/self.num_col))
        self.fig.suptitle(title)
        self.curr_ax = -1
        self.axs = []
        if self.num_col > 1:
            self.fig.subplots_adjust(wspace=0.3)
        if self.num_row > 1:
            self.fig.subplots_adjust(hspace=0.7)

    def add_subplot(self, xlabel, ylabel, title='', on_grid=True):
        """Add subplot to figure."""
        self.curr_ax += 1
        if self.curr_ax >= self.num_col * self.num_row:
            raise ValueError("Number of subplots in figure exceeded. Figure " \
                             "contains only %s subplots." % str(self.num_col +\
                             self.num_row))
        new_ax = self.fig.add_subplot(self.num_row, self.num_col, \
                                      self.curr_ax + 1)
        new_ax.set_xlabel(xlabel)
        new_ax.set_ylabel(ylabel)
        new_ax.set_title(title)
        new_ax.grid(on_grid)
        self.axs.append(new_ax)

    def add_curve(self, var_x, var_y, fmt=None, ax_num=None, **kwargs):
        """Add (x, y) curve to current subplot.

        Parameters:
            var_x (array): abcissa values
            var_y (array): ordinate values
            fmt (str, optional): format e.g 'k--'
            ax_num (int, optional): number of axis to plot. Current if None.
            kwargs (optional): arguments for plot matplotlib function
        """
        if self.curr_ax == -1:
            raise ValueError('Add a subplot first!')
        if ax_num is None:
            axis = self.axs[self.curr_ax]
        else:
            axis = self.axs[ax_num]
        if fmt is not None:
            axis.plot(var_x, var_y, fmt, **kwargs)
        else:
            axis.plot(var_x, var_y, **kwargs)
        if 'label' in kwargs:
            axis.legend(loc='best')
        axis.grid(True)

    def set_x_lim(self, var_x, ax_num=None):
        """Set x axis limits from minimum and maximum provided values.

        Parameters:
            var_x (array): abcissa values
            ax_num (int, optional): number of axis to plot. Current if None.
        """
        if ax_num is None:
            axis = self.axs[self.curr_ax]
        else:
            axis = self.axs[ax_num]
        axis.set_xlim([np.min(var_x), np.max(var_x)])

    def set_y_lim(self, var_y, ax_num=None):
        """Set y axis limits from minimum and maximum provided values.

        Parameters:
            var_y (array): ordinate values
            ax_num (int, optional): number of axis to plot. Current if None.
        """
        if ax_num is None:
            axis = self.axs[self.curr_ax]
        else:
            axis = self.axs[ax_num]
        axis.set_ylim([np.min(var_y), np.max(var_y)])

    def get_elems(self):
        """Return figure and list of all axes (of each subplot).

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        return self.fig, self.axs

def get_collection_arrays(array_sub):
    """ Get number of array rows and generate list of array if only one row

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

def make_map(num_sub=1, projection=ccrs.PlateCarree()):
    """Create map figure with cartopy.

    Parameters:
        num_sub (int): number of subplots in figure.
        projection (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.

    Returns:
        matplotlib.figure.Figure, np.array(cartopy.mpl.geoaxes.GeoAxesSubplot)
    """
    num_row, num_col = get_row_col_size(num_sub)
    fig, axis_sub = plt.subplots(num_row, num_col, figsize=(9, 13), \
                        subplot_kw=dict(projection=projection), squeeze=False)

    if not isinstance(axis_sub, np.ndarray):
        axis_sub = np.array([[axis_sub]])

    for axis in axis_sub.flatten():
        grid = axis.gridlines(draw_labels=True)
        grid.xlabels_top = grid.ylabels_right = False
        grid.xformatter = LONGITUDE_FORMATTER
        grid.yformatter = LATITUDE_FORMATTER

    fig.tight_layout()
    if num_col > 1:
        fig.subplots_adjust(wspace=0.3)
    if num_row > 1:
        fig.subplots_adjust(hspace=-0.5)

    return fig, axis_sub

def add_shapes(axis, projection=ccrs.PlateCarree()):
    """Overlay Earth's countries coastlines to matplotlib.pyplot axis.

    Parameters:
        axis (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy axis.
        projection (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.
    """
    shp_file = shapereader.natural_earth(resolution='10m', \
                category='cultural', name='admin_0_countries')
    shp = shapereader.Reader(shp_file)
    for geometry in shp.geometries():
        axis.add_geometries([geometry], projection, facecolor='', \
                            edgecolor='black')

def add_populated(axis, extent, projection=ccrs.PlateCarree()):
    """Add city names.

    Parameters:
        axis (cartopy.mpl.geoaxes.GeoAxesSubplot): cartopy axis.
        extent (): geographical limits.
        projection (cartopy.crs projection, optional): geographical projection,
            PlateCarree default.

    """
    shp_file = shapereader.natural_earth(resolution='110m', \
                           category='cultural', name='populated_places_simple')

    shp = shapereader.Reader(shp_file)
    cnt = 0
    for rec, point in zip(shp.records(), shp.geometries()):
        cnt += 1
        if (point.x <= extent[1]) and (point.x > extent[0]):
            if (point.y <= extent[3]) and (point.y > extent[2]):
                axis.plot(point.x, point.y, 'ko', markersize=7, \
                          transform=projection)
                axis.text(point.x, point.y, rec.attributes['name'], \
                    horizontalalignment='right', verticalalignment='bottom', \
                    transform=projection, fontsize=14)

def get_row_col_size(num_sub):
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
            num_row = int(num_sub/3)
        else:
            num_col = 2
            num_row = int(num_sub/2) + num_sub % 2
    return num_row, num_col

def get_borders(geo_coord):
    """Get min and max longitude and min and max latitude (in this order).

    Parameters:
        geo_coord (2d np.array): (lat, lon) for each point in a row.

    Returns:
        np.array
    """
    min_lon = max(np.min(geo_coord[:, 1]), -180)
    max_lon = min(np.max(geo_coord[:, 1]), 180)
    min_lat = max(np.min(geo_coord[:, 0]), -90)
    max_lat = min(np.max(geo_coord[:, 0]), 90)
    return [min_lon, max_lon, min_lat, max_lat]
