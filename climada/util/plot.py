"""
Define auxiliary functions for plots.
"""

__all__ = ['Graph2D',
           'geo_bin_from_array',
           'geo_im_from_array'
          ]

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# Number of pixels in one direction in rendered image
RESOLUTION = 250
# Degrees to add in the border
BUFFER_DEG = 1.0

def make_map(projection=ccrs.PlateCarree()):
    """Create map figure with cartopy."""
    fig, axis = plt.subplots(figsize=(9, 13), \
                             subplot_kw=dict(projection=projection))
    grid = axis.gridlines(draw_labels=True)
    grid.xlabels_top = grid.ylabels_right = False
    grid.xformatter = LONGITUDE_FORMATTER
    grid.yformatter = LATITUDE_FORMATTER
    return fig, axis

def add_shapes(axis, projection=ccrs.PlateCarree()):
    """Overlay Earth's countries coastlines to matplotlib.pyplot axis."""
    shp_file = shapereader.natural_earth(resolution='10m', \
                category='cultural', name='admin_0_countries')
    shp = shapereader.Reader(shp_file)
    for geometry in shp.geometries():
        axis.add_geometries([geometry], projection, facecolor='', \
                            edgecolor='black')

def add_populated(axis, extent, projection=ccrs.PlateCarree()):
    """Add cities names."""
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
        
def geo_bin_from_array(geo_coord, array_val, var_name, title):
    """Plot array values binned over input coordinates.

    Parameters:
        geo_coord (2d np.array): (lat, lon) for each point
        array_val (np.array): values at each point in geo_coord that are
            binned
        var_name (str): label to be shown in the colorbar
        title (str): figure title

    Returns:
        matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    if geo_coord.shape[0] != array_val.size:
        raise ValueError("Size mismatch in input array: %s != %s." % \
                         (geo_coord.shape[0], array_val.size))

    extent = get_borders(geo_coord)
    extent = ([extent[0] - BUFFER_DEG, extent[1] + BUFFER_DEG, extent[2] - \
               BUFFER_DEG, extent[3] + BUFFER_DEG])
    fig, axis = make_map()
    axis.set_extent((extent))
    add_shapes(axis)    
    add_populated(axis, extent)

    hex_bin = axis.hexbin(geo_coord[:, 1], geo_coord[:, 0], C=array_val, \
        cmap='Wistia', gridsize=int(array_val.size/2), \
        transform=ccrs.PlateCarree())

    # Create colorbar in this axes  
    divider = make_axes_locatable(axis)
    cbax = divider.append_axes('right', size="6.5%", pad=0.1, \
                               axes_class=plt.Axes)
    cbar = plt.colorbar(hex_bin, cax=cbax, orientation='vertical')
    cbar.set_label(var_name)
    axis.set_title(title)
    return fig, axis

def geo_im_from_array(geo_coord, array_im, var_name, title):
    """Image plot defined in array over input coordinates.

    Parameters:
        geo_coord (2d np.array): (lat, lon) for each point
        array_im (np.array): values at each point in geo_coord that are
            ploted
        var_name (str): label to be shown in the colorbar
        title (str): figure title

    Returns:
        matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot

    Raises:
        ValueError
    """
    if geo_coord.shape[0] != array_im.size:
        raise ValueError("Size mismatch in input array: %s != %s." % \
                         (geo_coord.shape[0], array_im.size))
    # Create regular grid where to interpolate the array
    extent = get_borders(geo_coord)
    grid_x, grid_y = np.mgrid[extent[0] : extent[1] : complex(0, RESOLUTION), \
                              extent[2] : extent[3] : complex(0, RESOLUTION)]
    grid_im = griddata((geo_coord[:, 1], geo_coord[:, 0]), array_im, \
                             (grid_x, grid_y))
    grid_im = np.squeeze(grid_im)

    # Colormesh with coastline
    fig, axis = make_map()
    axis.set_extent((extent))
    add_shapes(axis)       
    col_mesh = axis.pcolormesh(grid_x, grid_y, grid_im)  
    
    # Create colorbar in this axes  
    divider = make_axes_locatable(axis)
    cbax = divider.append_axes('right', size="6.5%", pad=0.1, \
                               axes_class=plt.Axes)
    cbar = plt.colorbar(col_mesh, cax=cbax, orientation='vertical')
    cbar.set_label(var_name)
    axis.set_title(title)
    return fig, axis

class Graph2D(object):
    """2D graph object. Handles various subplots and curves."""
    def __init__(self, title='', num_subplots=1, num_row=None, num_col=None):

        if (num_row is None) or (num_col is None):
            self._compute_size(num_subplots)
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

    def add_curve(self, var_x, var_y, var_style, var_label=None):
        """Add (x, y) curve to current subplot."""
        if self.curr_ax == -1:
            raise ValueError('Add a subplot first!')
        axis = self.axs[self.curr_ax]
        axis.plot(var_x, var_y, var_style, label=var_label)
        if var_label is not None:
            axis.legend(loc='best')
        axis.grid(True)

    def set_x_lim(self, var_x):
        """Set x axis limits from minimum and maximum provided values."""
        axis = self.axs[self.curr_ax]
        axis.set_xlim([np.min(var_x), np.max(var_x)])

    def set_y_lim(self, var_y):
        """Set y axis limits from minimum and maximum provided values."""
        axis = self.axs[self.curr_ax]
        axis.set_ylim([np.min(var_y), np.max(var_y)])

    def get_elems(self):
        """Return figure and list of all axes (of each subplot).

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        return self.fig, self.axs

    def _compute_size(self, num_sub):
        """Compute number of rows and columns of subplots in figure."""
        if num_sub <= 3:
            self.num_col = num_sub
            self.num_row = 1
        else:
            if num_sub % 3 == 0:
                self.num_col = 3
                self.num_row = int(num_sub/3)
            else:
                self.num_col = 2
                self.num_row = int(num_sub/2) + num_sub % 2

def get_borders(geo_coord):
    """Get min and max longitude and min and max latitude (in this order)."""
    return [np.min(geo_coord[:, 1]), np.max(geo_coord[:, 1]), \
        np.min(geo_coord[:, 0]), np.max(geo_coord[:, 0])]
